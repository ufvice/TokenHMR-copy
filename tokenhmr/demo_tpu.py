import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import tqdm

from lib.datasets.vitdet_dataset_tpu import ViTDetDatasetTPU
from lib.models.tokenhmr_tpu import load_tokenhmr_tpu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenHMR TPU/CPU 推理脚本")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重.pt路径")
    parser.add_argument("--model_config", type=str, required=True, help="模型配置yaml")
    parser.add_argument(
        "--video_root", type=str, required=True, help="存放视频的文件夹"
    )
    parser.add_argument(
        "--bbox_path", type=str, required=True, help="val_bbox风格的pt文件"
    )
    parser.add_argument("--output_path", type=str, required=True, help="输出pt文件路径")
    parser.add_argument(
        "--dataset_dir", type=str, default="", help="可选：数据目录，透传给Hydra配置"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="推理batch size")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="DataLoader worker数"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "xla"],
        help="推理设备选择，xla用于TPU",
    )
    parser.add_argument(
        "--video_exts",
        type=str,
        default="mp4,mov,avi,mkv",
        help="匹配视频文件的扩展名，逗号分隔",
    )
    parser.add_argument(
        "--sequence_list",
        type=str,
        default="",
        help="可选：只处理给定txt文件中的序列ID",
    )
    parser.add_argument(
        "--skip_missing_video",
        action="store_true",
        help="若某个序列缺视频则跳过而非报错",
    )
    parser.add_argument(
        "--intrinsic",
        type=str,
        default="",
        help="可选：3x3内参，提供npy/pt文件或9个数字",
    )
    parser.add_argument(
        "--extrinsic",
        type=str,
        default="",
        help="可选：4x4(或T,4,4)外参npy/pt文件，默认单位阵",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="显示tqdm进度条",
    )
    return parser.parse_args()


def _load_tensor_from_arg(arg: str, expected_shape: Tuple[int, ...]) -> torch.Tensor:
    if not arg:
        if len(expected_shape) == 2:
            return torch.eye(expected_shape[0], dtype=torch.float32)
        if len(expected_shape) == 3:
            base = torch.eye(expected_shape[1], dtype=torch.float32)
            return base.unsqueeze(0)
        raise ValueError("不支持的期望形状")

    path = Path(arg)
    if path.exists():
        if path.suffix in {".pt", ".pth"}:
            data = torch.load(path, map_location="cpu")
            tensor = torch.as_tensor(data, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(
                np.load(path, allow_pickle=True), dtype=torch.float32
            )
    else:
        values = [float(v) for v in arg.split(",") if v.strip()]
        tensor = torch.tensor(values, dtype=torch.float32)

    if len(expected_shape) == 2:
        if tensor.ndim == 1 and tensor.numel() == expected_shape[0] * expected_shape[1]:
            tensor = tensor.view(*expected_shape)
        if tensor.shape != expected_shape:
            raise ValueError(f"张量形状{tensor.shape}与期望{expected_shape}不符")
    elif len(expected_shape) == 3:
        spatial = expected_shape[1:]
        if tensor.ndim == 1 and tensor.numel() == spatial[0] * spatial[1]:
            tensor = tensor.view(*spatial)
        if tensor.ndim == 2 and tensor.shape == spatial:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[1:] != spatial:
            raise ValueError(f"张量形状{tensor.shape}与期望{expected_shape}不符")
    else:
        raise ValueError("仅支持2维或3维期望形状")
    return tensor.float()


def _to_intrinsic(arg: str) -> torch.Tensor:
    tensor = _load_tensor_from_arg(arg, (3, 3))
    if tensor.ndim != 2:
        raise ValueError("intrinsic必须是3x3")
    return tensor


def _to_extrinsic(arg: str) -> torch.Tensor:
    tensor = _load_tensor_from_arg(arg, (1, 4, 4))
    if tensor.ndim not in (2, 3):
        raise ValueError("extrinsic必须是4x4或(T,4,4)")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def _rotmat_to_aa(rotmats: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为轴角表示。
    输入可以是 (..., 3, 3)，输出为 (..., 3)。
    """
    orig_shape = rotmats.shape[:-2]
    rotmats_flat = rotmats.reshape(-1, 3, 3)

    # 按标准公式从旋转矩阵恢复轴角
    trace = rotmats_flat[:, 0, 0] + rotmats_flat[:, 1, 1] + rotmats_flat[:, 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    sin_theta = torch.sin(theta)
    kx = (rotmats_flat[:, 2, 1] - rotmats_flat[:, 1, 2]) / (2.0 * sin_theta)
    ky = (rotmats_flat[:, 0, 2] - rotmats_flat[:, 2, 0]) / (2.0 * sin_theta)
    kz = (rotmats_flat[:, 1, 0] - rotmats_flat[:, 0, 1]) / (2.0 * sin_theta)

    axis = torch.stack([kx, ky, kz], dim=-1)

    # 对于非常小的旋转角，退化为零向量，避免数值问题
    small_angle = sin_theta.abs() < 1e-4
    if small_angle.any():
        axis[small_angle] = 0.0
        theta = theta.clone()
        theta[small_angle] = 0.0

    aa = axis * theta.unsqueeze(-1)
    return aa.reshape(*orig_shape, 3)


def _resolve_device(requested: str) -> Tuple[torch.device, Optional[object]]:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), None
        try:
            import torch_xla.core.xla_model as xm  # type: ignore

            return xm.xla_device(), xm
        except Exception:
            return torch.device("cpu"), None
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        return torch.device("cuda"), None
    if requested == "cpu":
        return torch.device("cpu"), None
    if requested == "xla":
        import torch_xla.core.xla_model as xm  # type: ignore

        return xm.xla_device(), xm
    raise ValueError(f"未知设备{requested}")


def _read_sequence_list(path: str) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _find_video(video_root: Path, key: str, extensions: Sequence[str]) -> Path:
    for ext in extensions:
        candidate = video_root / f"{key}.{ext}"
        if candidate.exists():
            return candidate
    matches = list(video_root.glob(f"**/{key}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"未找到序列{key}对应视频")


def _read_video_frames(video_path: Path) -> List:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"{video_path}无有效帧")
    return frames


def _finalize_sequence_smplx_like(
    *,
    num_frames: int,
    per_frame_global_orient_aa: torch.Tensor,
    per_frame_body_pose_aa: torch.Tensor,
    per_frame_betas: torch.Tensor,
    per_frame_transl_c: torch.Tensor,
) -> Dict:
    """
    将单个序列整理为 tc3d-eval 期望的预测格式：
    {seq_name: {'smplx_data_c': {...}, 'smplx_data_w': {...}}}

    其中 body_pose 采用 21 个 body joints 的轴角展平后形状 [T,63]。
    """
    if num_frames == 0:
        raise RuntimeError("序列帧数不能为0")

    # 仅保留前 21 个 body joints，对应 63 维轴角
    if per_frame_body_pose_aa.ndim != 3:
        raise ValueError("per_frame_body_pose_aa 期望形状为[T, J, 3]")
    body_joints = per_frame_body_pose_aa.shape[1]
    target_joints = 21
    if body_joints < target_joints:
        raise ValueError(
            f"SMPL 关节数不足，期望至少 {target_joints} 个，实际为 {body_joints}"
        )

    body_pose_21 = per_frame_body_pose_aa[:, :target_joints, :]  # [T,21,3]
    body_pose_flat = body_pose_21.reshape(num_frames, target_joints * 3)  # [T,63]

    smplx_data_c = {
        "body_pose": body_pose_flat.clone(),
        "betas": per_frame_betas.clone(),
        "global_orient": per_frame_global_orient_aa.clone(),
        "transl": per_frame_transl_c.clone(),
    }

    # 当前没有精确的 world 坐标外参信息，先直接复制一份用于全局指标计算
    smplx_data_w = {
        "body_pose": body_pose_flat.clone(),
        "betas": per_frame_betas.clone(),
        "global_orient": per_frame_global_orient_aa.clone(),
        "transl": per_frame_transl_c.clone(),
    }

    return {
        "smplx_data_c": smplx_data_c,
        "smplx_data_w": smplx_data_w,
    }


def main():
    args = _parse_args()
    video_root = Path(args.video_root)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intrinsic = _to_intrinsic(args.intrinsic)
    extrinsic_template = _to_extrinsic(args.extrinsic)

    model, model_cfg = load_tokenhmr_tpu(
        checkpoint_path=args.checkpoint,
        model_cfg=args.model_config,
        dataset_dir=args.dataset_dir,
    )
    device, xm = _resolve_device(args.device)
    model = model.to(device)
    model.eval()

    bbox_dict = torch.load(args.bbox_path, map_location="cpu")
    sequence_filter = _read_sequence_list(args.sequence_list)
    extensions = [
        ext.strip().lstrip(".") for ext in args.video_exts.split(",") if ext.strip()
    ]

    all_keys = sorted(bbox_dict.keys())
    if sequence_filter is not None:
        sequence_set = set(sequence_filter)
        all_keys = [k for k in all_keys if k in sequence_set]

    results: Dict[str, Dict] = {}
    iterator = all_keys
    if args.progress:
        iterator = tqdm.tqdm(iterator, desc="TPU推理")

    for key in iterator:
        try:
            video_path = _find_video(video_root, key, extensions)
        except FileNotFoundError as exc:
            if args.skip_missing_video:
                print(f"[警告] {exc}，已跳过")
                continue
            raise

        frames = _read_video_frames(video_path)
        boxes = torch.as_tensor(bbox_dict[key], dtype=torch.float32).cpu().numpy()
        frame_count = len(frames)
        if len(boxes) != frame_count:
            min_len = min(len(boxes), frame_count)
            print(
                f"[提示] 序列{key}帧数与bbox不符({frame_count}!={len(boxes)}), 仅保留前{min_len}帧"
            )
            frames = frames[:min_len]
            boxes = boxes[:min_len]
            frame_count = min_len

        # 1) 根据 bbox 过滤掉全 0 的帧，这些帧视为“无人”，不做模型推理
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"序列{key}的bbox形状异常，期望[T,4]，实际为{boxes.shape}")
        zero_mask = np.all(boxes == 0.0, axis=1)
        valid_mask = ~zero_mask
        valid_indices = np.nonzero(valid_mask)[0]

        # 为整个序列预先分配 SMPLX 参数缓存（在 CPU 上）
        num_frames = frame_count
        target_joints = 21
        betas_dim = 10
        per_frame_global_orient_aa = torch.zeros((num_frames, 3), dtype=torch.float32)
        per_frame_body_pose_aa = torch.zeros(
            (num_frames, target_joints, 3), dtype=torch.float32
        )
        per_frame_betas = torch.zeros((num_frames, betas_dim), dtype=torch.float32)
        per_frame_transl_c = torch.zeros((num_frames, 3), dtype=torch.float32)

        if valid_indices.size == 0:
            # 整个序列都无人：直接写入全 0 姿态，占位以通过评估流程
            print(f"[提示] 序列{key}所有bbox为0，跳过模型推理，仅输出占位姿态")
            results[key] = _finalize_sequence_smplx_like(
                num_frames=num_frames,
                per_frame_global_orient_aa=per_frame_global_orient_aa,
                per_frame_body_pose_aa=per_frame_body_pose_aa,
                per_frame_betas=per_frame_betas,
                per_frame_transl_c=per_frame_transl_c,
            )
            continue

        frames_valid = [frames[i] for i in valid_indices]
        boxes_valid = boxes[valid_indices]

        dataset = ViTDetDatasetTPU(
            model_cfg,
            frames=frames_valid,
            boxes=boxes_valid,
            frame_indices=valid_indices,
            sequence_id=key,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )

        for batch in dataloader:
            imgs = batch["img"].to(device)
            batch_meta = {k: v for k, v in batch.items() if k != "img"}
            with torch.no_grad():
                out = model({"img": imgs})
            if xm is not None and device.type == "xla":
                xm.mark_step()

            smpl_params = out["pred_smpl_params"]
            global_orient = smpl_params["global_orient"].detach().cpu()  # [B,1,3,3]
            body_pose = smpl_params["body_pose"].detach().cpu()  # [B,J,3,3]
            betas = smpl_params["betas"].detach().cpu()  # [B,10]
            pred_cam_t = out["pred_cam_t"].detach().cpu()  # [B,3]

            # 将旋转矩阵转为轴角，并裁剪为前 21 个 body joints
            B = global_orient.shape[0]
            go_mat = global_orient[:, 0]  # [B,3,3]
            go_aa = _rotmat_to_aa(go_mat)  # [B,3]

            J = body_pose.shape[1]
            body_pose_mat = body_pose.view(B * J, 3, 3)
            body_pose_aa_all = _rotmat_to_aa(body_pose_mat).view(B, J, 3)
            if J < target_joints:
                raise ValueError(
                    f"SMPL body joints 数不足，期望至少 {target_joints}，实际为 {J}"
                )
            body_pose_aa_21 = body_pose_aa_all[:, :target_joints, :]  # [B,21,3]

            for i in range(B):
                frame_index = int(batch_meta["frame_index"][i])
                if frame_index < 0 or frame_index >= num_frames:
                    raise IndexError(
                        f"frame_index 超出范围: {frame_index} / {num_frames}"
                    )
                per_frame_global_orient_aa[frame_index] = go_aa[i]
                per_frame_body_pose_aa[frame_index] = body_pose_aa_21[i]
                per_frame_betas[frame_index] = betas[i]
                per_frame_transl_c[frame_index] = pred_cam_t[i]

        results[key] = _finalize_sequence_smplx_like(
            num_frames=num_frames,
            per_frame_global_orient_aa=per_frame_global_orient_aa,
            per_frame_body_pose_aa=per_frame_body_pose_aa,
            per_frame_betas=per_frame_betas,
            per_frame_transl_c=per_frame_transl_c,
        )

    torch.save(results, output_path)
    print(f"已保存推理结果到 {output_path}")


if __name__ == "__main__":
    main()
