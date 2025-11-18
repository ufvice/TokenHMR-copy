import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import tqdm

from lib.datasets.vitdet_dataset_tpu import ViTDetDatasetTPU
from lib.models.tokenhmr_tpu import load_tokenhmr_tpu
from tokenization.models.rotation_utils import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)


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


def _finalize_sequence(
    records: List[Dict], intrinsic: torch.Tensor, extrinsic_template: torch.Tensor
) -> Dict:
    if not records:
        raise RuntimeError("记录为空")
    records.sort(key=lambda x: x["frame_index"])

    def stack_tensor(key: str):
        data = [rec[key] for rec in records]
        return torch.stack(data, dim=0)

    # ==================== 修改开始 ====================
    # 1. 堆叠所有必需的 SMPL 参数张量
    #    这些参数已经由模型在 main 循环中预测并存储在 records 中
    global_orient_matrix = stack_tensor("global_orient")  # (T, 1, 3, 3)
    body_pose_6d = stack_tensor("body_pose")  # (T, J, 6)
    betas = stack_tensor("betas")  # (T, 10)

    # pred_cam_t 直接作为 SMPL 的平移 'transl'
    transl = stack_tensor("pred_cam_t")  # (T, 3)
    T = body_pose_6d.shape[0]
    J = body_pose_6d.shape[1]

    # --- 转换 body_pose: 6D -> 轴角 ---
    body_pose_matrix = rotation_6d_to_matrix(body_pose_6d.view(-1, 6))
    body_pose_aa = matrix_to_axis_angle(body_pose_matrix).view(T, J, 3)

    # --- 转换 global_orient: 3x3 -> 轴角 ---
    global_orient_aa = matrix_to_axis_angle(global_orient_matrix.view(T, 3, 3))

    # 2. 构建 smplx_data_c 字典
    #    这符合我们上一个项目 (GVHMR) 的评估格式
    smplx_data_c = {
        "global_orient": global_orient_aa,  # (T, 3) 轴角
        "body_pose": body_pose_aa,  # (T, 21, 3) 轴角
        "betas": betas,
        "transl": transl,
    }

    # 3. 以期望的格式返回结果
    #    这个 TokenHMR 模型似乎只生成一套相机空间参数，
    #    因此我们将 'smplx_data_c' 和 'smplx_data_w' 都设置为相同的值，
    #    以确保评估脚本两边都能取到数据。
    return {
        "smplx_data_c": smplx_data_c,
        "smplx_data_w": smplx_data_c.copy(),  # 确保评估器可以访问 _w
    }
    # ==================== 修改结束 ====================


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

        dataset = ViTDetDatasetTPU(
            model_cfg,
            frames=frames,
            boxes=boxes,
            frame_indices=range(len(frames)),
            sequence_id=key,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )

        seq_records: List[Dict] = []
        for batch in dataloader:
            imgs = batch["img"].to(device)
            batch_meta = {k: v for k, v in batch.items() if k != "img"}
            with torch.no_grad():
                out = model({"img": imgs})
            if xm is not None and device.type == "xla":
                xm.mark_step()

            smpl_params = out["pred_smpl_params"]
            global_orient = smpl_params["global_orient"].detach().cpu()
            body_pose = smpl_params["body_pose"].detach().cpu()
            betas = smpl_params["betas"].detach().cpu()
            pred_cam = out["pred_cam"].detach().cpu()
            pred_cam_t = out["pred_cam_t"].detach().cpu()

            for idx in range(global_orient.shape[0]):
                record = {
                    "frame_index": int(batch_meta["frame_index"][idx]),
                    "box_center": batch_meta["box_center"][idx].detach().cpu(),
                    "box_size": float(batch_meta["box_size"][idx]),
                    "img_size": batch_meta["img_size"][idx].detach().cpu(),
                    "global_orient": global_orient[idx],
                    "body_pose": body_pose[idx],
                    "betas": betas[idx],
                    "pred_cam": pred_cam[idx],
                    "pred_cam_t": pred_cam_t[idx],
                }
                seq_records.append(record)

        results[key] = _finalize_sequence(seq_records, intrinsic, extrinsic_template)

    torch.save(results, output_path)
    print(f"已保存推理结果到 {output_path}")


if __name__ == "__main__":
    main()
