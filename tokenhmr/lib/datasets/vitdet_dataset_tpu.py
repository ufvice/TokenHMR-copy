"""TPU-friendly video dataset that mirrors ViTDetDataset preprocessing."""

from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch
from skimage.filters import gaussian
from yacs.config import CfgNode

from .utils import (
    convert_cvimg_to_tensor,
    expand_to_aspect_ratio,
    generate_image_patch_cv2,
)

DEFAULT_MEAN = 255.0 * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255.0 * np.array([0.229, 0.224, 0.225])


def _ensure_frame_list(frames: Sequence[np.ndarray]) -> List[np.ndarray]:
    if isinstance(frames, np.ndarray):
        if frames.ndim == 4:
            return [frames[i] for i in range(frames.shape[0])]
        raise ValueError("frames ndarray must be 4-D (T,H,W,C)")
    return [np.asarray(frame) for frame in frames]


class ViTDetDatasetTPU(torch.utils.data.Dataset):

    def __init__(
        self,
        cfg: CfgNode,
        frames: Sequence[np.ndarray],
        boxes: np.ndarray,
        frame_indices: Optional[Iterable[int]] = None,
        sequence_id: Optional[str] = None,
        train: bool = False,
        use_skimage_antialias: bool = True,
    ):
        super().__init__()
        assert train is False, "ViTDetDatasetTPU仅用于推理"

        self.cfg = cfg
        self.frames = _ensure_frame_list(frames)
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes需要形状[T,4]")
        if len(self.frames) != len(boxes):
            raise ValueError(f"帧数({len(self.frames)})与bbox数({len(boxes)})不一致")

        self.train = False
        self.sequence_id = sequence_id
        self.use_skimage_antialias = use_skimage_antialias
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)

        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        if frame_indices is None:
            self.frame_indices = np.arange(len(boxes), dtype=np.int64)
        else:
            frame_indices = np.asarray(list(frame_indices), dtype=np.int64)
            if len(frame_indices) != len(boxes):
                raise ValueError("frame_indices长度须与bbox一致")
            self.frame_indices = frame_indices

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        img_cv2 = self.frames[idx]
        if img_cv2.ndim != 3:
            raise ValueError("输入帧需为HWC图像")

        center = self.center[idx].copy()
        center_x, center_y = center
        scale = self.scale[idx]
        bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        bbox_size = expand_to_aspect_ratio(
            scale * 200, target_aspect_ratio=bbox_shape
        ).max()

        patch_width = patch_height = self.img_size
        cvimg = img_cv2.copy()
        if self.use_skimage_antialias:
            downsampling_factor = (bbox_size * 1.0) / patch_width
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(
                    cvimg,
                    sigma=(downsampling_factor - 1) / 2,
                    channel_axis=2,
                    preserve_range=True,
                )

        img_patch_cv, _ = generate_image_patch_cv2(
            cvimg,
            center_x,
            center_y,
            bbox_size,
            bbox_size,
            patch_width,
            patch_height,
            False,
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        for n_c in range(min(cvimg.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[
                n_c
            ]

        img_h, img_w = cvimg.shape[:2]
        item = {
            "img": img_patch,
            "personid": int(self.personid[idx]),
            "frame_index": int(self.frame_indices[idx]),
            "box_center": self.center[idx].copy(),
            "box_size": bbox_size,
            "img_size": np.array([img_w, img_h], dtype=np.float32),
        }
        if self.sequence_id is not None:
            item["sequence_id"] = self.sequence_id
        return item


__all__ = ["ViTDetDatasetTPU", "DEFAULT_MEAN", "DEFAULT_STD"]
