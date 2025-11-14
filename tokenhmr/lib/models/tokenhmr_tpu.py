"""SMPL-free TokenHMR variant for TPU/CPU inference."""

from typing import Dict

import pytorch_lightning as pl
import torch
from yacs.config import CfgNode

from ..utils.misc import load_pretrained
from ..utils.pylogger import get_pylogger
from ..configs import get_config
from .backbones import create_backbone
from .heads import build_smpl_head

log = get_pylogger(__name__)


class TokenHMRTPU(pl.LightningModule):
    def __init__(self, cfg: CfgNode, is_train_state: bool = False):
        super().__init__()
        if is_train_state:
            raise ValueError("TokenHMRTPU仅支持推理场景")

        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        self.backbone = create_backbone(cfg, load_weights=False)
        self.smpl_head = build_smpl_head(cfg)
        self.backbone, self.smpl_head = load_pretrained(
            cfg, self.backbone, self.smpl_head, is_train_state
        )

        self.register_buffer("initialized", torch.tensor(True), persistent=False)

    def forward_step(self, batch: Dict[str, torch.Tensor], train: bool = False) -> Dict:
        if train:
            raise ValueError("TokenHMRTPU不支持训练前向")

        imgs = batch["img"]
        batch_size = imgs.shape[0]

        conditioning_feats = self.backbone(imgs)
        pred_smpl_params, pred_cam, pred_smpl_params_list = self.smpl_head(
            conditioning_feats
        )

        device = imgs.device
        dtype = imgs.dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(
            batch_size, 2, device=device, dtype=dtype
        )
        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2
                * focal_length[:, 0]
                / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )

        output = {
            "pred_cam": pred_cam,
            "pred_cam_t": pred_cam_t,
            "pred_smpl_params": {k: v.clone() for k, v in pred_smpl_params.items()},
            "focal_length": focal_length,
        }
        if self.cfg.MODEL.SMPL_HEAD.TYPE == "token":
            output["cls_logits_softmax"] = pred_smpl_params_list["cls_logits_softmax"]
        return output

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.forward_step(batch, train=False)


def _prepare_model_cfg(
    model_cfg: str, checkpoint_path: str, dataset_dir: str
) -> CfgNode:
    cfg = get_config(model_cfg)
    cfg.defrost()
    cfg.ckpt_path = checkpoint_path
    if (cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in cfg.MODEL):
        assert cfg.MODEL.IMAGE_SIZE == 256, "ViT骨干要求IMAGE_SIZE=256"
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    if dataset_dir:
        cfg.DATASETS.DATASET_DIR = dataset_dir
    cfg.freeze()
    return cfg


def load_tokenhmr_tpu(
    checkpoint_path: str = "",
    model_cfg: str = "",
    dataset_dir: str = "",
) -> (TokenHMRTPU, CfgNode):
    model_cfg_node = _prepare_model_cfg(model_cfg, checkpoint_path, dataset_dir)
    model = TokenHMRTPU(cfg=model_cfg_node, is_train_state=False)
    return model, model_cfg_node


__all__ = ["TokenHMRTPU", "load_tokenhmr_tpu"]
