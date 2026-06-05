import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _read_image_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Cannot read image data from: {path}")
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"cv2.imdecode failed for: {path}")
    return image


def _postprocess_mask(mask: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
    out = mask.astype(np.uint8)
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


class PairChangeSegBackend:
    """End-to-end pair-change segmentation backend based on MobileSAM."""

    def __init__(
        self,
        checkpoint_path: Path,
        backbone_checkpoint: Path,
        device: str = "cuda:0",
        decoder_arch: str = "auto",
        inference_mode: str = "sliding",
        crop_size: int = 1536,
        stride: int = 768,
        threshold: float = 0.5,
        open_kernel: int = 3,
        close_kernel: int = 5,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.backbone_checkpoint = Path(backbone_checkpoint).resolve()
        self.decoder_arch = str(decoder_arch).strip() or "auto"
        self.inference_mode = str(inference_mode).strip() or "sliding"
        self.crop_size = int(crop_size)
        self.stride = int(stride)
        self.threshold = float(threshold)
        self.open_kernel = int(open_kernel)
        self.close_kernel = int(close_kernel)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"pair_change_checkpoint not found: {self.checkpoint_path}")
        if not self.backbone_checkpoint.exists():
            raise FileNotFoundError(f"backbone checkpoint not found: {self.backbone_checkpoint}")

        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from tools.predict_mobile_sam_pair_change_seg import PairChangeSegModel, predict_probability

        self._predict_probability = predict_probability

        device_arg = str(device).strip()
        self.device = torch.device(
            device_arg if torch.cuda.is_available() or not device_arg.startswith("cuda") else "cpu"
        )
        state = torch.load(str(self.checkpoint_path), map_location=self.device)
        resolved_arch = self._resolve_decoder_arch(state)
        self.decoder_arch = resolved_arch

        model = PairChangeSegModel(
            backbone_checkpoint=self.backbone_checkpoint,
            decoder_arch=resolved_arch,
        ).to(self.device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"], strict=False)
        elif isinstance(state, dict) and "image_encoder.patch_embed.seq.0.c.weight" in state:
            model.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=False)
        model.eval()
        self.model = model

    def _resolve_decoder_arch(self, state: Any) -> str:
        if self.decoder_arch != "auto":
            return self.decoder_arch
        if isinstance(state, dict) and isinstance(state.get("decoder_arch"), str):
            return str(state["decoder_arch"]).strip() or "aspp_se"
        if isinstance(state, dict) and isinstance(state.get("summary"), dict):
            return str(state["summary"].get("decoder_arch", "aspp_se")).strip() or "aspp_se"
        return "simple"

    def __call__(self, base_path: str, current_path: str) -> np.ndarray:
        base_bgr = _read_image_unicode(Path(base_path), cv2.IMREAD_COLOR)
        current_bgr = _read_image_unicode(Path(current_path), cv2.IMREAD_COLOR)
        if base_bgr.shape[:2] != current_bgr.shape[:2]:
            base_bgr = cv2.resize(
                base_bgr,
                (current_bgr.shape[1], current_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
        current_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)

        prob = self._predict_probability(
            model=self.model,
            base_rgb=base_rgb,
            current_rgb=current_rgb,
            device=self.device,
            inference_mode=self.inference_mode,
            crop_size=self.crop_size,
            stride=self.stride,
        )
        raw_mask = (prob > self.threshold).astype(np.uint8)
        return _postprocess_mask(raw_mask, self.open_kernel, self.close_kernel)
