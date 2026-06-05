import sys
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import torch
import torch.nn as nn


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


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, kernel_size=3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(16, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ASPPBlock(nn.Module):
    def __init__(self, channels: int, rates: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        branch_channels = channels // 2
        self.branches = nn.ModuleList(
            [
                ConvNormAct(
                    channels,
                    branch_channels,
                    kernel_size=1 if rate == 1 else 3,
                    padding=0 if rate == 1 else rate,
                    dilation=rate,
                )
                for rate in rates
            ]
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, kernel_size=1),
            nn.GELU(),
        )
        self.project = ConvNormAct(branch_channels * (len(rates) + 1), channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        context = self.global_branch(x)
        features.append(torch.nn.functional.interpolate(context, size=size, mode="bilinear", align_corners=False))
        return self.project(torch.cat(features, dim=1))


class UpRefineBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            ResidualBlock(out_channels),
            SEBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


class ChangeDecoder(nn.Module):
    def __init__(self, in_channels: int = 256, hidden_channels: int = 256) -> None:
        super().__init__()
        fusion_channels = in_channels * 4
        self.reduce = nn.Sequential(
            nn.Conv2d(fusion_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.GELU(),
        )
        self.head = nn.Conv2d(hidden_channels // 4, 1, kernel_size=1)

    def forward(self, base_feat: torch.Tensor, current_feat: torch.Tensor) -> torch.Tensor:
        diff = current_feat - base_feat
        fusion = torch.cat([base_feat, current_feat, torch.abs(diff), diff], dim=1)
        x = self.reduce(fusion)
        x = self.block1(x)
        x = self.up1(x)
        x = self.block2(x)
        x = self.up2(x)
        return self.head(x)


class ASPPSEChangeDecoder(nn.Module):
    def __init__(self, in_channels: int = 256, hidden_channels: int = 256) -> None:
        super().__init__()
        fusion_channels = in_channels * 5
        self.reduce = ConvNormAct(fusion_channels, hidden_channels, kernel_size=1, padding=0)
        self.context = nn.Sequential(
            ASPPBlock(hidden_channels),
            SEBlock(hidden_channels),
            ResidualBlock(hidden_channels),
        )
        self.up1 = UpRefineBlock(hidden_channels, hidden_channels // 2)
        self.up2 = UpRefineBlock(hidden_channels // 2, hidden_channels // 4)
        self.head = nn.Sequential(
            ConvNormAct(hidden_channels // 4, hidden_channels // 4, kernel_size=3),
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=1),
        )

    def forward(self, base_feat: torch.Tensor, current_feat: torch.Tensor) -> torch.Tensor:
        diff = current_feat - base_feat
        product = base_feat * current_feat
        fusion = torch.cat([base_feat, current_feat, torch.abs(diff), diff, product], dim=1)
        x = self.reduce(fusion)
        x = self.context(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.head(x)


def build_change_decoder(decoder_arch: str, in_channels: int = 256, hidden_channels: int = 256) -> nn.Module:
    normalized = decoder_arch.strip().lower()
    if normalized in {"simple", "baseline"}:
        return ChangeDecoder(in_channels=in_channels, hidden_channels=hidden_channels)
    if normalized in {"aspp_se", "strong", "multiscale"}:
        return ASPPSEChangeDecoder(in_channels=in_channels, hidden_channels=hidden_channels)
    raise ValueError(f"Unsupported decoder architecture: {decoder_arch}")


class PairChangeSegModel(nn.Module):
    def __init__(self, backbone_checkpoint: Path, decoder_arch: str = "aspp_se") -> None:
        super().__init__()
        project_root = Path(__file__).resolve().parents[1]
        vendor_root = project_root / "src" / "vendor"
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        from mobile_sam import sam_model_registry

        sam = sam_model_registry["vit_t"](checkpoint=str(backbone_checkpoint))
        self.image_encoder = sam.image_encoder
        self.decoder_arch = decoder_arch.strip().lower()
        self.change_decoder = build_change_decoder(self.decoder_arch, in_channels=256, hidden_channels=256)
        self.register_buffer("pixel_mean", sam.pixel_mean.float().clone(), persistent=False)
        self.register_buffer("pixel_std", sam.pixel_std.float().clone(), persistent=False)
        self.img_size = int(getattr(self.image_encoder, "img_size", 1024))

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()
        return (tensor - self.pixel_mean) / self.pixel_std

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(self.preprocess(tensor))

    def forward(self, base_tensor: torch.Tensor, current_tensor: torch.Tensor) -> torch.Tensor:
        return self.change_decoder(self.encode(base_tensor), self.encode(current_tensor))


def _predict_pair_probability(
    model: PairChangeSegModel,
    base_rgb: np.ndarray,
    current_rgb: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    height, width = current_rgb.shape[:2]
    resized_base = cv2.resize(base_rgb, (model.img_size, model.img_size), interpolation=cv2.INTER_LINEAR)
    resized_current = cv2.resize(current_rgb, (model.img_size, model.img_size), interpolation=cv2.INTER_LINEAR)
    base_tensor = torch.from_numpy(resized_base).permute(2, 0, 1).contiguous().unsqueeze(0).to(device=device, dtype=torch.float32)
    current_tensor = torch.from_numpy(resized_current).permute(2, 0, 1).contiguous().unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(base_tensor, current_tensor)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)


def _window_starts(length: int, crop_size: int, stride: int) -> List[int]:
    if crop_size <= 0 or crop_size >= length:
        return [0]
    starts = list(range(0, max(1, length - crop_size + 1), max(1, stride)))
    last = length - crop_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_probability(
    model: PairChangeSegModel,
    base_rgb: np.ndarray,
    current_rgb: np.ndarray,
    device: torch.device,
    inference_mode: str,
    crop_size: int,
    stride: int,
) -> np.ndarray:
    if inference_mode == "full":
        return _predict_pair_probability(model, base_rgb, current_rgb, device)

    height, width = current_rgb.shape[:2]
    if crop_size <= 0:
        crop_size = min(height, width)
    if stride <= 0:
        stride = max(1, crop_size // 2)

    prob_full = np.zeros((height, width), dtype=np.float32)
    for y in _window_starts(height, crop_size, stride):
        for x in _window_starts(width, crop_size, stride):
            y2 = min(height, y + crop_size)
            x2 = min(width, x + crop_size)
            crop_prob = _predict_pair_probability(
                model=model,
                base_rgb=base_rgb[y:y2, x:x2],
                current_rgb=current_rgb[y:y2, x:x2],
                device=device,
            )
            prob_full[y:y2, x:x2] = np.maximum(prob_full[y:y2, x:x2], crop_prob)
    return prob_full


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

        prob = predict_probability(
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
