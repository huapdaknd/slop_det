import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from .roi_utils import build_box_mask, compute_crop_box_xyxy, remap_box_to_resized_crop
except ImportError:
    try:
        from roi_utils import build_box_mask, compute_crop_box_xyxy, remap_box_to_resized_crop
    except ImportError:
        from src.roi_utils import build_box_mask, compute_crop_box_xyxy, remap_box_to_resized_crop


CLASS_NAMES = [
    "rock_fall",
    "rock_spalling",
    "leaves",
    "slope_plant",
    "other_plant",
    "car",
    "change_color",
    "other",
]
BINARY_CLASS_NAMES = ["no_change", "change"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
DEFAULT_MOBILE_SAM_CKPT = str(Path(__file__).resolve().parents[2] / "models" / "backbones" / "mobile_sam.pt")
DEFAULT_SAM_VIT_H_CKPT = str(Path(__file__).resolve().parents[2] / "models" / "backbones" / "sam_vit_h_4b8939.pth")


def normalize_roi_view_mode(view_mode: str) -> str:
    mode = str(view_mode or "full_image").strip().lower()
    return mode if mode in {"full_image", "crop"} else "full_image"


def roi_pool_feature(
    feat: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    input_size: int,
    roi_size: int,
) -> torch.Tensor:
    batch, _, feat_h, feat_w = feat.shape
    pooled = []
    for i in range(batch):
        x1, y1, x2, y2 = boxes_xyxy[i]
        fx1 = int(torch.floor(x1 * feat_w / input_size).item())
        fy1 = int(torch.floor(y1 * feat_h / input_size).item())
        fx2 = int(torch.ceil(x2 * feat_w / input_size).item())
        fy2 = int(torch.ceil(y2 * feat_h / input_size).item())

        fx1 = max(0, min(fx1, feat_w - 1))
        fy1 = max(0, min(fy1, feat_h - 1))
        fx2 = max(fx1 + 1, min(fx2, feat_w))
        fy2 = max(fy1 + 1, min(fy2, feat_h))

        region = feat[i : i + 1, :, fy1:fy2, fx1:fx2]
        pooled.append(F.adaptive_max_pool2d(region, (roi_size, roi_size)))
    return torch.cat(pooled, dim=0)


class PairRoiClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        roi_size: int = 7,
        use_mask: bool = True,
        pretrained: bool = False,
        encoder_backbone: str = "mobile_sam_vit_t",
        encoder_checkpoint: str = DEFAULT_MOBILE_SAM_CKPT,
    ) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.use_mask = use_mask
        self.encoder_backbone = str(encoder_backbone).lower()
        self.encoder_input_size = 0

        if self.encoder_backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            channels = 512
            self.register_buffer("encoder_mean", IMAGENET_MEAN.clone(), persistent=False)
            self.register_buffer("encoder_std", IMAGENET_STD.clone(), persistent=False)
        elif self.encoder_backbone == "mobile_sam_vit_t":
            vendor_root = Path(__file__).resolve().parent / "vendor"
            if (vendor_root / "mobile_sam").exists() and str(vendor_root) not in sys.path:
                sys.path.insert(0, str(vendor_root))
            from mobile_sam import sam_model_registry

            ckpt_path = str(encoder_checkpoint or DEFAULT_MOBILE_SAM_CKPT)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"MobileSAM checkpoint not found: {ckpt_path}")

            sam = sam_model_registry["vit_t"](checkpoint=ckpt_path)
            self.encoder = sam.image_encoder
            self.encoder_input_size = int(getattr(self.encoder, "img_size", 1024))
            channels = 256
            self.register_buffer("encoder_mean", sam.pixel_mean.float().clone(), persistent=False)
            self.register_buffer("encoder_std", sam.pixel_std.float().clone(), persistent=False)
        elif self.encoder_backbone in {"sam_vit_h", "vit_h"}:
            vendor_root = Path(__file__).resolve().parent / "vendor"
            if (vendor_root / "segment_anything_model").exists() and str(vendor_root) not in sys.path:
                sys.path.insert(0, str(vendor_root))
            from segment_anything_model import sam_model_registry

            ckpt_path = str(encoder_checkpoint or DEFAULT_SAM_VIT_H_CKPT)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"SAM vit-h checkpoint not found: {ckpt_path}")

            sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
            self.encoder = sam.image_encoder
            self.encoder_input_size = int(getattr(self.encoder, "img_size", 1024))
            channels = 256
            self.register_buffer("encoder_mean", sam.pixel_mean.float().clone(), persistent=False)
            self.register_buffer("encoder_std", sam.pixel_std.float().clone(), persistent=False)
        else:
            raise ValueError(f"Unsupported encoder_backbone: {encoder_backbone}")

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = channels * roi_size * roi_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def encode_images(self, img_batch: torch.Tensor) -> torch.Tensor:
        if self.encoder_backbone in {"mobile_sam_vit_t", "sam_vit_h", "vit_h"}:
            if self.encoder_input_size > 0 and (
                img_batch.shape[-2] != self.encoder_input_size or img_batch.shape[-1] != self.encoder_input_size
            ):
                img_batch = F.interpolate(
                    img_batch,
                    size=(self.encoder_input_size, self.encoder_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            img_batch = img_batch * 255.0
            img_batch = (img_batch - self.encoder_mean) / self.encoder_std
            encoded = self.encoder(img_batch)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
            return encoded
        img_batch = (img_batch - self.encoder_mean) / self.encoder_std
        return self.encoder(img_batch)

    def build_fused_features(
        self,
        base_images: torch.Tensor,
        current_images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat_a = self.encode_images(base_images)
        feat_b = self.encode_images(current_images)
        fused = torch.cat([feat_a, feat_b, torch.abs(feat_b - feat_a)], dim=1)
        fused = self.fuse(fused)

        if self.use_mask and masks is not None:
            mask_feat = F.interpolate(masks, size=fused.shape[-2:], mode="nearest")
            fused = fused * (0.5 + 0.5 * mask_feat)
        return fused

    def extract_roi_features(
        self,
        base_images: torch.Tensor,
        current_images: torch.Tensor,
        masks: Optional[torch.Tensor],
        boxes: torch.Tensor,
        input_size: int,
    ) -> torch.Tensor:
        fused = self.build_fused_features(base_images, current_images, masks)
        return roi_pool_feature(
            fused,
            boxes,
            input_size=int(input_size),
            roi_size=self.roi_size,
        )


class PairProposalStageModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        roi_size: int = 7,
        use_mask: bool = True,
        pretrained: bool = False,
        encoder_backbone: str = "mobile_sam_vit_t",
        encoder_checkpoint: str = DEFAULT_MOBILE_SAM_CKPT,
        enable_bbox_head: bool = False,
    ) -> None:
        super().__init__()
        self.enable_bbox_head = bool(enable_bbox_head)
        self.backbone = PairRoiClassifier(
            num_classes=num_classes,
            roi_size=roi_size,
            use_mask=use_mask,
            pretrained=pretrained,
            encoder_backbone=encoder_backbone,
            encoder_checkpoint=encoder_checkpoint,
        )
        self.bbox_head: Optional[nn.Module]
        if self.enable_bbox_head:
            self.bbox_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.backbone.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 4),
            )
        else:
            self.bbox_head = None

    @property
    def roi_size(self) -> int:
        return self.backbone.roi_size

    @property
    def use_mask(self) -> bool:
        return self.backbone.use_mask

    @property
    def encoder_backbone(self) -> str:
        return self.backbone.encoder_backbone

    def set_encoder_trainable(self, trainable: bool) -> None:
        for param in self.backbone.encoder.parameters():
            param.requires_grad = trainable

    def extract_roi_features(
        self,
        base_images: torch.Tensor,
        current_images: torch.Tensor,
        masks: Optional[torch.Tensor],
        boxes: torch.Tensor,
        input_size: int,
    ) -> torch.Tensor:
        return self.backbone.extract_roi_features(base_images, current_images, masks, boxes, input_size)

    def forward(
        self,
        base_images: torch.Tensor,
        current_images: torch.Tensor,
        masks: Optional[torch.Tensor],
        boxes: torch.Tensor,
        input_size: int,
    ) -> Dict[str, torch.Tensor]:
        roi_feat = self.extract_roi_features(base_images, current_images, masks, boxes, input_size)
        outputs: Dict[str, torch.Tensor] = {
            "logits": self.backbone.classifier(roi_feat),
            "roi_feat": roi_feat,
        }
        if self.bbox_head is not None:
            outputs["bbox_deltas"] = self.bbox_head(roi_feat)
        return outputs


class PairChangeClassifier:
    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        mobile_sam_ckpt: Optional[str] = None,
        sam_vit_h_ckpt: Optional[str] = None,
        roi_view_mode: Optional[str] = None,
        roi_crop_context_ratio: Optional[float] = None,
        roi_crop_min_context: Optional[int] = None,
    ) -> None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        ckpt_args = checkpoint.get("args", {})
        self.input_size = int(ckpt_args.get("input_size", 1024))
        self.roi_size = int(ckpt_args.get("roi_size", 7))
        self.num_classes = int(ckpt_args.get("num_classes", len(CLASS_NAMES)))
        self.class_names = list(ckpt_args.get("class_names", CLASS_NAMES))
        self.use_mask = bool(ckpt_args.get("use_mask", True))
        self.roi_view_mode = normalize_roi_view_mode(
            ckpt_args.get("roi_view_mode", "full_image") if roi_view_mode is None else roi_view_mode
        )
        self.roi_crop_context_ratio = float(
            ckpt_args.get("roi_crop_context_ratio", 0.0)
            if roi_crop_context_ratio is None
            else roi_crop_context_ratio
        )
        self.roi_crop_min_context = int(
            ckpt_args.get("roi_crop_min_context", 0) if roi_crop_min_context is None else roi_crop_min_context
        )
        self.encoder_backbone = str(ckpt_args.get("encoder_backbone", "resnet18"))
        encoder_checkpoint = str(ckpt_args.get("encoder_checkpoint", ""))
        self.pretrained = bool(ckpt_args.get("pretrained", False))

        if self.encoder_backbone == "mobile_sam_vit_t":
            if mobile_sam_ckpt and os.path.exists(mobile_sam_ckpt):
                encoder_checkpoint = mobile_sam_ckpt
            elif not encoder_checkpoint or not os.path.exists(encoder_checkpoint):
                encoder_checkpoint = DEFAULT_MOBILE_SAM_CKPT
        elif self.encoder_backbone in {"sam_vit_h", "vit_h"}:
            if sam_vit_h_ckpt and os.path.exists(sam_vit_h_ckpt):
                encoder_checkpoint = sam_vit_h_ckpt
            elif not encoder_checkpoint or not os.path.exists(encoder_checkpoint):
                encoder_checkpoint = DEFAULT_SAM_VIT_H_CKPT

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        model = PairRoiClassifier(
            num_classes=self.num_classes,
            roi_size=self.roi_size,
            use_mask=self.use_mask,
            pretrained=self.pretrained,
            encoder_backbone=self.encoder_backbone,
            encoder_checkpoint=encoder_checkpoint,
        )
        ckpt_state = checkpoint["model_state"]
        save_head_only = bool(checkpoint.get("save_head_only", False))
        model.load_state_dict(ckpt_state, strict=not save_head_only)
        model.eval().to(self.device)
        self.model = model

    def _prep_rgb(self, bgr_img: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.input_size, self.input_size):
            rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        return tensor.unsqueeze(0)

    def _predict_logits_from_crops(
        self,
        base_img_bgr: np.ndarray,
        current_img_bgr: np.ndarray,
        boxes_xyxy: List[List[float]],
    ) -> torch.Tensor:
        h, w = current_img_bgr.shape[:2]
        base_tensors: List[torch.Tensor] = []
        current_tensors: List[torch.Tensor] = []
        remapped_boxes: List[List[float]] = []
        mask_tensors: List[torch.Tensor] = []

        for box in boxes_xyxy:
            crop_xyxy = compute_crop_box_xyxy(
                tuple(float(v) for v in box),
                image_width=w,
                image_height=h,
                context_ratio=self.roi_crop_context_ratio,
                min_context_pixels=self.roi_crop_min_context,
            )
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
            base_crop = base_img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            current_crop = current_img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            base_tensors.append(self._prep_rgb(base_crop))
            current_tensors.append(self._prep_rgb(current_crop))
            remapped = remap_box_to_resized_crop(
                tuple(float(v) for v in box),
                crop_xyxy,
                output_width=self.input_size,
                output_height=self.input_size,
            )
            remapped_boxes.append([float(v) for v in remapped])
            if self.use_mask:
                mask_np = build_box_mask(self.input_size, self.input_size, remapped)
                mask_tensors.append(torch.from_numpy(mask_np).to(self.device))

        base_batch = torch.cat(base_tensors, dim=0)
        current_batch = torch.cat(current_tensors, dim=0)
        feat_a = self.model.encode_images(base_batch)
        feat_b = self.model.encode_images(current_batch)
        fused = torch.cat([feat_a, feat_b, torch.abs(feat_b - feat_a)], dim=1)
        fused = self.model.fuse(fused)

        if self.use_mask and mask_tensors:
            mask_t = torch.stack(mask_tensors, dim=0).unsqueeze(1)
            mask_feat = F.interpolate(mask_t, size=fused.shape[-2:], mode="nearest")
            fused = fused * (0.5 + 0.5 * mask_feat)

        boxes_t = torch.tensor(remapped_boxes, dtype=torch.float32, device=self.device)
        roi_feat = roi_pool_feature(
            fused,
            boxes_t,
            input_size=self.input_size,
            roi_size=self.model.roi_size,
        )
        return self.model.classifier(roi_feat)

    def predict_instances(
        self,
        base_img_bgr: np.ndarray,
        current_img_bgr: np.ndarray,
        mask_bin: np.ndarray,
        contours: List[np.ndarray],
        min_instance_pixels: int,
        confidence_threshold: float = 0.0,
        contour_meta: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        h, w = mask_bin.shape[:2]
        kept: List[Dict[str, Any]] = []
        boxes_input: List[List[float]] = []

        sx = self.input_size / float(w)
        sy = self.input_size / float(h)

        for idx, cnt in enumerate(contours):
            meta = None
            if contour_meta is not None and idx < len(contour_meta):
                meta = contour_meta[idx]
            if cnt is None or len(cnt) < 3:
                continue
            local = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(local, [cnt.astype(np.int32)], 1)
            change_pixels = int(local.sum())
            if change_pixels < min_instance_pixels:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            kept.append(
                {
                    "cnt": cnt,
                    "change_pixels": change_pixels,
                    "group_id": None,
                    "label": "change",
                    "score": None,
                    "binary_change_score": None if meta is None else meta.get("change_score"),
                    "binary_no_change_score": None if meta is None else meta.get("no_change_score"),
                    "proposal_box": None if meta is None else meta.get("proposal_box"),
                }
            )
            boxes_input.append([x * sx, y * sy, (x + bw) * sx, (y + bh) * sy])

        if not kept:
            return kept

        with torch.no_grad():
            if self.roi_view_mode == "crop":
                logits = self._predict_logits_from_crops(base_img_bgr, current_img_bgr, boxes_input)
            else:
                a_tensor = self._prep_rgb(base_img_bgr)
                b_tensor = self._prep_rgb(current_img_bgr)
                feat_a = self.model.encode_images(a_tensor)
                feat_b = self.model.encode_images(b_tensor)
                fused = torch.cat([feat_a, feat_b, torch.abs(feat_b - feat_a)], dim=1)
                fused = self.model.fuse(fused)

                if self.use_mask:
                    mask_small = cv2.resize(
                        mask_bin.astype(np.uint8),
                        (self.input_size, self.input_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask_t = torch.from_numpy(mask_small).float().to(self.device).unsqueeze(0).unsqueeze(0)
                    mask_feat = F.interpolate(mask_t, size=fused.shape[-2:], mode="nearest")
                    fused = fused * (0.5 + 0.5 * mask_feat)

                n = len(kept)
                fused_rep = fused.repeat(n, 1, 1, 1)
                boxes_t = torch.tensor(boxes_input, dtype=torch.float32, device=self.device)
                roi_feat = roi_pool_feature(
                    fused_rep,
                    boxes_t,
                    input_size=self.input_size,
                    roi_size=self.model.roi_size,
                )
                logits = self.model.classifier(roi_feat)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        filtered: List[Dict[str, Any]] = []
        for idx, item in enumerate(kept):
            score = float(conf[idx].item())
            if score < float(confidence_threshold):
                continue
            class_id = int(pred[idx].item())
            label = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            item["group_id"] = class_id
            item["label"] = label
            item["score"] = score
            filtered.append(item)
        return filtered


class PairBinaryRoiClassifier:
    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        mobile_sam_ckpt: Optional[str] = None,
        sam_vit_h_ckpt: Optional[str] = None,
        roi_view_mode: Optional[str] = None,
        roi_crop_context_ratio: Optional[float] = None,
        roi_crop_min_context: Optional[int] = None,
    ) -> None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        ckpt_args = checkpoint.get("args", {})
        self.input_size = int(ckpt_args.get("input_size", 1024))
        self.roi_size = int(ckpt_args.get("roi_size", 7))
        self.num_classes = int(ckpt_args.get("num_classes", 2))
        self.class_names = list(ckpt_args.get("class_names", BINARY_CLASS_NAMES))
        self.use_mask = bool(ckpt_args.get("use_mask", True))
        self.roi_view_mode = normalize_roi_view_mode(
            ckpt_args.get("roi_view_mode", "full_image") if roi_view_mode is None else roi_view_mode
        )
        self.roi_crop_context_ratio = float(
            ckpt_args.get("roi_crop_context_ratio", 0.0)
            if roi_crop_context_ratio is None
            else roi_crop_context_ratio
        )
        self.roi_crop_min_context = int(
            ckpt_args.get("roi_crop_min_context", 0) if roi_crop_min_context is None else roi_crop_min_context
        )
        self.encoder_backbone = str(ckpt_args.get("encoder_backbone", "mobile_sam_vit_t"))
        encoder_checkpoint = str(ckpt_args.get("encoder_checkpoint", ""))
        self.pretrained = bool(ckpt_args.get("pretrained", False))

        if self.encoder_backbone == "mobile_sam_vit_t":
            if mobile_sam_ckpt and os.path.exists(mobile_sam_ckpt):
                encoder_checkpoint = mobile_sam_ckpt
            elif not encoder_checkpoint or not os.path.exists(encoder_checkpoint):
                encoder_checkpoint = DEFAULT_MOBILE_SAM_CKPT
        elif self.encoder_backbone in {"sam_vit_h", "vit_h"}:
            if sam_vit_h_ckpt and os.path.exists(sam_vit_h_ckpt):
                encoder_checkpoint = sam_vit_h_ckpt
            elif not encoder_checkpoint or not os.path.exists(encoder_checkpoint):
                encoder_checkpoint = DEFAULT_SAM_VIT_H_CKPT

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        model = PairRoiClassifier(
            num_classes=self.num_classes,
            roi_size=self.roi_size,
            use_mask=self.use_mask,
            pretrained=self.pretrained,
            encoder_backbone=self.encoder_backbone,
            encoder_checkpoint=encoder_checkpoint,
        )
        save_head_only = bool(checkpoint.get("save_head_only", False))
        model.load_state_dict(checkpoint["model_state"], strict=not save_head_only)
        model.eval().to(self.device)
        self.model = model

    def _prep_rgb(self, bgr_img: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.input_size, self.input_size):
            rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255.0
        return tensor.unsqueeze(0)

    def _predict_probs_from_crops(
        self,
        base_img_bgr: np.ndarray,
        current_img_bgr: np.ndarray,
        boxes_xyxy: List[List[float]],
    ) -> torch.Tensor:
        h, w = current_img_bgr.shape[:2]
        base_tensors: List[torch.Tensor] = []
        current_tensors: List[torch.Tensor] = []
        remapped_boxes: List[List[float]] = []
        mask_tensors: List[torch.Tensor] = []

        for box in boxes_xyxy:
            crop_xyxy = compute_crop_box_xyxy(
                tuple(float(v) for v in box),
                image_width=w,
                image_height=h,
                context_ratio=self.roi_crop_context_ratio,
                min_context_pixels=self.roi_crop_min_context,
            )
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
            base_crop = base_img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            current_crop = current_img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            base_tensors.append(self._prep_rgb(base_crop))
            current_tensors.append(self._prep_rgb(current_crop))
            remapped = remap_box_to_resized_crop(
                tuple(float(v) for v in box),
                crop_xyxy,
                output_width=self.input_size,
                output_height=self.input_size,
            )
            remapped_boxes.append([float(v) for v in remapped])
            if self.use_mask:
                mask_np = build_box_mask(self.input_size, self.input_size, remapped)
                mask_tensors.append(torch.from_numpy(mask_np).to(self.device))

        base_batch = torch.cat(base_tensors, dim=0)
        current_batch = torch.cat(current_tensors, dim=0)
        feat_a = self.model.encode_images(base_batch)
        feat_b = self.model.encode_images(current_batch)
        fused = torch.cat([feat_a, feat_b, torch.abs(feat_b - feat_a)], dim=1)
        fused = self.model.fuse(fused)

        if self.use_mask and mask_tensors:
            mask_t = torch.stack(mask_tensors, dim=0).unsqueeze(1)
            mask_feat = F.interpolate(mask_t, size=fused.shape[-2:], mode="nearest")
            fused = fused * (0.5 + 0.5 * mask_feat)

        boxes_t = torch.tensor(remapped_boxes, dtype=torch.float32, device=self.device)
        roi_feat = roi_pool_feature(
            fused,
            boxes_t,
            input_size=self.input_size,
            roi_size=self.model.roi_size,
        )
        logits = self.model.classifier(roi_feat)
        return torch.softmax(logits, dim=1)

    def predict_boxes(
        self,
        base_img_bgr: np.ndarray,
        current_img_bgr: np.ndarray,
        boxes_xyxy: List[List[float]],
        mask_bin: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if not boxes_xyxy:
            return []

        with torch.no_grad():
            if self.roi_view_mode == "crop":
                probs = self._predict_probs_from_crops(base_img_bgr, current_img_bgr, boxes_xyxy)
            else:
                base_tensor = self._prep_rgb(base_img_bgr)
                current_tensor = self._prep_rgb(current_img_bgr)
                feat_a = self.model.encode_images(base_tensor)
                feat_b = self.model.encode_images(current_tensor)
                fused = torch.cat([feat_a, feat_b, torch.abs(feat_b - feat_a)], dim=1)
                fused = self.model.fuse(fused)

                if self.use_mask and mask_bin is not None:
                    mask_small = cv2.resize(
                        mask_bin.astype(np.uint8),
                        (self.input_size, self.input_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask_t = torch.from_numpy(mask_small).float().to(self.device).unsqueeze(0).unsqueeze(0)
                    mask_feat = F.interpolate(mask_t, size=fused.shape[-2:], mode="nearest")
                    fused = fused * (0.5 + 0.5 * mask_feat)

                fused_rep = fused.repeat(len(boxes_xyxy), 1, 1, 1)
                boxes_t = torch.tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
                roi_feat = roi_pool_feature(
                    fused_rep,
                    boxes_t,
                    input_size=self.input_size,
                    roi_size=self.model.roi_size,
                )
                logits = self.model.classifier(roi_feat)
                probs = torch.softmax(logits, dim=1)

        outputs: List[Dict[str, Any]] = []
        for idx in range(probs.shape[0]):
            no_change_score = float(probs[idx, 0].item()) if probs.shape[1] > 0 else 0.0
            change_score = float(probs[idx, 1].item()) if probs.shape[1] > 1 else 1.0 - no_change_score
            pred = int(torch.argmax(probs[idx]).item())
            outputs.append(
                {
                    "box": boxes_xyxy[idx],
                    "pred_label": self.class_names[pred] if pred < len(self.class_names) else str(pred),
                    "change_score": change_score,
                    "no_change_score": no_change_score,
                    "is_change": int(pred == 1),
                }
            )
        return outputs
