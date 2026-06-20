import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

from .classifier import PairBinaryRoiClassifier, PairChangeClassifier
from .gescf import GeSCF
from .pair_change_backend import PairChangeSegBackend
from .veg_infer import VegetationCoverageService
from .yolo_infer import YoloDetectionService


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_ALLOWED_LABELS = [
    "rock_fall",
    "rock_spalling",
    "leaves",
    "slope_plant",
    "other_plant",
    "car",
    "change_color",
    "other",
]
BASE_UPDATE_LABELS = {"slope_plant", "other_plant", "rock_fall", "rock_spalling", "landslide"}
VEGETATION_LOCATION_LABELS = {"slope_plant", "other_plant"}
LABEL_GROUP_IDS = {
    "rock_fall": 0,
    "rock_spalling": 1,
    "landslide": 2,
    "slope_plant": 3,
    "other_plant": 4,
}


@dataclass
class DetectorConfig:
    data_root: str
    output_root: str
    base_roi_root: str = ""
    vegetation_config: str = ""
    proposal_backend: str = "gescf"
    classifier_ckpt: str = ""
    binary_classifier_ckpt: str = ""
    mobile_sam_ckpt: str = ""
    pair_change_checkpoint: str = ""
    pair_change_device: str = "cuda:0"
    pair_change_decoder_arch: str = "auto"
    pair_change_inference_mode: str = "sliding"
    pair_change_crop_size: int = 1536
    pair_change_stride: int = 768
    pair_change_threshold: float = 0.5
    pair_change_open_kernel: int = 3
    pair_change_close_kernel: int = 5
    sam_vit_h_ckpt: str = ""
    superpoint_ckpt: str = ""
    classifier_device: str = "auto"
    binary_classifier_device: str = "auto"
    allowed_labels: List[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_LABELS))
    export_all_as_label: str = ""
    binary_accepted_label: str = ""
    binary_rejected_label: str = ""
    enable_leaf_color_rule: bool = True
    leaf_color_min_ratio: float = 0.08
    leaf_gray_ratio_threshold: float = 0.12
    leaf_gray_increase_threshold: float = 0.08
    leaf_color_drop_threshold: float = 0.08
    leaf_hue_shift_threshold: float = 0.08

    min_change_pixels: int = 20000
    min_change_ratio: float = 0.001
    min_instance_pixels: int = 20000
    require_raw_change_for_instances: bool = True
    classifier_confidence_threshold: float = 0.6
    binary_classifier_change_threshold: float = 0.5
    update_base_on_change: bool = False
    vegetation_write_outputs: bool = False
    vegetation_save_debug_images: bool = False
    classifier_roi_view_mode: str = ""
    classifier_roi_crop_context_ratio: float = -1.0
    classifier_roi_crop_min_context: int = -1
    binary_roi_view_mode: str = ""
    binary_roi_crop_context_ratio: float = -1.0
    binary_roi_crop_min_context: int = -1

    test_dataset: str = "ChangeVPR"
    output_size: int = 512
    feature_facet: str = "key"
    feature_layer: int = 17
    embedding_layer: int = 32
    sam_backbone: str = "vit_t"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.7
    pseudo_backbone: str = "vit_t"
    pseudo_mask_mode: str = "default"
    pseudo_mask_open_kernel: int = 17
    pseudo_mask_seed_dilate_kernel: int = 33
    pseudo_mask_close_kernel: int = 0
    adaptive_threshold_scale: float = 1.0
    adaptive_threshold_mode: str = "mad"
    adaptive_threshold_min: Optional[float] = None
    adaptive_threshold_max: Optional[float] = None
    adaptive_threshold_mad_eps: float = 0.0
    adaptive_threshold_sim_std_eps: float = 0.0
    adaptive_threshold_candidate_percentile_min: float = 85.0
    adaptive_threshold_seed_percentile: float = 50.0
    adaptive_threshold_max_candidate_ratio: float = 0.25
    merge_same_label_instances: bool = False
    merge_same_label_kernel: int = 0
    enable_slope_vegetation_decrease_rule: bool = True
    slope_vegetation_override_label: str = "slope_plant"
    slope_vegetation_roi_overlap_threshold: float = 0.3
    slope_vegetation_roi_coverage_threshold: float = 0.3
    slope_vegetation_decrease_threshold: float = 0.1
    slope_vegetation_excluded_source_labels: List[str] = field(default_factory=lambda: ["leaves", "change_color"])
    slope_vegetation_allowed_source_labels: List[str] = field(default_factory=list)
    enable_slope_vegetation_hsv_bare_rule: bool = False
    slope_vegetation_base_green_min_ratio: float = 0.15
    slope_vegetation_current_green_max_ratio: float = 0.08
    slope_vegetation_green_drop_min_ratio: float = 0.08
    slope_vegetation_gray_increase_min_ratio: float = 0.08
    enable_yolo_detection: bool = False
    yolo_config: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectorConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed})

    @classmethod
    def from_json(cls, path: Path) -> "DetectorConfig":
        with path.open("r", encoding="utf-8-sig") as f:
            raw = json.load(f)
        return cls.from_dict(raw)


class OnlineChangeService:
    def __init__(self, config: DetectorConfig, config_base_dir: Optional[Path] = None) -> None:
        self.config = config
        self.config_base_dir = config_base_dir or Path.cwd()

        self.package_root = Path(__file__).resolve().parents[1]
        self.data_root = self._resolve_path(config.data_root)
        self.output_root = self._resolve_path(config.output_root)
        self.base_roi_root = self._resolve_path(config.base_roi_root) if config.base_roi_root else None
        self.vegetation_config = self._resolve_path(config.vegetation_config) if config.vegetation_config else None
        self.classifier_ckpt = self._resolve_path(config.classifier_ckpt) if config.classifier_ckpt else None
        self.binary_classifier_ckpt = self._resolve_path(config.binary_classifier_ckpt) if config.binary_classifier_ckpt else None
        self.mobile_sam_ckpt = self._resolve_path(config.mobile_sam_ckpt) if config.mobile_sam_ckpt else None
        self.pair_change_checkpoint = self._resolve_path(config.pair_change_checkpoint) if config.pair_change_checkpoint else None
        self.sam_vit_h_ckpt = self._resolve_path(config.sam_vit_h_ckpt) if config.sam_vit_h_ckpt else None
        self.superpoint_ckpt = self._resolve_path(config.superpoint_ckpt) if config.superpoint_ckpt else None
        self.yolo_config = self._resolve_path(config.yolo_config) if config.yolo_config else None
        self.allowed_labels: Set[str] = set(config.allowed_labels)

        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")
        if self.base_roi_root is not None and not self.base_roi_root.exists():
            raise FileNotFoundError(f"base_roi_root not found: {self.base_roi_root}")
        if self.vegetation_config is not None and not self.vegetation_config.exists():
            raise FileNotFoundError(f"vegetation_config not found: {self.vegetation_config}")

        self._prepare_runtime_env()
        self.model = self._build_mask_backend()
        self.binary_classifier = self._build_binary_classifier()
        self.classifier = self._build_classifier()
        self._veg_service: Optional[VegetationCoverageService] = None
        self._yolo_service: Optional[YoloDetectionService] = None

    @classmethod
    def from_config_file(cls, config_path: str) -> "OnlineChangeService":
        path = Path(config_path).resolve()
        cfg = DetectorConfig.from_json(path)
        return cls(cfg, config_base_dir=path.parent)

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p.resolve()
        return (self.config_base_dir / p).resolve()

    def _prepare_runtime_env(self) -> None:
        if self.mobile_sam_ckpt is not None:
            if not self.mobile_sam_ckpt.exists():
                raise FileNotFoundError(f"mobile_sam_ckpt not found: {self.mobile_sam_ckpt}")
            os.environ["EDGE_CD_MOBILE_SAM_CKPT"] = str(self.mobile_sam_ckpt)

        if self.sam_vit_h_ckpt is not None:
            if not self.sam_vit_h_ckpt.exists():
                raise FileNotFoundError(f"sam_vit_h_ckpt not found: {self.sam_vit_h_ckpt}")
            os.environ["EDGE_CD_SAM_VIT_H_CKPT"] = str(self.sam_vit_h_ckpt)

        if self.superpoint_ckpt is not None:
            if not self.superpoint_ckpt.exists():
                raise FileNotFoundError(f"superpoint_ckpt not found: {self.superpoint_ckpt}")
            os.environ["EDGE_CD_SUPERPOINT_CKPT"] = str(self.superpoint_ckpt)

    def _build_gescf(self):
        args = SimpleNamespace(
            test_dataset=self.config.test_dataset,
            output_size=self.config.output_size,
            feature_facet=self.config.feature_facet,
            feature_layer=self.config.feature_layer,
            embedding_layer=self.config.embedding_layer,
            sam_backbone=self.config.sam_backbone,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            pseudo_backbone=self.config.pseudo_backbone,
            pseudo_mask_mode=self.config.pseudo_mask_mode,
            pseudo_mask_open_kernel=self.config.pseudo_mask_open_kernel,
            pseudo_mask_seed_dilate_kernel=self.config.pseudo_mask_seed_dilate_kernel,
            pseudo_mask_close_kernel=self.config.pseudo_mask_close_kernel,
            adaptive_threshold_scale=self.config.adaptive_threshold_scale,
            adaptive_threshold_mode=self.config.adaptive_threshold_mode,
            adaptive_threshold_min=self.config.adaptive_threshold_min,
            adaptive_threshold_max=self.config.adaptive_threshold_max,
            adaptive_threshold_mad_eps=self.config.adaptive_threshold_mad_eps,
            adaptive_threshold_sim_std_eps=self.config.adaptive_threshold_sim_std_eps,
            adaptive_threshold_candidate_percentile_min=self.config.adaptive_threshold_candidate_percentile_min,
            adaptive_threshold_seed_percentile=self.config.adaptive_threshold_seed_percentile,
            adaptive_threshold_max_candidate_ratio=self.config.adaptive_threshold_max_candidate_ratio,
        )
        model = GeSCF(args)
        model.eval()
        torch.set_grad_enabled(False)
        return model

    def _build_mask_backend(self):
        backend_name = str(self.config.proposal_backend).strip().lower()
        if backend_name in {"pair_change_seg", "pair_change", "mobile_sam_pair"}:
            checkpoint_path = self.pair_change_checkpoint
            if checkpoint_path is None:
                raise FileNotFoundError("pair_change_checkpoint is required for proposal_backend=pair_change_seg")
            return PairChangeSegBackend(
                checkpoint_path=checkpoint_path,
                device=self.config.pair_change_device,
                decoder_arch=self.config.pair_change_decoder_arch,
                inference_mode=self.config.pair_change_inference_mode,
                crop_size=self.config.pair_change_crop_size,
                stride=self.config.pair_change_stride,
                threshold=self.config.pair_change_threshold,
                open_kernel=self.config.pair_change_open_kernel,
                close_kernel=self.config.pair_change_close_kernel,
            )
        return self._build_gescf()

    def _build_classifier(self):
        if self.classifier_ckpt is None:
            return None
        if not self.classifier_ckpt.exists():
            raise FileNotFoundError(f"classifier_ckpt not found: {self.classifier_ckpt}")
        roi_view_mode = str(self.config.classifier_roi_view_mode).strip() or None
        roi_crop_context_ratio = (
            float(self.config.classifier_roi_crop_context_ratio)
            if float(self.config.classifier_roi_crop_context_ratio) >= 0
            else None
        )
        roi_crop_min_context = (
            int(self.config.classifier_roi_crop_min_context)
            if int(self.config.classifier_roi_crop_min_context) >= 0
            else None
        )
        return PairChangeClassifier(
            ckpt_path=str(self.classifier_ckpt),
            device=self.config.classifier_device,
            mobile_sam_ckpt=str(self.mobile_sam_ckpt) if self.mobile_sam_ckpt else None,
            sam_vit_h_ckpt=str(self.sam_vit_h_ckpt) if self.sam_vit_h_ckpt else None,
            roi_view_mode=roi_view_mode,
            roi_crop_context_ratio=roi_crop_context_ratio,
            roi_crop_min_context=roi_crop_min_context,
        )

    def _build_binary_classifier(self):
        if self.binary_classifier_ckpt is None:
            return None
        if not self.binary_classifier_ckpt.exists():
            raise FileNotFoundError(f"binary_classifier_ckpt not found: {self.binary_classifier_ckpt}")
        roi_view_mode = str(self.config.binary_roi_view_mode).strip() or None
        roi_crop_context_ratio = (
            float(self.config.binary_roi_crop_context_ratio)
            if float(self.config.binary_roi_crop_context_ratio) >= 0
            else None
        )
        roi_crop_min_context = (
            int(self.config.binary_roi_crop_min_context)
            if int(self.config.binary_roi_crop_min_context) >= 0
            else None
        )
        return PairBinaryRoiClassifier(
            ckpt_path=str(self.binary_classifier_ckpt),
            device=self.config.binary_classifier_device,
            mobile_sam_ckpt=str(self.mobile_sam_ckpt) if self.mobile_sam_ckpt else None,
            sam_vit_h_ckpt=str(self.sam_vit_h_ckpt) if self.sam_vit_h_ckpt else None,
            roi_view_mode=roi_view_mode,
            roi_crop_context_ratio=roi_crop_context_ratio,
            roi_crop_min_context=roi_crop_min_context,
        )

    def _get_vegetation_service(self) -> Optional[VegetationCoverageService]:
        if self._veg_service is not None:
            return self._veg_service
        veg_config_path = self.vegetation_config
        if veg_config_path is None:
            default_path = self.package_root / "config" / "vegetation_config.json"
            if not default_path.exists():
                return None
            veg_config_path = default_path
        self._veg_service = VegetationCoverageService.from_config_file(str(veg_config_path))
        return self._veg_service

    def _get_yolo_service(self) -> Optional[YoloDetectionService]:
        if not bool(self.config.enable_yolo_detection):
            return None
        if self.yolo_config is None:
            return None
        if not self.yolo_config.exists():
            raise FileNotFoundError(f"yolo_config not found: {self.yolo_config}")
        if self._yolo_service is None:
            self._yolo_service = YoloDetectionService.from_config_file(str(self.yolo_config))
        return self._yolo_service

    @staticmethod
    def _read_image_unicode(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            raise FileNotFoundError(f"Cannot read image data from: {path}")
        img = cv2.imdecode(data, flags)
        if img is None:
            raise ValueError(f"cv2.imdecode failed for: {path}")
        return img

    @staticmethod
    def _write_image_unicode(path: Path, img: np.ndarray) -> None:
        ext = path.suffix or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise ValueError(f"cv2.imencode failed for: {path}")
        buf.tofile(str(path))

    @staticmethod
    def _unletterbox_mask(mask: np.ndarray, orig_shape: Tuple[int, int]) -> np.ndarray:
        tgt_h, tgt_w = mask.shape[:2]
        orig_h, orig_w = orig_shape
        if orig_h == 0 or orig_w == 0 or tgt_h == 0 or tgt_w == 0:
            return np.zeros((orig_h, orig_w), dtype=mask.dtype)

        scale = min(tgt_w / float(orig_w), tgt_h / float(orig_h))
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        if new_w <= 0 or new_h <= 0:
            return np.zeros((orig_h, orig_w), dtype=mask.dtype)

        left = (tgt_w - new_w) // 2
        top = (tgt_h - new_h) // 2
        cropped = mask[top:top + new_h, left:left + new_w]
        return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _list_scene_images(scene_dir: Path) -> List[Path]:
        return sorted([p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

    def _get_single_base(self, scene_dir: Path) -> Path:
        images = self._list_scene_images(scene_dir)
        if len(images) != 1:
            raise RuntimeError(f"Expected 1 base image in {scene_dir}, found {len(images)}")
        return images[0]

    def _next_output_dir(self, scene_name: str, current_stem: str) -> Path:
        scene_root = self.output_root / f"{scene_name}_images"
        scene_root.mkdir(parents=True, exist_ok=True)
        out = scene_root / current_stem
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _run_mask(self, base_path: Path, current_path: Path) -> np.ndarray:
        with torch.no_grad():
            return self.model(str(base_path), str(current_path))

    @staticmethod
    def _detect_change(mask: np.ndarray, min_pixels: int, min_ratio: float) -> Tuple[bool, int, float]:
        bin_mask = (mask > 0).astype(np.uint8)
        change_pixels = int(bin_mask.sum())
        change_ratio = change_pixels / float(bin_mask.size)
        is_change = (change_pixels >= min_pixels) and (change_ratio >= min_ratio)
        return is_change, change_pixels, change_ratio

    @staticmethod
    def _detect_change_from_exported_shapes(
        labelme: Dict[str, Any],
        image_size: int,
        min_pixels: int,
        min_ratio: float,
    ) -> Tuple[bool, int, float]:
        shapes = labelme.get("shapes", [])
        change_pixels = 0
        if isinstance(shapes, list):
            for shape in shapes:
                flags = shape.get("flags", {}) if isinstance(shape, dict) else {}
                pixels = int(flags.get("change_pixels", 0))
                if pixels <= 0 and isinstance(shape, dict):
                    try:
                        description = json.loads(shape.get("description") or "{}")
                    except Exception:
                        description = {}
                    pixels = int(description.get("change_pixels", 0) or 0)
                change_pixels += pixels
        change_ratio = change_pixels / float(image_size) if image_size > 0 else 0.0
        is_change = (change_pixels >= min_pixels) and (change_ratio >= min_ratio)
        return is_change, change_pixels, change_ratio

    @staticmethod
    def _load_json_unicode(path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            with path.open("r", encoding="utf-8-sig") as f:
                return json.load(f)

    @staticmethod
    def _scene_name_candidates(scene: str) -> List[str]:
        names = [scene]
        if scene.endswith("_images"):
            names.append(scene[:-7])
        else:
            names.append(f"{scene}_images")
        seen: Set[str] = set()
        ordered: List[str] = []
        for name in names:
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _resolve_base_roi_scene_dir(self, scene: str) -> Optional[Tuple[str, Path]]:
        if self.base_roi_root is None:
            return None
        for name in self._scene_name_candidates(scene):
            scene_dir = self.base_roi_root / name
            if not scene_dir.exists():
                continue
            return name, scene_dir
        return None

    def _find_base_roi_json(self, scene: str) -> Optional[Path]:
        resolved = self._resolve_base_roi_scene_dir(scene)
        if resolved is None:
            return None
        _, scene_dir = resolved
        json_files = sorted([p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
        if not json_files:
            return None
        image_files = self._list_scene_images(scene_dir)
        if len(image_files) == 1:
            stem_match = [p for p in json_files if p.stem == image_files[0].stem]
            if stem_match:
                return stem_match[0]
        if len(json_files) == 1:
            return json_files[0]
        return json_files[0]

    def _load_base_roi_mask(self, scene: str, image_height: int, image_width: int) -> Optional[np.ndarray]:
        roi_json = self._find_base_roi_json(scene)
        if roi_json is None:
            return None
        roi_payload = self._load_json_unicode(roi_json)
        roi_mask = self._build_mask_from_labelme(
            roi_payload,
            target_height=image_height,
            target_width=image_width,
            allowed_labels=None,
        )
        if int(roi_mask.sum()) <= 0:
            return None
        return roi_mask

    @staticmethod
    def _build_mask_from_labelme(
        payload: Dict[str, Any],
        target_height: int,
        target_width: int,
        allowed_labels: Optional[Set[str]] = None,
    ) -> np.ndarray:
        shapes = payload.get("shapes", [])
        if not isinstance(shapes, list):
            return np.zeros((target_height, target_width), dtype=np.uint8)

        ref_h = int(payload.get("imageHeight") or target_height)
        ref_w = int(payload.get("imageWidth") or target_width)
        if ref_h <= 0 or ref_w <= 0:
            ref_h = target_height
            ref_w = target_width

        sx = target_width / float(ref_w)
        sy = target_height / float(ref_h)
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)

        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label", "")).strip()
            if allowed_labels is not None and label not in allowed_labels:
                continue
            pts = shape.get("points")
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            shape_type = str(shape.get("shape_type", "polygon")).lower()

            scaled_points = []
            for pt in pts:
                if not isinstance(pt, list) or len(pt) != 2:
                    continue
                x = int(round(float(pt[0]) * sx))
                y = int(round(float(pt[1]) * sy))
                x = max(0, min(x, target_width - 1))
                y = max(0, min(y, target_height - 1))
                scaled_points.append([x, y])

            if shape_type == "rectangle" and len(scaled_points) >= 2:
                (x1, y1), (x2, y2) = scaled_points[0], scaled_points[1]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color=1, thickness=-1)
            elif len(scaled_points) >= 3:
                arr = np.asarray(scaled_points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(canvas, [arr], color=1)
        return canvas.astype(np.uint8)

    @staticmethod
    def _normalize_vegetation_metrics(veg_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "roi_pixels": veg_result.get("roi_pixels"),
            "base_pc": veg_result.get("coverage_a"),
            "current_pc": veg_result.get("coverage_b"),
            "p_decrease_rate": veg_result.get("decrease_rate"),
            "p_increase_rate": veg_result.get("increase_rate"),
            "decrease_threshold": veg_result.get("decrease_threshold"),
            "increase_threshold": veg_result.get("increase_threshold"),
            "decrease_detected": veg_result.get("decrease_detected"),
            "increase_detected": veg_result.get("increase_detected"),
            "change_detected": veg_result.get("change_detected"),
            "base_update_recommended": veg_result.get("base_update_recommended"),
            "base_updated": veg_result.get("base_updated"),
            "debug_output_dir": "." if veg_result.get("debug_files") else None,
            "debug_files": veg_result.get("debug_files"),
        }

    def _compute_vegetation_coverage_metrics(
        self,
        scene: str,
        current_path: Path,
        instances: List[Dict[str, Any]],
        image_height: int,
        image_width: int,
        output_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        roi_mask = self._load_base_roi_mask(scene, image_height, image_width)
        if roi_mask is None:
            return None
        veg_service = self._get_vegetation_service()
        if veg_service is None:
            return None

        resolved = self._resolve_base_roi_scene_dir(scene)
        veg_scene = resolved[0] if resolved is not None else scene
        veg_result = veg_service.process(
            scene=veg_scene,
            current_path=str(current_path),
            update_base=False,
            output_dir=str(output_dir) if output_dir is not None and self.config.vegetation_write_outputs else None,
            write_base_current=False,
            save_debug_images=bool(self.config.vegetation_save_debug_images),
            write_outputs=bool(self.config.vegetation_write_outputs),
        )
        return self._normalize_vegetation_metrics(veg_result)

    @staticmethod
    def _scale_min_pixels_for_image_size(
        min_pixels: int,
        source_image_size: int,
        target_image_size: int,
    ) -> int:
        if min_pixels <= 0:
            return 0
        if source_image_size <= 0 or target_image_size <= 0:
            return min_pixels
        scale = target_image_size / float(source_image_size)
        return max(1, int(round(min_pixels * scale)))

    @staticmethod
    def _has_base_update_label(labelme: Dict[str, Any]) -> bool:
        shapes = labelme.get("shapes", [])
        if not isinstance(shapes, list) or len(shapes) == 0:
            return False
        return any(str(shape.get("label", "")) in BASE_UPDATE_LABELS for shape in shapes if isinstance(shape, dict))

    @staticmethod
    def _should_update_base(labelme: Dict[str, Any], is_change: bool) -> bool:
        # Only interested change classes should replace the scene base.
        return bool(is_change) and OnlineChangeService._has_base_update_label(labelme)

    @staticmethod
    def _compute_contour_hsv_stats(image_bgr: np.ndarray, cnt: np.ndarray) -> Dict[str, Any]:
        h, w = image_bgr.shape[:2]
        local = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(local, [cnt.astype(np.int32)], 1)
        if int(local.sum()) <= 0:
            return {
                "pixel_count": 0,
                "green_ratio": 0.0,
                "yellow_ratio": 0.0,
                "red_ratio": 0.0,
                "gray_ratio": 0.0,
                "colorful_ratio": 0.0,
                "dominant_color": "",
            }

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        pixels = hsv[local > 0]
        if pixels.size == 0:
            return {
                "pixel_count": 0,
                "green_ratio": 0.0,
                "yellow_ratio": 0.0,
                "red_ratio": 0.0,
                "gray_ratio": 0.0,
                "colorful_ratio": 0.0,
                "dominant_color": "",
            }

        h_ch = pixels[:, 0]
        s_ch = pixels[:, 1]
        v_ch = pixels[:, 2]

        green = ((h_ch >= 35) & (h_ch <= 95) & (s_ch >= 40) & (v_ch >= 30))
        yellow = ((h_ch >= 15) & (h_ch < 35) & (s_ch >= 45) & (v_ch >= 45))
        red = (((h_ch < 15) | (h_ch >= 170)) & (s_ch >= 45) & (v_ch >= 40))
        gray = ((s_ch <= 45) & (v_ch >= 35) & (v_ch <= 230))

        pixel_count = float(len(pixels))
        green_ratio = float(green.sum() / pixel_count)
        yellow_ratio = float(yellow.sum() / pixel_count)
        red_ratio = float(red.sum() / pixel_count)
        gray_ratio = float(gray.sum() / pixel_count)
        colorful_ratio = green_ratio + yellow_ratio + red_ratio

        color_map = {
            "green": green_ratio,
            "yellow": yellow_ratio,
            "red": red_ratio,
        }
        dominant_color = max(color_map, key=color_map.get)
        if color_map[dominant_color] <= 0:
            dominant_color = ""

        return {
            "pixel_count": int(pixel_count),
            "green_ratio": green_ratio,
            "yellow_ratio": yellow_ratio,
            "red_ratio": red_ratio,
            "gray_ratio": gray_ratio,
            "colorful_ratio": colorful_ratio,
            "dominant_color": dominant_color,
        }

    def _apply_leaf_color_rule(
        self,
        instances: List[Dict[str, Any]],
        base_img_bgr: np.ndarray,
        current_img_bgr: np.ndarray,
    ) -> None:
        if not self.config.enable_leaf_color_rule:
            return
        for item in instances:
            label = str(item.get("label", ""))
            if label not in {"leaves", "change_color"}:
                continue
            cnt = item.get("cnt")
            if cnt is None or len(cnt) < 3:
                continue

            base_stats = self._compute_contour_hsv_stats(base_img_bgr, cnt)
            current_stats = self._compute_contour_hsv_stats(current_img_bgr, cnt)
            base_colorful = float(base_stats["colorful_ratio"])
            current_colorful = float(current_stats["colorful_ratio"])
            base_gray = float(base_stats["gray_ratio"])
            current_gray = float(current_stats["gray_ratio"])
            base_dom = str(base_stats["dominant_color"])
            current_dom = str(current_stats["dominant_color"])

            looks_like_leaf_drop = (
                base_colorful >= float(self.config.leaf_color_min_ratio)
                and current_gray >= float(self.config.leaf_gray_ratio_threshold)
                and (current_gray - base_gray) >= float(self.config.leaf_gray_increase_threshold)
                and (base_colorful - current_colorful) >= float(self.config.leaf_color_drop_threshold)
            )
            looks_like_color_shift = (
                base_colorful >= float(self.config.leaf_color_min_ratio)
                and current_colorful >= float(self.config.leaf_color_min_ratio)
                and base_dom != ""
                and current_dom != ""
                and base_dom != current_dom
                and abs(current_colorful - base_colorful) <= float(self.config.leaf_color_drop_threshold)
                and max(
                    abs(float(current_stats["green_ratio"]) - float(base_stats["green_ratio"])),
                    abs(float(current_stats["yellow_ratio"]) - float(base_stats["yellow_ratio"])),
                    abs(float(current_stats["red_ratio"]) - float(base_stats["red_ratio"])),
                ) >= float(self.config.leaf_hue_shift_threshold)
            )

            original_label = label
            if looks_like_leaf_drop:
                item["label"] = "leaves"
            elif looks_like_color_shift:
                item["label"] = "change_color"

            item["base_hsv_stats"] = base_stats
            item["current_hsv_stats"] = current_stats
            item["label_before_color_rule"] = original_label
            item["color_rule_applied"] = original_label != item["label"]

    @staticmethod
    def _normalized_kernel_size(kernel_size: int) -> int:
        kernel_size = int(kernel_size)
        if kernel_size <= 0:
            return 0
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def _merge_nearby_same_label_instances(
        self,
        instances: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
        special_export_labels: Set[str],
    ) -> List[Dict[str, Any]]:
        if not instances or not self.config.merge_same_label_instances:
            return instances

        kernel_size = self._normalized_kernel_size(self.config.merge_same_label_kernel)
        if kernel_size <= 1:
            return instances

        h, w = image_shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        passthrough: List[Dict[str, Any]] = []
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for ins in instances:
            label = str(ins.get("label", "change"))
            if label not in special_export_labels and self.allowed_labels and label not in self.allowed_labels:
                passthrough.append(ins)
                continue
            grouped.setdefault(label, []).append(ins)

        merged_instances: List[Dict[str, Any]] = []
        for label, label_instances in grouped.items():
            canvas = np.zeros((h, w), dtype=np.uint8)
            valid_instances: List[Dict[str, Any]] = []
            for ins in label_instances:
                cnt = ins.get("cnt")
                if cnt is None or len(cnt) < 3:
                    continue
                cv2.fillPoly(canvas, [cnt.astype(np.int32)], 1)
                valid_instances.append(ins)

            if not valid_instances:
                continue

            merged_mask = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cnt is None or len(cnt) < 3:
                    continue
                local = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(local, [cnt.astype(np.int32)], 1)
                pixels = int(local.sum())
                if pixels < self.config.min_instance_pixels:
                    continue

                overlapped: List[Dict[str, Any]] = []
                for ins in valid_instances:
                    ins_cnt = ins.get("cnt")
                    if ins_cnt is None or len(ins_cnt) < 3:
                        continue
                    ins_local = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(ins_local, [ins_cnt.astype(np.int32)], 1)
                    if int((local * ins_local).sum()) > 0:
                        overlapped.append(ins)

                if not overlapped:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                merged: Dict[str, Any] = {
                    "cnt": cnt,
                    "change_pixels": sum(int(item.get("change_pixels", 0)) for item in overlapped),
                    "group_id": None,
                    "label": label,
                    "proposal_box": [float(x), float(y), float(x + bw), float(y + bh)],
                }
                scores = [item.get("score") for item in overlapped if item.get("score") is not None]
                if scores:
                    merged["score"] = max(float(v) for v in scores)
                binary_change_scores = [
                    item.get("binary_change_score") for item in overlapped if item.get("binary_change_score") is not None
                ]
                if binary_change_scores:
                    merged["binary_change_score"] = max(float(v) for v in binary_change_scores)
                binary_no_change_scores = [
                    item.get("binary_no_change_score") for item in overlapped if item.get("binary_no_change_score") is not None
                ]
                if binary_no_change_scores:
                    merged["binary_no_change_score"] = min(float(v) for v in binary_no_change_scores)
                if any(item.get("color_rule_applied") for item in overlapped):
                    merged["color_rule_applied"] = True
                before_color = [item.get("label_before_color_rule") for item in overlapped if item.get("label_before_color_rule")]
                if before_color:
                    merged["label_before_color_rule"] = ",".join(sorted({str(v) for v in before_color}))
                before_slope_veg = [
                    item.get("label_before_slope_vegetation_rule")
                    for item in overlapped
                    if item.get("label_before_slope_vegetation_rule")
                ]
                if before_slope_veg:
                    merged["label_before_slope_vegetation_rule"] = ",".join(
                        sorted({str(v) for v in before_slope_veg})
                    )
                if any(item.get("slope_vegetation_rule_applied") for item in overlapped):
                    merged["slope_vegetation_rule_applied"] = True
                slope_decrease_rates = [
                    item.get("slope_vegetation_decrease_rate")
                    for item in overlapped
                    if item.get("slope_vegetation_decrease_rate") is not None
                ]
                if slope_decrease_rates:
                    merged["slope_vegetation_decrease_rate"] = max(float(v) for v in slope_decrease_rates)
                roi_overlap_pixels = [
                    item.get("roi_overlap_pixels") for item in overlapped if item.get("roi_overlap_pixels") is not None
                ]
                if roi_overlap_pixels:
                    merged["roi_overlap_pixels"] = max(int(v) for v in roi_overlap_pixels)
                roi_overlap_ratios = [
                    item.get("roi_overlap_ratio") for item in overlapped if item.get("roi_overlap_ratio") is not None
                ]
                if roi_overlap_ratios:
                    merged["roi_overlap_ratio"] = max(float(v) for v in roi_overlap_ratios)
                roi_coverage_ratios = [
                    item.get("roi_coverage_ratio") for item in overlapped if item.get("roi_coverage_ratio") is not None
                ]
                if roi_coverage_ratios:
                    merged["roi_coverage_ratio"] = max(float(v) for v in roi_coverage_ratios)
                merged_instances.append(merged)

        return passthrough + merged_instances

    @staticmethod
    def _assign_vegetation_location_labels(instances: List[Dict[str, Any]], roi_mask: Optional[np.ndarray]) -> None:
        if roi_mask is None:
            return
        h, w = roi_mask.shape[:2]
        for ins in instances:
            label = str(ins.get("label", "")).strip()
            if label not in VEGETATION_LOCATION_LABELS:
                continue
            cnt = ins.get("cnt")
            if cnt is None or len(cnt) < 3:
                continue
            local = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(local, [cnt.astype(np.int32)], 1)
            instance_pixels = int(local.sum())
            if instance_pixels <= 0:
                continue
            roi_overlap_pixels = int((local * roi_mask).sum())
            roi_overlap_ratio = roi_overlap_pixels / float(instance_pixels)
            ins["label_before_roi_rule"] = label
            ins["roi_overlap_pixels"] = roi_overlap_pixels
            ins["roi_overlap_ratio"] = roi_overlap_ratio
            ins["label"] = "slope_plant" if roi_overlap_ratio >= 0.5 else "other_plant"

    def _apply_slope_vegetation_decrease_rule(
        self,
        instances: List[Dict[str, Any]],
        roi_mask: Optional[np.ndarray],
        vegetation_metrics: Optional[Dict[str, Any]],
        base_img_bgr: np.ndarray,
        current_img_bgr: np.ndarray,
    ) -> None:
        if not self.config.enable_slope_vegetation_decrease_rule:
            return
        if roi_mask is None or vegetation_metrics is None:
            return

        decrease_rate = float(vegetation_metrics.get("p_decrease_rate") or 0.0)
        decrease_detected = bool(vegetation_metrics.get("decrease_detected"))
        if not decrease_detected and decrease_rate < float(self.config.slope_vegetation_decrease_threshold):
            return

        h, w = roi_mask.shape[:2]
        roi_pixels = int(roi_mask.sum())
        if roi_pixels <= 0:
            return

        overlap_threshold = float(self.config.slope_vegetation_roi_overlap_threshold)
        coverage_threshold = float(self.config.slope_vegetation_roi_coverage_threshold)
        base_green_min = float(self.config.slope_vegetation_base_green_min_ratio)
        current_green_max = float(self.config.slope_vegetation_current_green_max_ratio)
        green_drop_min = float(self.config.slope_vegetation_green_drop_min_ratio)
        gray_increase_min = float(self.config.slope_vegetation_gray_increase_min_ratio)
        override_label = str(self.config.slope_vegetation_override_label or "slope_plant").strip()
        excluded_source_labels = {
            str(label).strip()
            for label in self.config.slope_vegetation_excluded_source_labels
            if str(label).strip()
        }
        allowed_source_labels = {
            str(label).strip()
            for label in self.config.slope_vegetation_allowed_source_labels
            if str(label).strip()
        }
        if not override_label:
            return

        for ins in instances:
            original_label = str(ins.get("label", "")).strip()
            if original_label in excluded_source_labels:
                continue
            if allowed_source_labels and original_label not in allowed_source_labels:
                continue
            cnt = ins.get("cnt")
            if cnt is None or len(cnt) < 3:
                continue
            local = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(local, [cnt.astype(np.int32)], 1)
            instance_pixels = int(local.sum())
            if instance_pixels <= 0:
                continue

            overlap_pixels = int((local * roi_mask).sum())
            instance_overlap_ratio = overlap_pixels / float(instance_pixels)
            roi_coverage_ratio = overlap_pixels / float(roi_pixels)
            if instance_overlap_ratio < overlap_threshold or roi_coverage_ratio < coverage_threshold:
                continue

            base_stats = None
            current_stats = None
            if self.config.enable_slope_vegetation_hsv_bare_rule:
                base_stats = self._compute_contour_hsv_stats(base_img_bgr, cnt)
                current_stats = self._compute_contour_hsv_stats(current_img_bgr, cnt)
                base_green = float(base_stats.get("green_ratio", 0.0))
                current_green = float(current_stats.get("green_ratio", 0.0))
                base_gray = float(base_stats.get("gray_ratio", 0.0))
                current_gray = float(current_stats.get("gray_ratio", 0.0))
                if base_green < base_green_min:
                    continue
                if current_green > current_green_max:
                    continue
                if (base_green - current_green) < green_drop_min:
                    continue
                if (current_gray - base_gray) < gray_increase_min:
                    continue

            ins["label_before_slope_vegetation_rule"] = original_label
            ins["slope_vegetation_rule_applied"] = True
            ins["slope_vegetation_decrease_rate"] = decrease_rate
            if base_stats is not None:
                ins["base_hsv_stats"] = base_stats
            if current_stats is not None:
                ins["current_hsv_stats"] = current_stats
            ins["roi_overlap_pixels"] = overlap_pixels
            ins["roi_overlap_ratio"] = instance_overlap_ratio
            ins["roi_coverage_ratio"] = roi_coverage_ratio
            ins["label"] = override_label

    def _mask_to_labelme_json(
        self,
        mask: np.ndarray,
        image_path: Path,
        base_img_bgr: np.ndarray,
        current_img_bgr: np.ndarray,
        export_instances: bool = True,
        roi_mask: Optional[np.ndarray] = None,
        vegetation_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        h, w = mask.shape[:2]
        bin_mask = (mask > 0).astype(np.uint8)
        contours: List[np.ndarray] = []
        if export_instances:
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        forced_label = str(self.config.export_all_as_label or "").strip()
        binary_rejected_label = str(self.config.binary_rejected_label or "").strip()
        if forced_label:
            instances: List[Dict[str, Any]] = []
            for cnt in contours:
                if cnt is None or len(cnt) < 3:
                    continue
                local = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(local, [cnt.astype(np.int32)], 1)
                pixels = int(local.sum())
                if pixels < self.config.min_instance_pixels:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw <= 0 or bh <= 0:
                    continue
                instances.append(
                    {
                        "cnt": cnt,
                        "change_pixels": pixels,
                        "group_id": None,
                        "label": forced_label,
                        "score": None,
                        "proposal_box": [float(x), float(y), float(x + bw), float(y + bh)],
                    }
                )
            contours = []
        else:
            instances = []

        binary_meta: List[Dict[str, Any]] = []
        if not forced_label and self.binary_classifier is not None and contours:
            filtered_contours: List[np.ndarray] = []
            scaled_boxes: List[List[float]] = []
            original_boxes: List[List[float]] = []
            contour_pixels: List[int] = []
            scale_x = self.binary_classifier.input_size / float(max(1, w))
            scale_y = self.binary_classifier.input_size / float(max(1, h))
            for cnt in contours:
                if cnt is None or len(cnt) < 3:
                    continue
                local = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(local, [cnt.astype(np.int32)], 1)
                pixels = int(local.sum())
                if pixels < self.config.min_instance_pixels:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw <= 0 or bh <= 0:
                    continue
                filtered_contours.append(cnt)
                original_boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
                contour_pixels.append(pixels)
                scaled_boxes.append([x * scale_x, y * scale_y, (x + bw) * scale_x, (y + bh) * scale_y])

            if filtered_contours:
                binary_boxes = (
                    original_boxes
                    if getattr(self.binary_classifier, "roi_view_mode", "full_image") == "crop"
                    else scaled_boxes
                )
                binary_outputs = self.binary_classifier.predict_boxes(
                    base_img_bgr=base_img_bgr,
                    current_img_bgr=current_img_bgr,
                    boxes_xyxy=binary_boxes,
                    mask_bin=bin_mask,
                )
                kept_contours: List[np.ndarray] = []
                kept_meta: List[Dict[str, Any]] = []
                for cnt, box, pixels, output in zip(filtered_contours, original_boxes, contour_pixels, binary_outputs):
                    change_score = float(output.get("change_score", 0.0))
                    no_change_score = float(output.get("no_change_score", 0.0))
                    if change_score < float(self.config.binary_classifier_change_threshold):
                        if binary_rejected_label:
                            instances.append(
                                {
                                    "cnt": cnt,
                                    "change_pixels": pixels,
                                    "group_id": None,
                                    "label": binary_rejected_label,
                                    "score": None,
                                    "binary_change_score": change_score,
                                    "binary_no_change_score": no_change_score,
                                    "proposal_box": box,
                                }
                            )
                        continue
                    kept_contours.append(cnt)
                    kept_meta.append(
                        {
                            "change_score": change_score,
                            "no_change_score": no_change_score,
                            "proposal_box": box,
                        }
                    )
                contours = kept_contours
                binary_meta = kept_meta

        if not forced_label and self.classifier is not None:
            classified_instances = self.classifier.predict_instances(
                    base_img_bgr=base_img_bgr,
                    current_img_bgr=current_img_bgr,
                    mask_bin=bin_mask,
                    contours=contours,
                    min_instance_pixels=self.config.min_instance_pixels,
                    confidence_threshold=self.config.classifier_confidence_threshold,
                    contour_meta=binary_meta if binary_meta else None,
                )
            self._apply_leaf_color_rule(classified_instances, base_img_bgr, current_img_bgr)
            instances.extend(classified_instances)
        elif not forced_label:
            for idx, cnt in enumerate(contours):
                if cnt is None or len(cnt) < 3:
                    continue
                local = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(local, [cnt.astype(np.int32)], 1)
                pixels = int(local.sum())
                if pixels < self.config.min_instance_pixels:
                    continue
                accepted_label = str(self.config.binary_accepted_label or "change").strip() or "change"
                instances.append(
                    {
                        "cnt": cnt,
                        "change_pixels": pixels,
                        "group_id": None,
                        "label": accepted_label,
                        "score": None,
                    }
                )
                if idx < len(binary_meta):
                    instances[-1]["binary_change_score"] = binary_meta[idx]["change_score"]
                    instances[-1]["binary_no_change_score"] = binary_meta[idx]["no_change_score"]
                    instances[-1]["proposal_box"] = binary_meta[idx]["proposal_box"]

        self._assign_vegetation_location_labels(instances, roi_mask)
        self._apply_slope_vegetation_decrease_rule(
            instances,
            roi_mask,
            vegetation_metrics,
            base_img_bgr,
            current_img_bgr,
        )

        special_export_labels = {label for label in [forced_label, binary_rejected_label] if label}
        export_instances = self._merge_nearby_same_label_instances(instances, (h, w), special_export_labels)
        shapes = []
        for ins in export_instances:
            cnt = ins["cnt"]
            if cnt is None or len(cnt) < 3:
                continue
            label = str(ins.get("label", "change"))
            if label not in special_export_labels and self.allowed_labels and label not in self.allowed_labels:
                continue
            points = [[float(x), float(y)] for x, y in cnt.squeeze(1).tolist()]
            # Labelme only supports boolean shape flags in the edit dialog.
            # Keep debug/export metadata in description to avoid editLabel crashes.
            flags: Dict[str, bool] = {}
            if ins.get("color_rule_applied") is not None:
                flags["color_rule_applied"] = bool(ins["color_rule_applied"])
            description_payload: Dict[str, Any] = {
                "change_pixels": int(ins.get("change_pixels", 0)),
            }
            if ins.get("score") is not None:
                description_payload["score"] = round(float(ins["score"]), 6)
            if ins.get("binary_change_score") is not None:
                description_payload["binary_change_score"] = round(float(ins["binary_change_score"]), 6)
            if ins.get("binary_no_change_score") is not None:
                description_payload["binary_no_change_score"] = round(float(ins["binary_no_change_score"]), 6)
            if ins.get("proposal_box") is not None:
                description_payload["proposal_box"] = [round(float(v), 3) for v in ins["proposal_box"]]
            if ins.get("label_before_color_rule") is not None:
                description_payload["label_before_color_rule"] = str(ins["label_before_color_rule"])
            if ins.get("label_before_roi_rule") is not None:
                description_payload["label_before_roi_rule"] = str(ins["label_before_roi_rule"])
            if ins.get("label_before_slope_vegetation_rule") is not None:
                description_payload["label_before_slope_vegetation_rule"] = str(ins["label_before_slope_vegetation_rule"])
            if ins.get("slope_vegetation_rule_applied") is not None:
                description_payload["slope_vegetation_rule_applied"] = bool(ins["slope_vegetation_rule_applied"])
            if ins.get("slope_vegetation_decrease_rate") is not None:
                description_payload["slope_vegetation_decrease_rate"] = round(
                    float(ins["slope_vegetation_decrease_rate"]),
                    6,
                )
            if ins.get("roi_overlap_pixels") is not None:
                description_payload["roi_overlap_pixels"] = int(ins["roi_overlap_pixels"])
            if ins.get("roi_overlap_ratio") is not None:
                description_payload["roi_overlap_ratio"] = round(float(ins["roi_overlap_ratio"]), 6)
            if ins.get("roi_coverage_ratio") is not None:
                description_payload["roi_coverage_ratio"] = round(float(ins["roi_coverage_ratio"]), 6)
            if ins.get("base_hsv_stats") is not None:
                stats = ins["base_hsv_stats"]
                description_payload["base_hsv_stats"] = {
                    "green_ratio": round(float(stats.get("green_ratio", 0.0)), 4),
                    "yellow_ratio": round(float(stats.get("yellow_ratio", 0.0)), 4),
                    "red_ratio": round(float(stats.get("red_ratio", 0.0)), 4),
                    "gray_ratio": round(float(stats.get("gray_ratio", 0.0)), 4),
                    "colorful_ratio": round(float(stats.get("colorful_ratio", 0.0)), 4),
                    "dominant_color": str(stats.get("dominant_color", "")),
                }
            if ins.get("current_hsv_stats") is not None:
                stats = ins["current_hsv_stats"]
                description_payload["current_hsv_stats"] = {
                    "green_ratio": round(float(stats.get("green_ratio", 0.0)), 4),
                    "yellow_ratio": round(float(stats.get("yellow_ratio", 0.0)), 4),
                    "red_ratio": round(float(stats.get("red_ratio", 0.0)), 4),
                    "gray_ratio": round(float(stats.get("gray_ratio", 0.0)), 4),
                    "colorful_ratio": round(float(stats.get("colorful_ratio", 0.0)), 4),
                    "dominant_color": str(stats.get("dominant_color", "")),
                }
            shapes.append(
                {
                    "label": label,
                    "points": points,
                    "group_id": LABEL_GROUP_IDS.get(label, ins.get("group_id")),
                    "shape_type": "polygon",
                    "description": json.dumps(description_payload, ensure_ascii=False),
                    "flags": flags,
                    "line_color": None,
                    "fill_color": None,
                }
            )

        return (
            {
                "version": "3.15.0",
                "flags": {},
                "lineColor": [0, 255, 0, 128],
                "fillColor": [255, 0, 0, 128],
                "shapes": shapes,
                "imagePath": image_path.name,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w,
            },
            instances,
        )

    @staticmethod
    def _append_yolo_to_labelme(labelme: Dict[str, Any], yolo_result: Dict[str, Any]) -> None:
        detections = yolo_result.get("detections", [])
        if not isinstance(detections, list):
            detections = []

        for det in detections:
            box = det.get("box_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            labelme.setdefault("shapes", []).append(
                {
                    "label": str(det.get("label", "yolo_detection")),
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": LABEL_GROUP_IDS.get(str(det.get("label", "yolo_detection"))),
                    "shape_type": "rectangle",
                    "description": None,
                    "flags": {},
                    "line_color": None,
                    "fill_color": None,
                }
            )

        labelme["yolo_detection"] = {
            "has_detection": int(yolo_result.get("has_detection", 0)),
            "num_detections": int(yolo_result.get("num_detections", 0)),
            "class_counts": yolo_result.get("class_counts", {}),
        }

    @staticmethod
    def _render_labelme_outline_preview(
        image_bgr: np.ndarray,
        labelme: Dict[str, Any],
        line_color: Tuple[int, int, int] = (0, 0, 255),
        line_thickness: int = 4,
    ) -> np.ndarray:
        preview = image_bgr.copy()
        image_height, image_width = preview.shape[:2]
        for shape in labelme.get("shapes") or []:
            if not isinstance(shape, dict):
                continue
            points = np.asarray(shape.get("points") or [], dtype=np.float32)
            if points.shape[0] < 2:
                continue
            points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
            pts = np.round(points).astype(np.int32)
            shape_type = str(shape.get("shape_type") or "polygon").lower()
            if shape_type == "rectangle" and pts.shape[0] >= 2:
                x1, y1 = pts[0].tolist()
                x2, y2 = pts[1].tolist()
                cv2.rectangle(preview, (x1, y1), (x2, y2), line_color, line_thickness, cv2.LINE_AA)
            elif pts.shape[0] >= 3:
                cv2.polylines(
                    preview,
                    [pts.reshape(-1, 1, 2)],
                    True,
                    line_color,
                    line_thickness,
                    cv2.LINE_AA,
                )
        return preview

    def _update_scene_base(self, scene_dir: Path, current_path: Path) -> Path:
        new_base = scene_dir / current_path.name
        if new_base.resolve() != current_path.resolve():
            shutil.copy2(str(current_path), str(new_base))
        for p in self._list_scene_images(scene_dir):
            if p.resolve() != new_base.resolve():
                p.unlink()
        return new_base

    def process(
        self,
        scene: str,
        current_path: str,
        update_base: Optional[bool] = None,
        output_dir: Optional[str] = None,
        write_base_current: bool = True,
        write_mask_image: bool = True,
    ) -> Dict[str, Any]:
        current = Path(current_path).resolve()
        if not current.exists():
            raise FileNotFoundError(f"current image not found: {current}")

        scene_dir = self.data_root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(f"scene folder not found: {scene_dir}")

        base = self._get_single_base(scene_dir)
        mask = self._run_mask(base, current)

        current_stem = current.stem
        if output_dir is None:
            out_dir = self._next_output_dir(scene, current_stem)
        else:
            out_dir = Path(output_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
        base_out = out_dir / f"{current_stem}_base{base.suffix}"
        current_out = out_dir / f"{current_stem}_current{current.suffix}"
        masked_out = out_dir / f"{current_stem}_mask{current.suffix}"
        json_out = out_dir / f"{current_stem}_mask.json"

        if write_base_current:
            shutil.copy2(str(base), str(base_out))
            shutil.copy2(str(current), str(current_out))

        base_img = self._read_image_unicode(base, cv2.IMREAD_COLOR)
        current_img = self._read_image_unicode(current, cv2.IMREAD_COLOR)
        backend_name = str(self.config.proposal_backend).strip().lower()
        if backend_name in {"pair_change_seg", "pair_change", "mobile_sam_pair"}:
            mask_orig = (mask > 0).astype(np.uint8)
        else:
            mask_orig = self._unletterbox_mask((mask > 0).astype(np.uint8), (current_img.shape[0], current_img.shape[1]))
        raw_is_change, raw_change_pixels, raw_change_ratio = self._detect_change(
            mask_orig,
            self.config.min_change_pixels,
            self.config.min_change_ratio,
        )
        roi_mask = self._load_base_roi_mask(scene, current_img.shape[0], current_img.shape[1])

        veg_metrics = self._compute_vegetation_coverage_metrics(
            scene=scene,
            current_path=current,
            instances=[],
            image_height=current_img.shape[0],
            image_width=current_img.shape[1],
            output_dir=out_dir,
        )
        labelme, _ = self._mask_to_labelme_json(
            mask=mask_orig,
            image_path=current_out,
            base_img_bgr=base_img,
            current_img_bgr=current_img,
            export_instances=(raw_is_change or not self.config.require_raw_change_for_instances),
            roi_mask=roi_mask,
            vegetation_metrics=veg_metrics,
        )
        if veg_metrics is not None:
            labelme["vegetation_metrics"] = veg_metrics

        if backend_name in {"pair_change_seg", "pair_change", "mobile_sam_pair"}:
            scaled_min_change_pixels = int(self.config.min_change_pixels)
        else:
            scaled_min_change_pixels = self._scale_min_pixels_for_image_size(
                min_pixels=self.config.min_change_pixels,
                source_image_size=mask.size,
                target_image_size=mask_orig.size,
            )
        is_change, change_pixels, change_ratio = self._detect_change_from_exported_shapes(
            labelme=labelme,
            image_size=mask_orig.size,
            min_pixels=scaled_min_change_pixels,
            min_ratio=self.config.min_change_ratio,
        )
        if self._has_base_update_label(labelme):
            is_change = True
        base_update_recommended = self._should_update_base(labelme, is_change)
        should_commit_base_update = self.config.update_base_on_change if update_base is None else bool(update_base)
        do_update_base = bool(should_commit_base_update) and base_update_recommended

        base_after = base
        if do_update_base:
            base_after = self._update_scene_base(scene_dir, current)

        yolo_result = None
        yolo_service = self._get_yolo_service()
        if yolo_service is not None:
            yolo_result = yolo_service.predict_image(str(current))
            self._append_yolo_to_labelme(labelme, yolo_result)
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(labelme, f, ensure_ascii=False, indent=2)
        if write_mask_image:
            preview = self._render_labelme_outline_preview(current_img, labelme)
            self._write_image_unicode(masked_out, preview)

        result = {
            "scene": scene,
            "proposal_backend": backend_name or "gescf",
            "base_before": str(base),
            "current": str(current),
            "change": int(is_change),
            "change_pixels": change_pixels,
            "change_ratio": float(change_ratio),
            "raw_change": int(raw_is_change),
            "raw_change_pixels": raw_change_pixels,
            "raw_change_ratio": float(raw_change_ratio),
            "json_shapes": len(labelme.get("shapes", [])),
            "base_update_recommended": int(base_update_recommended),
            "base_updated": int(do_update_base),
            "base_after": str(base_after),
            "output_dir": str(out_dir),
        }
        if veg_metrics is not None:
            result["vegetation_metrics"] = veg_metrics
        if yolo_result is not None:
            result["yolo_detection"] = labelme.get("yolo_detection")
        return result
