import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


IMAGE_SIZE = 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class VegetationConfig:
    data_root: str
    output_root: str
    weights: str
    device: str = "cuda"
    change_threshold_percent: float = 5.0
    decrease_threshold: float = 0.05
    increase_threshold: float = 0.05
    mask_threshold: float = 0.5
    update_base_on_change: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VegetationConfig":
        return cls(**data)

    @classmethod
    def from_json(cls, path: Path) -> "VegetationConfig":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)


class VegetationCoverageService:
    def __init__(self, config: VegetationConfig, config_base_dir: Optional[Path] = None) -> None:
        self.config = config
        self.config_base_dir = config_base_dir or Path.cwd()

        self.data_root = self._resolve_path(config.data_root)
        self.output_root = self._resolve_path(config.output_root)
        self.weights_path = self._resolve_path(config.weights)

        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"weights not found: {self.weights_path}")

        vendor_root = Path(__file__).resolve().parent / "vendor"
        if (vendor_root / "mobile_sam").exists() and str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        from mobile_sam.build_sam import build_sam_vit_t  # type: ignore

        self._build_sam_vit_t = build_sam_vit_t
        self.device = torch.device(
            "cuda" if config.device.lower().startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model(self.weights_path, self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "VegetationCoverageService":
        path = Path(config_path).resolve()
        cfg = VegetationConfig.from_json(path)
        return cls(cfg, config_base_dir=path.parent)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path.resolve()
        return (self.config_base_dir / path).resolve()

    def _load_model(self, weights_path: Path, device: torch.device):
        model = self._build_sam_vit_t(checkpoint=None)
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _normalize_image(image: Image.Image) -> Image.Image:
        return image.convert("RGB") if image.mode == "RGBA" else image

    @staticmethod
    def _render_overlay(image: Image.Image, mask_np: np.ndarray) -> Image.Image:
        overlay_np = np.array(image).copy()
        color = np.array([0, 255, 0], dtype=np.uint8)
        fg = mask_np == 1
        overlay_np[fg] = (overlay_np[fg] * 0.7 + color * 0.3).astype(np.uint8)
        return Image.fromarray(overlay_np)

    @staticmethod
    def _mask_to_image(mask_np: np.ndarray) -> Image.Image:
        return Image.fromarray((mask_np * 255).astype(np.uint8))

    @staticmethod
    def _build_points(
        original_size: Tuple[int, int],
        foreground_points: Sequence[Tuple[int, int]],
        background_points: Sequence[Tuple[int, int]],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        width, height = original_size
        points: List[List[int]] = []
        labels: List[int] = []

        for x0, y0 in foreground_points:
            points.append([int(x0 * IMAGE_SIZE / width), int(y0 * IMAGE_SIZE / height)])
            labels.append(1)
        for x0, y0 in background_points:
            points.append([int(x0 * IMAGE_SIZE / width), int(y0 * IMAGE_SIZE / height)])
            labels.append(0)

        if not points:
            return None, None
        return (
            torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(0),
        )

    def _predict_mask(
        self,
        image: Image.Image,
        foreground_points: Sequence[Tuple[int, int]] = (),
        background_points: Sequence[Tuple[int, int]] = (),
    ) -> Tuple[np.ndarray, Image.Image, Image.Image]:
        image = self._normalize_image(image)
        original_size = image.size

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        points_tensor, labels_tensor = self._build_points(
            original_size, foreground_points, background_points, self.device
        )

        with torch.no_grad():
            processed_images = (image_tensor - self.model.pixel_mean) / self.model.pixel_std
            image_embeddings = self.model.image_encoder(processed_images)

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(points_tensor, labels_tensor) if points_tensor is not None else None,
                boxes=None,
                masks=None,
            )

            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upsampled_masks = torch.nn.functional.interpolate(
                low_res_masks,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )

            binary_mask = (torch.sigmoid(upsampled_masks) > self.config.mask_threshold).float()
            binary_mask = torch.nn.functional.interpolate(
                binary_mask,
                size=(original_size[1], original_size[0]),
                mode="nearest",
            )
            mask_np = binary_mask.squeeze().cpu().numpy().astype(np.uint8)

        overlay = self._render_overlay(image, mask_np)
        mask_img = self._mask_to_image(mask_np)
        return mask_np, overlay, mask_img

    @staticmethod
    def _compute_coverage(mask_np: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> float:
        if roi_mask is None:
            return float(mask_np.mean())
        denom = int(roi_mask.sum())
        if denom <= 0:
            return 0.0
        return float((mask_np * roi_mask).sum() / float(denom))

    @staticmethod
    def _build_change_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[Image.Image, int, int, np.ndarray]:
        decrease = (mask_a == 1) & (mask_b == 0)
        increase = (mask_a == 0) & (mask_b == 1)

        change_mask = np.zeros((mask_a.shape[0], mask_a.shape[1], 3), dtype=np.uint8)
        change_mask[decrease] = [255, 0, 0]
        change_mask[increase] = [0, 0, 255]

        updated_mask = mask_a.copy()
        updated_mask[decrease] = 0
        updated_mask[increase] = 1
        return Image.fromarray(change_mask), int(decrease.sum()), int(increase.sum()), updated_mask

    @staticmethod
    def _list_scene_images(scene_dir: Path) -> List[Path]:
        return sorted([p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

    @staticmethod
    def _list_scene_jsons(scene_dir: Path) -> List[Path]:
        return sorted([p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])

    def _get_single_base(self, scene_dir: Path) -> Path:
        images = self._list_scene_images(scene_dir)
        if len(images) != 1:
            raise RuntimeError(f"Expected 1 base image in {scene_dir}, found {len(images)}")
        return images[0]

    def _find_scene_roi_json(self, scene_dir: Path, base_path: Path) -> Optional[Path]:
        json_files = self._list_scene_jsons(scene_dir)
        if not json_files:
            return None

        stem_match = [p for p in json_files if p.stem == base_path.stem]
        if stem_match:
            return stem_match[0]
        return json_files[0]

    @staticmethod
    def _build_roi_mask_from_json(
        roi_json_path: Path,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        with roi_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        shapes = payload.get("shapes", [])
        if not isinstance(shapes, list) or len(shapes) == 0:
            raise RuntimeError(f"ROI json has no shapes: {roi_json_path}")

        ref_h = int(payload.get("imageHeight") or target_height)
        ref_w = int(payload.get("imageWidth") or target_width)
        if ref_h <= 0 or ref_w <= 0:
            ref_h = target_height
            ref_w = target_width

        sx = target_width / float(ref_w)
        sy = target_height / float(ref_h)

        canvas = Image.new("L", (target_width, target_height), 0)
        draw = ImageDraw.Draw(canvas)
        valid_shape_count = 0

        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            pts = shape.get("points")
            if not isinstance(pts, list) or len(pts) < 2:
                continue

            shape_type = str(shape.get("shape_type", "polygon")).lower()
            scaled_points: List[Tuple[float, float]] = []
            for pt in pts:
                if not isinstance(pt, list) or len(pt) != 2:
                    continue
                x = float(pt[0]) * sx
                y = float(pt[1]) * sy
                scaled_points.append((x, y))

            if shape_type == "rectangle":
                if len(scaled_points) < 2:
                    continue
                (x1, y1), (x2, y2) = scaled_points[0], scaled_points[1]
                draw.rectangle([x1, y1, x2, y2], fill=1)
                valid_shape_count += 1
            else:
                if len(scaled_points) < 3:
                    continue
                draw.polygon(scaled_points, fill=1)
                valid_shape_count += 1

        if valid_shape_count == 0:
            raise RuntimeError(f"ROI json has no valid polygon/rectangle shapes: {roi_json_path}")

        return (np.array(canvas, dtype=np.uint8) > 0).astype(np.uint8)

    def _next_output_dir(self, scene_name: str) -> Path:
        scene_root = self.output_root / f"{scene_name}_images"
        scene_root.mkdir(parents=True, exist_ok=True)
        max_idx = 0
        for p in scene_root.iterdir():
            if p.is_dir() and p.name.startswith("CD_") and p.name[3:].isdigit():
                max_idx = max(max_idx, int(p.name[3:]))
        out = scene_root / f"CD_{max_idx + 1}"
        out.mkdir(parents=True, exist_ok=False)
        return out

    def _update_scene_base(self, scene_dir: Path, current_path: Path) -> Path:
        new_base = scene_dir / current_path.name
        if new_base.resolve() != current_path.resolve():
            shutil.copy2(str(current_path), str(new_base))
        for path in self._list_scene_images(scene_dir):
            if path.resolve() != new_base.resolve():
                path.unlink()
        return new_base

    def process(
        self,
        scene: str,
        current_path: str,
        fg_points_a: Sequence[Tuple[int, int]] = (),
        bg_points_a: Sequence[Tuple[int, int]] = (),
        fg_points_b: Sequence[Tuple[int, int]] = (),
        bg_points_b: Sequence[Tuple[int, int]] = (),
        update_base: Optional[bool] = None,
        output_dir: Optional[str] = None,
        write_base_current: bool = True,
        save_debug_images: bool = True,
        write_outputs: bool = True,
    ) -> Dict[str, Any]:
        scene_dir = self.data_root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(f"scene folder not found: {scene_dir}")

        current = Path(current_path).resolve()
        if not current.exists():
            raise FileNotFoundError(f"current image not found: {current}")

        base = self._get_single_base(scene_dir)
        image_a = Image.open(base)
        image_b = Image.open(current)
        roi_json_path = self._find_scene_roi_json(scene_dir, base)

        mask_a, overlay_a, mask_a_img = self._predict_mask(image_a, fg_points_a, bg_points_a)
        mask_b, overlay_b, mask_b_img = self._predict_mask(image_b, fg_points_b, bg_points_b)
        raw_mask_a = mask_a.copy()
        raw_mask_b = mask_b.copy()
        raw_overlay_a = overlay_a
        raw_overlay_b = overlay_b
        raw_mask_a_img = mask_a_img
        raw_mask_b_img = mask_b_img

        roi_mask_a: Optional[np.ndarray] = None
        roi_mask_b: Optional[np.ndarray] = None
        if roi_json_path is not None:
            roi_mask_a = self._build_roi_mask_from_json(roi_json_path, mask_a.shape[0], mask_a.shape[1])
            roi_mask_b = self._build_roi_mask_from_json(roi_json_path, mask_b.shape[0], mask_b.shape[1])
            mask_a = (mask_a * roi_mask_a).astype(np.uint8)
            mask_b = (mask_b * roi_mask_b).astype(np.uint8)
            overlay_a = self._render_overlay(image_a, mask_a)
            overlay_b = self._render_overlay(image_b, mask_b)
            mask_a_img = self._mask_to_image(mask_a)
            mask_b_img = self._mask_to_image(mask_b)

        coverage_a = self._compute_coverage(mask_a, roi_mask_a)
        coverage_b = self._compute_coverage(mask_b, roi_mask_b)
        delta = coverage_b - coverage_a

        change_mask_img, _, _, updated_mask = self._build_change_mask(mask_a, mask_b)

        total_pixels = int(roi_mask_a.sum()) if roi_mask_a is not None else mask_a.size
        if total_pixels <= 0:
            raise RuntimeError("ROI pixel count is zero. Check ROI json polygons.")
        base_veg_pixels = int(mask_a.sum())
        base_nonveg_pixels = int(total_pixels - base_veg_pixels)

        decrease_bin = ((mask_a == 1) & (mask_b == 0)).astype(np.uint8)  # red
        increase_bin = ((mask_a == 0) & (mask_b == 1)).astype(np.uint8)  # blue
        decrease_pixels = int(decrease_bin.sum())
        increase_pixels = int(increase_bin.sum())

        # User-defined rates:
        # decrease_rate = red / b, increase_rate = blue / a
        decrease_rate = decrease_pixels / float(base_nonveg_pixels) if base_nonveg_pixels > 0 else 0.0
        increase_rate = increase_pixels / float(base_veg_pixels) if base_veg_pixels > 0 else 0.0

        decrease_detected = decrease_rate >= float(self.config.decrease_threshold)
        increase_detected = increase_rate >= float(self.config.increase_threshold)
        detected = bool(decrease_detected or increase_detected)

        out_dir: Optional[Path] = None
        debug_files: Dict[str, str] = {}
        if write_outputs:
            if output_dir is None:
                out_dir = self._next_output_dir(scene)
            else:
                out_dir = Path(output_dir).resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
            stem = current.stem

            base_out = out_dir / f"{stem}_base{base.suffix}"
            current_out = out_dir / f"{stem}_current{current.suffix}"
            overlay_a_out = out_dir / f"{stem}_overlay_base.png"
            overlay_b_out = out_dir / f"{stem}_overlay_current.png"
            mask_a_out = out_dir / f"{stem}_mask_base.png"
            mask_b_out = out_dir / f"{stem}_mask_current.png"
            overlay_a_raw_out = out_dir / f"{stem}_overlay_base_raw.png"
            overlay_b_raw_out = out_dir / f"{stem}_overlay_current_raw.png"
            mask_a_raw_out = out_dir / f"{stem}_mask_base_raw.png"
            mask_b_raw_out = out_dir / f"{stem}_mask_current_raw.png"
            roi_mask_out = out_dir / f"{stem}_roi_mask.png"
            change_mask_out = out_dir / f"{stem}_change_mask.png"
            updated_mask_out = out_dir / f"{stem}_updated_mask.png"
            metrics_out = out_dir / f"{stem}_metrics.json"

            out_dir.mkdir(parents=True, exist_ok=True)
            if write_base_current:
                shutil.copy2(str(base), str(base_out))
                shutil.copy2(str(current), str(current_out))
                debug_files["base_image"] = base_out.name
                debug_files["current_image"] = current_out.name
            change_mask_img.save(change_mask_out)
            debug_files["change_mask"] = change_mask_out.name
            if save_debug_images:
                raw_overlay_a.save(overlay_a_raw_out)
                raw_overlay_b.save(overlay_b_raw_out)
                raw_mask_a_img.save(mask_a_raw_out)
                raw_mask_b_img.save(mask_b_raw_out)
                overlay_a.save(overlay_a_out)
                overlay_b.save(overlay_b_out)
                mask_a_img.save(mask_a_out)
                mask_b_img.save(mask_b_out)
                Image.fromarray((updated_mask * 255).astype(np.uint8)).save(updated_mask_out)
                debug_files.update(
                    {
                        "overlay_base_raw": overlay_a_raw_out.name,
                        "overlay_current_raw": overlay_b_raw_out.name,
                        "mask_base_raw": mask_a_raw_out.name,
                        "mask_current_raw": mask_b_raw_out.name,
                        "overlay_base": overlay_a_out.name,
                        "overlay_current": overlay_b_out.name,
                        "mask_base": mask_a_out.name,
                        "mask_current": mask_b_out.name,
                        "updated_mask": updated_mask_out.name,
                    }
                )
                if roi_mask_a is not None:
                    Image.fromarray((roi_mask_a * 255).astype(np.uint8)).save(roi_mask_out)
                    debug_files["roi_mask"] = roi_mask_out.name

        summary = {
            "scene": scene,
            "base_before": str(base),
            "current": str(current),
            "roi_json": str(roi_json_path) if roi_json_path is not None else None,
            "roi_pixels": int(total_pixels),
            "a_base_veg_pixels": int(base_veg_pixels),
            "b_base_nonveg_pixels": int(base_nonveg_pixels),
            "coverage_a": coverage_a,
            "coverage_b": coverage_b,
            "coverage_delta": delta,
            "decrease_rate": decrease_rate,
            "increase_rate": increase_rate,
            "decrease_threshold": float(self.config.decrease_threshold),
            "increase_threshold": float(self.config.increase_threshold),
            "decrease_detected": bool(decrease_detected),
            "increase_detected": bool(increase_detected),
            "change_detected": bool(detected),
            "device": str(self.device),
            "weights": str(self.weights_path),
            "output_dir": str(out_dir) if out_dir is not None else None,
            "debug_files": debug_files,
        }
        if write_outputs and out_dir is not None:
            with metrics_out.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        base_update_recommended = bool(detected)
        # If caller doesn't specify, follow config default (safer for integrations).
        if update_base is None:
            do_update_base = bool(self.config.update_base_on_change) and base_update_recommended
        else:
            do_update_base = bool(update_base) and base_update_recommended

        base_after = base
        if do_update_base:
            base_after = self._update_scene_base(scene_dir, current)
        summary["base_update_recommended"] = int(base_update_recommended)
        summary["base_updated"] = int(do_update_base)
        summary["base_after"] = str(base_after)
        return summary
