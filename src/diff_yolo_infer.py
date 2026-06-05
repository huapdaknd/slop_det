import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .model_infer import LABEL_GROUP_IDS, OnlineChangeService
from .yolo_infer import YoloDetectionService


EDGE_OUTPUT_LABELS = {"rock_spalling", "slope_plant", "other_plant"}
YOLO_OUTPUT_LABELS = {"rock_fall", "rock_spalling", "landslide"}


class DiffYoloClassificationService:
    """Classify edge-change regions with YOLO rock_fall / landslide detections."""

    def __init__(
        self,
        cd_service: OnlineChangeService,
        yolo_service: YoloDetectionService,
        min_overlap_ratio: float = 0.15,
        min_overlap_pixels: int = 100,
    ) -> None:
        self.cd_service = cd_service
        self.yolo_service = yolo_service
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.min_overlap_pixels = int(min_overlap_pixels)

    @classmethod
    def from_config_files(
        cls,
        cd_config_path: str,
        yolo_config_path: str,
        min_overlap_ratio: float = 0.15,
        min_overlap_pixels: int = 100,
    ) -> "DiffYoloClassificationService":
        return cls(
            cd_service=OnlineChangeService.from_config_file(cd_config_path),
            yolo_service=YoloDetectionService.from_config_file(yolo_config_path),
            min_overlap_ratio=min_overlap_ratio,
            min_overlap_pixels=min_overlap_pixels,
        )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _shape_to_mask(shape: Dict[str, Any], height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.asarray(shape.get("points") or [], dtype=np.float32)
        if points.size == 0:
            return mask

        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        shape_type = str(shape.get("shape_type") or "polygon").lower()
        if shape_type == "rectangle" and len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            pt1 = int(round(min(x1, x2))), int(round(min(y1, y2)))
            pt2 = int(round(max(x1, x2))), int(round(max(y1, y2)))
            cv2.rectangle(mask, pt1, pt2, 1, thickness=-1)
        elif len(points) >= 3:
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        return mask

    @staticmethod
    def _box_bounds(box_xyxy: Sequence[float], height: int, width: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        left = int(max(0, min(width - 1, np.floor(min(x1, x2)))))
        top = int(max(0, min(height - 1, np.floor(min(y1, y2)))))
        right = int(max(left + 1, min(width, np.ceil(max(x1, x2)))))
        bottom = int(max(top + 1, min(height, np.ceil(max(y1, y2)))))
        return left, top, right, bottom

    @staticmethod
    def _box_area(bounds: Tuple[int, int, int, int]) -> int:
        left, top, right, bottom = bounds
        return max(0, right - left) * max(0, bottom - top)

    @staticmethod
    def _parse_description(raw: Any) -> Dict[str, Any]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError:
            return {"raw_description": str(raw)}

    @staticmethod
    def _red_mask_from_overlay(
        mask_image_path: Path,
        height: int,
        width: int,
        current_image_path: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        if not mask_image_path.exists():
            return None
        image = cv2.imdecode(np.fromfile(str(mask_image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

        blue = image[:, :, 0].astype(np.int16)
        green = image[:, :, 1].astype(np.int16)
        red = image[:, :, 2].astype(np.int16)
        mask = ((red >= 180) & (red - green >= 80) & (red - blue >= 80)).astype(np.uint8)
        if current_image_path is not None and current_image_path.exists():
            current = cv2.imdecode(np.fromfile(str(current_image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if current is not None:
                if current.shape[:2] != (height, width):
                    current = cv2.resize(current, (width, height), interpolation=cv2.INTER_LINEAR)
                diff = np.abs(image.astype(np.int16) - current.astype(np.int16)).sum(axis=2)
                mask = (mask & (diff >= 30)).astype(np.uint8)
        return mask

    def _load_change_regions(
        self,
        scene: str,
        edge_json_path: Path,
        mask_image_path: Optional[Path] = None,
        current_image_path: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
        payload = self._read_json(edge_json_path)
        height = int(payload.get("imageHeight") or 0)
        width = int(payload.get("imageWidth") or 0)
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid LabelMe image size in {edge_json_path}")

        red_mask = (
            self._red_mask_from_overlay(mask_image_path, height, width, current_image_path)
            if mask_image_path
            else None
        )
        regions: List[Dict[str, Any]] = []
        union_mask = np.zeros((height, width), dtype=np.uint8)
        for idx, shape in enumerate(payload.get("shapes") or []):
            label = str(shape.get("label") or "").strip()
            if label not in EDGE_OUTPUT_LABELS:
                continue
            mask = self._shape_to_mask(shape, height, width)
            pixels = int(mask.sum())
            if pixels <= 0:
                continue
            union_mask[mask > 0] = 1
            regions.append(
                {
                    "index": idx,
                    "shape": shape,
                    "mask": mask,
                    "pixels": pixels,
                    "matched_detection_indices": [],
                }
            )
        if red_mask is not None and int(red_mask.sum()) > 0:
            union_mask = red_mask.astype(np.uint8)
        return payload, regions, union_mask

    def _filter_yolo_detections(
        self,
        detections: Sequence[Dict[str, Any]],
        change_regions: List[Dict[str, Any]],
        change_mask: np.ndarray,
    ) -> List[Dict[str, Any]]:
        height, width = change_mask.shape[:2]
        kept: List[Dict[str, Any]] = []
        for det_idx, det in enumerate(detections):
            if str(det.get("label") or "") not in YOLO_OUTPUT_LABELS:
                continue
            bounds = self._box_bounds(det["box_xyxy"], height, width)
            area = self._box_area(bounds)
            if area <= 0:
                continue

            left, top, right, bottom = bounds
            overlap_pixels = int(change_mask[top:bottom, left:right].sum())
            overlap_ratio = overlap_pixels / float(area)
            if overlap_pixels < self.min_overlap_pixels or overlap_ratio < self.min_overlap_ratio:
                continue

            best_region_idx: Optional[int] = None
            best_region_overlap = 0
            for region in change_regions:
                region_overlap = int(region["mask"][top:bottom, left:right].sum())
                if region_overlap > best_region_overlap:
                    best_region_overlap = region_overlap
                    best_region_idx = int(region["index"])

            item = dict(det)
            item["detection_index"] = det_idx
            item["change_overlap_pixels"] = overlap_pixels
            item["change_overlap_ratio"] = overlap_ratio
            item["matched_change_region_index"] = best_region_idx
            item["matched_change_region_overlap_pixels"] = best_region_overlap
            kept.append(item)

            if best_region_idx is not None:
                for region in change_regions:
                    if int(region["index"]) == best_region_idx:
                        region["matched_detection_indices"].append(len(kept) - 1)
                        break
        return kept

    def _build_combined_labelme(
        self,
        edge_payload: Dict[str, Any],
        change_regions: List[Dict[str, Any]],
        diff_detections: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        shapes: List[Dict[str, Any]] = []
        for region in change_regions:
            matched = [diff_detections[i] for i in region["matched_detection_indices"]]
            original_shape = dict(region["shape"])
            description = self._parse_description(original_shape.get("description"))
            description["source"] = "edge"
            description["change_pixels"] = int(region["pixels"])
            description["edge_alarm"] = True
            description["matched_yolo_detections"] = [
                {
                    "label": str(det.get("label")),
                    "score": round(float(det.get("score") or 0.0), 6),
                    "box_xyxy": [round(float(v), 3) for v in det.get("box_xyxy", [])],
                    "change_overlap_ratio": round(float(det.get("change_overlap_ratio") or 0.0), 6),
                    "change_overlap_pixels": int(det.get("change_overlap_pixels") or 0),
                }
                for det in matched
            ]
            original_shape["description"] = json.dumps(description, ensure_ascii=False)
            original_shape["flags"] = original_shape.get("flags") or {}
            original_shape["group_id"] = LABEL_GROUP_IDS.get(str(original_shape.get("label") or ""), original_shape.get("group_id"))
            shapes.append(original_shape)

        for det in diff_detections:
            x1, y1, x2, y2 = [float(v) for v in det.get("box_xyxy", [])]
            description = {
                "source": "yolo",
                "score": round(float(det.get("score") or 0.0), 6),
                "class_id": int(det.get("class_id") or 0),
                "box_xyxy": [round(v, 3) for v in [x1, y1, x2, y2]],
                "overlaps_change": True,
                "change_overlap_ratio": round(float(det.get("change_overlap_ratio") or 0.0), 6),
                "change_overlap_pixels": int(det.get("change_overlap_pixels") or 0),
                "matched_change_region_index": det.get("matched_change_region_index"),
            }
            shapes.append(
                {
                    "label": str(det.get("label") or "yolo_detection"),
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": LABEL_GROUP_IDS.get(str(det.get("label") or "")),
                    "shape_type": "rectangle",
                    "description": json.dumps(description, ensure_ascii=False),
                    "flags": {},
                    "line_color": None,
                    "fill_color": None,
                }
            )

        combined = dict(edge_payload)
        combined["version"] = combined.get("version", "3.15.0")
        combined["flags"] = combined.get("flags") or {}
        combined["lineColor"] = combined.get("lineColor", [0, 255, 0, 128])
        combined["fillColor"] = combined.get("fillColor", [255, 0, 0, 128])
        combined["shapes"] = shapes
        combined["imageData"] = None
        combined["imageHeight"] = int(edge_payload.get("imageHeight") or 0)
        combined["imageWidth"] = int(edge_payload.get("imageWidth") or 0)
        return combined

    def process(
        self,
        scene: str,
        current_path: str,
        output_dir: Optional[str] = None,
        update_base: Optional[bool] = None,
        run_yolo_when_no_change: bool = False,
    ) -> Dict[str, Any]:
        current = Path(current_path).resolve()
        out_dir = Path(output_dir).resolve() if output_dir else None
        edge_result = self.cd_service.process(
            scene=scene,
            current_path=str(current),
            update_base=update_base,
            output_dir=str(out_dir) if out_dir is not None else None,
            write_base_current=True,
            write_mask_image=True,
        )
        edge_output_dir = Path(edge_result["output_dir"]).resolve()

        edge_json_path = edge_output_dir / f"{current.stem}_mask.json"
        mask_image_path = edge_output_dir / f"{current.stem}_mask{current.suffix}"
        current_image_path = edge_output_dir / f"{current.stem}_current{current.suffix}"
        edge_payload, change_regions, change_mask = self._load_change_regions(
            scene=scene,
            edge_json_path=edge_json_path,
            mask_image_path=mask_image_path,
            current_image_path=current_image_path,
        )
        has_red_change = int(change_mask.sum()) > 0
        should_run_yolo = has_red_change or run_yolo_when_no_change

        raw_detections: List[Dict[str, Any]] = []
        if should_run_yolo:
            result = next(iter(self.yolo_service._predict_results([current])))
            raw_detections = self.yolo_service._results_to_detections(result)

        diff_detections = self._filter_yolo_detections(raw_detections, change_regions, change_mask)
        combined_json = self._build_combined_labelme(
            edge_payload=edge_payload,
            change_regions=change_regions,
            diff_detections=diff_detections,
        )

        self._write_json(edge_json_path, combined_json)

        should_commit_base_update = (
            self.cd_service.config.update_base_on_change if update_base is None else bool(update_base)
        )
        edge_base_updated = bool(edge_result.get("base_updated"))
        yolo_base_update_recommended = bool(diff_detections)
        yolo_base_updated = False
        base_after = Path(edge_result.get("base_after") or edge_result.get("base_before") or "")
        if bool(should_commit_base_update) and yolo_base_update_recommended and not edge_base_updated:
            scene_dir = self.cd_service.data_root / scene
            base_after = self.cd_service._update_scene_base(scene_dir, current)
            yolo_base_updated = True

        class_counts = {label: 0 for label in sorted(EDGE_OUTPUT_LABELS | YOLO_OUTPUT_LABELS)}
        for region in change_regions:
            label = str(region["shape"].get("label") or "")
            if label in class_counts:
                class_counts[label] += 1
        for det in diff_detections:
            label = str(det.get("label") or "yolo_detection")
            class_counts[label] = class_counts.get(label, 0) + 1

        summary = {
            "scene": scene,
            "current": str(current),
            "output_dir": str(edge_output_dir),
            "edge": edge_result,
            "change_regions": len(change_regions),
            "raw_yolo_detections": len(raw_detections),
            "diff_yolo_detections": len(diff_detections),
            "class_counts": class_counts,
            "base_update_recommended": int(bool(edge_result.get("base_update_recommended")) or yolo_base_update_recommended),
            "base_updated": int(edge_base_updated or yolo_base_updated),
            "base_after": str(base_after),
            "yolo_base_update_recommended": int(yolo_base_update_recommended),
            "yolo_base_updated": int(yolo_base_updated),
            "min_overlap_ratio": self.min_overlap_ratio,
            "min_overlap_pixels": self.min_overlap_pixels,
            "red_change_pixels": int(change_mask.sum()),
            "mask_json": str(edge_json_path),
            "base_image": str(edge_output_dir / f"{current.stem}_base{Path(edge_result['base_before']).suffix}"),
            "current_image": str(edge_output_dir / f"{current.stem}_current{current.suffix}"),
            "mask_image": str(edge_output_dir / f"{current.stem}_mask{current.suffix}"),
        }
        return summary
