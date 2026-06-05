import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASS_NAMES = ["rock_fall", "landslide"]


@dataclass
class YoloDetectorConfig:
    model_path: str
    device: str = "0"
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.5
    max_det: int = 300
    class_names: List[str] = field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))

    @classmethod
    def from_json(cls, path: Path) -> "YoloDetectorConfig":
        with path.open("r", encoding="utf-8-sig") as f:
            raw = json.load(f)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in raw.items() if key in allowed})


class YoloDetectionService:
    def __init__(self, config: YoloDetectorConfig, config_base_dir: Optional[Path] = None) -> None:
        self.config = config
        self.config_base_dir = config_base_dir or Path.cwd()
        self.model_path = self._resolve_path(config.model_path)
        self.class_names = list(config.class_names or DEFAULT_CLASS_NAMES)

        if not self.model_path.exists():
            raise FileNotFoundError(f"model_path not found: {self.model_path}")

        self._model = None

    @classmethod
    def from_config_file(cls, config_path: str) -> "YoloDetectionService":
        path = Path(config_path).resolve()
        cfg = YoloDetectorConfig.from_json(path)
        return cls(cfg, config_base_dir=path.parent)

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p.resolve()
        return (self.config_base_dir / p).resolve()

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
    def _is_image(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def _predict_results(self, image_paths: Sequence[Path]):
        model = self._get_model()
        return model.predict(
            source=[str(p) for p in image_paths],
            imgsz=int(self.config.imgsz),
            conf=float(self.config.conf),
            iou=float(self.config.iou),
            device=str(self.config.device),
            max_det=int(self.config.max_det),
            save=False,
            stream=True,
            verbose=False,
        )

    def _results_to_detections(self, result: Any) -> List[Dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", None) or {}
        detections: List[Dict[str, Any]] = []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()

        for idx in range(len(xyxy)):
            class_id = int(classes[idx])
            label = names.get(class_id)
            if label is None:
                label = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"class_{class_id}"
            detections.append(
                {
                    "class_id": class_id,
                    "label": str(label),
                    "score": float(confs[idx]),
                    "box_xyxy": [float(v) for v in xyxy[idx].tolist()],
                }
            )
        return detections

    def _empty_class_counts(self) -> Dict[str, int]:
        return {name: 0 for name in self.class_names}

    def predict_image(self, current_path: str) -> Dict[str, Any]:
        image_path = self._resolve_path(current_path)
        if not self._is_image(image_path):
            raise FileNotFoundError(f"current image not found or not supported: {image_path}")

        source_img = self._read_image_unicode(image_path)
        image_height, image_width = source_img.shape[:2]
        result = next(iter(self._predict_results([image_path])))
        detections = self._results_to_detections(result)

        class_counts = self._empty_class_counts()
        for det in detections:
            class_counts[det["label"]] = class_counts.get(det["label"], 0) + 1

        return {
            "image_name": image_path.name,
            "source_path": str(image_path),
            "image_height": int(image_height),
            "image_width": int(image_width),
            "has_detection": int(bool(detections)),
            "num_detections": len(detections),
            "class_counts": class_counts,
            "detections": detections,
        }
