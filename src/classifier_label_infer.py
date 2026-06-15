import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import cv2
import numpy as np
import torch

from tools.classifier_pair_zero_shot_labelme import (
    DEFAULT_STATE_PROMPTS,
    DEFAULT_TRANSITION_PROMPTS,
    classify_batch,
    classify_prompt_ensemble,
    crop_pair,
    encode_prompt_ensemble,
    encode_text,
    focus_polygon,
    reject_out_of_scope_state,
    reject_uncertain_state,
    resolve_transition,
    select_view_result,
)
from tools.classifier_zero_shot_labelme import (
    batched,
    bgr_to_pil,
    clamp_crop_box,
    find_pair_image,
    load_model,
    read_image,
    read_json,
    resize_like,
    shape_points,
)


class ClassifierLabelMappingService:
    """Rewrite change-region labels using independent base/current classifier states."""

    def __init__(
        self,
        model_dir: Path,
        device: str = "cuda:0",
        precision: str = "fp16",
        batch_size: int = 32,
        pad_ratio: float = 0.25,
        min_pad: int = 32,
        logit_scale: float = 100.0,
        min_out_of_scope_score: float = 0.25,
        min_state_score: float = 0.30,
        min_state_margin: float = 0.08,
        min_pair_score: float = 0.55,
        min_pair_margin: float = 0.15,
        min_alarm_area_ratio: float = 0.005,
        keep_only_alarm_labels: bool = True,
        alarm_labels: Optional[List[str]] = None,
        label_map: Optional[Mapping[str, str]] = None,
        base_update_labels: Optional[List[str]] = None,
    ) -> None:
        self.model_dir = Path(model_dir).absolute()
        self.device = torch.device(device)
        self.precision = str(precision)
        self.batch_size = max(1, int(batch_size))
        self.pad_ratio = float(pad_ratio)
        self.min_pad = int(min_pad)
        self.logit_scale = float(logit_scale)
        self.min_out_of_scope_score = float(min_out_of_scope_score)
        self.min_state_score = float(min_state_score)
        self.min_state_margin = float(min_state_margin)
        self.min_pair_score = float(min_pair_score)
        self.min_pair_margin = float(min_pair_margin)
        self.min_alarm_area_ratio = float(min_alarm_area_ratio)
        self.keep_only_alarm_labels = bool(keep_only_alarm_labels)
        self.alarm_labels = {
            str(label).strip()
            for label in (
                alarm_labels
                or ["landslide", "rock_fall", "vegetation_loss"]
            )
            if str(label).strip()
        }
        self.label_map = dict(label_map or {})
        self.base_update_labels = {
            str(label).strip()
            for label in (base_update_labels or [])
            if str(label).strip()
        }

        open_clip, model, preprocess = load_model(
            self.model_dir,
            self.device,
            self.precision,
        )
        self.model = model
        self.preprocess = preprocess
        self.state_labels, self.text_features = encode_text(
            open_clip,
            model,
            self.device,
            DEFAULT_STATE_PROMPTS,
        )
        (
            self.transition_labels,
            self.transition_prompt_labels,
            self.transition_text_features,
        ) = encode_prompt_ensemble(
            open_clip,
            model,
            self.device,
            DEFAULT_TRANSITION_PROMPTS,
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "ClassifierLabelMappingService":
        path = Path(config_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_dir = Path(str(payload["model_dir"]))
        if not model_dir.is_absolute():
            model_dir = (path.parent / model_dir).absolute()
        return cls(
            model_dir=model_dir,
            device=str(payload.get("device", "cuda:0")),
            precision=str(payload.get("precision", "fp16")),
            batch_size=int(payload.get("batch_size", 32)),
            pad_ratio=float(payload.get("pad_ratio", 0.25)),
            min_pad=int(payload.get("min_pad", 32)),
            logit_scale=float(payload.get("logit_scale", 100.0)),
            min_out_of_scope_score=float(
                payload.get("min_out_of_scope_score", 0.25)
            ),
            min_state_score=float(payload.get("min_state_score", 0.30)),
            min_state_margin=float(payload.get("min_state_margin", 0.08)),
            min_pair_score=float(payload.get("min_pair_score", 0.55)),
            min_pair_margin=float(payload.get("min_pair_margin", 0.15)),
            min_alarm_area_ratio=float(
                payload.get("min_alarm_area_ratio", 0.005)
            ),
            keep_only_alarm_labels=bool(
                payload.get("keep_only_alarm_labels", True)
            ),
            alarm_labels=list(
                payload.get("alarm_labels")
                or ["landslide", "rock_fall", "vegetation_loss"]
            ),
            label_map=payload.get("label_map") or {},
            base_update_labels=list(payload.get("base_update_labels") or []),
        )

    @staticmethod
    def _find_json(output_dir: Path) -> Path:
        candidates = sorted(output_dir.glob("*_mask.json"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one *_mask.json in {output_dir}, found {len(candidates)}"
            )
        return candidates[0]

    @staticmethod
    def _write_outline_preview(
        current: np.ndarray,
        shapes: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        preview = current.copy()
        height, width = preview.shape[:2]
        for shape in shapes:
            points = np.asarray(shape.get("points") or [], dtype=np.float32)
            if points.shape[0] < 3:
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.polylines(
                preview,
                [np.round(points).astype(np.int32).reshape(-1, 1, 2)],
                True,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )
        ok, encoded = cv2.imencode(output_path.suffix or ".jpeg", preview)
        if not ok:
            raise ValueError(f"Failed to encode preview: {output_path}")
        encoded.tofile(str(output_path))

    def process_output(self, output_dir: str) -> Dict[str, Any]:
        root = Path(output_dir).resolve()
        json_path = self._find_json(root)
        payload = read_json(json_path)
        base_path = find_pair_image(json_path, "_base", payload)
        current_path = find_pair_image(json_path, "_current", payload)
        if base_path is None or current_path is None:
            raise FileNotFoundError(f"Missing base/current image beside {json_path}")

        current = read_image(current_path)
        base = resize_like(read_image(base_path), current.shape[:2])
        height, width = current.shape[:2]
        jobs: List[Dict[str, Any]] = []
        shapes = payload.get("shapes") or []
        for shape_index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                continue
            points = shape_points(shape)
            if points.shape[0] < 3:
                continue
            box = clamp_crop_box(
                points,
                width,
                height,
                self.pad_ratio,
                self.min_pad,
            )
            base_crop, current_crop, rel_points = crop_pair(
                base,
                current,
                points,
                box,
            )
            jobs.append(
                {
                    "shape_index": shape_index,
                    "base_context": bgr_to_pil(base_crop),
                    "current_context": bgr_to_pil(current_crop),
                    "base_focus": bgr_to_pil(
                        focus_polygon(base_crop, rel_points)
                    ),
                    "current_focus": bgr_to_pil(
                        focus_polygon(current_crop, rel_points)
                    ),
                    "pair_focus": bgr_to_pil(
                        cv2.hconcat(
                            [
                                focus_polygon(base_crop, rel_points),
                                focus_polygon(current_crop, rel_points),
                            ]
                        )
                    ),
                }
            )

        transition_counts: Counter[str] = Counter()
        mapped_counts: Counter[str] = Counter()
        accepted_shapes: List[Dict[str, Any]] = []
        accepted_change_pixels = 0
        for batch in batched(jobs, self.batch_size):
            base_context_results = classify_batch(
                self.model,
                self.preprocess,
                [job["base_context"] for job in batch],
                self.text_features,
                self.state_labels,
                self.device,
                self.precision,
                self.logit_scale,
            )
            current_context_results = classify_batch(
                self.model,
                self.preprocess,
                [job["current_context"] for job in batch],
                self.text_features,
                self.state_labels,
                self.device,
                self.precision,
                self.logit_scale,
            )
            base_focus_results = classify_batch(
                self.model,
                self.preprocess,
                [job["base_focus"] for job in batch],
                self.text_features,
                self.state_labels,
                self.device,
                self.precision,
                self.logit_scale,
            )
            current_focus_results = classify_batch(
                self.model,
                self.preprocess,
                [job["current_focus"] for job in batch],
                self.text_features,
                self.state_labels,
                self.device,
                self.precision,
                self.logit_scale,
            )
            pair_results = classify_prompt_ensemble(
                self.model,
                self.preprocess,
                [job["pair_focus"] for job in batch],
                self.transition_text_features,
                self.transition_labels,
                self.transition_prompt_labels,
                self.device,
                self.precision,
                self.logit_scale,
            )

            for job, base_context, current_context, base_focus, current_focus, pair_result in zip(
                batch,
                base_context_results,
                current_context_results,
                base_focus_results,
                current_focus_results,
                pair_results,
            ):
                base_result = select_view_result(base_context, base_focus)
                current_result = select_view_result(current_context, current_focus)
                base_result = reject_out_of_scope_state(
                    base_result,
                    self.min_out_of_scope_score,
                )
                current_result = reject_out_of_scope_state(
                    current_result,
                    self.min_out_of_scope_score,
                )
                base_result = reject_uncertain_state(
                    base_result,
                    self.min_state_score,
                    self.min_state_margin,
                )
                current_result = reject_uncertain_state(
                    current_result,
                    self.min_state_score,
                    self.min_state_margin,
                )
                transition, transition_source = resolve_transition(
                    base_result,
                    current_result,
                    pair_result,
                    self.min_pair_score,
                    self.min_pair_margin,
                )
                mapped_label = self.label_map.get(transition, transition)
                transition_counts[transition] += 1
                mapped_counts[mapped_label] += 1

                shape = shapes[int(job["shape_index"])]
                source_description = shape.get("description")
                try:
                    source_metadata = json.loads(source_description or "{}")
                except (TypeError, json.JSONDecodeError):
                    source_metadata = {}
                description = {
                    key: source_metadata[key]
                    for key in ("change_pixels",)
                    if key in source_metadata
                }
                description.update(
                    {
                        "source": "pair_change_seg_with_classifier",
                        "classifier_transition": transition,
                        "classifier_transition_source": transition_source,
                        "classifier_score": round(
                            float(
                                pair_result["score"]
                                if transition_source
                                in {"pair_classifier", "strong_pair_classifier"}
                                else current_result["score"]
                            ),
                            6,
                        ),
                        "classifier_margin": round(
                            float(
                                pair_result["margin"]
                                if transition_source
                                in {"pair_classifier", "strong_pair_classifier"}
                                else current_result["margin"]
                            ),
                            6,
                        ),
                    }
                )
                shape["label"] = mapped_label
                shape["group_id"] = None
                shape["description"] = (
                    json.dumps(description, ensure_ascii=False)
                    if description
                    else None
                )
                shape["flags"] = shape.get("flags") or {}
                if (
                    not self.keep_only_alarm_labels
                    or mapped_label in self.alarm_labels
                ):
                    accepted_shapes.append(shape)
                    accepted_change_pixels += int(
                        source_metadata.get("change_pixels", 0) or 0
                    )

        image_pixels = int(height * width)
        accepted_change_ratio = (
            accepted_change_pixels / float(image_pixels)
            if image_pixels > 0
            else 0.0
        )
        area_filter_applied = (
            accepted_change_ratio < self.min_alarm_area_ratio
        )
        if area_filter_applied:
            accepted_shapes = []
            accepted_change_pixels = 0
            accepted_change_ratio = 0.0
        if self.keep_only_alarm_labels:
            payload["shapes"] = accepted_shapes

        payload.pop("classifier_classification", None)
        payload["imagePath"] = current_path.name
        payload["imageData"] = None
        payload["flags"] = payload.get("flags") or {}
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mask_candidates = sorted(
            path
            for path in root.glob(f"{json_path.stem}.*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if mask_candidates:
            self._write_outline_preview(
                current,
                list(payload.get("shapes") or []),
                mask_candidates[0],
            )
        base_update_recommended = any(
            label in self.base_update_labels and count > 0
            for label, count in mapped_counts.items()
        )
        return {
            "json_path": str(json_path),
            "num_classified_shapes": len(jobs),
            "num_accepted_shapes": len(payload.get("shapes") or []),
            "accepted_change_pixels": accepted_change_pixels,
            "accepted_change_ratio": accepted_change_ratio,
            "area_filter_applied": int(area_filter_applied),
            "transition_counts": dict(transition_counts),
            "mapped_label_counts": dict(mapped_counts),
            "base_update_recommended": int(base_update_recommended),
        }
