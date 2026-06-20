import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_MODEL_DIR = "models/classifier/zero_shot_vit_l14"
DEFAULT_PROMPTS: Dict[str, List[str]] = {
    "landslide": [
        "a landslide on a mountain road",
        "a collapsed soil slope",
        "a fresh landslide scar on a slope",
    ],
    "rockfall": [
        "a rockfall on a mountain road",
        "loose rocks and gravel on a slope",
        "fallen rocks near a road",
    ],
    "exposed_soil_slope": [
        "an exposed soil slope",
        "bare soil on a slope",
        "a brown exposed earth slope",
    ],
    "slope_vegetation_loss": [
        "a slope area with vegetation loss",
        "fallen vegetation on a slope",
        "missing green vegetation on a slope",
    ],
    "fallen_leaves": [
        "fallen leaves on vegetation",
        "brown leaves on trees",
        "seasonal leaf color change",
    ],
    "normal_vegetation": [
        "normal green vegetation",
        "trees and bushes with no hazard",
        "healthy vegetation on a hillside",
    ],
    "vehicle": [
        "a vehicle on a road",
        "a car or truck",
        "traffic on a mountain road",
    ],
    "construction_equipment": [
        "construction equipment",
        "an excavator or construction vehicle",
        "road construction machinery",
    ],
    "shadow_or_lighting_change": [
        "a lighting or shadow change",
        "sun glare or illumination change",
        "a shadow on the ground",
    ],
    "other_irrelevant_change": [
        "other irrelevant change",
        "an unimportant background change",
        "a miscellaneous object",
    ],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (project_root() / path).resolve()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Cannot read image data: {path}")
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"cv2.imdecode failed: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"cv2.imencode failed: {path}")
    encoded.tofile(str(path))


def find_pair_image(json_path: Path, suffix: str, payload: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    stem = json_path.name.replace("_mask.json", suffix)
    for ext in IMAGE_EXTENSIONS:
        candidate = json_path.parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate.resolve()
    if suffix == "_current" and payload:
        image_path = str(payload.get("imagePath") or "").strip()
        if image_path:
            candidate = (json_path.parent / image_path).resolve()
            if candidate.exists():
                return candidate
    return None


def shape_points(shape: Mapping[str, Any]) -> np.ndarray:
    points = shape.get("points") or []
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    shape_type = str(shape.get("shape_type") or "polygon").lower()
    if shape_type == "rectangle" and arr.shape[0] >= 2:
        x1, y1 = arr[0]
        x2, y2 = arr[1]
        arr = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return arr


def clamp_crop_box(points: np.ndarray, width: int, height: int, pad_ratio: float, min_pad: int) -> Tuple[int, int, int, int]:
    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))
    pad = max(int(min_pad), int(max(x2 - x1, y2 - y1) * float(pad_ratio)))
    cx1 = max(0, int(np.floor(x1)) - pad)
    cy1 = max(0, int(np.floor(y1)) - pad)
    cx2 = min(width, int(np.ceil(x2)) + pad)
    cy2 = min(height, int(np.ceil(y2)) + pad)
    if cx2 <= cx1:
        cx2 = min(width, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(height, cy1 + 1)
    return cx1, cy1, cx2, cy2


def resize_like(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def overlay_shape(image: np.ndarray, rel_points: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    if rel_points.shape[0] < 3:
        return out
    mask = np.zeros(out.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(rel_points).astype(np.int32)], 255)
    color = np.zeros_like(out)
    color[:, :] = (0, 0, 255)
    active = mask > 0
    out[active] = cv2.addWeighted(out[active], 1.0 - alpha, color[active], alpha, 0)
    return out


def draw_tag(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], max(90, 10 * len(text))), 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_classifier_crop(
    base: np.ndarray,
    current: np.ndarray,
    points: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    classifier_view: str,
) -> np.ndarray:
    cx1, cy1, cx2, cy2 = crop_box
    base_crop = base[cy1:cy2, cx1:cx2].copy()
    current_crop = current[cy1:cy2, cx1:cx2].copy()
    rel = points.copy()
    rel[:, 0] -= cx1
    rel[:, 1] -= cy1
    masked = overlay_shape(current_crop, rel)

    mode = str(classifier_view).lower()
    if mode == "current":
        return current_crop
    if mode == "masked":
        return masked
    if mode != "triplet":
        raise ValueError(f"Unsupported classifier_view: {classifier_view}")

    base_crop = draw_tag(base_crop, "base")
    current_crop = draw_tag(current_crop, "current")
    masked = draw_tag(masked, "mask")
    h = max(base_crop.shape[0], current_crop.shape[0], masked.shape[0])
    panels = []
    for panel in [base_crop, current_crop, masked]:
        if panel.shape[0] != h:
            scale_w = max(1, int(round(panel.shape[1] * h / max(1, panel.shape[0]))))
            panel = cv2.resize(panel, (scale_w, h), interpolation=cv2.INTER_LINEAR)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def normalize_prompts(prompts_path: Optional[Path]) -> Dict[str, List[str]]:
    if prompts_path is None:
        return DEFAULT_PROMPTS
    payload = read_json(prompts_path)
    if isinstance(payload, dict) and "classes" in payload:
        payload = payload["classes"]
    prompts: Dict[str, List[str]] = {}
    if not isinstance(payload, dict):
        raise ValueError("Prompt JSON must be a dict of label -> prompt/list.")
    for label, raw in payload.items():
        if isinstance(raw, str):
            values = [raw]
        else:
            values = [str(v) for v in raw]
        values = [v.strip() for v in values if v and str(v).strip()]
        if values:
            prompts[str(label).strip()] = values
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompts_path}")
    return prompts


def encode_text(open_clip: Any, model: torch.nn.Module, device: torch.device, prompts: Mapping[str, Sequence[str]]) -> Tuple[List[str], torch.Tensor]:
    labels = list(prompts.keys())
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    class_features = []
    with torch.no_grad():
        for label in labels:
            tokens = tokenizer(list(prompts[label])).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            feature = text_features.mean(dim=0)
            feature = feature / feature.norm()
            class_features.append(feature)
    return labels, torch.stack(class_features, dim=0)


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_model(model_dir: Path, device: torch.device, precision: str):
    import open_clip

    config = read_json(model_dir / "classifier_config.json")
    preprocess_cfg = config.get("preprocess_cfg") or {}
    image_mean = tuple(float(v) for v in preprocess_cfg.get("mean", [])) or None
    image_std = tuple(float(v) for v in preprocess_cfg.get("std", [])) or None
    model_path = model_dir / "classifier_model.bin"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained=str(model_path),
        precision=precision,
        device=device,
        image_mean=image_mean,
        image_std=image_std,
    )
    model.eval()
    return open_clip, model, preprocess


def cast_image_batch(images: torch.Tensor, precision: str) -> torch.Tensor:
    if precision == "fp16":
        return images.half()
    if precision == "bf16":
        return images.bfloat16()
    return images


def parse_description(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
        return value if isinstance(value, dict) else {"raw_description": str(raw)}
    except Exception:
        return {"raw_description": str(raw)}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "json_path",
        "shape_index",
        "source_label",
        "classifier_label",
        "classifier_score",
        "classifier_margin",
        "top3",
        "crop_path",
        "current_path",
        "base_path",
        "x1",
        "y1",
        "x2",
        "y2",
        "change_pixels",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_labelme(path: Path, source_payload: Mapping[str, Any], shapes: Sequence[Mapping[str, Any]], image_name: str) -> None:
    payload = dict(source_payload)
    payload["imagePath"] = image_name
    payload["imageData"] = None
    payload["shapes"] = list(shapes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root)
    model_dir_arg = Path(args.model_dir)
    model_dir = (
        model_dir_arg.absolute()
        if model_dir_arg.is_absolute()
        else (project_root() / model_dir_arg).absolute()
    )
    prompts = normalize_prompts(resolve_path(args.prompts) if args.prompts else None)
    device = torch.device(args.device)
    open_clip, model, preprocess = load_model(model_dir, device, args.precision)
    labels, text_features = encode_text(open_clip, model, device, prompts)

    json_paths = sorted(input_root.rglob("*_mask.json"))
    if args.limit > 0:
        json_paths = json_paths[: args.limit]
    jobs: List[Dict[str, Any]] = []
    payloads: Dict[Path, Dict[str, Any]] = {}
    classifier_shapes: Dict[Path, List[Dict[str, Any]]] = {}

    for json_path in json_paths:
        try:
            payload = read_json(json_path)
            current_path = find_pair_image(json_path, "_current", payload)
            base_path = find_pair_image(json_path, "_base", payload)
            if current_path is None:
                continue
            if base_path is None:
                base_path = current_path
            current = read_image(current_path)
            base = resize_like(read_image(base_path), current.shape[:2])
        except Exception as exc:
            print(f"[WARN] skip {json_path}: {exc}")
            continue

        h, w = current.shape[:2]
        payloads[json_path] = payload
        classifier_shapes[json_path] = []
        for idx, shape in enumerate(payload.get("shapes") or []):
            if not isinstance(shape, dict):
                continue
            points = shape_points(shape)
            if points.shape[0] < 3:
                continue
            crop_box = clamp_crop_box(points, w, h, args.pad_ratio, args.min_pad)
            crop = build_classifier_crop(base, current, points, crop_box, args.classifier_view)
            rel = json_path.relative_to(input_root)
            crop_path = output_root / "crops" / rel.parent / f"{json_path.stem}_shape{idx:04d}.jpg"
            write_image(crop_path, crop)
            desc = parse_description(shape.get("description"))
            jobs.append(
                {
                    "json_path": json_path,
                    "shape_index": idx,
                    "source_shape": shape,
                    "source_label": str(shape.get("label", "")),
                    "description": desc,
                    "crop_path": crop_path,
                    "current_path": current_path,
                    "base_path": base_path,
                    "crop_box": crop_box,
                    "pil": bgr_to_pil(crop),
                }
            )

    rows: List[Dict[str, Any]] = []
    for batch in batched(jobs, max(1, int(args.batch_size))):
        images = torch.stack([preprocess(job["pil"]) for job in batch], dim=0).to(device)
        images = cast_image_batch(images, args.precision)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = args.logit_scale * image_features @ text_features.t()
            probs = torch.softmax(logits, dim=-1)
            top_scores, top_indices = torch.topk(probs, k=min(3, len(labels)), dim=-1)

        for job, scores, indices in zip(batch, top_scores.cpu().tolist(), top_indices.cpu().tolist()):
            top = [(labels[int(i)], float(s)) for s, i in zip(scores, indices)]
            label = top[0][0]
            score = top[0][1]
            margin = score - (top[1][1] if len(top) > 1 else 0.0)
            x1, y1, x2, y2 = job["crop_box"]
            desc = dict(job["description"])
            desc.update(
                {
                    "classifier_source_label": job["source_label"],
                    "classifier_label": label,
                    "classifier_score": round(float(score), 6),
                    "classifier_margin": round(float(margin), 6),
                    "classifier_top3": [{"label": name, "score": round(float(value), 6)} for name, value in top],
                    "classifier_view": args.classifier_view,
                    "classifier_model": str(model_dir),
                }
            )
            row = {
                "json_path": str(job["json_path"]),
                "shape_index": int(job["shape_index"]),
                "source_label": job["source_label"],
                "classifier_label": label,
                "classifier_score": round(float(score), 6),
                "classifier_margin": round(float(margin), 6),
                "top3": json.dumps(desc["classifier_top3"], ensure_ascii=False),
                "crop_path": str(job["crop_path"]),
                "current_path": str(job["current_path"]),
                "base_path": str(job["base_path"]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "change_pixels": job["description"].get("change_pixels", ""),
            }
            rows.append(row)
            if args.write_json:
                shape = dict(job["source_shape"])
                shape["label"] = label
                shape["flags"] = shape.get("flags") or {}
                shape["description"] = json.dumps(desc, ensure_ascii=False)
                classifier_shapes[job["json_path"]].append(shape)

    write_csv(output_root / "classifier_zero_shot_results.csv", rows)
    if args.write_json:
        for json_path, shapes in classifier_shapes.items():
            if not shapes:
                continue
            rel = json_path.relative_to(input_root)
            source_payload = payloads[json_path]
            current_path = find_pair_image(json_path, "_current", source_payload)
            image_name = current_path.name if current_path is not None else str(source_payload.get("imagePath") or "")
            write_labelme(output_root / "labelme_classifier" / rel, source_payload, shapes, image_name)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model_dir": str(model_dir),
        "classifier_view": args.classifier_view,
        "num_json": len(json_paths),
        "num_shapes": len(rows),
        "labels": labels,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot classification on LabelMe change proposals.")
    parser.add_argument("--input-root", required=True, help="Root containing *_mask.json outputs.")
    parser.add_argument("--output-root", required=True, help="Directory for CSV, crops, and optional LabelMe JSON.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Local classifier model directory.")
    parser.add_argument("--prompts", default="", help="Optional JSON dict of label -> prompt/list.")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--classifier-view", default="triplet", choices=["current", "masked", "triplet"])
    parser.add_argument("--pad-ratio", type=float, default=0.20)
    parser.add_argument("--min-pad", type=int, default=24)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of JSON files for smoke tests.")
    parser.add_argument("--write-json", action="store_true", help="Write LabelMe JSON with classifier labels.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
