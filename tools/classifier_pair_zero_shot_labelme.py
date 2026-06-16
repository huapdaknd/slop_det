import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    from classifier_zero_shot_labelme import (
        batched,
        bgr_to_pil,
        clamp_crop_box,
        find_pair_image,
        load_model,
        parse_description,
        project_root,
        read_image,
        read_json,
        resize_like,
        resolve_path,
        shape_points,
        write_image,
    )
except ImportError:
    from tools.classifier_zero_shot_labelme import (
        batched,
        bgr_to_pil,
        clamp_crop_box,
        find_pair_image,
        load_model,
        parse_description,
        project_root,
        read_image,
        read_json,
        resize_like,
        resolve_path,
        shape_points,
        write_image,
    )


DEFAULT_STATE_PROMPTS: Dict[str, List[str]] = {
    "healthy_green_vegetation": [
        "an intact vegetated hillside with healthy green tree canopy and shrubs",
        "dense living vegetation covering a slope without fresh bare soil or debris",
        "standing green trees and bushes with continuous canopy cover",
        "normal green slope vegetation with no excavation no collapse and no fallen trunks",
    ],
    "seasonal_leaf_color": [
        "intact standing tree crowns with yellow red or brown seasonal foliage",
        "a hillside whose trees changed leaf color but still keep continuous canopy cover",
        "autumn colored woodland without fresh bare soil fallen trunks or excavation",
        "seasonal color change only, not tree removal not leafless branches and not disturbed ground",
    ],
    "dry_or_leafless_vegetation": [
        "standing trees with dry sparse leaves and many visible bare branches",
        "dry or leafless vegetation where trunks remain upright and rooted in place",
        "a hillside with dead foliage substantial leaf loss or exposed branches but no fresh cut slope",
        "withered standing vegetation without logging debris, large fallen trunks, or obvious excavation",
    ],
    "cut_or_fallen_trees": [
        "cut trees, felled trunks, and logging debris on a hillside",
        "fallen tree trunks, broken stems, and collapsed vegetation on a slope",
        "uprooted or toppled trees lying on the ground rather than standing dry trees",
        "fresh tree fall or tree clearing with visible woody debris",
    ],
    "cleared_or_excavated_slope": [
        "an engineered cut slope with a smooth regular bare face beside a road",
        "a manmade road-cut or widened slope with straight edges benches terraces or machine tracks",
        "a mechanically cleared bare hillside with excavation marks and geometric shaping",
        "construction excavation rather than a chaotic natural collapse or scattered rockfall",
        "a regular cut face that looks reshaped by machinery",
    ],
    "natural_landslide": [
        "an irregular natural slope collapse with a visible head scarp and displaced earth",
        "a chaotic unengineered slope failure with broken soil rock debris and torn vegetation",
        "a natural landslide with a downslope movement path and material accumulated at the slope toe",
        "an uneven collapsed hillside rather than a smooth regular machine-cut face",
        "fresh natural failure with disrupted ground geometry and no engineered benching",
    ],
    "exposed_rock_or_soil": [
        "a stable continuous exposed bedrock or bare soil surface on a hillside",
        "an intact in-place rocky or earthy slope without a movement path or toe deposit",
        "long-standing bare earth or rock face with no head scarp and no displaced debris",
        "continuous in-place rock or soil rather than detached loose stones",
        "a bare slope surface that is not obviously excavated and not an active collapse",
    ],
    "loose_rock_or_gravel": [
        "loose detached rocks and gravel scattered on a slope or road shoulder",
        "separate angular stones and rock fragments lying on the ground",
        "fresh small rock debris accumulated beside a mountain road",
        "detached rocks lying on a road surface or roadside shoulder",
        "a local cluster of loose stones at the toe of a slope, not a whole bare cut face",
        "discrete rockfall-like debris rather than continuous exposed bedrock or soil",
    ],
    "vehicle": [
        "a car pickup van or truck on a road",
        "the selected region is mainly a vehicle viewed from above",
        "a passenger car parked or moving in a mountain road scene",
        "vehicle body roof windows and wheels dominate the region",
    ],
    "construction_equipment": [
        "an excavator loader bulldozer or other construction machine",
        "road construction machinery with an arm bucket tracks or heavy equipment body",
        "construction equipment rather than an ordinary passenger vehicle",
    ],
    "road_surface": [
        "a close-up region dominated by asphalt concrete or dirt road surface",
        "the main subject is pavement with lane markings cracks or road texture",
        "road surface fills most of the selected region rather than slope material",
    ],
    "person": [
        "a person standing or walking outdoors",
        "one or more small human figures in a road or slope scene",
        "a pedestrian rather than a vehicle or machine",
    ],
    "manmade_structure": [
        "a guardrail retaining wall building sign fence or utility pole",
        "ordinary roadside infrastructure rather than slope damage or vegetation change",
        "a fixed manmade structure with regular edges and construction materials",
    ],
}

DEFAULT_TRANSITION_PROMPTS: Dict[str, List[str]] = {
    "construction_clearing": [
        "left before and right after: an already bare or partially bare roadside slope was mechanically reshaped into a regular engineered cut face",
        "left before and right after: road construction widened or regraded a slope with benches tracks or geometric excavation marks",
        "left before and right after: mechanical excavation occurred without the main evidence being loss of tree canopy",
    ],
    "vegetation_loss_candidate": [
        "before and after images: dense vegetation changed into cut fallen or removed trees",
        "a green vegetated slope before and clear loss of plant cover after",
        "trees or shrubs were cleared, felled, buried, or replaced by bare soil rock or an engineered cut slope",
        "the main change is disappearance of living vegetation, regardless of whether it was caused by clearing excavation or damage",
    ],
    "landslide_candidate": [
        "left before and right after: an intact natural hillside developed an irregular collapse with displaced soil and debris",
        "left before and right after: a new head scarp movement path and chaotic landslide deposit appeared",
        "left before and right after: slope material moved downslope naturally and accumulated at the toe without regular excavation geometry",
        "the after image shows irregular failure rather than a smooth machine-cut slope",
    ],
    "rockfall_candidate": [
        "before and after images: new detached rocks appeared on the road roadside shoulder or slope toe",
        "a relatively clear roadside area before and fresh local rock debris after",
        "the after image shows a small concentrated rockfall deposit, not a whole bare slope conversion",
    ],
    "seasonal_leaf_color_change": [
        "before and after images: intact green tree crowns changed to intact yellow red or brown seasonal foliage",
        "normal seasonal leaf color change with canopy still present and no disturbed soil",
        "only leaf hue changed while vegetation structure stayed intact",
    ],
    "leaf_drying_or_fall": [
        "before and after images: tree leaves became dry sparse or leafless while trunks remained standing",
        "healthy foliage before and dried leaves bare branches or substantial leaf loss after",
        "vegetation remains standing but foliage density decreases clearly",
    ],
    "non_target_change": [
        "before and after images show a vehicle person machine road or ordinary roadside structure change",
        "an unrelated roadside object changed without vegetation loss landslide or rockfall",
        "the change belongs to traffic equipment or infrastructure rather than slope material",
    ],
    "no_meaningful_change": [
        "before and after images show the same hillside road and vegetation with no meaningful physical change",
        "no important structural change between the before image and the after image, with no clear canopy loss and no new bare slope",
        "the selected region remains essentially the same scene content, without new exposed soil rock debris or excavation",
    ],
    "other_visual_change": [
        "before and after images show another local visual change that is not vegetation loss landslide rockfall or non-target object change",
        "an unclassified difference between two views that does not match the main business categories",
    ],
}

UNKNOWN_STATE = "unknown_or_uncertain"
OUT_OF_SCOPE_STATE = "out_of_scope"
NON_TARGET_STATES = {
    "vehicle",
    "person",
    "road_surface",
    "manmade_structure",
}
VEGETATION_STATES = {
    "healthy_green_vegetation",
    "seasonal_leaf_color",
    "dry_or_leafless_vegetation",
}
CONTEXT_OBJECT_STATES = {
    "vehicle",
    "person",
    "construction_equipment",
    "manmade_structure",
}
MOVABLE_OBJECT_STATES = {
    "vehicle",
    "person",
    "construction_equipment",
}
SLOPE_SURFACE_STATES = {
    "cleared_or_excavated_slope",
    "natural_landslide",
    "exposed_rock_or_soil",
}


def load_prompts(path: str) -> Dict[str, List[str]]:
    if not path:
        return DEFAULT_STATE_PROMPTS
    payload = read_json(resolve_path(path))
    if isinstance(payload, dict) and "classes" in payload:
        payload = payload["classes"]
    if not isinstance(payload, dict):
        raise ValueError("Prompt JSON must be a dict of label -> prompt/list.")
    result: Dict[str, List[str]] = {}
    for label, raw in payload.items():
        values = [raw] if isinstance(raw, str) else list(raw)
        values = [str(value).strip() for value in values if str(value).strip()]
        if values:
            result[str(label).strip()] = values
    return result


def encode_text(open_clip: Any, model: torch.nn.Module, device: torch.device, prompts: Mapping[str, Sequence[str]]):
    labels = list(prompts)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    features = []
    with torch.no_grad():
        for label in labels:
            tokens = tokenizer(list(prompts[label])).to(device)
            encoded = model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            mean = encoded.mean(dim=0)
            features.append(mean / mean.norm())
    return labels, torch.stack(features)


def encode_prompt_ensemble(
    open_clip: Any,
    model: torch.nn.Module,
    device: torch.device,
    prompts: Mapping[str, Sequence[str]],
):
    labels = list(prompts)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    prompt_labels: List[str] = []
    features = []
    with torch.no_grad():
        for label in labels:
            values = list(prompts[label])
            tokens = tokenizer(values).to(device)
            encoded = model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            features.append(encoded)
            prompt_labels.extend([label] * len(values))
    return labels, prompt_labels, torch.cat(features, dim=0)


def _aggregate_prompt_scores(
    similarities: torch.Tensor,
    labels: Sequence[str],
    prompt_labels: Sequence[str],
    top_k: int = 2,
) -> torch.Tensor:
    class_scores = []
    for label in labels:
        indices = [i for i, value in enumerate(prompt_labels) if value == label]
        values = similarities[:, indices]
        k = min(max(1, int(top_k)), values.shape[1])
        class_scores.append(torch.topk(values, k=k, dim=1).values.mean(dim=1))
    return torch.stack(class_scores, dim=1)


def classify_prompt_ensemble(
    model: torch.nn.Module,
    preprocess: Any,
    images: Sequence[Any],
    prompt_features: torch.Tensor,
    labels: Sequence[str],
    prompt_labels: Sequence[str],
    device: torch.device,
    precision: str,
    logit_scale: float,
    coarse_ensemble: Optional[Tuple[Sequence[str], Sequence[str], torch.Tensor]] = None,
) -> List[Dict[str, Any]]:
    tensor = torch.stack([preprocess(image) for image in images]).to(device)
    tensor = cast_images(tensor, precision)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        similarities = features @ prompt_features.t()
        fine_scores = _aggregate_prompt_scores(
            similarities, labels, prompt_labels
        )
        selected_groups: List[Optional[str]] = [None] * len(images)
        probs = torch.softmax(logit_scale * fine_scores, dim=-1)
        scores, indices = torch.topk(probs, k=min(3, len(labels)), dim=-1)
    results = []
    for row_index, (row_scores, row_indices) in enumerate(
        zip(scores.cpu().tolist(), indices.cpu().tolist())
    ):
        top3 = [
            {"label": labels[int(idx)], "score": round(float(score), 6)}
            for score, idx in zip(row_scores, row_indices)
        ]
        results.append(
            {
                "label": top3[0]["label"],
                "raw_label": top3[0]["label"],
                "score": top3[0]["score"],
                "margin": round(
                    top3[0]["score"]
                    - (top3[1]["score"] if len(top3) > 1 else 0.0),
                    6,
                ),
                "top3": top3,
                "coarse_group": selected_groups[row_index],
            }
        )
    return results


def cast_images(images: torch.Tensor, precision: str) -> torch.Tensor:
    if precision == "fp16":
        return images.half()
    if precision == "bf16":
        return images.bfloat16()
    return images


def classify_batch(
    model: torch.nn.Module,
    preprocess: Any,
    images: Sequence[Any],
    text_features: torch.Tensor,
    labels: Sequence[str],
    device: torch.device,
    precision: str,
    logit_scale: float,
) -> List[Dict[str, Any]]:
    tensor = torch.stack([preprocess(image) for image in images]).to(device)
    tensor = cast_images(tensor, precision)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        probs = torch.softmax(logit_scale * features @ text_features.t(), dim=-1)
        scores, indices = torch.topk(probs, k=min(3, len(labels)), dim=-1)
    results = []
    for row_scores, row_indices in zip(scores.cpu().tolist(), indices.cpu().tolist()):
        top3 = [{"label": labels[int(idx)], "score": round(float(score), 6)} for score, idx in zip(row_scores, row_indices)]
        results.append(
            {
                "label": top3[0]["label"],
                "raw_label": top3[0]["label"],
                "score": top3[0]["score"],
                "margin": round(top3[0]["score"] - (top3[1]["score"] if len(top3) > 1 else 0.0), 6),
                "top3": top3,
            }
        )
    return results


def reject_uncertain_state(result: Dict[str, Any], min_score: float, min_margin: float) -> Dict[str, Any]:
    if float(result["score"]) < min_score:
        result["label"] = UNKNOWN_STATE
        result["uncertainty_reason"] = "low_score"
        return result
    if float(result["margin"]) < min_margin:
        top3 = result.get("top3") or []
        top_labels = {str(item.get("label")) for item in top3[:2]}
        if len(top_labels) == 2 and top_labels.issubset(SLOPE_SURFACE_STATES):
            result["display_label"] = f"{result['raw_label']} [subtype uncertain]"
            result["coarse_label"] = "disturbed_or_exposed_slope"
            result["uncertainty_reason"] = "slope_subtype"
            return result
        result["label"] = UNKNOWN_STATE
        result["uncertainty_reason"] = "low_margin"
    return result


def reject_out_of_scope_state(
    result: Dict[str, Any],
    min_score: float,
) -> Dict[str, Any]:
    if float(result["score"]) < min_score:
        result["label"] = OUT_OF_SCOPE_STATE
        result["display_label"] = (
            f"{OUT_OF_SCOPE_STATE} [candidate={result['raw_label']}]"
        )
        result["uncertainty_reason"] = "very_low_score"
    return result


def select_view_result(context: Dict[str, Any], focused: Dict[str, Any]) -> Dict[str, Any]:
    focused_is_object = (
        focused["raw_label"] in CONTEXT_OBJECT_STATES
        and float(focused["score"]) >= 0.35
        and float(focused["margin"]) >= 0.08
    )
    context_is_strong_object = (
        context["raw_label"] in CONTEXT_OBJECT_STATES
        and float(context["score"]) >= 0.55
        and float(context["margin"]) >= 0.15
        and (
            context["raw_label"] == focused["raw_label"]
            or float(context["score"]) >= float(focused["score"]) + 0.15
        )
    )
    if focused_is_object and context_is_strong_object:
        view, result = max(
            [("context", context), ("focused", focused)],
            key=lambda item: (float(item[1]["margin"]), float(item[1]["score"])),
        )
    elif focused_is_object:
        view, result = "focused", focused
    elif context_is_strong_object:
        view, result = "context", context
    else:
        view, result = "focused", focused
    selected = dict(result)
    selected["selected_view"] = view
    selected["context_top1"] = context["raw_label"]
    selected["focused_top1"] = focused["raw_label"]
    selected["context_score"] = context["score"]
    selected["focused_score"] = focused["score"]
    selected["context_margin"] = context["margin"]
    selected["focused_margin"] = focused["margin"]
    return selected


def resolve_transition(
    base_result: Mapping[str, Any],
    current_result: Mapping[str, Any],
    pair_result: Mapping[str, Any],
    min_pair_score: float,
    min_pair_margin: float,
) -> Tuple[str, str]:
    base_label = str(base_result["label"])
    current_label = str(current_result["label"])
    state_transition = transition_label(base_label, current_label)
    pair_transition = str(pair_result["label"])
    pair_is_confident = (
        float(pair_result["score"]) >= min_pair_score
        and float(pair_result["margin"]) >= min_pair_margin
    )
    if not pair_is_confident:
        return state_transition, "state_rule"

    pair_is_strong = (
        float(pair_result["score"]) >= max(min_pair_score, 0.75)
        and float(pair_result["margin"]) >= max(min_pair_margin, 0.30)
    )
    current_top3 = {
        str(item.get("label")): float(item.get("score", 0.0))
        for item in current_result.get("top3", [])
        if isinstance(item, Mapping)
    }
    current_context_label = str(current_result.get("context_top1"))
    current_focused_label = str(current_result.get("focused_top1"))
    current_context_score = float(current_result.get("context_score", 0.0))
    current_focused_score = float(current_result.get("focused_score", 0.0))
    current_has_movable_object = (
        (
            current_context_label in MOVABLE_OBJECT_STATES
            and current_context_score >= 0.30
        )
        or (
            current_focused_label in MOVABLE_OBJECT_STATES
            and current_focused_score >= 0.30
        )
        or current_top3.get("vehicle", 0.0) >= 0.18
        or current_top3.get("construction_equipment", 0.0) >= 0.18
        or current_top3.get("person", 0.0) >= 0.18
    )
    if (
        pair_is_strong
        and pair_transition == "vegetation_loss_candidate"
        and base_label in VEGETATION_STATES
        and (
            current_label in {
                "cut_or_fallen_trees",
                "exposed_rock_or_soil",
                "loose_rock_or_gravel",
            }
            or current_top3.get("cut_or_fallen_trees", 0.0) >= 0.15
            or current_top3.get("exposed_rock_or_soil", 0.0) >= 0.25
            or current_top3.get("loose_rock_or_gravel", 0.0) >= 0.25
        )
    ):
        return pair_transition, "strong_pair_classifier"

    if (
        pair_transition == "vegetation_loss_candidate"
        and str(current_result.get("context_top1")) == "cleared_or_excavated_slope"
        and str(current_result.get("focused_top1")) in {
            "cleared_or_excavated_slope",
            "exposed_rock_or_soil",
            "natural_landslide",
        }
    ):
        if base_label in VEGETATION_STATES:
            return "vegetation_loss_candidate", "pair_classifier"
        return "other_visual_change", "pair_classifier"
    if (
        pair_transition == "construction_clearing"
        and base_label in VEGETATION_STATES
        and current_label in {
            UNKNOWN_STATE,
            "cleared_or_excavated_slope",
            "exposed_rock_or_soil",
            "natural_landslide",
            "loose_rock_or_gravel",
        }
    ):
        return "vegetation_loss_candidate", "pair_classifier"
    if pair_transition == "construction_clearing" and current_label in {
        UNKNOWN_STATE,
        "cleared_or_excavated_slope",
        "exposed_rock_or_soil",
        "natural_landslide",
        "loose_rock_or_gravel",
    }:
        return "other_visual_change", "pair_classifier"
    if (
        pair_is_strong
        and pair_transition == "landslide_candidate"
        and current_label in {
            "natural_landslide",
            "exposed_rock_or_soil",
            "loose_rock_or_gravel",
        }
    ):
        return pair_transition, "strong_pair_classifier"
    if (
        pair_is_strong
        and pair_transition == "rockfall_candidate"
        and current_label == "loose_rock_or_gravel"
        and not current_has_movable_object
        and current_context_label == "road_surface"
    ):
        return pair_transition, "strong_pair_classifier"
    if pair_transition == "rockfall_candidate" and current_has_movable_object:
        return "non_target_change", "pair_classifier"
    if (
        pair_transition == "no_meaningful_change"
        and base_label in VEGETATION_STATES
        and (
            current_label in {
                "cut_or_fallen_trees",
                "cleared_or_excavated_slope",
                "natural_landslide",
                "exposed_rock_or_soil",
                "loose_rock_or_gravel",
            }
            or current_top3.get("cut_or_fallen_trees", 0.0) >= 0.12
            or current_top3.get("cleared_or_excavated_slope", 0.0) >= 0.20
            or current_top3.get("natural_landslide", 0.0) >= 0.20
            or current_top3.get("exposed_rock_or_soil", 0.0) >= 0.20
            or current_top3.get("loose_rock_or_gravel", 0.0) >= 0.20
        )
    ):
        return "vegetation_loss_candidate", "state_rule"
    if pair_transition in {
        "vegetation_loss_candidate",
        "landslide_candidate",
        "rockfall_candidate",
        "seasonal_leaf_color_change",
        "leaf_drying_or_fall",
    } and state_transition in {
        pair_transition,
        "uncertain_change",
        "other_visual_change",
    }:
        return pair_transition, "pair_classifier"
    if pair_transition in {"non_target_change", "no_meaningful_change"} and state_transition in {
        pair_transition,
        "uncertain_change",
        "other_visual_change",
    }:
        return pair_transition, "pair_classifier"
    return state_transition, "state_rule"


def transition_label(base_label: str, current_label: str) -> str:
    if OUT_OF_SCOPE_STATE in {base_label, current_label}:
        return "unclassified_change"
    if UNKNOWN_STATE in {base_label, current_label}:
        return "uncertain_change"
    if current_label == "construction_equipment":
        return "construction_activity"
    if (
        base_label in VEGETATION_STATES
        and current_label in {
            "cleared_or_excavated_slope",
            "natural_landslide",
            "exposed_rock_or_soil",
            "loose_rock_or_gravel",
        }
    ):
        return "vegetation_loss_candidate"
    if current_label == "cleared_or_excavated_slope" and base_label != "cleared_or_excavated_slope":
        return "other_visual_change"
    if current_label == "cut_or_fallen_trees" and base_label != "cut_or_fallen_trees":
        return "vegetation_loss_candidate"
    if current_label == "natural_landslide" and base_label != "natural_landslide":
        return "landslide_candidate"
    if current_label == "loose_rock_or_gravel" and base_label not in {"loose_rock_or_gravel", "natural_landslide"}:
        return "rockfall_candidate"
    if (
        current_label == "seasonal_leaf_color"
        and base_label != "seasonal_leaf_color"
    ):
        return "seasonal_leaf_color_change"
    if (
        current_label == "dry_or_leafless_vegetation"
        and base_label not in {"dry_or_leafless_vegetation", "cut_or_fallen_trees"}
    ):
        return "leaf_drying_or_fall"
    if base_label in NON_TARGET_STATES or current_label in NON_TARGET_STATES:
        if base_label == current_label:
            return "no_meaningful_change"
        return "non_target_change"
    if base_label == current_label:
        return "no_meaningful_change"
    return "other_visual_change"


def risk_group(
    transition: str,
    base_margin: float,
    current_margin: float,
    min_margin: float,
    transition_source: str,
    pair_margin: float,
    min_pair_margin: float,
) -> str:
    if (
        transition_source not in {"pair_classifier", "strong_pair_classifier"}
        and min(base_margin, current_margin) < min_margin
    ) or (
        transition_source in {"pair_classifier", "strong_pair_classifier"}
        and pair_margin < min_pair_margin
    ):
        return "review"
    if transition in {"vegetation_loss_candidate", "landslide_candidate", "rockfall_candidate"}:
        return "alarm"
    if transition in {
        "non_target_change",
        "construction_activity",
        "construction_clearing",
        "seasonal_leaf_color_change",
        "leaf_drying_or_fall",
        "no_meaningful_change",
    }:
        return "filter"
    return "review"


RISK_COLORS = {
    "alarm": (40, 40, 220),
    "filter": (40, 160, 40),
    "review": (0, 150, 230),
}


def draw_region(image: np.ndarray, rel_points: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    poly = np.round(rel_points).astype(np.int32)
    if len(poly) >= 3:
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], color)
        out = cv2.addWeighted(out, 0.82, overlay, 0.18, 0)
        cv2.polylines(out, [poly], True, color, 3, cv2.LINE_AA)
    return out


def draw_text_line(canvas: np.ndarray, text: str, origin: Tuple[int, int], scale: float, color: Tuple[int, int, int]) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def fit_text_scale(text: str, max_width: int, preferred: float, minimum: float = 0.38) -> float:
    scale = preferred
    while scale > minimum:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        if width <= max_width:
            break
        scale -= 0.04
    return max(minimum, scale)


def render_preview(
    base_crop: np.ndarray,
    current_crop: np.ndarray,
    rel_points: np.ndarray,
    source_label: str,
    base_result: Mapping[str, Any],
    current_result: Mapping[str, Any],
    transition: str,
    group: str,
) -> np.ndarray:
    color = RISK_COLORS[group]
    base_panel = draw_region(base_crop, rel_points, color)
    current_panel = draw_region(current_crop, rel_points, color)
    height = 420
    panel_width = 680

    def fit_panel(panel: np.ndarray) -> np.ndarray:
        scale = min(panel_width / max(1, panel.shape[1]), height / max(1, panel.shape[0]))
        new_w = max(1, int(round(panel.shape[1] * scale)))
        new_h = max(1, int(round(panel.shape[0] * scale)))
        resized = cv2.resize(panel, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        canvas = np.full((height, panel_width, 3), 28, dtype=np.uint8)
        x = (panel_width - new_w) // 2
        y = (height - new_h) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas

    base_panel = fit_panel(base_panel)
    current_panel = fit_panel(current_panel)
    cv2.rectangle(base_panel, (0, 0), (panel_width, 44), (0, 0, 0), -1)
    cv2.rectangle(current_panel, (0, 0), (panel_width, 44), (0, 0, 0), -1)
    base_text = f"BASE  {base_result.get('display_label', base_result['label'])}  {float(base_result['score']):.3f}"
    current_text = (
        f"CURRENT  {current_result.get('display_label', current_result['label'])}  "
        f"{float(current_result['score']):.3f}"
    )
    draw_text_line(
        base_panel,
        base_text,
        (10, 31),
        fit_text_scale(base_text, panel_width - 20, 0.64),
        (255, 255, 255),
    )
    draw_text_line(
        current_panel,
        current_text,
        (10, 31),
        fit_text_scale(current_text, panel_width - 20, 0.64),
        (255, 255, 255),
    )

    body = cv2.hconcat([base_panel, current_panel])
    info = np.full((118, body.shape[1], 3), 245, dtype=np.uint8)
    cv2.rectangle(info, (0, 0), (16, info.shape[0]), color, -1)
    gt_text = f"GT: {source_label}"
    transition_text = f"TRANSITION: {transition}"
    risk_text = (
        f"RISK: {group.upper()}    margins base={float(base_result['margin']):.3f} "
        f"current={float(current_result['margin']):.3f}"
    )
    draw_text_line(info, gt_text, (30, 30), fit_text_scale(gt_text, body.shape[1] - 50, 0.64), (35, 35, 35))
    draw_text_line(
        info,
        transition_text,
        (30, 64),
        fit_text_scale(transition_text, body.shape[1] - 50, 0.64),
        color,
    )
    draw_text_line(
        info,
        risk_text,
        (30, 98),
        fit_text_scale(risk_text, body.shape[1] - 50, 0.58),
        (35, 35, 35),
    )
    return cv2.vconcat([body, info])


def crop_pair(
    base: Any,
    current: Any,
    points: Any,
    crop_box: Tuple[int, int, int, int],
) -> Tuple[Any, Any, Any]:
    x1, y1, x2, y2 = crop_box
    base_crop = base[y1:y2, x1:x2].copy()
    current_crop = current[y1:y2, x1:x2].copy()
    rel_points = points.copy()
    rel_points[:, 0] -= x1
    rel_points[:, 1] -= y1
    return base_crop, current_crop, rel_points


def focus_polygon(crop: np.ndarray, rel_points: np.ndarray) -> np.ndarray:
    polygon = np.round(rel_points).astype(np.int32)
    if len(polygon) < 3:
        return crop
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=18, sigmaY=18)
    background = cv2.addWeighted(blurred, 0.25, np.full_like(crop, 127), 0.75, 0)
    return np.where(mask[..., None] > 0, crop, background)


def write_results_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "json_path",
        "shape_index",
        "source_label",
        "base_label",
        "base_raw_label",
        "base_selected_view",
        "base_context_top1",
        "base_focused_top1",
        "base_score",
        "base_margin",
        "current_label",
        "current_display_label",
        "current_raw_label",
        "current_selected_view",
        "current_context_top1",
        "current_focused_top1",
        "current_score",
        "current_margin",
        "transition_label",
        "transition_source",
        "pair_label",
        "pair_score",
        "pair_margin",
        "pair_top3",
        "risk_group",
        "base_top3",
        "current_top3",
        "preview_path",
        "change_pixels",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(args: argparse.Namespace) -> None:
    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root)
    model_dir_arg = Path(args.model_dir)
    model_dir = (
        model_dir_arg.absolute()
        if model_dir_arg.is_absolute()
        else (project_root() / model_dir_arg).absolute()
    )
    prompts = load_prompts(args.prompts)
    device = torch.device(args.device)
    open_clip, model, preprocess = load_model(model_dir, device, args.precision)
    state_labels, text_features = encode_text(
        open_clip, model, device, prompts
    )
    transition_labels, transition_prompt_labels, transition_text_features = (
        encode_prompt_ensemble(
        open_clip, model, device, DEFAULT_TRANSITION_PROMPTS
        )
    )

    json_paths = sorted(input_root.rglob("*_mask.json"))
    if args.limit > 0:
        json_paths = json_paths[: args.limit]
    jobs: List[Dict[str, Any]] = []

    for json_path in json_paths:
        try:
            payload = read_json(json_path)
            current_path = find_pair_image(json_path, "_current", payload)
            base_path = find_pair_image(json_path, "_base", payload)
            if current_path is None or base_path is None:
                continue
            current = read_image(current_path)
            base = resize_like(read_image(base_path), current.shape[:2])
        except Exception as exc:
            print(f"[WARN] skip {json_path}: {exc}")
            continue

        height, width = current.shape[:2]
        for shape_index, shape in enumerate(payload.get("shapes") or []):
            if not isinstance(shape, dict):
                continue
            points = shape_points(shape)
            if points.shape[0] < 3:
                continue
            box = clamp_crop_box(points, width, height, args.pad_ratio, args.min_pad)
            base_crop, current_crop, rel_points = crop_pair(base, current, points, box)
            base_focus = focus_polygon(base_crop, rel_points)
            current_focus = focus_polygon(current_crop, rel_points)
            pair_focus = cv2.hconcat([base_focus, current_focus])
            relative = json_path.relative_to(input_root)
            preview_path = output_root / "previews" / relative.parent / f"{json_path.stem}_shape{shape_index:04d}.jpg"
            jobs.append(
                {
                    "json_path": json_path,
                    "shape_index": shape_index,
                    "source_label": str(shape.get("label") or "").strip(),
                    "description": parse_description(shape.get("description")),
                    "base_context_pil": bgr_to_pil(base_crop),
                    "current_context_pil": bgr_to_pil(current_crop),
                    "base_pil": bgr_to_pil(base_focus),
                    "current_pil": bgr_to_pil(current_focus),
                    "pair_pil": bgr_to_pil(pair_focus),
                    "base_crop": base_crop,
                    "current_crop": current_crop,
                    "rel_points": rel_points,
                    "preview_path": preview_path,
                }
            )

    rows: List[Dict[str, Any]] = []
    source_transition = defaultdict(Counter)
    for batch in batched(jobs, max(1, args.batch_size)):
        base_context_results = classify_batch(
            model, preprocess, [job["base_context_pil"] for job in batch], text_features, state_labels,
            device, args.precision, args.logit_scale,
        )
        current_context_results = classify_batch(
            model, preprocess, [job["current_context_pil"] for job in batch], text_features, state_labels,
            device, args.precision, args.logit_scale,
        )
        base_focused_results = classify_batch(
            model, preprocess, [job["base_pil"] for job in batch], text_features, state_labels,
            device, args.precision, args.logit_scale,
        )
        current_focused_results = classify_batch(
            model, preprocess, [job["current_pil"] for job in batch], text_features, state_labels,
            device, args.precision, args.logit_scale,
        )
        if args.transition_mode == "hybrid":
            pair_results = classify_prompt_ensemble(
                model, preprocess, [job["pair_pil"] for job in batch],
                transition_text_features, transition_labels,
                transition_prompt_labels, device, args.precision,
                args.logit_scale,
            )
        else:
            pair_results = [
                {
                    "label": "",
                    "raw_label": "",
                    "score": 0.0,
                    "margin": 0.0,
                    "top3": [],
                }
                for _ in batch
            ]
        for job, base_context, current_context, base_focused, current_focused, pair_result in zip(
            batch,
            base_context_results,
            current_context_results,
            base_focused_results,
            current_focused_results,
            pair_results,
        ):
            base_result = select_view_result(base_context, base_focused)
            current_result = select_view_result(current_context, current_focused)
            base_result = reject_out_of_scope_state(
                base_result, float(args.min_out_of_scope_score)
            )
            current_result = reject_out_of_scope_state(
                current_result, float(args.min_out_of_scope_score)
            )
            if args.allow_unknown:
                base_result = reject_uncertain_state(
                    base_result, float(args.min_state_score), float(args.min_state_margin)
                )
                current_result = reject_uncertain_state(
                    current_result, float(args.min_state_score), float(args.min_state_margin)
                )
            if args.transition_mode == "state_only":
                transition = transition_label(
                    str(base_result["label"]),
                    str(current_result["label"]),
                )
                transition_source = "state_rule"
            else:
                transition, transition_source = resolve_transition(
                    base_result,
                    current_result,
                    pair_result,
                    float(args.min_pair_score),
                    float(args.min_pair_margin),
                )
            if (
                transition == "construction_clearing"
                and current_result["label"] == UNKNOWN_STATE
            ):
                current_result["display_label"] = "cleared_or_excavated_slope [pair]"
            group = risk_group(
                transition,
                float(base_result["margin"]),
                float(current_result["margin"]),
                float(args.min_margin),
                transition_source,
                float(pair_result["margin"]),
                float(args.min_pair_margin),
            )
            source_transition[job["source_label"]][transition] += 1
            preview = render_preview(
                job["base_crop"],
                job["current_crop"],
                job["rel_points"],
                job["source_label"],
                base_result,
                current_result,
                transition,
                group,
            )
            write_image(job["preview_path"], preview)
            rows.append(
                {
                    "json_path": str(job["json_path"]),
                    "shape_index": job["shape_index"],
                    "source_label": job["source_label"],
                    "base_label": base_result["label"],
                    "base_raw_label": base_result["raw_label"],
                    "base_selected_view": base_result["selected_view"],
                    "base_context_top1": base_result["context_top1"],
                    "base_focused_top1": base_result["focused_top1"],
                    "base_score": base_result["score"],
                    "base_margin": base_result["margin"],
                    "current_label": current_result["label"],
                    "current_display_label": current_result.get(
                        "display_label", current_result["label"]
                    ),
                    "current_raw_label": current_result["raw_label"],
                    "current_selected_view": current_result["selected_view"],
                    "current_context_top1": current_result["context_top1"],
                    "current_focused_top1": current_result["focused_top1"],
                    "current_score": current_result["score"],
                    "current_margin": current_result["margin"],
                    "transition_label": transition,
                    "transition_source": transition_source,
                    "pair_label": pair_result["label"],
                    "pair_score": pair_result["score"],
                    "pair_margin": pair_result["margin"],
                    "pair_top3": json.dumps(pair_result["top3"], ensure_ascii=False),
                    "risk_group": group,
                    "base_top3": json.dumps(base_result["top3"], ensure_ascii=False),
                    "current_top3": json.dumps(current_result["top3"], ensure_ascii=False),
                    "preview_path": str(job["preview_path"]),
                    "change_pixels": job["description"].get("change_pixels", ""),
                }
            )

    write_results_csv(output_root / "classifier_pair_zero_shot_results.csv", rows)
    write_html(output_root / "index.html", output_root, rows)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model_dir": str(model_dir),
        "num_json": len(json_paths),
        "num_shapes": len(rows),
        "state_labels": state_labels,
        "transition_labels": transition_labels,
        "unknown_state": UNKNOWN_STATE,
        "out_of_scope_state": OUT_OF_SCOPE_STATE,
        "min_out_of_scope_score": args.min_out_of_scope_score,
        "allow_unknown": args.allow_unknown,
        "non_target_states": sorted(NON_TARGET_STATES),
        "context_object_states": sorted(CONTEXT_OBJECT_STATES),
        "min_state_score": args.min_state_score,
        "min_state_margin": args.min_state_margin,
        "transition_mode": args.transition_mode,
        "min_pair_score": args.min_pair_score,
        "min_pair_margin": args.min_pair_margin,
        "source_label_counts": dict(Counter(row["source_label"] for row in rows)),
        "transition_counts": dict(Counter(row["transition_label"] for row in rows)),
        "risk_counts": dict(Counter(row["risk_group"] for row in rows)),
        "source_to_transition": {source: dict(counts) for source, counts in source_transition.items()},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_html(path: Path, output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    cards = []
    for row in rows:
        image_path = Path(str(row["preview_path"]))
        try:
            relative_image = image_path.relative_to(output_root).as_posix()
        except ValueError:
            relative_image = image_path.as_posix()
        source = html.escape(str(row["source_label"]))
        transition = html.escape(str(row["transition_label"]))
        group = html.escape(str(row["risk_group"]))
        cards.append(
            f"""
            <article class="item {group}" data-risk="{group}" data-source="{source}" data-transition="{transition}">
              <img src="{html.escape(relative_image)}" loading="lazy" alt="{transition}">
              <div class="meta">
                <strong>{transition}</strong>
                <span>GT: {source}</span>
                <span>{html.escape(str(row["base_label"]))} -> {html.escape(str(row["current_label"]))}</span>
              </div>
            </article>
            """
        )
    risks = sorted({str(row["risk_group"]) for row in rows})
    sources = sorted({str(row["source_label"]) for row in rows})
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Classifier pair zero-shot review</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #f3f5f6; color: #202326; }}
header {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #ccd2d6; padding: 12px 18px; }}
h1 {{ font-size: 20px; margin: 0 0 10px; }}
.filters {{ display: flex; gap: 10px; flex-wrap: wrap; }}
select {{ height: 34px; border: 1px solid #aeb7bd; background: #fff; padding: 0 30px 0 10px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 12px; padding: 12px; }}
.item {{ background: #fff; border: 1px solid #cbd1d5; border-left: 6px solid #e09a00; }}
.item.alarm {{ border-left-color: #dc2828; }}
.item.filter {{ border-left-color: #299044; }}
.item img {{ width: 100%; display: block; }}
.meta {{ display: grid; grid-template-columns: 1.3fr .7fr 1fr; gap: 8px; padding: 9px 12px; font-size: 13px; }}
.hidden {{ display: none; }}
</style>
</head>
<body>
<header>
  <h1>Classifier base/current zero-shot review ({len(rows)} regions)</h1>
  <div class="filters">
    <select id="risk"><option value="">All risk groups</option>{''.join(f'<option>{html.escape(v)}</option>' for v in risks)}</select>
    <select id="source"><option value="">All GT labels</option>{''.join(f'<option>{html.escape(v)}</option>' for v in sources)}</select>
  </div>
</header>
<main>{''.join(cards)}</main>
<script>
const risk = document.getElementById('risk');
const source = document.getElementById('source');
function applyFilters() {{
  document.querySelectorAll('.item').forEach(item => {{
    const visible = (!risk.value || item.dataset.risk === risk.value) &&
                    (!source.value || item.dataset.source === source.value);
    item.classList.toggle('hidden', !visible);
  }});
}}
risk.addEventListener('change', applyFilters);
source.addEventListener('change', applyFilters);
</script>
</body>
</html>"""
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify base/current LabelMe crops separately with OpenCLIP.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-dir", default="models/classifier/zero_shot_vit_l14")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pad-ratio", type=float, default=0.25)
    parser.add_argument("--min-pad", type=int, default=32)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--min-state-score", type=float, default=0.20)
    parser.add_argument("--min-state-margin", type=float, default=0.05)
    parser.add_argument(
        "--min-out-of-scope-score",
        type=float,
        default=0.25,
        help="Reject a state entirely when its best 13-class score is below this value.",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Reject low-score or low-margin states as unknown_or_uncertain.",
    )
    parser.add_argument(
        "--transition-mode",
        choices=["state_only", "hybrid"],
        default="hybrid",
        help="Use only base/current state rules, or allow pair-classifier transition overrides.",
    )
    parser.add_argument("--min-pair-score", type=float, default=0.45)
    parser.add_argument("--min-pair-margin", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
