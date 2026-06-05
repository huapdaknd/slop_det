import math
from typing import Tuple

import numpy as np


def clamp_box_xyxy(
    box_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    max_x = max(1, int(image_width)) - 1
    max_y = max(1, int(image_height)) - 1
    x1 = min(max(x1, 0.0), float(max_x))
    y1 = min(max(y1, 0.0), float(max_y))
    x2 = min(max(x2, x1 + 1.0), float(image_width))
    y2 = min(max(y2, y1 + 1.0), float(image_height))
    return x1, y1, x2, y2


def compute_crop_box_xyxy(
    box_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    context_ratio: float = 0.0,
    min_context_pixels: int = 0,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = clamp_box_xyxy(box_xyxy, image_width, image_height)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ratio = max(0.0, float(context_ratio))
    min_context = max(0, int(min_context_pixels))

    pad_x = max(float(min_context), bw * ratio)
    pad_y = max(float(min_context), bh * ratio)

    crop_x1 = max(0, int(math.floor(x1 - pad_x)))
    crop_y1 = max(0, int(math.floor(y1 - pad_y)))
    crop_x2 = min(int(image_width), int(math.ceil(x2 + pad_x)))
    crop_y2 = min(int(image_height), int(math.ceil(y2 + pad_y)))

    if crop_x2 <= crop_x1:
        crop_x2 = min(int(image_width), crop_x1 + 1)
    if crop_y2 <= crop_y1:
        crop_y2 = min(int(image_height), crop_y1 + 1)
    return crop_x1, crop_y1, crop_x2, crop_y2


def remap_box_to_resized_crop(
    box_xyxy: Tuple[float, float, float, float],
    crop_xyxy: Tuple[int, int, int, int],
    output_width: int,
    output_height: int,
) -> Tuple[float, float, float, float]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
    crop_w = max(1, int(crop_x2 - crop_x1))
    crop_h = max(1, int(crop_y2 - crop_y1))
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    scale_x = float(output_width) / float(crop_w)
    scale_y = float(output_height) / float(crop_h)

    rx1 = (x1 - crop_x1) * scale_x
    ry1 = (y1 - crop_y1) * scale_y
    rx2 = (x2 - crop_x1) * scale_x
    ry2 = (y2 - crop_y1) * scale_y

    rx1 = min(max(rx1, 0.0), float(max(1, output_width) - 1))
    ry1 = min(max(ry1, 0.0), float(max(1, output_height) - 1))
    rx2 = min(max(rx2, rx1 + 1.0), float(output_width))
    ry2 = min(max(ry2, ry1 + 1.0), float(output_height))
    return rx1, ry1, rx2, ry2


def build_box_mask(
    output_width: int,
    output_height: int,
    box_xyxy: Tuple[float, float, float, float],
) -> np.ndarray:
    mask = np.zeros((int(output_height), int(output_width)), dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    ix1 = max(0, min(int(round(x1)), int(output_width) - 1))
    iy1 = max(0, min(int(round(y1)), int(output_height) - 1))
    ix2 = max(ix1 + 1, min(int(round(x2)), int(output_width)))
    iy2 = max(iy1 + 1, min(int(round(y2)), int(output_height)))
    mask[iy1:iy2, ix1:ix2] = 1.0
    return mask
