"""
Generalizable Scene Change Detection Framework (GeSCF)
"""

import logging
import cv2
import numpy as np
from scipy.stats import skew

import torch
import torch.nn as nn
import torchvision

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Internal utilities
from .utils import calculate_iou, sanity_check_args, load_backbones

# Modules
from .pseudo_generator import PseudoGenerator
from .registration import coarse_transform


def imread_unicode(path, flags):
    """Read images with Unicode paths on Windows using cv2.imdecode."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Cannot read image data from: {path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"cv2.imdecode failed for: {path}")
    return img


def letterbox_image(img, target_shape, pad_value=0, interpolation=cv2.INTER_LINEAR):
    """Resize image to target_shape with preserved aspect ratio and padding."""
    src_h, src_w = img.shape[:2]
    tgt_h, tgt_w = target_shape
    if src_h == 0 or src_w == 0 or tgt_h == 0 or tgt_w == 0:
        if img.ndim == 2:
            return np.zeros((tgt_h, tgt_w), dtype=img.dtype)
        return np.zeros((tgt_h, tgt_w, img.shape[2]), dtype=img.dtype)
    if (src_h, src_w) == (tgt_h, tgt_w):
        return img

    scale = min(tgt_w / float(src_w), tgt_h / float(src_h))
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    if new_w <= 0 or new_h <= 0:
        if img.ndim == 2:
            return np.zeros((tgt_h, tgt_w), dtype=img.dtype)
        return np.zeros((tgt_h, tgt_w, img.shape[2]), dtype=img.dtype)

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    if img.ndim == 2:
        canvas = np.full((tgt_h, tgt_w), pad_value, dtype=img.dtype)
    else:
        canvas = np.full((tgt_h, tgt_w, img.shape[2]), pad_value, dtype=img.dtype)
    top = (tgt_h - new_h) // 2
    left = (tgt_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


class GeSCF(nn.Module):
    def __init__(self, args):
        sanity_check_args(args)
        super(GeSCF, self).__init__()

        # Dataset settings
        self.dataset = args.test_dataset
        self.dataset_bias = self.dataset == 'VL_CMU_CD'
        
        self.img_size = (256, 256) if self.dataset == 'TSUNAMI' else (512, 512)
        self.output_size = (args.output_size, args.output_size)
        
        logging.info(f'dataset name: {self.dataset}')

        # Feature extraction settings
        self.feature_facet = args.feature_facet
        self.feature_layer = args.feature_layer
        self.embedding_layer = args.embedding_layer
        self.pseudo_mask_mode = getattr(args, 'pseudo_mask_mode', 'default')
        self.pseudo_mask_open_kernel = max(1, int(getattr(args, 'pseudo_mask_open_kernel', 17)))
        self.pseudo_mask_seed_dilate_kernel = max(1, int(getattr(args, 'pseudo_mask_seed_dilate_kernel', 33)))
        self.pseudo_mask_close_kernel = max(0, int(getattr(args, 'pseudo_mask_close_kernel', 0)))
        self.adaptive_threshold_scale = float(getattr(args, 'adaptive_threshold_scale', 1.0))
        self.adaptive_threshold_mode = str(getattr(args, 'adaptive_threshold_mode', 'mad') or 'mad').lower()
        self.adaptive_threshold_min = getattr(args, 'adaptive_threshold_min', None)
        self.adaptive_threshold_max = getattr(args, 'adaptive_threshold_max', None)
        self.adaptive_threshold_mad_eps = float(getattr(args, 'adaptive_threshold_mad_eps', 0.0))
        self.adaptive_threshold_sim_std_eps = float(getattr(args, 'adaptive_threshold_sim_std_eps', 0.0))
        self.adaptive_threshold_candidate_percentile_min = float(
            getattr(args, 'adaptive_threshold_candidate_percentile_min', 85.0)
        )
        self.adaptive_threshold_seed_percentile = float(getattr(args, 'adaptive_threshold_seed_percentile', 50.0))
        self.adaptive_threshold_max_candidate_ratio = float(
            getattr(args, 'adaptive_threshold_max_candidate_ratio', 0.25)
        )

        # Default hyperparameters
        self.z_value = -0.52
        self.Ni = -0.2
        self.Nj = 0.2
        self.alpha_t = 0.65
        self.cosine_thr = 0.88
        
        # Build SAM and pseudo backbone
        self.sam_backbone, self.pseudo_backbone = load_backbones(args.sam_backbone, args.pseudo_backbone)
    
        # Build automatic mask generator
        if args.sam_backbone == "vit_t":
            from mobile_sam import SamAutomaticMaskGenerator as SamAutomaticMaskGenerator
        else:
            from segment_anything_model import SamAutomaticMaskGenerator as SamAutomaticMaskGenerator

        self.automatic_mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_backbone,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=0
        )

        # Build pseudo generator
        self.pseudo_generator = PseudoGenerator(
            feature_layer=self.feature_layer,
            embedding_layer=self.embedding_layer,
            img_size=self.img_size,
            backbone=self.pseudo_backbone
        )
        
    
    def load_img(self, img_path):
        """Load and preprocess an image (RGB and grayscale)."""

        # Load and letterbox RGB image
        bgr_img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = letterbox_image(rgb_img, self.img_size, pad_value=0, interpolation=cv2.INTER_LINEAR)
        rgb_img = np.array(rgb_img)

        # Load and letterbox grayscale image
        gray_img = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
        gray_img = letterbox_image(gray_img, self.img_size, pad_value=0, interpolation=cv2.INTER_LINEAR) / 255.
        gray_img = np.array(gray_img)

        # Transform RGB image to tensor
        input_tensor = self.transform()(rgb_img).unsqueeze(0)

        return rgb_img, gray_img, input_tensor


    def transform(self):
        """Return composed torchvision transform for RGB image preprocessing."""
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        
        
    def get_skewness_and_type(self, key, query, value, img_t1, flag):
        """Calculate skewness, type, and moderate region mask based on feature similarity map."""

        # Select similarity feature map
        if self.feature_facet == 'key':
            sim_map = key
        elif self.feature_facet == 'query':
            sim_map = query
        else:
            sim_map = value

        sim_map = sim_map.detach().cpu().numpy()[0]

        # Flatten relevant regions based on flag
        if flag:
            oov_idx = np.where(np.any(img_t1 != [0, 0, 0], axis=-1))
            flat_sim_map = sim_map[oov_idx].flatten()
        else:
            oov_idx = None
            flat_sim_map = sim_map.flatten()

        # Compute skewness
        skewness = skew(flat_sim_map)

        # Determine skewness type
        if skewness >= self.Nj:
            skew_type = 'Right-skewed'
        elif skewness <= self.Ni:
            skew_type = 'Left-skewed'
        else:
            skew_type = 'Moderate'

        # Generate moderate region mask using z-score thresholding
        mean = np.mean(sim_map)
        std = np.std(sim_map)
        z_score = (sim_map - mean) / std
        moderate_mask = (z_score < self.z_value).astype(np.float32)

        return skewness, skew_type, sim_map, flat_sim_map, moderate_mask, oov_idx


    def threshold(self, skewness):
        """Calculate dynamic threshold based on image size and skewness type."""
        h = self.img_size[0]
        w = self.img_size[1]
        
        # Threshold parameters
        b_left = 0.7
        b_right = 0.05
        s_left = 1.0
        s_right = 0.1
        mu = 2.5e5 / (h * w)
        c = 1.0 / mu**3

        # Threshold calculation based on skewness
        if skewness >= self.Nj:
            threshold = b_right + s_right * skewness * c
        elif skewness <= self.Ni:
            threshold = b_left - s_left * skewness * c
        else:
            threshold = 0.0

        threshold = float(threshold) * self.adaptive_threshold_scale
        if self.adaptive_threshold_min is not None:
            threshold = max(threshold, float(self.adaptive_threshold_min))
        if self.adaptive_threshold_max is not None:
            threshold = min(threshold, float(self.adaptive_threshold_max))
        return threshold


    def _modified_change_scores(self, flat_sim_map):
        """Convert similarity values into robust low-similarity change scores."""

        median = np.median(flat_sim_map)
        mad = np.median(np.abs(flat_sim_map - median))
        if self.adaptive_threshold_mad_eps > 0:
            mad = max(float(mad), self.adaptive_threshold_mad_eps)
        if mad <= 0 or not np.isfinite(mad):
            return np.zeros_like(flat_sim_map, dtype=np.float32)
        modified_z_scores = 0.6745 * (flat_sim_map - median) / mad
        return (-modified_z_scores).astype(np.float32, copy=False)


    def _otsu_threshold_from_scores(self, scores):
        """Find a data-driven split point from the positive change-score tail."""

        positive = scores[np.isfinite(scores) & (scores > 0)]
        if positive.size < 2:
            return 0.0

        score_min = float(positive.min())
        score_max = float(positive.max())
        if score_max - score_min < 1e-8:
            return score_max

        scaled = np.clip(np.round((positive - score_min) / (score_max - score_min) * 255.0), 0, 255).astype(np.uint8)
        otsu_value, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return score_min + (float(otsu_value) / 255.0) * (score_max - score_min)


    def _clip_adaptive_threshold(self, value):
        value = float(value)
        if self.adaptive_threshold_min is not None:
            value = max(value, float(self.adaptive_threshold_min))
        if self.adaptive_threshold_max is not None:
            value = min(value, float(self.adaptive_threshold_max))
        return value


    def hysteresis_threshold_function(self, flat_sim_map):
        """Build high-confidence seeds and connected weak candidates from one image pair."""

        scores = self._modified_change_scores(flat_sim_map)
        finite_scores = scores[np.isfinite(scores)]
        if self.adaptive_threshold_sim_std_eps > 0:
            sim_std = float(np.std(flat_sim_map))
            if sim_std <= self.adaptive_threshold_sim_std_eps:
                empty = np.zeros_like(flat_sim_map, dtype=bool)
                return empty, empty, {
                    "candidate": 0.0,
                    "candidate_otsu": 0.0,
                    "candidate_percentile": 0.0,
                    "seed": 0.0,
                    "positive_pixels": int((scores > 0).sum()),
                    "tail_pixels": 0,
                    "candidate_pixels": 0,
                }

        if finite_scores.size == 0:
            empty = np.zeros_like(flat_sim_map, dtype=bool)
            return empty, empty, {
                "candidate": 0.0,
                "candidate_otsu": 0.0,
                "candidate_percentile": 0.0,
                "seed": 0.0,
                "positive_pixels": 0,
                "tail_pixels": 0,
                "candidate_pixels": 0,
            }

        otsu_threshold = self._otsu_threshold_from_scores(scores)
        percentile_min = float(np.clip(self.adaptive_threshold_candidate_percentile_min, 0.0, 100.0))
        percentile_threshold = float(np.percentile(finite_scores, percentile_min))
        candidate_threshold = self._clip_adaptive_threshold(max(otsu_threshold, percentile_threshold))
        candidate_outliers = scores >= candidate_threshold
        max_candidate_ratio = float(self.adaptive_threshold_max_candidate_ratio)
        if 0.0 < max_candidate_ratio < 1.0:
            max_candidate_pixels = max(1, int(round(scores.size * max_candidate_ratio)))
            candidate_pixels = int(candidate_outliers.sum())
            if candidate_pixels > max_candidate_pixels:
                capped_threshold = float(np.partition(finite_scores, -max_candidate_pixels)[-max_candidate_pixels])
                candidate_threshold = self._clip_adaptive_threshold(max(candidate_threshold, capped_threshold))
                candidate_outliers = scores >= candidate_threshold

        tail_scores = scores[candidate_outliers]
        if tail_scores.size == 0:
            empty = np.zeros_like(flat_sim_map, dtype=bool)
            return empty, empty, {
                "candidate": float(candidate_threshold),
                "candidate_otsu": float(otsu_threshold),
                "candidate_percentile": float(percentile_threshold),
                "seed": float(candidate_threshold),
                "positive_pixels": int((scores > 0).sum()),
                "tail_pixels": 0,
                "candidate_pixels": 0,
            }

        seed_percentile = float(np.clip(self.adaptive_threshold_seed_percentile, 0.0, 100.0))
        seed_threshold = self._clip_adaptive_threshold(np.percentile(tail_scores, seed_percentile))
        if seed_threshold < candidate_threshold:
            seed_threshold = candidate_threshold
        seed_outliers = scores >= seed_threshold

        return seed_outliers, candidate_outliers, {
            "candidate": float(candidate_threshold),
            "candidate_otsu": float(otsu_threshold),
            "candidate_percentile": float(percentile_threshold),
            "seed": float(seed_threshold),
            "positive_pixels": int((scores > 0).sum()),
            "tail_pixels": int(tail_scores.size),
            "candidate_pixels": int(candidate_outliers.sum()),
        }
    
    
    def adaptive_threshold_function(self, flat_sim_map, skewness):
        """Detect outliers in the similarity map using MAD-based adaptive thresholding."""

        change_scores = self._modified_change_scores(flat_sim_map)

        # Determine threshold from skewness
        threshold = self.threshold(skewness)

        # Identify outliers
        outliers = change_scores > threshold

        return outliers


    @staticmethod
    def _build_ellipse_kernel(kernel_size):
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


    def _outliers_to_mask(self, outliers, sim_map, oov_idx, flag):
        binary_mask = np.zeros_like(sim_map, dtype=np.uint8)
        if flag:
            binary_mask[oov_idx[0][outliers], oov_idx[1][outliers]] = 1
        else:
            binary_mask[outliers.reshape(self.img_size)] = 1
        return binary_mask


    def _hysteresis_region_grow(self, seed_mask, candidate_mask):
        seed_mask = (seed_mask > 0).astype(np.uint8)
        candidate_mask = (candidate_mask > 0).astype(np.uint8)
        if not seed_mask.any():
            return seed_mask

        support_kernel = self._build_ellipse_kernel(self.pseudo_mask_seed_dilate_kernel)
        seed_support = cv2.dilate(seed_mask, support_kernel, iterations=1)
        num_labels, labels = cv2.connectedComponents(candidate_mask, connectivity=8)
        grown = seed_mask.copy()
        for label_id in range(1, num_labels):
            component = labels == label_id
            if np.any(seed_support[component] > 0):
                grown[component] = 1
        return grown.astype(np.uint8)


    def _compose_initial_pseudo_mask(self, skew_type, binary_mask_outliers, moderate_mask):
        """Compose the initial pseudo mask before morphology."""

        if self.pseudo_mask_mode == 'preserve_large':
            core_mask = (binary_mask_outliers > 0).astype(np.uint8)
            moderate_region = (moderate_mask > 0).astype(np.uint8)

            if core_mask.any():
                dilate_kernel = self._build_ellipse_kernel(self.pseudo_mask_seed_dilate_kernel)
                core_support = cv2.dilate(core_mask, dilate_kernel, iterations=1)
                expanded_region = np.logical_and(core_support > 0, moderate_region > 0)
                initial_mask = np.logical_or(core_mask > 0, expanded_region)
            else:
                initial_mask = moderate_region > 0
            return initial_mask.astype(np.uint8)

        if skew_type in ['Left-skewed', 'Right-skewed']:
            return (binary_mask_outliers > 0).astype(np.uint8)
        return (moderate_mask > 0).astype(np.uint8)


    def _refine_initial_pseudo_mask(self, initial_pseudo_mask):
        """Apply morphology to the initial pseudo mask."""

        refined = initial_pseudo_mask.astype(np.uint8)
        open_kernel = self._build_ellipse_kernel(self.pseudo_mask_open_kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, open_kernel)

        if self.pseudo_mask_mode == 'preserve_large' and self.pseudo_mask_close_kernel > 0:
            close_kernel = self._build_ellipse_kernel(self.pseudo_mask_close_kernel)
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_kernel)

        return refined

    def _prepare_pair_inputs(self, img_t0_path, img_t1_path):
        """Load images, generate SAM proposals, and apply optional coarse alignment."""

        img_t0, gray_img_t0, input_t0 = self.load_img(img_t0_path)
        img_t1, gray_img_t1, input_t1 = self.load_img(img_t1_path)

        masks_t0 = self.automatic_mask_generator.generate(img_t0)
        masks_t1 = self.automatic_mask_generator.generate(img_t1)

        aligned_img_t1, H, flag = coarse_transform(
            self.dataset, self.img_size, img_t0, img_t1, gray_img_t0, gray_img_t1
        )
        if self.dataset == 'Remote_Sensing':
            flag = False

        if flag:
            img_t1 = np.array(aligned_img_t1)
            input_t1 = self.transform()(img_t1).unsqueeze(0)
            if self.dataset == 'ChangeSim':
                masks_t1 = self.automatic_mask_generator.generate(img_t1)

        return {
            "img_t0": img_t0,
            "img_t1": img_t1,
            "input_t0": input_t0,
            "input_t1": input_t1,
            "masks_t0": masks_t0,
            "masks_t1": masks_t1,
            "H": H,
            "flag": flag,
        }


    def _generate_initial_pseudo_outputs(self, img_t0, img_t1, input_t0, input_t1, H, flag):
        """Run the initial pseudo-mask branch and retain debug artifacts."""

        inputs = torch.cat([input_t0, input_t1], dim=1).to(device='cuda')
        embed_t0, embed_t1, key, query, value = self.pseudo_generator(inputs)

        skewness, skew_type, sim_map, flat_sim_map, moderate_mask, oov_idx = self.get_skewness_and_type(
            key, query, value, img_t1, flag
        )

        threshold_meta = {
            "candidate": float(self.threshold(skewness)),
            "seed": None,
            "positive_pixels": None,
            "tail_pixels": None,
        }
        binary_mask_candidate = None
        if self.adaptive_threshold_mode == 'hysteresis':
            seed_outliers, candidate_outliers, threshold_meta = self.hysteresis_threshold_function(flat_sim_map)
            binary_mask_seed = self._outliers_to_mask(seed_outliers, sim_map, oov_idx, flag)
            binary_mask_candidate = self._outliers_to_mask(candidate_outliers, sim_map, oov_idx, flag)
            binary_mask_outliers = self._hysteresis_region_grow(binary_mask_seed, binary_mask_candidate)
        else:
            outliers = self.adaptive_threshold_function(flat_sim_map, skewness)
            binary_mask_outliers = self._outliers_to_mask(outliers, sim_map, oov_idx, flag)

        binary_mask_outliers_raw = binary_mask_outliers.copy()
        moderate_mask_raw = moderate_mask.copy()

        if self.dataset == 'VL_CMU_CD':
            oov_mask = np.all(img_t0 == [0, 0, 0], axis=-1)
            binary_mask_outliers[oov_mask] = 0
            moderate_mask[oov_mask] = 0

        if flag:
            warped_oov_mask = np.all(img_t1 == [0, 0, 0], axis=-1)
            binary_mask_outliers[warped_oov_mask] = 0
            moderate_mask[warped_oov_mask] = 0

            padding_size = 10
            kernel_size = 2 * padding_size + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            warped_oov_mask_uint8 = warped_oov_mask.astype(np.uint8) * 255
            dilated_mask = cv2.dilate(warped_oov_mask_uint8, kernel, iterations=1).astype(bool)

            binary_mask_outliers[dilated_mask] = 0
            moderate_mask[dilated_mask] = 0

            if self.dataset not in ['ChangeSim', 'VL_CMU_CD']:
                H_inv = np.linalg.inv(H)
                binary_mask_outliers = cv2.warpPerspective(binary_mask_outliers, H_inv, self.img_size)
                moderate_mask = cv2.warpPerspective(moderate_mask, H_inv, self.img_size)

        initial_pseudo_mask_before_open = self._compose_initial_pseudo_mask(
            skew_type=skew_type,
            binary_mask_outliers=binary_mask_outliers,
            moderate_mask=moderate_mask,
        )

        initial_pseudo_mask = self._refine_initial_pseudo_mask(initial_pseudo_mask_before_open)

        contours, _ = cv2.findContours(initial_pseudo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                cv2.drawContours(initial_pseudo_mask, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)

        return {
            "embed_t0": embed_t0,
            "embed_t1": embed_t1,
            "sim_map": sim_map,
            "skewness": float(skewness),
            "skew_type": skew_type,
            "adaptive_threshold": float(threshold_meta["candidate"]),
            "adaptive_threshold_mode": self.adaptive_threshold_mode,
            "adaptive_threshold_candidate": float(threshold_meta["candidate"]),
            "adaptive_threshold_candidate_otsu": float(threshold_meta.get("candidate_otsu", threshold_meta["candidate"])),
            "adaptive_threshold_candidate_percentile": float(
                threshold_meta.get("candidate_percentile", threshold_meta["candidate"])
            ),
            "adaptive_threshold_seed": (
                None if threshold_meta["seed"] is None else float(threshold_meta["seed"])
            ),
            "adaptive_threshold_positive_pixels": threshold_meta["positive_pixels"],
            "adaptive_threshold_tail_pixels": threshold_meta["tail_pixels"],
            "adaptive_threshold_candidate_pixels": threshold_meta.get("candidate_pixels"),
            "adaptive_threshold_max_candidate_ratio": self.adaptive_threshold_max_candidate_ratio,
            "pseudo_mask_mode": self.pseudo_mask_mode,
            "pseudo_mask_open_kernel": int(self.pseudo_mask_open_kernel),
            "pseudo_mask_seed_dilate_kernel": int(self.pseudo_mask_seed_dilate_kernel),
            "pseudo_mask_close_kernel": int(self.pseudo_mask_close_kernel),
            "binary_mask_outliers_raw": binary_mask_outliers_raw.astype(np.uint8),
            "binary_mask_candidate": (
                None if binary_mask_candidate is None else binary_mask_candidate.astype(np.uint8)
            ),
            "binary_mask_outliers": binary_mask_outliers.astype(np.uint8),
            "moderate_mask_raw": moderate_mask_raw.astype(np.uint8),
            "moderate_mask": moderate_mask.astype(np.uint8),
            "initial_pseudo_mask_before_open": initial_pseudo_mask_before_open.astype(np.uint8),
            "initial_pseudo_mask": initial_pseudo_mask.astype(np.uint8),
        }


    def forward(self, img_t0_path, img_t1_path):
        '''Generate final change mask from a pair of input images.'''

        pair_data = self._prepare_pair_inputs(img_t0_path, img_t1_path)
        masks_t0 = pair_data["masks_t0"]
        masks_t1 = pair_data["masks_t1"]

        pseudo_outputs = self._generate_initial_pseudo_outputs(
            img_t0=pair_data["img_t0"],
            img_t1=pair_data["img_t1"],
            input_t0=pair_data["input_t0"],
            input_t1=pair_data["input_t1"],
            H=pair_data["H"],
            flag=pair_data["flag"],
        )
        embed_t0 = pseudo_outputs["embed_t0"]
        embed_t1 = pseudo_outputs["embed_t1"]
        initial_pseudo_mask = pseudo_outputs["initial_pseudo_mask"]

        ####################################################
        # Geometric-Semantic Mask Matching 
        ####################################################

        mask_idx_t0 = []
        mask_idx_t1 = []

        # Geometric + Semantic matching: masks_t0
        for i in range(len(masks_t0)):
            iou, overlap_mask = calculate_iou(initial_pseudo_mask, masks_t0[i]['segmentation'])
            if iou >= self.alpha_t:
                mask_embedding_t0 = embed_t0[overlap_mask].mean(axis=0)
                mask_embedding_t1 = embed_t1[overlap_mask].mean(axis=0)
                cosine_similarity = torch.nn.functional.cosine_similarity(mask_embedding_t0, mask_embedding_t1, dim=0)
                if cosine_similarity < self.cosine_thr:
                    mask_idx_t0.append(i)

        x = np.zeros_like(initial_pseudo_mask)
        for j in mask_idx_t0:
            x = np.logical_or(x, masks_t0[j]['segmentation'])

        # Geometric + Semantic matching: masks_t1
        if not self.dataset_bias:
            for k in range(len(masks_t1)):
                iou, overlap_mask = calculate_iou(initial_pseudo_mask, masks_t1[k]['segmentation'])
                if iou >= self.alpha_t:
                    mask_embedding_t0 = embed_t0[overlap_mask].mean(axis=0)
                    mask_embedding_t1 = embed_t1[overlap_mask].mean(axis=0)
                    cosine_similarity = torch.nn.functional.cosine_similarity(mask_embedding_t0, mask_embedding_t1, dim=0)
                    if cosine_similarity < self.cosine_thr:
                        mask_idx_t1.append(k)

            y = np.zeros_like(initial_pseudo_mask)
            for l in mask_idx_t1:
                y = np.logical_or(y, masks_t1[l]['segmentation'])

            final_change_mask = np.logical_or(x, y)
        else:
            final_change_mask = x
            
        # Resize final_change_mask to output-size and ensure binary output
        final_change_mask = final_change_mask.astype(np.uint8) * 255  # ensure dtype and scale for cv2
        final_change_mask = cv2.resize(final_change_mask, self.output_size, interpolation=cv2.INTER_LINEAR)
        final_change_mask = (final_change_mask > 127).astype(np.uint8)  # threshold to binary

        return final_change_mask


    def debug_forward(self, img_t0_path, img_t1_path):
        """Run GeSCF and return intermediate pseudo-mask artifacts for debugging."""

        pair_data = self._prepare_pair_inputs(img_t0_path, img_t1_path)
        pseudo_outputs = self._generate_initial_pseudo_outputs(
            img_t0=pair_data["img_t0"],
            img_t1=pair_data["img_t1"],
            input_t0=pair_data["input_t0"],
            input_t1=pair_data["input_t1"],
            H=pair_data["H"],
            flag=pair_data["flag"],
        )

        debug_payload = {
            "img_t0": pair_data["img_t0"],
            "img_t1": pair_data["img_t1"],
            "flag": bool(pair_data["flag"]),
            "num_masks_t0": len(pair_data["masks_t0"]),
            "num_masks_t1": len(pair_data["masks_t1"]),
        }
        debug_payload.update(pseudo_outputs)
        return debug_payload
                
        
        
        
