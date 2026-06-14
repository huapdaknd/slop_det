from pathlib import Path
from typing import Dict, Optional

from .clip_label_infer import ClipLabelMappingService
from .diff_yolo_infer import DiffYoloClassificationService
from .model_infer import OnlineChangeService
from .veg_infer import VegetationCoverageService
from .yolo_infer import YoloDetectionService


class SlopDetService:
    """Unified class API for edge, vegetation, and diff+YOLO classification."""

    def __init__(
        self,
        cd_config_path: Optional[str] = None,
        veg_config_path: Optional[str] = None,
        yolo_config_path: Optional[str] = None,
        clip_config_path: Optional[str] = None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        default_cd = root / "config" / "model_config.json"
        default_veg = root / "config" / "vegetation_config.json"
        default_yolo = root / "config" / "model_config_yolo.json"
        default_clip = root / "config" / "clip_label_config.json"

        self.cd_config_path = str(Path(cd_config_path).resolve()) if cd_config_path else str(default_cd.resolve())
        self.veg_config_path = (
            str(Path(veg_config_path).resolve()) if veg_config_path else str(default_veg.resolve())
        )
        self.yolo_config_path = (
            str(Path(yolo_config_path).resolve()) if yolo_config_path else str(default_yolo.resolve())
        )
        self.clip_config_path = (
            str(Path(clip_config_path).resolve()) if clip_config_path else str(default_clip.resolve())
        )
        self._cd_service: Optional[OnlineChangeService] = None
        self._veg_service: Optional[VegetationCoverageService] = None
        self._yolo_service: Optional[YoloDetectionService] = None
        self._diff_yolo_service: Optional[DiffYoloClassificationService] = None
        self._clip_service: Optional[ClipLabelMappingService] = None

    def _get_cd_service(self) -> OnlineChangeService:
        if self._cd_service is None:
            self._cd_service = OnlineChangeService.from_config_file(self.cd_config_path)
        return self._cd_service

    def _get_veg_service(self) -> VegetationCoverageService:
        if self._veg_service is None:
            self._veg_service = VegetationCoverageService.from_config_file(self.veg_config_path)
        return self._veg_service

    def _get_yolo_service(self) -> YoloDetectionService:
        if self._yolo_service is None:
            self._yolo_service = YoloDetectionService.from_config_file(self.yolo_config_path)
        return self._yolo_service

    def _get_clip_service(self) -> ClipLabelMappingService:
        if self._clip_service is None:
            self._clip_service = ClipLabelMappingService.from_config_file(
                self.clip_config_path
            )
        return self._clip_service

    def _get_diff_yolo_service(
        self,
        min_overlap_ratio: float = 0.15,
        min_overlap_pixels: int = 100,
    ) -> DiffYoloClassificationService:
        if self._diff_yolo_service is None:
            self._diff_yolo_service = DiffYoloClassificationService(
                cd_service=self._get_cd_service(),
                yolo_service=self._get_yolo_service(),
                min_overlap_ratio=min_overlap_ratio,
                min_overlap_pixels=min_overlap_pixels,
            )
        else:
            self._diff_yolo_service.min_overlap_ratio = float(min_overlap_ratio)
            self._diff_yolo_service.min_overlap_pixels = int(min_overlap_pixels)
        return self._diff_yolo_service

    def run(
        self,
        scene: str,
        current: str,
        update_base: Optional[bool] = None,
        output_dir: Optional[str] = None,
        write_base_current: bool = True,
        write_mask_image: bool = True,
        min_overlap_ratio: float = 0.15,
        min_overlap_pixels: int = 100,
        run_yolo_when_no_change: bool = False,
    ) -> Dict:
        """Run MobileSAM change segmentation, then map polygon labels with CLIP."""
        _ = (
            min_overlap_ratio,
            min_overlap_pixels,
            run_yolo_when_no_change,
        )
        return self.run_clip(
            scene=scene,
            current=current,
            output_dir=output_dir,
            update_base=update_base,
            write_base_current=write_base_current,
            write_mask_image=write_mask_image,
        )

    def run_edge(
        self,
        scene: str,
        current: str,
        update_base: Optional[bool] = None,
        output_dir: Optional[str] = None,
        write_base_current: bool = True,
        write_mask_image: bool = True,
    ) -> Dict:
        """Run edge change detection with scene + current image path."""
        return self._get_cd_service().process(
            scene=scene,
            current_path=current,
            update_base=update_base,
            output_dir=output_dir,
            write_base_current=write_base_current,
            write_mask_image=write_mask_image,
        )

    def run_vegetation(
        self,
        scene: str,
        current: str,
        output_dir: Optional[str] = None,
        update_base: Optional[bool] = None,
        write_base_current: bool = True,
        save_debug_images: bool = True,
    ) -> Dict:
        """Run vegetation coverage change detection with scene + current image path."""
        return self._get_veg_service().process(
            scene=scene,
            current_path=current,
            update_base=update_base,
            output_dir=output_dir,
            write_base_current=write_base_current,
            save_debug_images=save_debug_images,
        )

    def run_clip(
        self,
        scene: str,
        current: str,
        output_dir: Optional[str] = None,
        update_base: Optional[bool] = None,
        write_base_current: bool = True,
        write_mask_image: bool = True,
    ) -> Dict:
        """Write the four edge files, then replace polygon labels with CLIP mappings."""
        cd_service = self._get_cd_service()
        result = cd_service.process(
            scene=scene,
            current_path=current,
            update_base=False,
            output_dir=output_dir,
            write_base_current=write_base_current,
            write_mask_image=write_mask_image,
        )
        clip_result = self._get_clip_service().process_output(
            str(result["output_dir"])
        )
        result["clip_classification"] = clip_result
        result["json_shapes"] = int(clip_result["num_accepted_shapes"])
        result["change_pixels"] = int(
            clip_result["accepted_change_pixels"]
        )
        result["change_ratio"] = float(
            clip_result["accepted_change_ratio"]
        )
        result["change"] = int(result["json_shapes"] > 0)
        should_update = (
            bool(cd_service.config.update_base_on_change)
            if update_base is None
            else bool(update_base)
        )
        do_update = should_update
        result["base_update_policy"] = "always_after_detection"
        result["base_update_recommended"] = int(should_update)
        result["base_updated"] = int(do_update)
        if do_update:
            scene_dir = cd_service.data_root / scene
            base_after = cd_service._update_scene_base(
                scene_dir,
                Path(current).resolve(),
            )
            result["base_after"] = str(base_after)
        return result

    def run_diff_yolo(
        self,
        scene: str,
        current: str,
        output_dir: Optional[str] = None,
        update_base: Optional[bool] = None,
        min_overlap_ratio: float = 0.15,
        min_overlap_pixels: int = 100,
        run_yolo_when_no_change: bool = False,
    ) -> Dict:
        """Run edge first, then write YOLO boxes that overlap red mask regions into mask.json."""
        return self._get_diff_yolo_service(
            min_overlap_ratio=min_overlap_ratio,
            min_overlap_pixels=min_overlap_pixels,
        ).process(
            scene=scene,
            current_path=current,
            output_dir=output_dir,
            update_base=update_base,
            run_yolo_when_no_change=run_yolo_when_no_change,
        )
