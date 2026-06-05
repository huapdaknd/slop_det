import argparse
import json
from pathlib import Path

from src import SlopDetService


DEFAULT_SCENE = ""
DEFAULT_CURRENT = ""
DEFAULT_OUTPUT_DIR = ""


class SlopDetApp:
    """Programmatic launcher for edge change detection with YOLO overlap classification."""

    def __init__(
        self,
        scene: str,
        current: str,
        config_path: str | None = None,
        yolo_config_path: str | None = None,
    ) -> None:
        self.scene = scene
        self.current = current
        self.config_path = config_path
        self.yolo_config_path = yolo_config_path
        self.service = SlopDetService(
            cd_config_path=config_path,
            yolo_config_path=yolo_config_path,
        )

    def run(
        self,
        output_dir: str | None = None,
        update_base: bool | None = None,
        min_overlap_ratio: float = 0.15,
        min_overlap_pixels: int = 100,
    ) -> dict:
        return self.service.run_diff_yolo(
            scene=self.scene,
            current=self.current,
            update_base=update_base,
            output_dir=output_dir,
            min_overlap_ratio=min_overlap_ratio,
            min_overlap_pixels=min_overlap_pixels,
        )


def _discover_sample() -> tuple[str, str] | None:
    current_root = Path("current_data")
    if not current_root.exists():
        return None
    for scene_dir in sorted(p for p in current_root.iterdir() if p.is_dir()):
        for image_path in sorted(p for p in scene_dir.iterdir() if p.is_file()):
            return scene_dir.name, str(image_path)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run edge change detection, then write YOLO boxes that overlap red mask regions into mask.json."
    )
    parser.add_argument("--scene", help="Scene folder name under data_root.")
    parser.add_argument("--current", help="Path to the current image.")
    parser.add_argument("--output-dir", help="Optional output directory.")
    parser.add_argument("--config", help="Optional path to config/model_config.json.")
    parser.add_argument("--yolo-config", help="Optional path to config/model_config_yolo.json.")
    parser.add_argument("--min-overlap-ratio", type=float, default=0.15)
    parser.add_argument("--min-overlap-pixels", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = args.scene or DEFAULT_SCENE
    current = args.current or DEFAULT_CURRENT
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR or None

    if not scene or not current:
        sample = _discover_sample()
        if sample is None:
            raise SystemExit("No sample found under current_data. Pass --scene and --current explicitly.")
        scene, current = sample

    app = SlopDetApp(
        scene=scene,
        current=current,
        config_path=args.config,
        yolo_config_path=args.yolo_config,
    )
    result = app.run(
        output_dir=output_dir,
        min_overlap_ratio=args.min_overlap_ratio,
        min_overlap_pixels=args.min_overlap_pixels,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
