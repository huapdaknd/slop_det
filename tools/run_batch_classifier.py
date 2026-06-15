from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_example import SlopDetApp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V3A change segmentation and classifier labeling in scene order."
    )
    parser.add_argument("--current-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "model_config.json"),
    )
    parser.add_argument(
        "--classifier-config",
        default=str(PROJECT_ROOT / "config" / "classifier_label_config.json"),
    )
    parser.add_argument("--expected-images", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--delete-no-change-output", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def list_samples(current_root: Path) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    for scene_dir in sorted(path for path in current_root.iterdir() if path.is_dir()):
        for image_path in sorted(
            path
            for path in scene_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            samples.append((scene_dir.name, image_path))
    return samples


def read_completed(summary_path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not summary_path.exists():
        return completed
    for line in summary_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ok"):
            completed.add((str(row.get("scene", "")), str(row.get("image", ""))))
    return completed


def main() -> None:
    args = parse_args()
    current_root = Path(args.current_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.config).resolve()
    classifier_config_path = Path(args.classifier_config).resolve()
    if not current_root.is_dir():
        raise FileNotFoundError(f"current root not found: {current_root}")

    samples = list_samples(current_root)
    if args.expected_images and len(samples) != args.expected_images:
        raise RuntimeError(
            f"expected {args.expected_images} images, found {len(samples)}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_summary.jsonl"
    meta_path = output_root / "batch_meta.json"
    completed = read_completed(summary_path) if args.resume else set()

    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "current_root": str(current_root),
        "output_root": str(output_root),
        "config": str(config_path),
        "classifier_config": str(classifier_config_path),
        "image_count": len(samples),
        "resume": bool(args.resume),
        "completed_before_start": len(completed),
        "base_update_policy": "always_after_detection",
        "delete_no_change_output": bool(args.delete_no_change_output),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    first_scene, first_image = samples[0]
    app = SlopDetApp(
        scene=first_scene,
        current=str(first_image),
        config_path=str(config_path),
        classifier_config_path=str(classifier_config_path),
    )

    summary_mode = "a" if args.resume else "w"
    with summary_path.open(summary_mode, encoding="utf-8") as summary_file:
        for index, (scene, image_path) in enumerate(samples, start=1):
            if (scene, image_path.name) in completed:
                continue
            free_gb = shutil.disk_usage(output_root).free / (1024**3)
            if free_gb < args.min_free_gb:
                raise RuntimeError(
                    f"free disk space {free_gb:.2f} GB is below "
                    f"{args.min_free_gb:.2f} GB"
                )

            item_output = output_root / f"{scene}_images" / image_path.stem
            row = {
                "index": index,
                "scene": scene,
                "image": image_path.name,
                "image_path": str(image_path),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                app.scene = scene
                app.current = str(image_path)
                result = app.run(
                    output_dir=str(item_output),
                    update_base=None,
                )
                row["ok"] = True
                row["result"] = result
                if (
                    args.delete_no_change_output
                    and int(result.get("change", 0)) != 1
                    and item_output.is_dir()
                    and output_root in item_output.parents
                ):
                    shutil.rmtree(item_output)
            except Exception as exc:
                row["ok"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
            row["finished_at"] = datetime.now().isoformat(timespec="seconds")
            summary_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary_file.flush()
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(samples),
                        "scene": scene,
                        "image": image_path.name,
                        "ok": row["ok"],
                        "change": int(
                            (row.get("result") or {}).get("change", 0)
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
