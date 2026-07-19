#!/usr/bin/env python3
"""
UA-DETRAC → YOLO Converter
==========================

Converts the downloaded UA-DETRAC dataset XML annotations to YOLO format,
organizing images and labels into train/val/test splits matching data.yaml.

Usage:
    python scripts/convert_to_yolo.py                     # default: 50 train, 10 val
    python scripts/convert_to_yolo.py --split 40 20       # custom split
    python scripts/convert_to_yolo.py --dataset-path ./custom-path/
    python scripts/convert_to_yolo.py --no-copy-images    # labels only, skip copying images

Output:
    datasets/vehicles/
    ├── train/images/   (.jpg copies)
    ├── train/labels/   (.txt, YOLO normalized format)
    ├── val/images/
    ├── val/labels/
    └── test/images/    (only if test annotations exist)
        test/labels/

Vehicle type mapping (UA-DETRAC → data.yaml):
    car    → 0 (car)
    bus    → 2 (bus)
    van    → 4 (van)
    others → skipped

Requirements:
    pip install opencv-python tqdm
    (opencv optional — falls back to default 960x540 image dims)
"""

import argparse
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "vehicles"


def _find_dataset_path() -> Path:
    """Auto-discover the cached UA-DETRAC dataset path via kagglehub or glob."""
    # Try kagglehub first (most reliable)
    try:
        import kagglehub
        return Path(kagglehub.dataset_download("bratjay/ua-detrac-orig"))
    except Exception:
        pass

    # Fallback: find the latest version in the cache
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "bratjay" / "ua-detrac-orig"
    if cache_root.exists():
        versions = sorted(cache_root.glob("versions/*"), reverse=True)
        if versions:
            return versions[0]

    return cache_root / "versions" / "2"  # last-resort fallback


DEFAULT_DATASET_PATH = _find_dataset_path()

# ── Kaggle mirror directory layout ─────────────────────────────────────────

# Nested structure in this mirror: DETRAC-Images/DETRAC-Images/MVI_XXXXX/
IMAGES_DIR = Path("DETRAC-Images") / "DETRAC-Images"
TRAIN_ANNO_DIR = Path("DETRAC-Train-Annotations-XML") / "DETRAC-Train-Annotations-XML"
TEST_ANNO_DIR = Path("DETRAC-Test-Annotations-XML") / "DETRAC-Test-Annotations-XML"

# ── Vehicle type mapping ──────────────────────────────────────────────────

# UA-DETRAC only annotates car, bus, van ("others" skipped).
DETRAC_TO_YOLO = {"car": 0, "bus": 1, "van": 2}
# Derive class names from the mapping so they stay in sync.
CLASS_NAMES = [name for name, _ in sorted(DETRAC_TO_YOLO.items(), key=lambda x: x[1])]

# Standard UA-DETRAC resolution
DEFAULT_IMG_SIZE = (960, 540)

# ── Stats tracking ─────────────────────────────────────────────────────────

class Stats:
    """Tracks conversion statistics."""
    def __init__(self):
        self.frames = 0
        self.vehicles = 0
        self.skipped_others = 0
        self.by_class: Dict[int, int] = {}


# ═══════════════════════════════════════════════════════════════════════════
#  XML Parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_detrac_xml(xml_path: Path) -> Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]]:
    """Parse a UA-DETRAC XML file.

    Args:
        xml_path: Path to MVI_XXXXX.xml.

    Returns:
        {frame_num: [(vehicle_type, (left, top, width, height)), ...]}
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    annotations: Dict[int, List[Tuple[str, Tuple[float, float, float, float]]]] = {}

    for frame_elem in root.findall("frame"):
        frame_num = int(frame_elem.get("num", "1"))

        target_list = frame_elem.find("target_list")
        if target_list is None:
            continue

        frame_annos = []
        for target in target_list.findall("target"):
            box = target.find("box")
            attr = target.find("attribute")
            if box is None or attr is None:
                continue

            left = float(box.get("left", "0"))
            top = float(box.get("top", "0"))
            width = float(box.get("width", "0"))
            height = float(box.get("height", "0"))
            vehicle_type = attr.get("vehicle_type", "others")

            frame_annos.append((vehicle_type, (left, top, width, height)))

        annotations[frame_num] = frame_annos

    return annotations


# ═══════════════════════════════════════════════════════════════════════════
#  YOLO Conversion
# ═══════════════════════════════════════════════════════════════════════════

def bbox_to_yolo(
    left: float, top: float, width: float, height: float,
    img_w: int, img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert pixel (left, top, width, height) to YOLO normalized format.

    Returns:
        (x_center, y_center, width, height) all normalized to [0, 1].
    """
    x_center = (left + width / 2.0) / img_w
    y_center = (top + height / 2.0) / img_h
    w_norm = width / img_w
    h_norm = height / img_h

    # Clamp
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, w_norm)),
        max(0.0, min(1.0, h_norm)),
    )


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Get image width and height. Falls back to DEFAULT_IMG_SIZE."""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            return (img.shape[1], img.shape[0])
    except ImportError:
        pass
    except Exception:
        pass
    return DEFAULT_IMG_SIZE


def convert_sequence(
    seq_name: str,
    images_root: Path,
    xml_path: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    copy_images: bool,
    stats: Stats,
) -> int:
    """Convert a single sequence to YOLO format.

    Returns:
        Number of frames processed in this sequence.
    """
    seq_img_dir = images_root / seq_name
    if not seq_img_dir.exists():
        print(f"  WARNING: {seq_name} image dir not found, skipping")
        return 0

    seq_frames = 0

    # Parse XML
    try:
        annotations = parse_detrac_xml(xml_path)
    except Exception as e:
        print(f"  ERROR parsing {xml_path.name}: {e}")
        return 0

    # Get sorted images
    img_files = sorted(
        [f for f in seq_img_dir.iterdir() if f.suffix.lower() == ".jpg"]
    )
    if not img_files:
        print(f"  WARNING: No .jpg files in {seq_img_dir}")
        return 0

    # Get dimensions from first image
    img_w, img_h = get_image_dimensions(img_files[0])

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    for img_file in img_files:
        stem = img_file.stem  # "img00001"
        try:
            frame_num = int("".join(ch for ch in stem if ch.isdigit()))
        except ValueError:
            continue

        unique_name = f"{seq_name}_{stem}"  # "MVI_20011_img00001"

        # Get annotations for this frame
        frame_annos = annotations.get(frame_num, [])
        yolo_lines = []

        for vehicle_type, (left, top, w, h) in frame_annos:
            yolo_class = DETRAC_TO_YOLO.get(vehicle_type)
            if yolo_class is None:
                stats.skipped_others += 1
                continue

            xc, yc, wn, hn = bbox_to_yolo(left, top, w, h, img_w, img_h)
            yolo_lines.append(f"{yolo_class} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

            stats.by_class[yolo_class] = stats.by_class.get(yolo_class, 0) + 1
            stats.vehicles += 1

        # Write label file
        with open(output_label_dir / f"{unique_name}.txt", "w") as f:
            f.writelines(yolo_lines)

        # Copy image
        if copy_images:
            dest = output_img_dir / f"{unique_name}.jpg"
            if not dest.exists():
                shutil.copy2(img_file, dest)

        stats.frames += 1
        seq_frames += 1

    return seq_frames


# ═══════════════════════════════════════════════════════════════════════════
#  Sequence Discovery & Splitting
# ═══════════════════════════════════════════════════════════════════════════

def discover_sequences(dataset_path: Path) -> Tuple[List[str], List[str]]:
    """Discover train and test sequence names from XML files."""
    train_seqs = []
    test_seqs = []

    train_dir = dataset_path / TRAIN_ANNO_DIR
    if train_dir.exists():
        train_seqs = sorted([f.stem for f in train_dir.glob("*.xml")])

    test_dir = dataset_path / TEST_ANNO_DIR
    if test_dir.exists():
        test_seqs = sorted([f.stem for f in test_dir.glob("*.xml")])

    return train_seqs, test_seqs


def split_train_val(train_seqs: List[str], num_val: int, seed: int) -> Tuple[List[str], List[str]]:
    """Randomly split training sequences into train and val."""
    random.seed(seed)
    shuffled = list(train_seqs)
    random.shuffle(shuffled)
    return sorted(shuffled[num_val:]), sorted(shuffled[:num_val])


# ═══════════════════════════════════════════════════════════════════════════
#  data.yaml Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_data_yaml(output_dir: Path, class_names: List[str]) -> Path:
    """Write a YOLO data.yaml to the output directory.

    Returns:
        Path to the generated data.yaml.
    """
    yaml_path = output_dir / "data.yaml"

    lines = [
        "# AI Speed Radar - YOLO Dataset Configuration",
        "# Auto-generated by convert_to_yolo.py",
        "",
        f"path: {output_dir.as_posix()}",
        "train: train/images",
        "val: val/images",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return yaml_path


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert UA-DETRAC to YOLO format")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH),
                        help="Path to downloaded UA-DETRAC dataset")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for YOLO dataset")
    parser.add_argument("--split", "-s", nargs=2, type=int, default=[50, 10],
                        metavar=("N_TRAIN", "N_VAL"),
                        help="Number of train/val sequences (default: 50 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--no-copy-images", action="store_true",
                        help="Only generate labels, don't copy images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without running")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("  Run scripts/download_dataset.py first, or use --dataset-path")
        return 1

    # ── Discover ───────────────────────────────────────────────────────
    train_seqs, test_seqs = discover_sequences(dataset_path)
    if not train_seqs:
        print(f"ERROR: No XML files found in {dataset_path / TRAIN_ANNO_DIR}")
        return 1

    print(f"Found: {len(train_seqs)} train sequences, {len(test_seqs)} test sequences")

    train_final, val_final = split_train_val(train_seqs, args.split[1], args.seed)
    train_final = train_final[:args.split[0]]

    if args.split[0] + args.split[1] > len(train_seqs):
        print(f"WARNING: Requested {args.split[0]}+{args.split[1]} sequences "
              f"but only {len(train_seqs)} available")

    print(f"Split:  {len(train_final)} train, {len(val_final)} val, {len(test_seqs)} test")

    if not test_seqs:
        print("Note: Test annotations unavailable (withheld for benchmark) — skipping test split")

    if args.dry_run:
        print("\n[Dry run — no conversion]")
        print(f"  Train: {train_final[:3]}... ({len(train_final)} total)")
        print(f"  Val:   {val_final[:3]}... ({len(val_final)} total)")
        return 0

    # ── Convert ────────────────────────────────────────────────────────
    images_root = dataset_path / IMAGES_DIR
    copy_images = not args.no_copy_images

    splits = [
        ("train", train_final, TRAIN_ANNO_DIR),
        ("val", val_final, TRAIN_ANNO_DIR),
        ("test", test_seqs, TEST_ANNO_DIR),
    ]

    all_stats: Dict[str, Stats] = {}

    for split_name, sequences, anno_subdir in splits:
        if not sequences:
            continue

        stat = Stats()
        print(f"\n{'='*50}\n  {split_name.upper()} ({len(sequences)} sequences)\n{'='*50}")

        anno_root = dataset_path / anno_subdir
        out_img = output_dir / split_name / "images"
        out_lbl = output_dir / split_name / "labels"

        for i, seq in enumerate(sequences):
            xml_path = anno_root / f"{seq}.xml"
            if not xml_path.exists():
                print(f"  SKIP: {seq}.xml not found")
                continue

            seq_frames = convert_sequence(
                seq, images_root, xml_path, out_img, out_lbl, copy_images, stat
            )
            # Print per-sequence stats (suppress after first 6 train sequences)
            if split_name != "train" or i < 6:
                print(f"  {seq}: {seq_frames} frames, {stat.vehicles} vehicles cumul.")

        all_stats[split_name] = stat

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  CONVERSION COMPLETE")
    print(f"{'='*55}")
    total_f = total_v = 0
    for name, s in all_stats.items():
        print(f"  {name:>6}: {s.frames:>6} frames, {s.vehicles:>6} vehicles "
              f"(skipped {s.skipped_others} 'others')")
        for cls_id, count in sorted(s.by_class.items()):
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "?"
            print(f"          class {cls_id} ({name}): {count}")
        total_f += s.frames
        total_v += s.vehicles
    print(f"  {'TOTAL':>6}: {total_f:>6} frames, {total_v:>6} vehicles")
    print(f"\n  Output: {output_dir}")

    # ── Generate data.yaml ────────────────────────────────────────────
    yaml_path = generate_data_yaml(output_dir, CLASS_NAMES)
    print(f"  Data config: {yaml_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
