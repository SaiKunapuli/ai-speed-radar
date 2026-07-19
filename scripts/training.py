#!/usr/bin/env python3

from __future__ import annotations

"""
AI Speed Radar - YOLO Model Training
====================================

Trains a YOLO11 model on the converted UA-DETRAC dataset
with automatic GPU/device detection across platforms.

Usage:
    python scripts/training.py
    python scripts/training.py --epochs 200 --batch 32 --device cpu

Requirements:
    pip install ultralytics torch

Pre-requisite:
    python scripts/download_dataset.py
    python scripts/convert_to_yolo.py
"""

import argparse
import platform
import sys
from pathlib import Path

from ultralytics import YOLO


# ── Default Paths ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "datasets" / "vehicles" / "data.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"


# ═══════════════════════════════════════════════════════════════════════════
#  Device Auto-Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_device() -> str:
    """Auto-detect the best available device for YOLO training.

    Returns:
        Device string compatible with ultralytics (e.g. "0", "mps", "cpu").
    """
    try:
        import torch

        # NVIDIA CUDA
        if torch.cuda.is_available():
            return "0"

        # Apple MPS (macOS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

    except ImportError:
        pass

    return "cpu"


def describe_device(device: str) -> str:
    """Return a human-readable description of the selected device."""
    try:
        import torch
    except ImportError:
        return f"{device} (torch not installed)"

    if device == "cpu":
        return "CPU"
    elif device == "mps":
        return "Apple MPS (Metal)"
    elif device.isdigit():
        if torch.cuda.is_available():
            if getattr(torch.version, "hip", None):
                name = torch.cuda.get_device_name(int(device))
                return f"AMD ROCm - {name}"
            name = torch.cuda.get_device_name(int(device))
            return f"CUDA - {name}"
    return str(device)


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def train(
    data_yaml: Path,
    output_dir: Path,
    model_name: str = "yolo11m.pt",
    device: str | None = None,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 416,
    lr0: float = 0.001,
    warmup_epochs: int = 3,
    patience: int = 20,
    weight_decay: float = 0.0005,
    cos_lr: bool = True,
    close_mosaic: int = 10,
    mosaic: float = 1.0,
    fliplr: float = 0.55,
    mixup: float = 0.1,
    scale: float = 0.5,
    perspective: float = 0.00025,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    save_period: int = 50,
    val: bool = True,
    resume: bool = False,
) -> None:
    """Train a YOLO model with the given hyperparameters.

    All augmentation and schedule parameters are surfaced as keyword
    arguments so they can be tuned via CLI or programmatic use.
    """

    if device is None:
        device = detect_device()

    desc = describe_device(device)
    print("=" * 55)
    print("  AI SPEED RADAR - YOLO Training")
    print("=" * 55)
    print(f"  Platform:  {platform.system()} ({platform.machine()})")
    print(f"  Device:    {desc}")
    print(f"  Data:      {data_yaml}")
    print(f"  Model:     {model_name}")
    print(f"  Epochs:    {epochs}")
    print(f"  Batch:     {batch}")
    print(f"  Image sz:  {imgsz}")
    print(f"  Output:    {output_dir}")
    print("-" * 55)

    if not data_yaml.exists():
        print(f"\nERROR: data.yaml not found at {data_yaml}")
        print("  Run: python scripts/download_dataset.py")
        print("  then: python scripts/convert_to_yolo.py")
        sys.exit(1)

    model = YOLO(model_name)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        device=device,
        lr0=lr0,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        mosaic=mosaic,
        fliplr=fliplr,
        mixup=mixup,
        scale=scale,
        perspective=perspective,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        save_period=save_period,
        val=val,
        resume=resume,
        project=str(output_dir),
        name="exp",
        augment=True,
        verbose=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO for AI Speed Radar vehicle detection"
    )
    p.add_argument("--data", default=str(DEFAULT_DATA_YAML),
                   help="Path to data.yaml (default: datasets/vehicles/data.yaml)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                   help="Output directory for model checkpoints (default: models/)")
    p.add_argument("--model", default="yolo11m.pt",
                   help="YOLO model to start from (default: yolo11m.pt)")
    p.add_argument("--device", default=None,
                   help='Device override: "0" for GPU, "mps", "cpu"')
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--no-cos-lr", dest="cos_lr", action="store_false",
                   help="Disable cosine LR schedule")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from last checkpoint")
    p.add_argument("--cpu", action="store_true",
                   help="Shortcut for --device cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cpu" if args.cpu else args.device

    train(
        data_yaml=Path(args.data),
        output_dir=Path(args.output),
        model_name=args.model,
        device=device,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,
        resume=args.resume,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())