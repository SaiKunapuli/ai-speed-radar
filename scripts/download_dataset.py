#!/usr/bin/env python3
"""
UA-DETRAC Dataset Downloader
============================

Downloads the UA-DETRAC dataset via kagglehub and prints a summary
of what was downloaded (sequence names, file counts, directory structure).

Usage:
    python scripts/download_dataset.py

Requirements:
    pip install kagglehub
"""

import os
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Kaggle dataset handle ──────────────────────────────────────────────────

KAGGLE_DATASET = "bratjay/ua-detrac-orig"

# DETRAC directory names inside the downloaded archive.
# This kaggle mirror nests directories one level deeper than the official release.
IMAGES_DIR = Path("DETRAC-Images") / "DETRAC-Images"          # All sequence dirs here
TRAIN_ANNO_DIR = Path("DETRAC-Train-Annotations-XML") / "DETRAC-Train-Annotations-XML"
TEST_ANNO_DIR = Path("DETRAC-Test-Annotations-XML") / "DETRAC-Test-Annotations-XML"


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Download
# ═══════════════════════════════════════════════════════════════════════════

def download_dataset() -> Path:
    """Download the UA-DETRAC dataset via kagglehub.

    Returns:
        Path to the cached dataset directory.
    """
    print("=" * 60)
    print("  UA-DETRAC Dataset Downloader")
    print(f"  Source: kagglehub -> {KAGGLE_DATASET}")
    print("=" * 60)

    try:
        import kagglehub
    except ImportError:
        print(
            "\nERROR: kagglehub is not installed.\n"
            "  Run: pip install kagglehub"
        )
        sys.exit(1)

    print("\nDownloading (this may take a while — ~4 GB)...")
    print("Subsequent runs will use the cached copy.\n")

    try:
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Authenticate: kagglehub.login()")
        print("     Or go to https://www.kaggle.com/settings/account")
        print("     Create an API token → place kaggle.json at ~/.kaggle/")
        print("  2. Check your internet connection")
        sys.exit(1)

    return Path(dataset_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Directory Listing
# ═══════════════════════════════════════════════════════════════════════════

def count_files(directory: Path, extensions: tuple = (".jpg", ".png", ".bmp")) -> int:
    """Count files with given extensions in a directory (recursive)."""
    if not directory.exists():
        return 0
    count = 0
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            count += 1
    return count


def list_subdirs(directory: Path) -> list:
    """List immediate subdirectories."""
    if not directory.exists():
        return []
    return sorted([d.name for d in directory.iterdir() if d.is_dir()])


def print_dataset_info(dataset_path: Path) -> None:
    """Print information about the downloaded dataset structure."""
    print("\n" + "=" * 60)
    print("  Dataset downloaded successfully!")
    print("=" * 60)

    print(f"\n  Location: {dataset_path}\n")

    # Check each expected directory
    sections = [
        ("Images (all sequences)", IMAGES_DIR, ".jpg"),
        ("Training annotations", TRAIN_ANNO_DIR, ".xml"),
        ("Test annotations", TEST_ANNO_DIR, ".xml"),
    ]

    for label, subdir, ext in sections:
        full_path = dataset_path / subdir
        if full_path.exists():
            subdirs = list_subdirs(full_path)
            file_count = count_files(full_path, (ext,))
            print(f"  {label}:")
            print(f"    {subdir}/  ->  {len(subdirs)} sequences, {file_count} {ext} files")
            if subdirs:
                sample = subdirs[:3]
                print(f"    Sample sequences: {', '.join(sample)}")
                if len(subdirs) > 3:
                    print(f"    ... and {len(subdirs) - 3} more")
        else:
            print(f"  {label}: NOT FOUND (expected {subdir}/)")

    print("\n" + "=" * 60)
    print("  Ready for conversion to YOLO format.")
    print("  Run: python scripts/convert_to_yolo.py")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    """Download UA-DETRAC and print summary. Returns 0 on success."""
    dataset_path = download_dataset()
    print_dataset_info(dataset_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
