#!/usr/bin/env python3
import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np
import cv2


def common_prefix(strings: list[str]) -> str:
    """Return common prefix for a list of strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        # shrink prefix until it matches
        while not s.startswith(prefix) and prefix:
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix


def sanitize_base_name(name: str) -> str:
    name = name.strip().strip("._- ")
    return name if name else "masks"


def infer_output_name(mask_paths: list[Path], mask_dir: Path) -> str:
    names = [p.stem for p in mask_paths]  # without extension
    pref = common_prefix(names)
    # Often filenames like: video_000001, video_000002 -> prefix "video_"
    # Remove trailing separators
    pref = sanitize_base_name(pref)
    if pref and pref != "masks":
        return pref + ".npz"
    # fallback to folder name
    return sanitize_base_name(mask_dir.name) + ".npz"


def main():
    ap = argparse.ArgumentParser(
        description="Convert a folder of PNG masks to a single NPZ file with key 'masks' (T,H,W)."
    )
    ap.add_argument(
        "mask_dir",
        help="Path to folder containing PNG mask images (one per frame).",
    )
    ap.add_argument(
        "-o", "--output",
        default=None,
        help="Output .npz path. If omitted, inferred from PNG filenames (common prefix) or folder name.",
    )
    ap.add_argument(
        "--pattern",
        default="*.png",
        help="Glob pattern for mask files inside mask_dir (default: *.png).",
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Binarization threshold (0..255). Pixel > threshold -> 1 else 0. Default: 127.",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Search for PNGs recursively (mask_dir/**/pattern).",
    )
    ap.add_argument(
        "--no-binarize",
        action="store_true",
        help="Do not binarize; store uint8 0..255 as-is (still saved as uint8).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing output.",
    )

    args = ap.parse_args()

    mask_dir = Path(args.mask_dir).expanduser().resolve()
    if not mask_dir.exists() or not mask_dir.is_dir():
        print(f"ERROR: mask_dir is not a directory: {mask_dir}", file=sys.stderr)
        sys.exit(1)

    if args.recursive:
        glob_pattern = str(mask_dir / "**" / args.pattern)
        paths = [Path(p) for p in glob.glob(glob_pattern, recursive=True)]
    else:
        glob_pattern = str(mask_dir / args.pattern)
        paths = [Path(p) for p in glob.glob(glob_pattern)]

    paths = sorted(paths)
    if not paths:
        print(f"ERROR: No files matched: {glob_pattern}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).expanduser().resolve() if args.output else (mask_dir / infer_output_name(paths, mask_dir))
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(".npz")

    # Read first mask to get shape
    first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        print(f"ERROR: Failed to read image: {paths[0]}", file=sys.stderr)
        sys.exit(1)
    H, W = first.shape[:2]

    masks = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"ERROR: Failed to read image: {p}", file=sys.stderr)
            sys.exit(1)
        if im.shape[:2] != (H, W):
            print(
                f"ERROR: Size mismatch: {p} is {im.shape[:2]}, expected {(H, W)}. "
                "Resizing is intentionally not done here to avoid silent mistakes.",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.no_binarize:
            m = im.astype(np.uint8)
        else:
            m = (im > args.threshold).astype(np.uint8)

        masks.append(m)

    masks = np.stack(masks, axis=0)  # (T,H,W)
    if args.dry_run:
        print(f"DRY RUN")
        print(f"  mask_dir: {mask_dir}")
        print(f"  files: {len(paths)} (first: {paths[0].name})")
        print(f"  output: {out_path}")
        print(f"  masks shape: {masks.shape}, dtype: {masks.dtype}, min/max: {masks.min()}/{masks.max()}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), masks=masks)
    print(f"saved {out_path}")
    print(f"shape {masks.shape} dtype {masks.dtype} min/max {masks.min()}/{masks.max()}")


if __name__ == "__main__":
    main()
