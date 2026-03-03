"""
Usage:
    python demo_external_masks.py --video demo/V24_7.avi --masks demo_masks.npz

The .npz file must contain an array named "masks" with shape (T, H, W).
"""

import argparse
import os
from pathlib import Path
import logging

import numpy as np

from flatsam.config import Config, load_config
from flatsam.flatsam import flatsam_track
from flatsam.utils.geom import H_warp
import flatsam.utils.geom as gu
import flatsam.utils.vis as vu
from flatsam.utils.io import GeneralVideoCapture, VideoWriter


logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run WOFTSAM using per-frame external masks from an .npz file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--video', type=Path, required=True, help='Path to the input video.')
    parser.add_argument('--masks', type=Path, required=True, help='Path to .npz file with array "masks" of shape (T, H, W).')
    parser.add_argument('--config', type=Path, help='Optional path to tracker config file.')
    parser.add_argument('--output', type=Path, default=Path('demo_external_masks_out.mp4'), help='Path to output video.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging.')
    parser.add_argument('--gpu', help='CUDA device')
    args = parser.parse_args()

    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


def load_frames(video_path):
    frames = []
    cap = GeneralVideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    return frames


def get_init_coords(mask):
    bbox = gu.mask2bbox(mask > 0)
    return np.asarray(bbox.as_points(), dtype=np.float32).T


def run(args):
    if args.config is not None:
        conf = load_config(args.config)
    else:
        conf = Config()
        conf.track_function = False
        conf.hough_lines = Config()
        conf.hough_lines.enabled = True
        conf.hough_lines.intersection_corners = False
        conf.resolve_symmetry = Config()
        conf.resolve_symmetry.enabled = False
        conf.resolve_symmetry.template_update = False
        conf.name = 'demo_external_masks'

    masks_data = np.load(args.masks)
    if 'masks' not in masks_data:
        raise ValueError(f'Expected array "masks" in {args.masks}.')
    external_masks = masks_data['masks']

    frames = load_frames(str(args.video))
    if not frames:
        raise ValueError(f'No frames loaded from {args.video}.')

    init_coords = get_init_coords(external_masks[0])
    seq_name = args.video.stem

    track_function = conf.track_function if conf.track_function else flatsam_track
    video_writer = VideoWriter(str(args.output))
    all_corners = []

    for frame_i, sam_mask, _, info in track_function(
        None,
        conf,
        frames,
        init_coords,
        seq_name,
        external_masks=external_masks,
    ):
        frame = frames[frame_i]
        try:
            if 'output_H' in info:
                H_init2current = info['output_H']
            else:
                H_init2current = np.linalg.inv(info['output_H2init'])
            current_corners = H_warp(H_init2current, init_coords)
        except Exception:
            current_corners = all_corners[-1].copy()

        vis = vu.draw_corners(vu.to_gray_3ch(frame), current_corners, vu.RED)
        vis = vu.blend_mask(vis, sam_mask > 0, alpha=0.2)
        video_writer.write(vis)
        all_corners.append(current_corners.copy())

    video_writer.close()
    logger.info(f'Wrote {args.output}')
    return 0


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    main()
