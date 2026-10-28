#!/usr/bin/env python3
"""Turn a series of PNG images into a video (MP4 via ffmpeg, or animated GIF).

Frames are taken in name order, so pltNNNNN_*.png come out in time order.
All frames are resized to the size of the first one.

    python tools/make_video.py "plt*_density.png" -o density.mp4 --fps 10
    python tools/make_video.py "plt*_schlieren.png" -o schlieren.gif --fps 8
    python tools/make_video.py frames/ -o movie.mp4      # every PNG in a folder

A typical workflow in a run folder:
    python3 ../../tools/plot_riemann.py --all
    python3 ../../tools/make_video.py "plt*_density.png" -o density.mp4
"""
import argparse
import glob
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def collect(source):
    p = Path(source)
    files = sorted(p.glob("*.png")) if p.is_dir() else sorted(Path(f) for f in glob.glob(source))
    if not files:
        sys.exit(f"no PNG files match '{source}'")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help='glob pattern in quotes, e.g. "plt*_density.png", or a folder')
    ap.add_argument("-o", "--output", type=Path, default=Path("movie.mp4"),
                    help="output .mp4 (needs ffmpeg) or .gif (default movie.mp4)")
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second (default 10)")
    ap.add_argument("--hold", type=int, default=1, help="repeat the last frame this many times")
    args = ap.parse_args()

    files = collect(args.source)
    first = Image.open(files[0]).convert("RGB")
    w, h = first.size
    w, h = w - w % 2, h - h % 2                  # H.264 needs even dimensions
    frames = [Image.open(f).convert("RGB").resize((w, h), Image.LANCZOS) for f in files]
    frames += [frames[-1]] * (args.hold - 1)
    print(f"{len(files)} images, {w} x {h}, {args.fps:g} fps")

    if args.output.suffix.lower() == ".gif":
        frames[0].save(args.output, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0)
    else:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            sys.exit("ffmpeg not found: install it (e.g. sudo apt install ffmpeg) or use -o movie.gif")
        with tempfile.TemporaryDirectory() as tmp:
            for i, fr in enumerate(frames):
                fr.save(Path(tmp) / f"f{i:05d}.png")
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{args.fps:g}",
                   "-i", str(Path(tmp) / "f%05d.png"), "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", "18", str(args.output)]
            subprocess.run(cmd, check=True)
    print(args.output)


if __name__ == "__main__":
    main()
