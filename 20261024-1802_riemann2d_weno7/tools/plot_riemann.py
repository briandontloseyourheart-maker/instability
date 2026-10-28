#!/usr/bin/env python3
"""Plot a 2D Riemann problem plotfile: filled density map plus density contours.

    python tools/plot_riemann.py          # newest plotfile in the current directory
    python tools/plot_riemann.py plt00480
    python tools/plot_riemann.py plt00480 --var pressure --cmap viridis --contours 0
    python tools/plot_riemann.py plt00480 --zoom 0.0 0.5 0.0 0.5 -o zoom.png
    python tools/plot_riemann.py --all    # every plotfile here, same colour range (video frames)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amrex_plotfile as apf  # noqa: E402


def render(path, args, vmin, vmax, out):
    pf = apf.read(path, [args.var])
    img = apf.composite(pf, args.var)
    ny, nx = img.shape
    vmin = vmin if vmin is not None else float(np.nanmin(img))
    vmax = vmax if vmax is not None else float(np.nanmax(img))
    x0, y0 = pf.prob_lo
    x1, y1 = pf.prob_hi

    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(img, origin="lower", extent=(x0, x1, y0, y1), cmap=args.cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    if args.contours > 0:
        xc = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        yc = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        ax.contour(xc, yc, img, levels=np.linspace(vmin, vmax, args.contours + 2)[1:-1],
                   colors="k", linewidths=0.35, alpha=0.6)
    if args.zoom:
        ax.set_xlim(args.zoom[0], args.zoom[1])
        ax.set_ylim(args.zoom[2], args.zoom[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(f"{args.var}   t = {pf.time:.3f}   {nx} x {ny}")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotfile", type=Path, nargs="?", default=Path("."),
                    help="plotfile or run directory (default: newest plotfile in the current directory)")
    ap.add_argument("--all", action="store_true",
                    help="plot every plotfile in the run directory with one shared colour range")
    ap.add_argument("--var", default="density")
    ap.add_argument("--cmap", default="RdBu_r", help="matplotlib colormap (default RdBu_r)")
    ap.add_argument("--contours", type=int, default=30, help="number of contour lines (0 = none)")
    ap.add_argument("--zoom", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("-o", "--output", type=Path, help="output PNG (single plotfile only)")
    args = ap.parse_args()

    if not args.all:
        path = apf.resolve(args.plotfile)
        out = args.output or path.with_name(path.name + f"_{args.var}.png")
        render(path, args, args.vmin, args.vmax, out)
        return

    paths = apf.all_plotfiles(args.plotfile)
    vmin, vmax = args.vmin, args.vmax
    if vmin is None or vmax is None:     # shared range so a video does not flicker
        lo, hi = np.inf, -np.inf
        for p in paths:
            img = apf.composite(apf.read(p, [args.var]), args.var)
            lo, hi = min(lo, float(np.nanmin(img))), max(hi, float(np.nanmax(img)))
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    for p in paths:
        render(p, args, vmin, vmax, p.with_name(p.name + f"_{args.var}.png"))


if __name__ == "__main__":
    main()
