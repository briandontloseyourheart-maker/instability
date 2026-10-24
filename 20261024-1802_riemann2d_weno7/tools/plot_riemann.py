#!/usr/bin/env python3
"""Plot a 2D Riemann problem plotfile: filled density map plus density contours.

    python tools/plot_riemann.py plt00480
    python tools/plot_riemann.py plt00480 --var pressure --cmap viridis --contours 0
    python tools/plot_riemann.py plt00480 --zoom 0.0 0.5 0.0 0.5 -o zoom.png
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotfile", type=Path)
    ap.add_argument("--var", default="density")
    ap.add_argument("--cmap", default="RdBu_r", help="matplotlib colormap (default RdBu_r)")
    ap.add_argument("--contours", type=int, default=30, help="number of contour lines (0 = none)")
    ap.add_argument("--zoom", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()

    pf = apf.read(args.plotfile, [args.var])
    img = apf.composite(pf, args.var)
    ny, nx = img.shape
    vmin = args.vmin if args.vmin is not None else float(np.nanmin(img))
    vmax = args.vmax if args.vmax is not None else float(np.nanmax(img))
    x0, y0 = pf.prob_lo
    x1, y1 = pf.prob_hi
    extent = (x0, x1, y0, y1)

    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(img, origin="lower", extent=extent, cmap=args.cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
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
    out = args.output or args.plotfile.with_name(args.plotfile.name.rstrip("/") + f"_{args.var}.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
