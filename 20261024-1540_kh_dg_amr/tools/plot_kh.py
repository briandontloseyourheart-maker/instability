#!/usr/bin/env python3
"""Render the surface density of a Kelvin-Helmholtz plotfile.

Left panel: the whole box. Right panel (optional): a zoom on a small window
with the outlines of the AMR cells drawn on top, like the reference figures.

    python tools/plot_kh.py plt00400
    python tools/plot_kh.py plt00400 --zoom 0.52 0.70 0.10 --grid
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amrex_plotfile as apf  # noqa: E402

# Dark blue -> cyan -> yellow, similar to MATLAB's "parula".
PARULA = LinearSegmentedColormap.from_list("parula_like", [
    (0.00, "#1a1a5e"), (0.12, "#2c3593"), (0.28, "#1569d6"), (0.42, "#0b96d6"),
    (0.55, "#22b5b2"), (0.68, "#7cc46c"), (0.80, "#d6b93a"), (0.90, "#fbbf2c"),
    (1.00, "#f9f40f")])


def draw_zoom_grid(ax, pf, window):
    """Outline every leaf cell (cells not covered by a finer level) in the window."""
    x0, x1, y0, y1 = window
    xlo, ylo = pf.prob_lo
    segs = set()
    nlev = len(pf.levels)
    for lev, level in enumerate(pf.levels):
        dx, dy = level.dx
        r = pf.ref_ratio[lev] if lev < nlev - 1 else 1
        fine = pf.levels[lev + 1].patches if lev < nlev - 1 else []
        i0 = int(np.floor((x0 - xlo) / dx)); i1 = int(np.ceil((x1 - xlo) / dx))
        j0 = int(np.floor((y0 - ylo) / dy)); j1 = int(np.ceil((y1 - ylo) / dy))
        for p in level.patches:
            for j in range(max(j0, p.lo[1]), min(j1, p.hi[1] + 1)):
                for i in range(max(i0, p.lo[0]), min(i1, p.hi[0] + 1)):
                    covered = any(fp.lo[0] <= i * r <= fp.hi[0] and fp.lo[1] <= j * r <= fp.hi[1]
                                  for fp in fine)
                    if covered:
                        continue
                    xa, xb = xlo + i * dx, xlo + (i + 1) * dx
                    ya, yb = ylo + j * dy, ylo + (j + 1) * dy
                    segs.update({((xa, ya), (xb, ya)), ((xa, yb), (xb, yb)),
                                 ((xa, ya), (xa, yb)), ((xb, ya), (xb, yb))})
    ax.add_collection(LineCollection(list(segs), colors="k", linewidths=0.3, alpha=0.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotfile", type=Path, nargs="?", default=Path("."),
                    help="plotfile or run directory (default: newest plotfile in the current directory)")
    ap.add_argument("--var", default="density")
    ap.add_argument("--zoom", nargs=3, type=float, metavar=("XC", "YC", "W"),
                    help="centre and width of a zoom window")
    ap.add_argument("--grid", action="store_true", help="draw AMR cells in the zoom")
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()
    args.plotfile = apf.resolve(args.plotfile)

    pf = apf.read(args.plotfile, [args.var])
    img = apf.composite(pf, args.var)
    vmin = args.vmin if args.vmin is not None else np.nanmin(img)
    vmax = args.vmax if args.vmax is not None else np.nanmax(img)
    extent = (pf.prob_lo[0], pf.prob_hi[0], pf.prob_lo[1], pf.prob_hi[1])

    ncols = 2 if args.zoom else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(img, origin="lower", extent=extent, cmap=PARULA, vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_axis_off()
    nlev = len(pf.levels)
    eff = pf.levels[-1].domain_hi[0] + 1
    ax.set_title(f"{args.var}   t = {pf.time:.3f}   levels = {nlev}   effective {eff}$^2$", fontsize=10)

    if args.zoom:
        xc, yc, w = args.zoom
        win = (xc - w / 2, xc + w / 2, yc - w / 2, yc + w / 2)
        ax.add_patch(Rectangle((win[0], win[2]), w, w, fill=False, ec="white", lw=1.2))
        az = axes[0, 1]
        az.imshow(img, origin="lower", extent=extent, cmap=PARULA, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        if args.grid:
            draw_zoom_grid(az, pf, win)
        az.set_xlim(win[0], win[1])
        az.set_ylim(win[2], win[3])
        az.set_axis_off()

    fig.tight_layout()
    out = args.output or args.plotfile.with_name(args.plotfile.name.rstrip("/") + f"_{args.var}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
