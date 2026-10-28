#!/usr/bin/env python3
"""Numerical schlieren and synthetic interferogram images of a plotfile.

Schlieren (density gradient magnitude, Quirk 1994):
    S = exp(-k |grad rho| / max|grad rho|)
    Shocks, contact lines and vortices appear as dark lines on white.

Interferogram (Mach-Zehnder, line-of-sight phase proportional to density):
    I = cos^2( pi (rho - rho_min) / d_rho ),   d_rho = (rho_max - rho_min) / N
    "infinite fringe" mode: each dark fringe is a line of constant density.
    With --carrier n the reference beam is tilted, adding n straight fringes
    across the box ("finite fringe" mode), which the flow then bends.

    python tools/plot_optics.py                      # newest plotfile: schlieren + interferogram
    python tools/plot_optics.py plt00726 --mode schlieren -k 20
    python tools/plot_optics.py plt00726 --mode interferogram --fringes 30
    python tools/plot_optics.py plt00726 --mode interferogram --carrier 40
    python tools/plot_optics.py --all --mode schlieren   # every plotfile (video frames)
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


def schlieren(rho, dx, dy, k):
    gy, gx = np.gradient(rho, dy, dx)
    g = np.hypot(gx, gy)
    return np.exp(-k * g / max(g.max(), 1e-300))


def interferogram(rho, x, y, rmin, rmax, fringes, carrier, angle):
    phase = np.pi * (rho - rmin) / ((rmax - rmin) / fringes)
    if carrier > 0:      # tilted reference beam: carrier fringes across the unit length
        X, Y = np.meshgrid(x, y)
        th = np.deg2rad(angle)
        phase = phase + np.pi * carrier * (X * np.cos(th) + Y * np.sin(th)) / (x[-1] - x[0])
    return np.cos(phase) ** 2


def render(path, args, rmin, rmax, out):
    pf = apf.read(path, ["density"])
    rho = apf.composite(pf, "density")
    ny, nx = rho.shape
    x0, y0 = pf.prob_lo
    x1, y1 = pf.prob_hi
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    x = x0 + (np.arange(nx) + 0.5) * dx
    y = y0 + (np.arange(ny) + 0.5) * dy
    rmin = float(rho.min()) if rmin is None else rmin
    rmax = float(rho.max()) if rmax is None else rmax

    panels = []
    if args.mode in ("schlieren", "both"):
        panels.append(("numerical schlieren", schlieren(rho, dx, dy, args.k)))
    if args.mode in ("interferogram", "both"):
        name = "interferogram" + (f" (carrier {args.carrier:g})" if args.carrier > 0 else "")
        panels.append((name, interferogram(rho, x, y, rmin, rmax, args.fringes,
                                           args.carrier, args.angle)))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.3), squeeze=False)
    for ax, (name, img) in zip(axes[0], panels):
        ax.imshow(img, origin="lower", extent=(x0, x1, y0, y1), cmap=args.cmap,
                  vmin=0, vmax=1, interpolation="bilinear")
        if args.zoom:
            ax.set_xlim(args.zoom[0], args.zoom[1])
            ax.set_ylim(args.zoom[2], args.zoom[3])
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{name}   t = {pf.time:.3f}   {nx} x {ny}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotfile", type=Path, nargs="?", default=Path("."),
                    help="plotfile or run directory (default: newest plotfile in the current directory)")
    ap.add_argument("--all", action="store_true", help="process every plotfile in the run directory")
    ap.add_argument("--mode", choices=["schlieren", "interferogram", "both"], default="both")
    ap.add_argument("-k", type=float, default=15.0, help="schlieren contrast (larger = more detail)")
    ap.add_argument("--fringes", type=float, default=20.0,
                    help="dark fringes across the density range (interferogram)")
    ap.add_argument("--carrier", type=float, default=0.0,
                    help="carrier fringes across the box; 0 = infinite-fringe mode")
    ap.add_argument("--angle", type=float, default=90.0,
                    help="carrier direction in degrees (90 = horizontal fringes)")
    ap.add_argument("--cmap", default="gray")
    ap.add_argument("--zoom", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("-o", "--output", type=Path, help="output PNG (single plotfile only)")
    args = ap.parse_args()

    if not args.all:
        path = apf.resolve(args.plotfile)
        out = args.output or path.with_name(path.name + f"_{args.mode}.png")
        render(path, args, None, None, out)
        return

    paths = apf.all_plotfiles(args.plotfile)
    # one density range for all frames, so fringes keep the same meaning in a video
    lo, hi = np.inf, -np.inf
    if args.mode != "schlieren":
        for p in paths:
            r = apf.composite(apf.read(p, ["density"]), "density")
            lo, hi = min(lo, float(r.min())), max(hi, float(r.max()))
    for p in paths:
        render(p, args, None if lo == np.inf else lo, None if hi == -np.inf else hi,
               p.with_name(p.name + f"_{args.mode}.png"))


if __name__ == "__main__":
    main()
