#!/usr/bin/env python3
"""CTest smoke test for the 2D Riemann solver (WENO7 + Roe + SSPRK(10,4)).

1. Accuracy: advect a smooth density wave on 32^2 and 64^2 periodic grids and
   check that the L1 error of the cell averages drops at (at least) 4th order.
2. Riemann problem, configuration 3 on 128^2 up to t = 0.3: density and
   pressure stay positive, the solution is symmetric about the diagonal
   x = y (as the initial data are), and a plotfile is written.

usage: smoke_test.py <riemann2d executable> <rp2d_test.inputs> <work dir>
"""
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import amrex_plotfile as apf  # noqa: E402


def run(exe, inputs, workdir, *overrides):
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), str(inputs), *overrides]
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    (workdir / "log.txt").write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")


def last_plotfile(d):
    plts = sorted(p for p in d.glob("plt*") if (p / "Header").exists())
    if not plts:
        sys.exit(f"FAILED: no plotfile in {d}")
    return plts[-1]


def main():
    exe, inputs, work = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    if work.exists():
        shutil.rmtree(work)

    # 1) order of accuracy on a smooth periodic wave ------------------------
    t_end = 0.25
    errs = []
    for n in (32, 64):
        d = work / f"wave_{n}"
        run(exe, inputs, d, "prob.type=1", f"amr.n_cell={n} {n}", "amr.max_grid_size=32",
            "geometry.is_periodic=1 1", "cns.lo_bc=0 0", "cns.hi_bc=0 0",
            f"stop_time={t_end}", "cns.cfl=0.8", "amr.plot_per=-1", "amr.plot_int=-1", "amr.v=0")
        pf = apf.read(last_plotfile(d), ["density"])
        rho = apf.composite(pf, "density")
        h = 1.0 / n
        xc = (np.arange(n) + 0.5) * h
        X, Y = np.meshgrid(xc, xc)
        # exact cell average of 1 + 0.2 sin(2 pi (x + y - 2t))
        damp = (math.sin(math.pi * h) / (math.pi * h)) ** 2
        exact = 1.0 + 0.2 * damp * np.sin(2 * math.pi * (X + Y - 2 * pf.time))
        errs.append(np.abs(rho - exact).mean())
    order = math.log2(errs[0] / errs[1])
    print(f"density wave L1 errors {errs}, observed order {order:.2f}")
    if order < 3.8:
        sys.exit("FAILED: observed order below 3.8")

    # 2) configuration 3 ------------------------------------------------------
    d = work / "riemann"
    run(exe, inputs, d, "amr.n_cell=128 128", "amr.max_grid_size=32", "stop_time=0.3",
        "amr.plot_per=-1", "amr.plot_int=-1", "amr.v=0")
    rows = list(csv.DictReader(open(d / "diagnostics.csv")))
    if min(float(r["rho_min"]) for r in rows) <= 0 or min(float(r["p_min"]) for r in rows) <= 0:
        sys.exit("FAILED: non-positive density or pressure")
    pf = apf.read(last_plotfile(d), ["density"])
    rho = apf.composite(pf, "density")
    asym = np.abs(rho - rho.T).max() / rho.max()
    print(f"Riemann problem: t = {pf.time:.3f}, rho in [{rho.min():.4f}, {rho.max():.4f}], "
          f"diagonal asymmetry {asym:.2e}")
    if not np.isfinite(rho).all() or asym > 1e-8:
        sys.exit("FAILED: solution not finite or not symmetric about x = y")
    print("PASSED")


if __name__ == "__main__":
    main()
