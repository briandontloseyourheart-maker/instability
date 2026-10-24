#!/usr/bin/env python3
"""CTest smoke test.

1. Accuracy: advect a smooth density wave on 16^2 and 32^2 grids and check
   that the error drops by roughly 2^4 (4th-order DG).
2. Kelvin-Helmholtz with AMR: run the small test case for a short time and
   check that mass, momentum and energy are conserved to round-off, density
   and pressure stay positive, refinement happened, and a plotfile was written.

usage: smoke_test.py <kh_dg executable> <kh_test.inputs> <work dir>
"""
import csv
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(exe, inputs, workdir, *overrides):
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), str(inputs), *overrides]
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    (workdir / "log.txt").write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return res.stdout


def main():
    exe, inputs, work = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    if work.exists():
        shutil.rmtree(work)

    # 1) order of accuracy ------------------------------------------------
    errs = []
    for n in (16, 32):
        out = run(exe, inputs, work / f"wave_{n}", "prob.type=1", f"amr.n_cell={n} {n}",
                  "amr.max_level=0", "amr.blocking_factor=4", "stop_time=0.25",
                  "dg.rk_order=4", "dg.cfl=0.3", "dg.do_filter=0",
                  "amr.plot_files_output=0", "amr.plot_int=-1", "amr.v=0", "dg.v=0")
        m = re.search(r"density-wave L2 error = ([0-9.eE+-]+)", out)
        errs.append(float(m.group(1)))
    order = math.log2(errs[0] / errs[1])
    print(f"density wave L2 errors {errs}, observed order {order:.2f}")
    if order < 3.5:
        sys.exit("FAILED: observed order of accuracy below 3.5")

    # 2) KH with AMR -------------------------------------------------------
    kh = work / "kh"
    run(exe, inputs, kh, "stop_time=0.05", "amr.plot_int=-1", "amr.plot_per=-1", "amr.v=0")
    rows = list(csv.DictReader(open(kh / "diagnostics.csv")))
    first, last = rows[0], rows[-1]
    for key in ("mass", "x_momentum", "energy"):
        a, b = float(first[key]), float(last[key])
        rel = abs(b - a) / abs(a)
        print(f"{key}: {a:.15g} -> {b:.15g} (rel. change {rel:.2e})")
        if rel > 1e-11:
            sys.exit(f"FAILED: {key} not conserved")
    if abs(float(last["y_momentum"])) > 1e-11:
        sys.exit("FAILED: y-momentum drifted from zero")
    if min(float(r["rho_min"]) for r in rows) <= 0 or min(float(r["p_min"]) for r in rows) <= 0:
        sys.exit("FAILED: non-positive density or pressure")
    if int(last["finest_level"]) < 1:
        sys.exit("FAILED: no refinement happened")
    plts = sorted(kh.glob("plt*"))
    if not plts or not (plts[-1] / "Header").exists():
        sys.exit("FAILED: no plotfile written")
    print(f"KH run OK: {len(rows) - 1} coarse steps, finest level {last['finest_level']}, "
          f"plotfile {plts[-1].name}")
    print("PASSED")


if __name__ == "__main__":
    main()
