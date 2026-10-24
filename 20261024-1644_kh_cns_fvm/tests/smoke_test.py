#!/usr/bin/env python3
"""CTest smoke test: short Kelvin-Helmholtz run with AMR.

Checks that mass, momentum and energy are conserved to round-off (periodic box
+ refluxing), density and pressure stay positive, refinement happened and a
plotfile was written.

usage: smoke_test.py <kh_cns executable> <kh_test.inputs> <work dir>
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    exe, inputs, work = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cmd = [str(exe), str(inputs), "stop_time=0.05", "amr.plot_int=-1", "amr.v=0"]
    res = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    (work / "log.txt").write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")

    rows = list(csv.DictReader(open(work / "diagnostics.csv")))
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
    plts = sorted(work.glob("plt*"))
    if not plts or not (plts[-1] / "Header").exists():
        sys.exit("FAILED: no plotfile written")
    print(f"KH run OK: {len(rows) - 1} coarse steps, finest level {last['finest_level']}, "
          f"plotfile {plts[-1].name}")
    print("PASSED")


if __name__ == "__main__":
    main()
