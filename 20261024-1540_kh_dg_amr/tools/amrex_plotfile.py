"""Minimal reader for 2D AMReX plotfiles (native "HyperCLaw-V1.1" format).

Only NumPy is required. Each level is returned as a list of patches with the
cell index box and the data of every variable, which is enough to build a
composite image or draw the AMR grid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Patch:
    lo: tuple[int, int]
    hi: tuple[int, int]
    data: dict[str, np.ndarray]          # var -> array[ny, nx]


@dataclass
class Level:
    dx: tuple[float, float]
    domain_lo: tuple[int, int]
    domain_hi: tuple[int, int]
    patches: list[Patch] = field(default_factory=list)


@dataclass
class Plotfile:
    path: Path
    variables: list[str]
    time: float
    prob_lo: tuple[float, float]
    prob_hi: tuple[float, float]
    ref_ratio: list[int]
    levels: list[Level]


_BOX = re.compile(r"\(\((-?\d+),(-?\d+)\)\s*\((-?\d+),(-?\d+)\)")


def _read_fab(path: Path, offset: int, nvar: int):
    with open(path, "rb") as f:
        f.seek(offset)
        header = f.readline().decode()
        m = re.search(r"\(\((-?\d+),(-?\d+)\)\s*\((-?\d+),(-?\d+)\)\s*\((\d+),(\d+)\)\)\s*(\d+)", header)
        if m is None:
            raise ValueError(f"cannot parse FAB header: {header!r}")
        lo = (int(m.group(1)), int(m.group(2)))
        hi = (int(m.group(3)), int(m.group(4)))
        ncomp = int(m.group(7))
        # Header: FAB ((8, (<real format>)),(8, (<byte order>)))... ;
        # byte order "8 7 6 5 4 3 2 1" means little-endian.
        orders = re.findall(r"\(8, \(([\d ]+)\)\)", header)
        little = len(orders) < 2 or orders[1].split()[0] == "8"
        nx, ny = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1
        dt = np.dtype("<f8" if little else ">f8")
        count = nx * ny * ncomp
        raw = np.frombuffer(f.read(count * 8), dtype=dt, count=count)
    return lo, hi, raw.reshape(ncomp, ny, nx)


def read(path: str | Path, variables: list[str] | None = None) -> Plotfile:
    path = Path(path)
    lines = (path / "Header").read_text().splitlines()
    it = iter(lines)
    next(it)                                  # version string
    nvar = int(next(it))
    names = [next(it).strip() for _ in range(nvar)]
    dim = int(next(it))
    if dim != 2:
        raise ValueError("only 2D plotfiles are supported")
    time = float(next(it))
    finest = int(next(it))
    prob_lo = tuple(float(v) for v in next(it).split())
    prob_hi = tuple(float(v) for v in next(it).split())
    ref_ratio = [int(v) for v in next(it).split()] if finest > 0 else []
    if finest == 0:
        next(it)                              # empty ref-ratio line
    domains = _BOX.findall(next(it))
    next(it)                                  # level steps
    dxs = [tuple(float(v) for v in next(it).split()) for _ in range(finest + 1)]
    next(it)                                  # coord sys
    next(it)                                  # bwidth

    want = names if variables is None else [v for v in variables if v in names]
    idx = {v: names.index(v) for v in want}

    levels = []
    for lev in range(finest + 1):
        hdr = next(it).split()
        ngrids = int(hdr[1])
        next(it)                              # level step
        for _ in range(ngrids * dim):
            next(it)                          # physical extents of each grid
        rel = next(it).strip()                # e.g. Level_0/Cell
        d = domains[lev]
        level = Level(dx=dxs[lev], domain_lo=(int(d[0]), int(d[1])), domain_hi=(int(d[2]), int(d[3])))
        cell_h = (path / (rel + "_H")).read_text().splitlines()
        for line in cell_h:
            if line.startswith("FabOnDisk:"):
                _, fname, off = line.split()
                lo, hi, arr = _read_fab(path / Path(rel).parent / fname, int(off), nvar)
                level.patches.append(Patch(lo, hi, {v: arr[i] for v, i in idx.items()}))
        levels.append(level)

    return Plotfile(path, names, time, prob_lo, prob_hi, ref_ratio, levels)


def composite(pf: Plotfile, var: str, max_level: int | None = None) -> np.ndarray:
    """Uniform image at the resolution of `max_level` (default: finest level).

    Coarse cells are replicated (no interpolation) and overwritten by finer data.
    """
    top = len(pf.levels) - 1 if max_level is None else min(max_level, len(pf.levels) - 1)
    lv = pf.levels[top]
    nx = lv.domain_hi[0] - lv.domain_lo[0] + 1
    ny = lv.domain_hi[1] - lv.domain_lo[1] + 1
    img = np.full((ny, nx), np.nan)
    factor = 1
    factors = [1] * (top + 1)
    for lev in range(top - 1, -1, -1):
        factor *= pf.ref_ratio[lev]
        factors[lev] = factor
    for lev in range(top + 1):
        f = factors[lev]
        for p in pf.levels[lev].patches:
            block = np.repeat(np.repeat(p.data[var], f, axis=0), f, axis=1)
            img[p.lo[1] * f:(p.hi[1] + 1) * f, p.lo[0] * f:(p.hi[0] + 1) * f] = block
    return img
