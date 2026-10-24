# kh_dg_amr — Kelvin-Helmholtz instability with 4th-order DG and AMR (AMReX)

A self-contained 2D simulation of the Kelvin-Helmholtz instability. The
compressible Euler equations are solved with a **4th-order modal
discontinuous Galerkin (DG)** method on an **adaptive mesh** (AMReX
`AmrLevel`, CNS-style code layout). The production setup starts from a
64² base grid and refines by six levels of factor 2 down to an effective
resolution of **4096²** cells. The quantity shown is the (surface) density of
the fluid.

Why DG? Each cell stores not only its average but also the higher moments of
the solution (here the 16 Legendre modes of a bicubic polynomial per variable).
These moments are evolved directly, so no reconstruction step is needed. This
gives inherent conservation, much lower advection and diffusion errors than a
finite-volume scheme of the same cost, and a compact stencil (only face
neighbours) with lots of local work, which suits massively parallel machines.

## Build

Requirements: a C++17 compiler, CMake ≥ 3.20, Python 3 (tests/plots), and an
AMReX checkout. The AMReX commit used for development is pinned below.

```bash
git clone https://github.com/AMReX-Codes/amrex.git
git -C amrex checkout 2399452da8054029a92c5309a4bccb3102c109fd
git clone https://github.com/briandontloseyourheart-maker/instability.git
cd instability/20261024-1540_kh_dg_amr
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAMREX_SOURCE_DIR=$PWD/../../amrex
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

An already installed 2D AMReX (built with `AMReX_AMRLEVEL=ON`) can be used
instead: `-DAMReX_DIR=<prefix>/lib/cmake/AMReX`.

Compile-time options:

| CMake option | default | meaning |
|---|---|---|
| `DG_ORDER` | 4 | order of accuracy = polynomial degree + 1 (1 … 8) |
| `DG_NQUAD` | `DG_ORDER` | Gauss points per direction for volume/face integrals |
| `KH_USE_OMP` | ON | OpenMP threading (AMReX source build only) |
| `KH_NATIVE` | OFF | add `-march=native` (≈20% faster, not portable) |

## Run

```bash
mkdir -p runs/test && cd runs/test
OMP_NUM_THREADS=4 ../../build/bin/kh_dg ../../inputs/kh_test.inputs
python3 ../../tools/plot_kh.py plt00050 --zoom 0.5 0.75 0.1 --grid
```

| inputs file | grid | purpose |
|---|---|---|
| `inputs/kh_test.inputs` | 64² + 2 levels (256² effective), t ≤ 0.2 | quick check, a few minutes on a laptop |
| `inputs/kh_4096.inputs` | 64² + 6 levels (**4096² effective**), t ≤ 3 | the full simulation (needs a workstation/cluster and hours) |

Any parameter can be overridden on the command line, e.g.
`amr.max_level=4 stop_time=1.0`.

Output in the run directory:

* `pltNNNNN/` — AMReX plotfiles with `density`, `x_velocity`, `y_velocity`,
  `pressure`, `vorticity` and `dg_sensor` (cell averages; readable by
  ParaView, VisIt, yt, or `tools/plot_kh.py`). Set `amr.plot_vars = ALL` to
  also store every DG mode.
* `chkNNNNN/` — checkpoints (full modal state; restart with `amr.restart=chkNNNNN`).
* `diagnostics.csv` — per coarse step: total mass, momentum and energy,
  min/max density, minimum pressure, number of cells on each level.

`tools/plot_kh.py` draws the density with a blue-to-yellow colormap; with
`--zoom XC YC W --grid` it adds a zoomed panel with the AMR cells outlined.

## Physical setup

Doubly periodic unit square, ideal gas with γ = 5/3:

* 0.25 < y < 0.75: ρ = 2, vₓ = +0.5; elsewhere ρ = 1, vₓ = −0.5; p = 2.5;
* the interfaces are smoothed with a tanh profile of width `prob.delta`;
* seed: v_y = w₀ sin(4πx) [exp(−(y−0.25)²/2σ²) + exp(−(y−0.75)²/2σ²)],
  w₀ = 0.1, σ = 0.05/√2, plus optional multi-mode noise (`prob.noise_*`)
  that seeds the small secondary billows.

## Numerical method

* **Basis**: tensor-product Legendre polynomials φₖ(ξ)φₗ(η), k,l ≤ 3, scaled
  so the mass matrix is the identity and mode (0,0) is the cell average
  (`src/dg/DG_Basis.*`).
* **Spatial operator**: volume integral ∫F·∇φ with Gauss quadrature and
  surface integrals with the HLLC Riemann solver at the face Gauss points
  (`src/dg/DG_Kernels.H`, `src/dg/DG_Euler.H`).
* **Time stepping**: SSP-RK3 (default) or classical RK4 (`dg.rk_order`),
  Δt = CFL / ((2p+1) Σ_d (|u_d|+c)/Δx_d).
* **Stabilisation** (both leave the cell average untouched, so conservation is exact):
  a troubled-cell modal filter driven by the Persson–Peraire smoothness
  sensor, and the Zhang–Shu positivity limiter for density and pressure.
* **AMR** (`src/amr/`): Berger–Oliger subcycling in time, regridding on the
  intra-cell density variation. Coarse→fine and fine→coarse transfers are
  exact L2 projections of the polynomials (`src/dg/DG_Interp.*`).
  Flux registers correct **all** DG moments of coarse cells next to a fine
  patch, using the fine fluxes integrated against the coarse test functions,
  so mass, momentum and energy are conserved to round-off across levels.

Verification (`ctest`): advecting a smooth density wave, the L2 error drops by
a factor ≈ 16 per grid doubling (observed order ≈ 4.0); the KH run with AMR
conserves mass, momentum and energy to ~1e-15 relative and stays positive.

## Code layout

```text
kh_dg_amr/
├── CMakeLists.txt
├── inputs/            kh_test.inputs, kh_4096.inputs
├── src/
│   ├── main.cpp       driver (amrex::Amr time loop)
│   ├── dg/            basis & quadrature, Euler physics, DG kernels, AMR projections
│   ├── amr/           DGLevel (AmrLevel subclass): setup, time step, tagging, reflux, output
│   └── problem/       Kelvin-Helmholtz initial condition and prob.* parameters
├── tools/             plotfile reader and plotting script
└── tests/             CTest smoke test (order of accuracy + conservation)
```

## Main run-time parameters

| parameter | default | meaning |
|---|---|---|
| `dg.cfl` | 0.6 | CFL number (scaled by 1/(2p+1)) |
| `dg.rk_order` | 3 | 3 = SSP-RK3, 4 = classical RK4 |
| `dg.gamma` | 5/3 | adiabatic index |
| `dg.refine_dengrad` | 0.02 | refine where the density varies by more than this fraction across a cell |
| `dg.refine_vort` | off | refine where the velocity changes by more than this across a cell |
| `dg.refine_box_lo/hi` | — | optional fixed refinement box |
| `dg.do_filter`, `dg.filter_alpha`, `dg.sensor_s0`, `dg.sensor_kappa` | 1, 2, −1, 1 | troubled-cell filter |
| `dg.do_positivity` | 1 | positivity limiter |
| `prob.*` | see above | initial condition (`prob.type=1` = density-wave accuracy test) |

AMReX is distributed under its own license, see `LICENSE-AMReX`.
