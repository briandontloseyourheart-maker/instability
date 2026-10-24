# instability

2D compressible-flow simulations with AMReX: Kelvin-Helmholtz instability
with adaptive mesh refinement (64² base grid, refined to an effective 4096²)
and a 2D Riemann problem with a high-order WENO scheme. Each project is a
standalone CMake build in its own timestamped folder (UTC, `YYYYMMDD-HHMM`):

| folder | method |
|---|---|
| [`20261024-1540_kh_dg_amr`](20261024-1540_kh_dg_amr) | 4th-order discontinuous Galerkin (DG), AmrLevel, CNS-style layout |
| [`20261024-1644_kh_cns_fvm`](20261024-1644_kh_cns_fvm) | AMReX CNS module, 2nd-order finite volume (FVM) |
| [`20261024-1802_riemann2d_weno7`](20261024-1802_riemann2d_weno7) | 2D Riemann problem (config 3), 2048², AMReX CNS + Roe flux, WENO7, SSPRK(10,4) |

See the README inside each folder for build and run instructions.
