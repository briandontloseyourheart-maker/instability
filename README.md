# instability

2D Kelvin-Helmholtz instability simulations with AMReX and adaptive mesh
refinement (64² base grid, refined to an effective 4096²). Each project is a
standalone CMake build in its own timestamped folder (UTC, `YYYYMMDD-HHMM`):

| folder | method |
|---|---|
| [`20261024-1540_kh_dg_amr`](20261024-1540_kh_dg_amr) | 4th-order discontinuous Galerkin (DG), AmrLevel, CNS-style layout |
| [`20261024-1644_kh_cns_fvm`](20261024-1644_kh_cns_fvm) | AMReX CNS module, 2nd-order finite volume (FVM) |

See the README inside each folder for build and run instructions.
