#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

// 2D Riemann problem: four constant states in the quadrants around (x0, y0).
// Defaults are configuration 3 of Lax & Liu (1998) / Schulz-Rinne et al.
// (1993): four shocks, the classic "mushroom" jet along the diagonal.
// Quadrant index: 0 = upper right, 1 = upper left, 2 = lower left, 3 = lower right.
struct ProbParm
{
    int type = 0;                 // 0: 2D Riemann problem, 1: smooth density wave (accuracy test)
    amrex::Real x0 = 0.8;
    amrex::Real y0 = 0.8;
    amrex::Real rho[4] = {1.5,   0.5323, 0.138, 0.5323};
    amrex::Real u[4]   = {0.0,   1.206,  1.206, 0.0};
    amrex::Real v[4]   = {0.0,   0.0,    1.206, 1.206};
    amrex::Real p[4]   = {1.5,   0.3,    0.029, 0.3};
};

#endif
