#ifndef CNS_PROB_H_
#define CNS_PROB_H_

// Initial conditions:
//  type 0 - 2D Riemann problem (four quadrant states, see cns_prob_parm.H)
//  type 1 - smooth density wave rho = 1 + 0.2 sin(2 pi (x+y)), u = v = 1, p = 1
//           on a periodic box (exact solution = shifted profile), used to
//           measure the order of accuracy of WENO7 + SSPRK(10,4).
// Each cell receives the average of 4x4 sub-samples, so quadrant boundaries
// that cut a cell (e.g. x0 = 0.8 on odd grids) are represented conservatively.

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <cmath>
#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "cns_prob_parm.H"

AMREX_GPU_HOST_DEVICE inline
void rp_primitive (amrex::Real x, amrex::Real y, ProbParm const& pp, amrex::Real W[4]) noexcept
{
    using amrex::Real;
    if (pp.type == 1) {
        const Real pi = Real(3.14159265358979323846);
        W[0] = Real(1.0) + Real(0.2)*std::sin(Real(2.0)*pi*(x + y));
        W[1] = Real(1.0); W[2] = Real(1.0); W[3] = Real(1.0);
        return;
    }
    int q;
    if (y >= pp.y0) { q = (x >= pp.x0) ? 0 : 1; }
    else            { q = (x >= pp.x0) ? 3 : 2; }
    W[0] = pp.rho[q]; W[1] = pp.u[q]; W[2] = pp.v[q]; W[3] = pp.p[q];
}

// Exact density of the smooth wave at time t (type 1).
AMREX_GPU_HOST_DEVICE inline
amrex::Real density_wave_exact (amrex::Real x, amrex::Real y, amrex::Real t) noexcept
{
    const amrex::Real pi = amrex::Real(3.14159265358979323846);
    return amrex::Real(1.0) + amrex::Real(0.2)*std::sin(amrex::Real(2.0)*pi*(x + y - amrex::Real(2.0)*t));
}

AMREX_GPU_DEVICE
inline
void
cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& pp)
{
    using amrex::Real;
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx      = geomdata.CellSize();

    // Smooth data: exact cell averages from 6-point Gauss quadrature;
    // discontinuous data: 4x4 sub-cell sampling.
    Real rho = 0, mx = 0, my = 0, E = 0;
    if (pp.type == 1) {
        const Real gx[6] = {Real(-0.9324695142031521), Real(-0.6612093864662645), Real(-0.2386191860831969),
                            Real( 0.2386191860831969), Real( 0.6612093864662645), Real( 0.9324695142031521)};
        const Real gw[6] = {Real(0.1713244923791704), Real(0.3607615730481386), Real(0.4679139345726910),
                            Real(0.4679139345726910), Real(0.3607615730481386), Real(0.1713244923791704)};
        for (int b = 0; b < 6; ++b) {
            const Real y = prob_lo[1] + (j + Real(0.5) + Real(0.5)*gx[b])*dx[1];
            for (int a = 0; a < 6; ++a) {
                const Real x = prob_lo[0] + (i + Real(0.5) + Real(0.5)*gx[a])*dx[0];
                Real W[4];
                rp_primitive(x, y, pp, W);
                const Real w = Real(0.25)*gw[a]*gw[b];
                rho += w*W[0]; mx += w*W[0]*W[1]; my += w*W[0]*W[2];
                E   += w*(W[3]/(parm.eos_gamma - Real(1.0)) + Real(0.5)*W[0]*(W[1]*W[1] + W[2]*W[2]));
            }
        }
    } else {
        // Count the sub-samples in each quadrant, then combine the quadrant
        // states as f0*U0 + f2*U2 + (f1*U1 + f3*U3). This grouping makes the
        // data bitwise symmetric about x = y when x0 = y0 (the UL and LR
        // states are mirror images), so the solution keeps that symmetry.
        constexpr int ns = 4;
        int cnt[4] = {0, 0, 0, 0};
        for (int sy = 0; sy < ns; ++sy) {
            const Real y = prob_lo[1] + (j + (sy + Real(0.5))/ns)*dx[1];
            for (int sx = 0; sx < ns; ++sx) {
                const Real x = prob_lo[0] + (i + (sx + Real(0.5))/ns)*dx[0];
                if (y >= pp.y0) { ++cnt[(x >= pp.x0) ? 0 : 1]; }
                else            { ++cnt[(x >= pp.x0) ? 3 : 2]; }
            }
        }
        Real U[4][4];
        for (int q = 0; q < 4; ++q) {
            const Real r = pp.rho[q], u = pp.u[q], v = pp.v[q];
            U[q][0] = r; U[q][1] = r*u; U[q][2] = r*v;
            U[q][3] = pp.p[q]/(parm.eos_gamma - Real(1.0)) + Real(0.5)*r*(u*u + v*v);
        }
        Real cons[4];
        for (int n = 0; n < 4; ++n) {
            const Real f0 = cnt[0]*U[0][n], f1 = cnt[1]*U[1][n];
            const Real f2 = cnt[2]*U[2][n], f3 = cnt[3]*U[3][n];
            cons[n] = (f0 + f2 + (f1 + f3))/Real(ns*ns);
        }
        rho = cons[0]; mx = cons[1]; my = cons[2]; E = cons[3];
    }

    const Real ei = E - Real(0.5)*(mx*mx + my*my)/rho;
    state(i,j,k,URHO ) = rho;
    state(i,j,k,UMX  ) = mx;
    state(i,j,k,UMY  ) = my;
    state(i,j,k,UEDEN) = E;
    state(i,j,k,UEINT) = ei;
    state(i,j,k,UTEMP) = ei/(rho*parm.cv);
}

#endif
