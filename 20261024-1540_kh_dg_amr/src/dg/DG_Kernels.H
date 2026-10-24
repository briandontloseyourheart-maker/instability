#ifndef DG_KERNELS_H_
#define DG_KERNELS_H_

// Cell- and face-local DG kernels. Everything here works on one cell or one
// face at a time and only touches the immediate neighbour, which is what makes
// DG compact and well suited to massively parallel machines.

#include "DG_Basis.H"
#include "DG_Euler.H"

#include <AMReX_Array4.H>
#include <cmath>

namespace dg {

using amrex::Array4;

// Load all modes of cell (i,j): a[v][ky][kx].
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void load_cell (Array4<Real const> const& s, int i, int j, Real a[NVAR][NP][NP]) noexcept
{
    for (int v = 0; v < NVAR; ++v) {
        for (int ky = 0; ky < NP; ++ky) {
            for (int kx = 0; kx < NP; ++kx) {
                a[v][ky][kx] = s(i,j,0,comp(v,kx,ky));
            }
        }
    }
}

// Evaluate the state at a reference point given phi_kx(xi) and phi_ky(eta).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_point (const Real a[NVAR][NP][NP], const Real px[NP], const Real py[NP], Real U[NVAR]) noexcept
{
    for (int v = 0; v < NVAR; ++v) {
        Real sum = 0.0;
        for (int ky = 0; ky < NP; ++ky) {
            Real t = 0.0;
            for (int kx = 0; kx < NP; ++kx) { t += a[v][ky][kx]*px[kx]; }
            sum += t*py[ky];
        }
        U[v] = sum;
    }
}

// Sum-factorised evaluation at all NQ x NQ volume Gauss points: U[v][qy][qx].
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_volume (const Real a[NVAR][NP][NP], const Basis& B, Real U[NVAR][NQ][NQ]) noexcept
{
    for (int v = 0; v < NVAR; ++v) {
        Real t[NP][NQ];   // t[ky][qx] = sum_kx a phi_kx(x_qx)
        for (int ky = 0; ky < NP; ++ky) {
            for (int qx = 0; qx < NQ; ++qx) {
                Real sum = 0.0;
                for (int kx = 0; kx < NP; ++kx) { sum += a[v][ky][kx]*B.phi[qx][kx]; }
                t[ky][qx] = sum;
            }
        }
        for (int qy = 0; qy < NQ; ++qy) {
            for (int qx = 0; qx < NQ; ++qx) {
                Real sum = 0.0;
                for (int ky = 0; ky < NP; ++ky) { sum += t[ky][qx]*B.phi[qy][ky]; }
                U[v][qy][qx] = sum;
            }
        }
    }
}

// 1D polynomial on a cell face: c[v][k] are the coefficients along the face.
// dir = 0: face x = const with phi_kx(side) folded in (c indexed by ky);
// dir = 1: face y = const (c indexed by kx).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void face_trace (const Real a[NVAR][NP][NP], int dir, const Real side[NP], Real c[NVAR][NP]) noexcept
{
    for (int v = 0; v < NVAR; ++v) {
        for (int k = 0; k < NP; ++k) {
            Real sum = 0.0;
            for (int m = 0; m < NP; ++m) {
                sum += (dir == 0) ? a[v][k][m]*side[m] : a[v][m][k]*side[m];
            }
            c[v][k] = sum;
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eval_trace (const Real c[NVAR][NP], const Real phi[NP], Real U[NVAR]) noexcept
{
    for (int v = 0; v < NVAR; ++v) {
        Real sum = 0.0;
        for (int k = 0; k < NP; ++k) { sum += c[v][k]*phi[k]; }
        U[v] = sum;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int floor_div (int i, int r) noexcept { return (i >= 0) ? i/r : -((-i-1)/r) - 1; }

// ---------------------------------------------------------------------------
// Numerical flux through face (i,j) normal to dir (face between cells
// (i-1,j)/(i,j) for dir=0 and (i,j-1)/(i,j) for dir=1).
//
// Stores the face moments seen from the lower cell:
//   G(v,kx,ky) = phi_kn(+1) * < Fhat_v * phi_kt >_face
// where kn/kt are the normal/tangential mode indices and < >_face is the face
// average. The upper cell uses phi_kn(-1) = (-1)^kn phi_kn(+1).
//
// If gc is non-null (fine level with reflux), also store the same quantity but
// with the tangential test function of the *coarse* parent face, so that the
// sum over the r fine faces reproduces the coarse-face integral exactly.
// ---------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE inline
void face_flux (int i, int j, int dir,
                Array4<Real const> const& s, Array4<Real> const& g,
                Array4<Real> const& gc, bool do_gc, int ratio,
                const Basis& B, const Refine& R, const EulerParm& ep) noexcept
{
    Real aL[NVAR][NP][NP], aR[NVAR][NP][NP];
    if (dir == 0) { load_cell(s, i-1, j, aL); }
    else          { load_cell(s, i, j-1, aL); }
    load_cell(s, i, j, aR);

    Real H [NVAR][NP] = {};   // < Fhat phi_kt >
    Real Hc[NVAR][NP] = {};   // same with coarse tangential basis

    const int it = (dir == 0) ? j : i;  // tangential index
    const int sub = do_gc ? (it - ratio*floor_div(it, ratio)) : 0;

    Real cL[NVAR][NP], cR[NVAR][NP];
    face_trace(aL, dir, B.phiR, cL);   // upper face of the lower cell
    face_trace(aR, dir, B.phiL, cR);   // lower face of the upper cell

    for (int q = 0; q < NQ; ++q) {
        Real UL[NVAR], UR[NVAR];
        eval_trace(cL, B.phi[q], UL);
        eval_trace(cR, B.phi[q], UR);
        Real F[NVAR];
        hllc_flux(UL, UR, dir, ep, F);
        for (int v = 0; v < NVAR; ++v) {
            const Real wf = B.wq[q]*F[v];
            for (int k = 0; k < NP; ++k) {
                H[v][k] += wf*B.phi[q][k];
                if (do_gc) { Hc[v][k] += wf*R.phic[sub][q][k]; }
            }
        }
    }

    for (int v = 0; v < NVAR; ++v) {
        for (int kn = 0; kn < NP; ++kn) {
            for (int kt = 0; kt < NP; ++kt) {
                const int kx = (dir == 0) ? kn : kt;
                const int ky = (dir == 0) ? kt : kn;
                g(i,j,0,comp(v,kx,ky)) = B.phiR[kn]*H[v][kt];
                if (do_gc) { gc(i,j,0,comp(v,kx,ky)) = B.phiR[kn]*Hc[v][kt]; }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Right-hand side dU/dt for every mode of cell (i,j):
//   volume integral  < F . grad(phi_m) >  minus surface terms.
// ---------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE inline
void cell_rhs (int i, int j, Array4<Real const> const& s,
               Array4<Real const> const& gx, Array4<Real const> const& gy,
               Array4<Real> const& dudt, Real dxinv, Real dyinv,
               const Basis& B, const EulerParm& ep) noexcept
{
    Real a[NVAR][NP][NP];
    load_cell(s, i, j, a);

    // Volume term with sum factorisation.
    Real U[NVAR][NQ][NQ];
    eval_volume(a, B, U);
    Real Fx[NVAR][NQ][NQ], Fy[NVAR][NQ][NQ];
    for (int qy = 0; qy < NQ; ++qy) {
        for (int qx = 0; qx < NQ; ++qx) {
            Real u[NVAR], fx[NVAR], fy[NVAR];
            for (int v = 0; v < NVAR; ++v) { u[v] = U[v][qy][qx]; }
            euler_flux(u, 0, ep, fx);
            euler_flux(u, 1, ep, fy);
            const Real w = B.wq[qx]*B.wq[qy];
            for (int v = 0; v < NVAR; ++v) {
                Fx[v][qy][qx] = 2.0*dxinv*w*fx[v];
                Fy[v][qy][qx] = 2.0*dyinv*w*fy[v];
            }
        }
    }
    Real vol[NVAR][NP][NP];
    for (int v = 0; v < NVAR; ++v) {
        // X[qy][kx] = sum_qx Fx dphi_kx,  Y[qy][kx] = sum_qx Fy phi_kx
        Real X[NQ][NP], Y[NQ][NP];
        for (int qy = 0; qy < NQ; ++qy) {
            for (int kx = 0; kx < NP; ++kx) {
                Real sx = 0.0, sy = 0.0;
                for (int qx = 0; qx < NQ; ++qx) {
                    sx += Fx[v][qy][qx]*B.dphi[qx][kx];
                    sy += Fy[v][qy][qx]*B.phi[qx][kx];
                }
                X[qy][kx] = sx; Y[qy][kx] = sy;
            }
        }
        for (int ky = 0; ky < NP; ++ky) {
            for (int kx = 0; kx < NP; ++kx) {
                Real sum = 0.0;
                for (int qy = 0; qy < NQ; ++qy) {
                    sum += X[qy][kx]*B.phi[qy][ky] + Y[qy][kx]*B.dphi[qy][ky];
                }
                vol[v][ky][kx] = sum;
            }
        }
    }

    for (int v = 0; v < NVAR; ++v) {
        for (int ky = 0; ky < NP; ++ky) {
            const Real sy = (ky % 2 == 0) ? 1.0 : -1.0;
            for (int kx = 0; kx < NP; ++kx) {
                const Real sx = (kx % 2 == 0) ? 1.0 : -1.0;
                const int c = comp(v,kx,ky);
                dudt(i,j,0,c) = vol[v][ky][kx]
                    - dxinv*(gx(i+1,j,0,c) - sx*gx(i,j,0,c))
                    - dyinv*(gy(i,j+1,0,c) - sy*gy(i,j,0,c));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stabilisation, applied after every Runge-Kutta stage.
//
// 1) Troubled-cell modal filter (Persson & Peraire smoothness sensor on the
//    density): in cells whose highest modes carry too much energy, damp the
//    high modes with an exponential filter exp(-alpha*eps*k(k+1)/(P(P+1))).
//    This acts like a small, local artificial viscosity and leaves the cell
//    average (mode 0) untouched, so conservation is exact.
// 2) Zhang & Shu positivity limiter: scale the high modes towards the cell
//    average until density and pressure are positive at all quadrature and
//    face points. Also conservative.
// ---------------------------------------------------------------------------
struct LimiterParm
{
    int  do_filter     = 1;
    Real sensor_s0     = -1.0;   // log10 threshold offset (added to -4*log10(P))
    Real sensor_kappa  = 1.0;    // ramp width in decades
    Real filter_alpha  = 2.0;    // strength of the modal damping
    int  do_positivity = 1;
    Real eps_pos       = 1.0e-10;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real smoothness_eps (const Real a[NVAR][NP][NP], const LimiterParm& lp) noexcept
{
    if (P == 0) { return 0.0; }
    Real top = 0.0, all = 0.0;
    for (int ky = 0; ky < NP; ++ky) {
        for (int kx = 0; kx < NP; ++kx) {
            const Real e = a[RHO][ky][kx]*a[RHO][ky][kx];
            all += e;
            if (kx == P || ky == P) { top += e; }
        }
    }
    if (all <= 0.0 || top <= 0.0) { return 0.0; }
    const Real se = std::log10(top/all);
    const Real s0 = lp.sensor_s0 - 4.0*std::log10(static_cast<Real>(P));
    const Real k  = lp.sensor_kappa;
    if (se < s0 - k) { return 0.0; }
    if (se > s0 + k) { return 1.0; }
    return 0.5*(1.0 + std::sin(0.5*3.14159265358979323846*(se - s0)/k));
}

AMREX_GPU_HOST_DEVICE inline
void limit_cell (int i, int j, Array4<Real> const& s, const Basis& B,
                 const EulerParm& ep, const LimiterParm& lp) noexcept
{
    if (P == 0) { return; }
    Real a[NVAR][NP][NP];
    for (int v = 0; v < NVAR; ++v) {
        for (int ky = 0; ky < NP; ++ky) {
            for (int kx = 0; kx < NP; ++kx) { a[v][ky][kx] = s(i,j,0,comp(v,kx,ky)); }
        }
    }

    bool changed = false;

    if (lp.do_filter) {
        const Real eps = smoothness_eps(a, lp);
        if (eps > 0.0) {
            const Real norm = 1.0/static_cast<Real>(P*(P+1));
            for (int ky = 0; ky < NP; ++ky) {
                for (int kx = 0; kx < NP; ++kx) {
                    const Real f = std::exp(-lp.filter_alpha*eps*(kx*(kx+1) + ky*(ky+1))*norm);
                    for (int v = 0; v < NVAR; ++v) { a[v][ky][kx] *= f; }
                }
            }
            changed = true;
        }
    }

    if (lp.do_positivity) {
        // Check points: interior Gauss points and Gauss points on the 4 faces.
        constexpr int NPTS = NQ*NQ + 4*NQ;
        auto all_points = [&] (Real Up[NVAR][NPTS]) {
            Real Uv[NVAR][NQ][NQ];
            eval_volume(a, B, Uv);
            for (int v = 0; v < NVAR; ++v) {
                for (int q = 0; q < NQ*NQ; ++q) { Up[v][q] = Uv[v][q/NQ][q%NQ]; }
            }
            Real c[NVAR][NP], u[NVAR];
            const Real* sides[2] = {B.phiL, B.phiR};
            int n = NQ*NQ;
            for (int dir = 0; dir < 2; ++dir) {
                for (int sd = 0; sd < 2; ++sd) {
                    face_trace(a, dir, sides[sd], c);
                    for (int q = 0; q < NQ; ++q, ++n) {
                        eval_trace(c, B.phi[q], u);
                        for (int v = 0; v < NVAR; ++v) { Up[v][n] = u[v]; }
                    }
                }
            }
        };

        Real mean[NVAR];
        for (int v = 0; v < NVAR; ++v) { mean[v] = a[v][0][0]; }
        const Real eps_rho = amrex::min(lp.eps_pos, 0.5*mean[RHO]);
        const Real pmean   = pressure(mean, ep);
        const Real eps_p   = amrex::min(lp.eps_pos, 0.5*pmean);

        Real Up[NVAR][NPTS];
        all_points(Up);

        // (a) density
        Real rmin = mean[RHO];
        for (int n = 0; n < NPTS; ++n) { rmin = amrex::min(rmin, Up[RHO][n]); }
        if (rmin < eps_rho) {
            const Real th = amrex::min(1.0, (mean[RHO] - eps_rho)/(mean[RHO] - rmin));
            for (int ky = 0; ky < NP; ++ky) {
                for (int kx = 0; kx < NP; ++kx) {
                    if (kx + ky > 0) { a[RHO][ky][kx] *= th; }
                }
            }
            for (int n = 0; n < NPTS; ++n) { Up[RHO][n] = mean[RHO] + th*(Up[RHO][n] - mean[RHO]); }
            changed = true;
        }

        // (b) pressure: largest t in [0,1] with p(mean + t(U-mean)) >= eps_p
        Real theta = 1.0;
        for (int n = 0; n < NPTS; ++n) {
            Real U[NVAR];
            for (int v = 0; v < NVAR; ++v) { U[v] = Up[v][n]; }
            if (pressure(U, ep) < eps_p) {
                Real lo = 0.0, hi = 1.0;
                for (int it = 0; it < 30; ++it) {
                    const Real t = 0.5*(lo + hi);
                    Real W[NVAR];
                    for (int v = 0; v < NVAR; ++v) { W[v] = mean[v] + t*(U[v] - mean[v]); }
                    if (pressure(W, ep) >= eps_p) { lo = t; } else { hi = t; }
                }
                theta = amrex::min(theta, lo);
            }
        }
        if (theta < 1.0) {
            for (int v = 0; v < NVAR; ++v) {
                for (int ky = 0; ky < NP; ++ky) {
                    for (int kx = 0; kx < NP; ++kx) {
                        if (kx + ky > 0) { a[v][ky][kx] *= theta; }
                    }
                }
            }
            changed = true;
        }
    }

    if (changed) {
        for (int v = 0; v < NVAR; ++v) {
            for (int ky = 0; ky < NP; ++ky) {
                for (int kx = 0; kx < NP; ++kx) { s(i,j,0,comp(v,kx,ky)) = a[v][ky][kx]; }
            }
        }
    }
}

// Maximum of (|u|+c)/dx + (|v|+c)/dy over the cell average.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real cell_wave_rate (int i, int j, Array4<Real const> const& s, Real dxinv, Real dyinv,
                     const EulerParm& ep) noexcept
{
    Real U[NVAR];
    for (int v = 0; v < NVAR; ++v) { U[v] = s(i,j,0,comp(v,0,0)); }
    return max_speed(U, 0, ep)*dxinv + max_speed(U, 1, ep)*dyinv;
}

} // namespace dg

#endif
