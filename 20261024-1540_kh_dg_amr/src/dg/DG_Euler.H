#ifndef DG_EULER_H_
#define DG_EULER_H_

// Pointwise physics of the 2D compressible Euler equations (ideal gas):
// conserved <-> primitive conversion, physical flux and the HLLC Riemann solver.

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Algorithm.H>
#include <cmath>

namespace dg {

using amrex::Real;

struct EulerParm
{
    Real gamma     = 5.0/3.0;
    Real small_rho = 1.0e-10;
    Real small_p   = 1.0e-10;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure (const Real U[4], const EulerParm& e) noexcept
{
    const Real rho = U[0];
    const Real ke  = 0.5*(U[1]*U[1] + U[2]*U[2])/rho;
    return (e.gamma - 1.0)*(U[3] - ke);
}

// Physical flux in direction dir (0 = x, 1 = y).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void euler_flux (const Real U[4], int dir, const EulerParm& e, Real F[4]) noexcept
{
    const Real rho = amrex::max(U[0], e.small_rho);
    const Real u   = U[1]/rho;
    const Real v   = U[2]/rho;
    const Real p   = amrex::max(pressure(U, e), e.small_p);
    const Real un  = (dir == 0) ? u : v;
    F[0] = U[0]*un;
    F[1] = U[1]*un + ((dir == 0) ? p : 0.0);
    F[2] = U[2]*un + ((dir == 1) ? p : 0.0);
    F[3] = (U[3] + p)*un;
}

// Largest signal speed |u_n| + c, used for the time step.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real max_speed (const Real U[4], int dir, const EulerParm& e) noexcept
{
    const Real rho = amrex::max(U[0], e.small_rho);
    const Real p   = amrex::max(pressure(U, e), e.small_p);
    const Real c   = std::sqrt(e.gamma*p/rho);
    return std::abs(U[1+dir]/rho) + c;
}

// HLLC approximate Riemann solver (Toro), normal direction dir.
AMREX_GPU_HOST_DEVICE inline
void hllc_flux (const Real UL[4], const Real UR[4], int dir, const EulerParm& e, Real F[4]) noexcept
{
    const int n = 1 + dir;        // normal momentum component
    const int t = 2 - dir;        // tangential momentum component

    const Real rl = amrex::max(UL[0], e.small_rho);
    const Real rr = amrex::max(UR[0], e.small_rho);
    const Real ul = UL[n]/rl, ur = UR[n]/rr;
    const Real vl = UL[t]/rl, vr = UR[t]/rr;
    const Real pl = amrex::max(pressure(UL, e), e.small_p);
    const Real pr = amrex::max(pressure(UR, e), e.small_p);
    const Real cl = std::sqrt(e.gamma*pl/rl);
    const Real cr = std::sqrt(e.gamma*pr/rr);

    const Real SL = amrex::min(ul - cl, ur - cr);
    const Real SR = amrex::max(ul + cl, ur + cr);

    Real FL[4], FR[4];
    euler_flux(UL, dir, e, FL);
    euler_flux(UR, dir, e, FR);

    if (SL >= 0.0) { for (int m = 0; m < 4; ++m) { F[m] = FL[m]; } return; }
    if (SR <= 0.0) { for (int m = 0; m < 4; ++m) { F[m] = FR[m]; } return; }

    const Real SM = (pr - pl + rl*ul*(SL - ul) - rr*ur*(SR - ur))
                  / (rl*(SL - ul) - rr*(SR - ur));

    auto star = [&] (const Real U[4], const Real FK[4], Real rk, Real uk, Real vk,
                     Real pk, Real SK, Real Fo[4])
    {
        const Real fac = rk*(SK - uk)/(SK - SM);
        Real Us[4];
        Us[0] = fac;
        Us[n] = fac*SM;
        Us[t] = fac*vk;
        Us[3] = fac*(U[3]/rk + (SM - uk)*(SM + pk/(rk*(SK - uk))));
        for (int m = 0; m < 4; ++m) { Fo[m] = FK[m] + SK*(Us[m] - U[m]); }
    };

    if (SM >= 0.0) { star(UL, FL, rl, ul, vl, pl, SL, F); }
    else           { star(UR, FR, rr, ur, vr, pr, SR, F); }
}

} // namespace dg

#endif
