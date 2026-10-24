#ifndef CNS_WENO_ROE_K_H_
#define CNS_WENO_ROE_K_H_

// 7th-order WENO reconstruction in characteristic variables + Roe flux.
//
// For the face between cells (i-1) and i in direction dir:
//  1. Roe-average the two neighbouring cell states and build the left/right
//     eigenvector matrices of the Euler flux Jacobian in that direction.
//  2. Project the 8 cells i-4 ... i+3 onto characteristic variables.
//  3. WENO7 (Balsara & Shu 2000; optional WENO-Z weights) gives the left state
//     from cells i-4..i+2 and the right state from cells i-3..i+3.
//  4. Back to conserved variables, then the Roe flux with Harten's entropy fix.
// If a reconstructed state has non-positive density or pressure, that face
// falls back to first order (cell values), which keeps strong shocks robust.

#include "CNS_index_macros.H"
#include "CNS_parm.H"

#include <AMReX_FArrayBox.H>
#include <cmath>

namespace weno_roe {

using amrex::Real;

// Conserved state of the 4 Euler variables in the order (rho, m_n, m_t, E),
// i.e. normal and tangential momentum for face direction dir.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void load (amrex::Array4<Real const> const& s, int i, int j, int k, int dir, Real U[4]) noexcept
{
    U[0] = s(i,j,k,URHO);
    U[1] = (dir == 0) ? s(i,j,k,UMX) : s(i,j,k,UMY);
    U[2] = (dir == 0) ? s(i,j,k,UMY) : s(i,j,k,UMX);
    U[3] = s(i,j,k,UEDEN);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure (const Real U[4], Real gamma) noexcept
{
    return (gamma - Real(1.0))*(U[3] - Real(0.5)*(U[1]*U[1] + U[2]*U[2])/U[0]);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void phys_flux (const Real U[4], Real gamma, Real F[4]) noexcept
{
    const Real un = U[1]/U[0];
    const Real p  = pressure(U, gamma);
    F[0] = U[1];
    F[1] = U[1]*un + p;
    F[2] = U[2]*un;
    F[3] = (U[3] + p)*un;
}

// Roe average of two states -> (un, ut, H, c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void roe_average (const Real UL[4], const Real UR[4], Real gamma,
                  Real& un, Real& ut, Real& H, Real& c) noexcept
{
    const Real sl = std::sqrt(UL[0]), sr = std::sqrt(UR[0]);
    const Real HL = (UL[3] + pressure(UL, gamma))/UL[0];
    const Real HR = (UR[3] + pressure(UR, gamma))/UR[0];
    const Real w  = Real(1.0)/(sl + sr);
    un = (UL[1]/sl + UR[1]/sr)*w;
    ut = (UL[2]/sl + UR[2]/sr)*w;
    H  = (sl*HL + sr*HR)*w;
    const Real c2 = (gamma - Real(1.0))*(H - Real(0.5)*(un*un + ut*ut));
    c = std::sqrt(amrex::max(c2, Real(1.e-14)));
}

// Right (columns) and left (rows) eigenvectors for (rho, m_n, m_t, E).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void eigenvectors (Real un, Real ut, Real H, Real c, Real gamma,
                   Real R[4][4], Real L[4][4]) noexcept
{
    const Real q2 = un*un + ut*ut;
    const Real b1 = (gamma - Real(1.0))/(c*c);
    const Real b2 = Real(0.5)*b1*q2;
    // R[row][col]: columns are the waves (u-c, shear, entropy, u+c)
    R[0][0] = 1;          R[0][1] = 0;  R[0][2] = 1;              R[0][3] = 1;
    R[1][0] = un - c;     R[1][1] = 0;  R[1][2] = un;             R[1][3] = un + c;
    R[2][0] = ut;         R[2][1] = 1;  R[2][2] = ut;             R[2][3] = ut;
    R[3][0] = H - un*c;   R[3][1] = ut; R[3][2] = Real(0.5)*q2;   R[3][3] = H + un*c;

    L[0][0] = Real(0.5)*(b2 + un/c); L[0][1] = -Real(0.5)*(b1*un + Real(1.0)/c);
    L[0][2] = -Real(0.5)*b1*ut;      L[0][3] =  Real(0.5)*b1;
    L[1][0] = -ut;                   L[1][1] = 0; L[1][2] = 1; L[1][3] = 0;
    L[2][0] = Real(1.0) - b2;        L[2][1] = b1*un; L[2][2] = b1*ut; L[2][3] = -b1;
    L[3][0] = Real(0.5)*(b2 - un/c); L[3][1] = -Real(0.5)*(b1*un - Real(1.0)/c);
    L[3][2] = -Real(0.5)*b1*ut;      L[3][3] =  Real(0.5)*b1;
}

// WENO7 value at the right face of the centre cell v[3] (v[0..6] = cells -3..+3).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real weno7 (const Real v[7], int use_z) noexcept
{
    const Real a = v[0], b = v[1], c = v[2], d = v[3], e = v[4], f = v[5], g = v[6];
    const Real q0 = (-Real(3.0)*a + Real(13.0)*b - Real(23.0)*c + Real(25.0)*d)/Real(12.0);
    const Real q1 = (        b - Real(5.0)*c + Real(13.0)*d + Real(3.0)*e)/Real(12.0);
    const Real q2 = (-c + Real(7.0)*d + Real(7.0)*e - f)/Real(12.0);
    const Real q3 = ( Real(3.0)*d + Real(13.0)*e - Real(5.0)*f + g)/Real(12.0);

    const Real b0 = a*(Real(547.0)*a - Real(3882.0)*b + Real(4642.0)*c - Real(1854.0)*d)
                  + b*(Real(7043.0)*b - Real(17246.0)*c + Real(7042.0)*d)
                  + c*(Real(11003.0)*c - Real(9402.0)*d) + Real(2107.0)*d*d;
    const Real b1 = b*(Real(267.0)*b - Real(1642.0)*c + Real(1602.0)*d - Real(494.0)*e)
                  + c*(Real(2843.0)*c - Real(5966.0)*d + Real(1922.0)*e)
                  + d*(Real(3443.0)*d - Real(2522.0)*e) + Real(547.0)*e*e;
    const Real b2 = c*(Real(547.0)*c - Real(2522.0)*d + Real(1922.0)*e - Real(494.0)*f)
                  + d*(Real(3443.0)*d - Real(5966.0)*e + Real(1602.0)*f)
                  + e*(Real(2843.0)*e - Real(1642.0)*f) + Real(267.0)*f*f;
    const Real b3 = d*(Real(2107.0)*d - Real(9402.0)*e + Real(7042.0)*f - Real(1854.0)*g)
                  + e*(Real(11003.0)*e - Real(17246.0)*f + Real(4642.0)*g)
                  + f*(Real(7043.0)*f - Real(3882.0)*g) + Real(547.0)*g*g;

    constexpr Real d0 = Real(1.0)/Real(35.0), d1 = Real(12.0)/Real(35.0);
    constexpr Real d2 = Real(18.0)/Real(35.0), d3 = Real(4.0)/Real(35.0);
    Real a0, a1, a2, a3;
    if (use_z) {
        constexpr Real eps = Real(1.e-40);
        const Real tau = std::abs(b0 - b1 - b2 + b3);
        a0 = d0*(Real(1.0) + (tau/(b0 + eps))*(tau/(b0 + eps)));
        a1 = d1*(Real(1.0) + (tau/(b1 + eps))*(tau/(b1 + eps)));
        a2 = d2*(Real(1.0) + (tau/(b2 + eps))*(tau/(b2 + eps)));
        a3 = d3*(Real(1.0) + (tau/(b3 + eps))*(tau/(b3 + eps)));
    } else {
        constexpr Real eps = Real(1.e-6);
        a0 = d0/((eps + b0)*(eps + b0));
        a1 = d1/((eps + b1)*(eps + b1));
        a2 = d2/((eps + b2)*(eps + b2));
        a3 = d3/((eps + b3)*(eps + b3));
    }
    return (a0*q0 + a1*q1 + a2*q2 + a3*q3)/(a0 + a1 + a2 + a3);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool admissible (const Real U[4], Real gamma, Real smallr, Real smallp) noexcept
{
    return U[0] > smallr && pressure(U, gamma) > smallp;
}

// Roe flux with Harten entropy fix; F in (rho, m_n, m_t, E) ordering.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void roe_flux (const Real UL[4], const Real UR[4], Real gamma, Real F[4]) noexcept
{
    Real un, ut, H, c, R[4][4], L[4][4];
    roe_average(UL, UR, gamma, un, ut, H, c);
    eigenvectors(un, ut, H, c, gamma, R, L);
    Real FL[4], FR[4];
    phys_flux(UL, gamma, FL);
    phys_flux(UR, gamma, FR);
    const Real lam[4] = {un - c, un, un, un + c};
    const Real delta = Real(0.1)*(std::abs(un) + c);
    Real alpha[4];
    for (int w = 0; w < 4; ++w) {
        Real sum = 0;
        for (int v = 0; v < 4; ++v) { sum += L[w][v]*(UR[v] - UL[v]); }
        Real al = std::abs(lam[w]);
        if (al < delta) { al = (lam[w]*lam[w] + delta*delta)/(Real(2.0)*delta); }
        alpha[w] = al*sum;
    }
    for (int v = 0; v < 4; ++v) {
        Real diss = 0;
        for (int w = 0; w < 4; ++w) { diss += R[v][w]*alpha[w]; }
        F[v] = Real(0.5)*(FL[v] + FR[v] - diss);
    }
}

} // namespace weno_roe

// Flux through the low face of cell (i,j,k) in direction dir.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void cns_weno7_roe_flux (int i, int j, int k, int dir,
                         amrex::Array4<amrex::Real const> const& s,
                         amrex::Array4<amrex::Real> const& flx,
                         Parm const& parm, int use_z) noexcept
{
    using namespace weno_roe;
    const Real gamma = parm.eos_gamma;

    // 8 cells: m = 0..7 <-> offsets -4..+3 along dir
    Real U[8][4];
    for (int m = 0; m < 8; ++m) {
        const int o = m - 4;
        if (dir == 0) { load(s, i+o, j, k, dir, U[m]); }
        else          { load(s, i, j+o, k, dir, U[m]); }
    }

    Real un, ut, H, c, R[4][4], L[4][4];
    roe_average(U[3], U[4], gamma, un, ut, H, c);
    eigenvectors(un, ut, H, c, gamma, R, L);

    Real W[8][4];
    for (int m = 0; m < 8; ++m) {
        for (int w = 0; w < 4; ++w) {
            W[m][w] = L[w][0]*U[m][0] + L[w][1]*U[m][1] + L[w][2]*U[m][2] + L[w][3]*U[m][3];
        }
    }

    Real WL[4], WR[4];
    for (int w = 0; w < 4; ++w) {
        const Real vl[7] = {W[0][w], W[1][w], W[2][w], W[3][w], W[4][w], W[5][w], W[6][w]};
        const Real vr[7] = {W[7][w], W[6][w], W[5][w], W[4][w], W[3][w], W[2][w], W[1][w]};
        WL[w] = weno7(vl, use_z);
        WR[w] = weno7(vr, use_z);
    }

    Real UL[4], UR[4];
    for (int v = 0; v < 4; ++v) {
        UL[v] = R[v][0]*WL[0] + R[v][1]*WL[1] + R[v][2]*WL[2] + R[v][3]*WL[3];
        UR[v] = R[v][0]*WR[0] + R[v][1]*WR[1] + R[v][2]*WR[2] + R[v][3]*WR[3];
    }
    if (!admissible(UL, gamma, parm.smallr, parm.smallp) ||
        !admissible(UR, gamma, parm.smallr, parm.smallp)) {
        for (int v = 0; v < 4; ++v) { UL[v] = U[3][v]; UR[v] = U[4][v]; }
    }

    Real F[4];
    roe_flux(UL, UR, gamma, F);

    flx(i,j,k,URHO ) = F[0];
    flx(i,j,k,UMX  ) = (dir == 0) ? F[1] : F[2];
    flx(i,j,k,UMY  ) = (dir == 0) ? F[2] : F[1];
    flx(i,j,k,UEDEN) = F[3];
    flx(i,j,k,UEINT) = Real(0.0);
    flx(i,j,k,UTEMP) = Real(0.0);
}

#endif
