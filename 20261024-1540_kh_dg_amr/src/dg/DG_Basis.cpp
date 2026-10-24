#include "DG_Basis.H"

#include <AMReX.H>
#include <cmath>
#include <vector>

namespace dg {

namespace {
void legendre_pd (int k, Real x, Real& p, Real& dp)
{
    // Bonnet recursion for P_k and P_k'.
    Real p0 = 1.0, p1 = x, d0 = 0.0, d1 = 1.0;
    if (k == 0) { p = 1.0; dp = 0.0; return; }
    for (int n = 1; n < k; ++n) {
        Real p2 = ((2*n+1)*x*p1 - n*p0) / (n+1);
        Real d2 = d0 + (2*n+1)*p1;
        p0 = p1; p1 = p2; d0 = d1; d1 = d2;
    }
    p = p1; dp = d1;
}
}

Real legendre_orthonormal (int k, Real x)
{
    Real p, dp; legendre_pd(k, x, p, dp);
    return std::sqrt(2.0*k+1.0) * p;
}

Real legendre_orthonormal_deriv (int k, Real x)
{
    Real p, dp; legendre_pd(k, x, p, dp);
    return std::sqrt(2.0*k+1.0) * dp;
}

void gauss_legendre (int n, Real* x, Real* w)
{
    const Real pi = 3.14159265358979323846;
    for (int i = 0; i < n; ++i) {
        Real z = std::cos(pi*(i+0.75)/(n+0.5));
        for (int it = 0; it < 100; ++it) {
            Real p, dp; legendre_pd(n, z, p, dp);
            Real dz = p/dp;
            z -= dz;
            if (std::abs(dz) < 1e-15) { break; }
        }
        Real p, dp; legendre_pd(n, z, p, dp);
        x[n-1-i] = z;                               // ascending order
        w[n-1-i] = 2.0/((1.0-z*z)*dp*dp);
    }
}

Basis make_basis ()
{
    Basis b{};
    Real w[NQ];
    gauss_legendre(NQ, b.xq, w);
    for (int q = 0; q < NQ; ++q) {
        b.wq[q] = 0.5*w[q];
        for (int k = 0; k < NP; ++k) {
            b.phi[q][k]  = legendre_orthonormal(k, b.xq[q]);
            b.dphi[q][k] = legendre_orthonormal_deriv(k, b.xq[q]);
        }
    }
    for (int k = 0; k < NP; ++k) {
        b.phiR[k] = legendre_orthonormal(k,  1.0);
        b.phiL[k] = legendre_orthonormal(k, -1.0);
    }
    return b;
}

Refine make_refine (int r)
{
    if (r < 1 || r > MAX_RATIO) { amrex::Abort("dg::make_refine: unsupported refinement ratio"); }
    Refine R{};
    R.r = r;
    Real x[NP], w[NP];
    gauss_legendre(NP, x, w);   // exact for degree 2P products
    for (int s = 0; s < r; ++s) {
        for (int m = 0; m < NP; ++m) {
            for (int k = 0; k < NP; ++k) {
                Real sum = 0.0;
                for (int q = 0; q < NP; ++q) {
                    Real xc = (x[q] + 2*s + 1 - r) / r;
                    sum += 0.5*w[q]*legendre_orthonormal(m, x[q])*legendre_orthonormal(k, xc);
                }
                R.T[s][m][k] = sum;
            }
        }
    }
    const Basis& b = basis();
    for (int s = 0; s < r; ++s) {
        for (int q = 0; q < NQ; ++q) {
            for (int k = 0; k < NP; ++k) {
                R.phic[s][q][k] = legendre_orthonormal(k, (b.xq[q] + 2*s + 1 - r) / r);
            }
        }
    }
    return R;
}

const Basis& basis ()
{
    static const Basis b = make_basis();
    return b;
}

const Refine& refine (int r)
{
    static const Refine R2 = make_refine(2);
    static const Refine R4 = make_refine(4);
    if (r == 2) { return R2; }
    if (r == 4) { return R4; }
    amrex::Abort("dg::refine: only refinement ratios 2 and 4 are supported");
    return R2;
}

} // namespace dg
