#ifndef DG_BASIS_H_
#define DG_BASIS_H_

// Tensor-product Legendre basis for a modal discontinuous Galerkin scheme.
//
// Each cell is mapped to the reference square [-1,1]^2. The solution of one
// conserved variable inside a cell is
//
//     u(xi, eta) = sum_{kx,ky} a(kx,ky) * phi_kx(xi) * phi_ky(eta),
//
// with phi_k = sqrt(2k+1) P_k (P_k = Legendre polynomial). This scaling makes
// the basis orthonormal with respect to the cell average, so
//   * the mass matrix is the identity, and
//   * a(0,0) is exactly the cell average (used for conservation, output, AMR).

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

#ifndef DG_ORDER
#define DG_ORDER 4
#endif
#ifndef DG_NQUAD
#define DG_NQUAD DG_ORDER
#endif

namespace dg {

using amrex::Real;

constexpr int NP    = DG_ORDER;      // modes per direction (= order of accuracy)
constexpr int P     = NP - 1;        // polynomial degree
constexpr int NQ    = DG_NQUAD;      // Gauss points per direction
constexpr int NDOF  = NP * NP;       // modes per variable per cell
constexpr int NVAR  = 4;             // rho, rho*u, rho*v, rho*E
constexpr int NCOMP = NVAR * NDOF;   // MultiFab components

enum Var { RHO = 0, MX = 1, MY = 2, ENER = 3 };

static_assert(NP >= 1 && NP <= 8, "DG_ORDER must be between 1 and 8");
static_assert(NQ >= NP, "need at least NP quadrature points");

// Component index in the MultiFab for variable v and mode (kx,ky).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int comp (int v, int kx, int ky) noexcept { return v*NDOF + ky*NP + kx; }

// Precomputed 1D tables. Small, trivially copyable: captured by value in kernels.
struct Basis
{
    Real xq[NQ];         // Gauss-Legendre nodes on [-1,1]
    Real wq[NQ];         // weights normalised to sum 1 (i.e. w/2)
    Real phi[NQ][NP];    // phi_k(x_q)
    Real dphi[NQ][NP];   // d phi_k / d xi at x_q
    Real phiR[NP];       // phi_k(+1)
    Real phiL[NP];       // phi_k(-1)
};

// Coarse <-> fine L2 projection matrices for refinement ratio r:
//   T[s][m][k] = 1/2 int_{-1}^{1} phi_m(xi) phi_k((xi + 2s + 1 - r)/r) dxi
// s = 0..r-1 is the position of the fine cell inside the coarse cell.
constexpr int MAX_RATIO = 4;
struct Refine
{
    int  r = 2;
    Real T[MAX_RATIO][NP][NP];
    // phi_k evaluated at the fine-face Gauss points, expressed in the coarse
    // face coordinate: phic[s][q][k] = phi_k((x_q + 2s + 1 - r)/r).
    // Used to accumulate fine fluxes against coarse test functions (reflux).
    Real phic[MAX_RATIO][NQ][NP];
};

// Host-side helpers (DG_Basis.cpp)
Real legendre_orthonormal (int k, Real x);
Real legendre_orthonormal_deriv (int k, Real x);
void gauss_legendre (int n, Real* x, Real* w);   // w sums to 2
Basis  make_basis ();
Refine make_refine (int r);

// Global, initialised once in main().
const Basis&  basis ();
const Refine& refine (int r);

} // namespace dg

#endif
