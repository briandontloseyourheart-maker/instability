#include "DG_Interp.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

using namespace amrex;

namespace dg {

DGInterp dg_interp;

Box DGInterp::CoarseBox (const Box& fine, int ratio)
{
    return amrex::coarsen(fine, ratio);
}

Box DGInterp::CoarseBox (const Box& fine, const IntVect& ratio)
{
    return amrex::coarsen(fine, ratio);
}

void DGInterp::interp (const FArrayBox& crse, int crse_comp,
                       FArrayBox& fine, int fine_comp, int ncomp,
                       const Box& fine_region, const IntVect& ratio,
                       const Geometry& /*crse_geom*/, const Geometry& /*fine_geom*/,
                       Vector<BCRec> const& /*bcr*/,
                       int actual_comp, int /*actual_state*/, RunOn /*runon*/)
{
    AMREX_ALWAYS_ASSERT(ratio[0] == ratio[1]);
    const int r = ratio[0];
    const Refine& R = refine(r);

    auto const c = crse.const_array();
    auto const f = fine.array();
    const auto lo = amrex::lbound(fine_region);
    const auto hi = amrex::ubound(fine_region);

    // Work variable by variable. Normally FillPatch asks for whole variables
    // (all NDOF modes); a partial request (e.g. only cell averages for a
    // derived quantity) falls back to piecewise-constant injection.
    for (int n0 = 0; n0 < ncomp; ) {
        const int gcomp = actual_comp + n0;       // global state component
        const int v     = gcomp / NDOF;
        const int first = v*NDOF;
        const bool whole = (gcomp == first) && (n0 + NDOF <= ncomp);

        if (whole) {
            for (int j = lo.y; j <= hi.y; ++j) {
                const int jc = (j >= 0) ? j/r : -((-j-1)/r) - 1;
                const int sy = j - r*jc;
                for (int i = lo.x; i <= hi.x; ++i) {
                    const int ic = (i >= 0) ? i/r : -((-i-1)/r) - 1;
                    const int sx = i - r*ic;
                    // b(m,n) = sum_{k,l} T[sx][m][k] T[sy][n][l] a(k,l)
                    Real tmp[NP][NP];   // tmp[l][m] = sum_k T[sx][m][k] a(k,l)
                    for (int l = 0; l < NP; ++l) {
                        for (int m = 0; m < NP; ++m) {
                            Real sum = 0.0;
                            for (int k = 0; k < NP; ++k) {
                                sum += R.T[sx][m][k]*c(ic,jc,0,crse_comp+n0+l*NP+k);
                            }
                            tmp[l][m] = sum;
                        }
                    }
                    for (int nn = 0; nn < NP; ++nn) {
                        for (int m = 0; m < NP; ++m) {
                            Real sum = 0.0;
                            for (int l = 0; l < NP; ++l) { sum += R.T[sy][nn][l]*tmp[l][m]; }
                            f(i,j,0,fine_comp+n0+nn*NP+m) = sum;
                        }
                    }
                }
            }
            n0 += NDOF;
        } else {
            const bool is_mean = (gcomp == first);
            for (int j = lo.y; j <= hi.y; ++j) {
                const int jc = (j >= 0) ? j/r : -((-j-1)/r) - 1;
                for (int i = lo.x; i <= hi.x; ++i) {
                    const int ic = (i >= 0) ? i/r : -((-i-1)/r) - 1;
                    f(i,j,0,fine_comp+n0) = is_mean ? c(ic,jc,0,crse_comp+n0) : 0.0;
                }
            }
            n0 += 1;
        }
    }
}

void average_down (const MultiFab& fine, MultiFab& crse,
                   const Geometry& crse_geom, const IntVect& ratio)
{
    AMREX_ALWAYS_ASSERT(ratio[0] == ratio[1]);
    AMREX_ALWAYS_ASSERT(fine.nComp() == NCOMP && crse.nComp() == NCOMP);
    const int r = ratio[0];
    const Refine& R = refine(r);
    const Real inv = 1.0/static_cast<Real>(r*r);

    BoxArray cba = fine.boxArray();
    cba.coarsen(ratio);
    MultiFab ctmp(cba, fine.DistributionMap(), NCOMP, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(ctmp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const c = ctmp.array(mfi);
        auto const f = fine.const_array(mfi);
        amrex::ParallelFor(bx, [=] (int ic, int jc, int) noexcept
        {
            for (int v = 0; v < NVAR; ++v) {
                Real a[NP][NP] = {};   // a[l][k]
                for (int sy = 0; sy < r; ++sy) {
                    for (int sx = 0; sx < r; ++sx) {
                        const int i = r*ic + sx, j = r*jc + sy;
                        // a(k,l) += sum_{m,n} T[sx][m][k] T[sy][n][l] b(m,n)
                        Real tmp[NP][NP];   // tmp[n][k] = sum_m T[sx][m][k] b(m,n)
                        for (int nn = 0; nn < NP; ++nn) {
                            for (int k = 0; k < NP; ++k) {
                                Real sum = 0.0;
                                for (int m = 0; m < NP; ++m) {
                                    sum += R.T[sx][m][k]*f(i,j,0,comp(v,m,nn));
                                }
                                tmp[nn][k] = sum;
                            }
                        }
                        for (int l = 0; l < NP; ++l) {
                            for (int k = 0; k < NP; ++k) {
                                Real sum = 0.0;
                                for (int nn = 0; nn < NP; ++nn) { sum += R.T[sy][nn][l]*tmp[nn][k]; }
                                a[l][k] += sum;
                            }
                        }
                    }
                }
                for (int l = 0; l < NP; ++l) {
                    for (int k = 0; k < NP; ++k) { c(ic,jc,0,comp(v,k,l)) = inv*a[l][k]; }
                }
            }
        });
    }

    crse.ParallelCopy(ctmp, 0, 0, NCOMP, 0, 0, crse_geom.periodicity());
}

} // namespace dg
