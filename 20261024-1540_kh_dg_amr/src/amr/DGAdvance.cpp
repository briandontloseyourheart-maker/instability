// Time integration of one AMR level: explicit Runge-Kutta in Butcher form.
#include "DGLevel.H"

#include <AMReX_MultiFab.H>

using namespace amrex;
using namespace dg;

namespace {

struct ButcherTableau
{
    int  stages;
    Real a[4][4];
    Real b[4];
    Real c[4];
};

// 3rd-order strong-stability-preserving RK (Shu & Osher).
constexpr ButcherTableau SSPRK3 = {
    3,
    {{0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.25, 0.25, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
    {1.0/6.0, 1.0/6.0, 2.0/3.0, 0.0},
    {0.0, 1.0, 0.5, 0.0}
};

// Classical 4th-order RK (matches the spatial order of DG_ORDER=4).
constexpr ButcherTableau RK4 = {
    4,
    {{0.0, 0.0, 0.0, 0.0}, {0.5, 0.0, 0.0, 0.0}, {0.0, 0.5, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}},
    {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0},
    {0.0, 0.5, 0.5, 1.0}
};

} // namespace

Real DGLevel::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
    BL_PROFILE("DGLevel::advance()");

    for (int k = 0; k < NUM_STATE_TYPE; ++k) {
        state[k].allocOldData();
        state[k].swapTimeLevels(dt);
    }

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);
    MultiFab::Copy(S_new, S_old, 0, 0, NCOMP, 0);

    const ButcherTableau& rk = (rk_order == 4) ? RK4 : SSPRK3;

    MultiFab Sborder(grids, dmap, NCOMP, NUM_GROW);
    Vector<MultiFab> K(rk.stages);
    for (auto& k : K) { k.define(grids, dmap, NCOMP, 0); }

    std::array<MultiFab,AMREX_SPACEDIM> flux, flux_cb;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const BoxArray fba = amrex::convert(grids, IntVect::TheDimensionVector(d));
        flux[d].define(fba, dmap, NCOMP, 0);
    }

    FluxRegister* fr_as_crse = nullptr;
    if (do_reflux && level < parent->finestLevel()) {
        fr_as_crse = &getLevel(level+1).flux_reg;
        fr_as_crse->setVal(0.0);
    }
    FluxRegister* fr_as_fine = (do_reflux && level > 0) ? &flux_reg : nullptr;
    Real fine_scale = 0.0;
    if (fr_as_fine) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            flux_cb[d].define(flux[d].boxArray(), dmap, NCOMP, 0);
        }
        // Each coarse face is covered by r fine faces in 2D.
        fine_scale = 1.0/static_cast<Real>(parent->refRatio(level-1)[0]);
    }

    for (int s = 0; s < rk.stages; ++s) {
        // Stage solution U_s = U^n + dt * sum_j a_sj K_j
        if (s > 0) {
            MultiFab::Copy(S_new, S_old, 0, 0, NCOMP, 0);
            for (int j = 0; j < s; ++j) {
                if (rk.a[s][j] != 0.0) { MultiFab::Saxpy(S_new, dt*rk.a[s][j], K[j], 0, 0, NCOMP, 0); }
            }
            apply_limiter(S_new);
        }

        fill_stage(Sborder, S_new, time + rk.c[s]*dt);
        compute_rhs(Sborder, K[s], flux, fr_as_fine ? &flux_cb : nullptr);

        // Time-integrated interface fluxes, weighted like the final RK update.
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (fr_as_crse) {
                fr_as_crse->CrseInit(flux[d], d, 0, 0, NCOMP, -rk.b[s]*dt, FluxRegister::ADD);
            }
            if (fr_as_fine) {
                fr_as_fine->FineAdd(flux_cb[d], d, 0, 0, NCOMP, rk.b[s]*dt*fine_scale);
            }
        }
    }

    MultiFab::Copy(S_new, S_old, 0, 0, NCOMP, 0);
    for (int s = 0; s < rk.stages; ++s) {
        MultiFab::Saxpy(S_new, dt*rk.b[s], K[s], 0, 0, NCOMP, 0);
    }
    apply_limiter(S_new);

    return dt;
}

// Fill ghost cells of the stage state. FillPatch provides the coarse/fine
// ghost cells at the stage time (time-interpolated coarse data projected onto
// the fine basis); the valid region and same-level ghosts are then taken from
// the current stage solution itself.
void DGLevel::fill_stage (MultiFab& Sborder, const MultiFab& U, Real t)
{
    FillPatch(*this, Sborder, NUM_GROW, t, State_Type, 0, NCOMP);
    MultiFab::Copy(Sborder, U, 0, 0, NCOMP, 0);
    Sborder.FillBoundary(geom.periodicity());
}

void DGLevel::compute_rhs (const MultiFab& S, MultiFab& dUdt,
                           std::array<MultiFab,AMREX_SPACEDIM>& flux,
                           std::array<MultiFab,AMREX_SPACEDIM>* flux_cb)
{
    BL_PROFILE("DGLevel::compute_rhs()");

    const Basis B = basis();
    const Refine R = (level > 0) ? refine(parent->refRatio(level-1)[0]) : refine(2);
    const int ratio = R.r;
    const bool do_gc = (flux_cb != nullptr);
    const auto ep = euler_parm;
    const Real dxinv = geom.InvCellSize(0);
    const Real dyinv = geom.InvCellSize(1);

    // Pass 1: one Riemann problem per face Gauss point.
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(flux[d], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& fbx = mfi.tilebox();
            auto const s  = S.const_array(mfi);
            auto const g  = flux[d].array(mfi);
            auto const gc = do_gc ? (*flux_cb)[d].array(mfi) : g;
            amrex::ParallelFor(fbx, [=] (int i, int j, int) noexcept
            {
                face_flux(i, j, d, s, g, gc, do_gc, ratio, B, R, ep);
            });
        }
    }

    // Pass 2: volume integral + surface terms for every cell.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(dUdt, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const s  = S.const_array(mfi);
        auto const gx = flux[0].const_array(mfi);
        auto const gy = flux[1].const_array(mfi);
        auto const du = dUdt.array(mfi);
        amrex::ParallelFor(bx, [=] (int i, int j, int) noexcept
        {
            cell_rhs(i, j, s, gx, gy, du, dxinv, dyinv, B, ep);
        });
    }
}

void DGLevel::apply_limiter (MultiFab& S)
{
    BL_PROFILE("DGLevel::apply_limiter()");
    const Basis B = basis();
    const auto ep = euler_parm;
    const auto lp = limiter_parm;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const s = S.array(mfi);
        amrex::ParallelFor(bx, [=] (int i, int j, int) noexcept
        {
            limit_cell(i, j, s, B, ep, lp);
        });
    }
}
