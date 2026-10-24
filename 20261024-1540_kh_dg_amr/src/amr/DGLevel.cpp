#include "DGLevel.H"
#include "DG_Interp.H"

#include <AMReX_ParmParse.H>
#include <AMReX_TagBox.H>
#include <AMReX_Utility.H>
#include <AMReX_MultiFabUtil.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

using namespace amrex;
using namespace dg;

DGLevel::DGLevel () = default;

DGLevel::DGLevel (Amr& papa, int lev, const Geometry& level_geom, const BoxArray& bl,
                  const DistributionMapping& dm, Real time)
    : AmrLevel(papa, lev, level_geom, bl, dm, time)
{
    const Real* dx = geom.CellSize();
    if (std::abs(dx[0] - dx[1]) > 1.e-12*dx[0]) {
        amrex::Abort("DGLevel: the solver assumes square cells (dx == dy)");
    }
    define_flux_register();
}

DGLevel::~DGLevel () = default;

void DGLevel::define_flux_register ()
{
    if (do_reflux && level > 0) {
        flux_reg.define(grids, dmap, parent->refRatio(level-1), level, NCOMP);
    }
}

void DGLevel::restart (Amr& papa, std::istream& is, bool bReadSpecial)
{
    AmrLevel::restart(papa, is, bReadSpecial);
    define_flux_register();
}

// New level data from the previous version of the same level (regrid).
void DGLevel::init (AmrLevel& old)
{
    auto& oldlev = dynamic_cast<DGLevel&>(old);
    const Real dt_new    = parent->dtLevel(level);
    const Real cur_time  = oldlev.state[State_Type].curTime();
    const Real prev_time = oldlev.state[State_Type].prevTime();
    setTimeLevel(cur_time, cur_time - prev_time, dt_new);

    MultiFab& S_new = get_new_data(State_Type);
    FillPatch(old, S_new, 0, cur_time, State_Type, 0, NCOMP);
}

// A level that did not exist before: project from the coarse level.
void DGLevel::init ()
{
    const Real dt        = parent->dtLevel(level);
    const Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    const Real prev_time = getLevel(level-1).state[State_Type].prevTime();
    const Real dt_old    = (cur_time - prev_time)/static_cast<Real>(parent->MaxRefRatio(level-1));
    setTimeLevel(cur_time, dt_old, dt);

    MultiFab& S_new = get_new_data(State_Type);
    FillCoarsePatch(S_new, 0, cur_time, State_Type, 0, NCOMP);
}

// L2 projection of the analytic initial condition onto the DG basis, using a
// high-order Gauss rule so that thin interfaces are integrated accurately.
void DGLevel::initData ()
{
    constexpr int NI = 12;
    struct InitQuad { Real x[NI]; Real w[NI]; Real phi[NI][NP]; } iq{};
    {
        Real w[NI];
        gauss_legendre(NI, iq.x, w);
        for (int q = 0; q < NI; ++q) {
            iq.w[q] = 0.5*w[q];
            for (int k = 0; k < NP; ++k) { iq.phi[q][k] = legendre_orthonormal(k, iq.x[q]); }
        }
    }

    const auto problo = geom.ProbLoArray();
    const auto dx     = geom.CellSizeArray();
    const auto pp     = prob_parm;
    const Real gamma  = euler_parm.gamma;

    MultiFab& S_new = get_new_data(State_Type);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const s = S_new.array(mfi);
        amrex::ParallelFor(bx, [=] (int i, int j, int) noexcept
        {
            Real a[NVAR][NP][NP] = {};
            for (int qy = 0; qy < NI; ++qy) {
                const Real y = problo[1] + (j + 0.5 + 0.5*iq.x[qy])*dx[1];
                for (int qx = 0; qx < NI; ++qx) {
                    const Real x = problo[0] + (i + 0.5 + 0.5*iq.x[qx])*dx[0];
                    Real W[4];
                    kh::initial_primitive(x, y, pp, W);
                    const Real U[4] = { W[0], W[0]*W[1], W[0]*W[2],
                                        W[3]/(gamma - 1.0) + 0.5*W[0]*(W[1]*W[1] + W[2]*W[2]) };
                    const Real w = iq.w[qx]*iq.w[qy];
                    for (int v = 0; v < NVAR; ++v) {
                        for (int ky = 0; ky < NP; ++ky) {
                            for (int kx = 0; kx < NP; ++kx) {
                                a[v][ky][kx] += w*U[v]*iq.phi[qx][kx]*iq.phi[qy][ky];
                            }
                        }
                    }
                }
            }
            for (int v = 0; v < NVAR; ++v) {
                for (int ky = 0; ky < NP; ++ky) {
                    for (int kx = 0; kx < NP; ++kx) { s(i,j,0,comp(v,kx,ky)) = a[v][ky][kx]; }
                }
            }
        });
    }
    apply_limiter(S_new);
}

Real DGLevel::estTimeStep ()
{
    const Real dxinv = geom.InvCellSize(0);
    const Real dyinv = geom.InvCellSize(1);
    const auto ep = euler_parm;
    const MultiFab& S = get_new_data(State_Type);

    const Real rate = amrex::ReduceMax(S, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& s) -> Real
        {
            Real r = 0.0;
            amrex::Loop(bx, [&] (int i, int j, int) {
                r = amrex::max(r, cell_wave_rate(i, j, s, dxinv, dyinv, ep));
            });
            return r;
        });
    Real global_rate = rate;
    ParallelDescriptor::ReduceRealMax(global_rate);

    // Explicit RK-DG stability: dt ~ CFL / ((2P+1) * sum_d (|u_d|+c)/dx_d)
    return cfl/(static_cast<Real>(2*P + 1)*global_rate);
}

void DGLevel::computeInitialDt (int finest_level, int /*sub_cycle*/, Vector<int>& n_cycle,
                                const Vector<IntVect>& /*ref_ratio*/, Vector<Real>& dt_level,
                                Real stop_time)
{
    if (level > 0) { return; }

    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; ++i) {
        dt_level[i] = getLevel(i).estTimeStep();
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0, n_factor*dt_level[i]);
    }

    const Real eps = 0.001*dt_0;
    const Real cur_time = state[State_Type].curTime();
    if (stop_time >= 0.0 && (cur_time + dt_0) > (stop_time - eps)) {
        dt_0 = stop_time - cur_time;
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; ++i) {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void DGLevel::computeNewDt (int finest_level, int /*sub_cycle*/, Vector<int>& n_cycle,
                            const Vector<IntVect>& /*ref_ratio*/, Vector<Real>& dt_min,
                            Vector<Real>& dt_level, Real stop_time, int post_regrid_flag)
{
    if (level > 0) { return; }

    for (int i = 0; i <= finest_level; ++i) {
        dt_min[i] = getLevel(i).estTimeStep();
    }

    constexpr Real change_max = 1.1;
    for (int i = 0; i <= finest_level; ++i) {
        dt_min[i] = (post_regrid_flag == 1) ? std::min(dt_min[i], dt_level[i])
                                            : std::min(dt_min[i], change_max*dt_level[i]);
    }

    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; ++i) {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0, n_factor*dt_min[i]);
    }

    const Real eps = 0.001*dt_0;
    const Real cur_time = state[State_Type].curTime();
    if (stop_time >= 0.0 && (cur_time + dt_0) > (stop_time - eps)) {
        dt_0 = stop_time - cur_time;
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; ++i) {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void DGLevel::post_regrid (int /*lbase*/, int /*new_finest*/) {}

void DGLevel::post_timestep (int /*iteration*/)
{
    if (level < parent->finestLevel()) {
        if (do_reflux) { reflux(); }
        avgDown();
    }
}

// Correct the coarse cells next to the fine region with the difference between
// the time-integrated fine and coarse face fluxes, for every DG mode.
//
// The register stores face moments as seen from the cell *below/left of* the
// face. A coarse cell on the other side sees the same moment multiplied by
// phi_kn(-1)/phi_kn(+1) = (-1)^kn, so odd normal modes flip sign on the
// high-side registers before applying them.
void DGLevel::reflux ()
{
    auto& fr = getLevel(level+1).flux_reg;
    MultiFab& S_new = get_new_data(State_Type);
    MultiFab volume(grids, dmap, 1, 0);
    volume.setVal(geom.CellSize(0));   // fluxes are face averages and dx == dy

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const Orientation lo_face(d, Orientation::low);
        const Orientation hi_face(d, Orientation::high);
        FabSet& hi_reg = fr[hi_face];
        for (FabSetIter fsi(hi_reg); fsi.isValid(); ++fsi) {
            FArrayBox& fab = hi_reg[fsi];
            for (int v = 0; v < NVAR; ++v) {
                for (int ky = 0; ky < NP; ++ky) {
                    for (int kx = 0; kx < NP; ++kx) {
                        const int kn = (d == 0) ? kx : ky;
                        if (kn % 2 == 1) { fab.mult<RunOn::Host>(-1.0, comp(v,kx,ky), 1); }
                    }
                }
            }
        }
        fr.Reflux(S_new, volume, lo_face, 1.0, 0, 0, NCOMP, geom);
        fr.Reflux(S_new, volume, hi_face, 1.0, 0, 0, NCOMP, geom);
    }
}

void DGLevel::avgDown ()
{
    if (level == parent->finestLevel()) { return; }
    auto& fine_lev = getLevel(level+1);
    dg::average_down(fine_lev.get_new_data(State_Type), get_new_data(State_Type),
                     geom, parent->refRatio(level));
}

void DGLevel::post_init (Real /*stop_time*/)
{
    if (level > 0) { return; }
    for (int k = parent->finestLevel()-1; k >= 0; --k) {
        getLevel(k).avgDown();
    }
    write_diagnostics(state[State_Type].curTime(), true);
}

void DGLevel::post_restart () {}

void DGLevel::postCoarseTimeStep (Real time)
{
    if (level == 0) { write_diagnostics(time, false); }
}

// Totals are taken from level 0, which after averaging down holds the exact
// composite cell averages. min/max are over the valid cells of all levels.
void DGLevel::write_diagnostics (Real time, bool header)
{
    const MultiFab& S0 = get_new_data(State_Type);
    const Real cell_area = geom.CellSize(0)*geom.CellSize(1);
    std::array<Real,NVAR> tot{};
    for (int v = 0; v < NVAR; ++v) {
        tot[v] = S0.sum(comp(v,0,0), true)*cell_area;
    }
    ParallelDescriptor::ReduceRealSum(tot.data(), NVAR);

    Real rho_min = std::numeric_limits<Real>::max(), rho_max = -rho_min;
    Real p_min = rho_min;
    Vector<Long> ncells(parent->finestLevel()+1);
    const auto ep = euler_parm;
    for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
        const MultiFab& S = getLevel(lev).get_new_data(State_Type);
        rho_min = std::min(rho_min, S.min(comp(RHO,0,0)));
        rho_max = std::max(rho_max, S.max(comp(RHO,0,0)));
        const Real pm = amrex::ReduceMin(S, 0,
            [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& s) -> Real
            {
                Real r = std::numeric_limits<Real>::max();
                amrex::Loop(bx, [&] (int i, int j, int) {
                    Real U[NVAR];
                    for (int v = 0; v < NVAR; ++v) { U[v] = s(i,j,0,comp(v,0,0)); }
                    r = amrex::min(r, pressure(U, ep));
                });
                return r;
            });
        p_min = std::min(p_min, pm);
        ncells[lev] = S.boxArray().numPts();
    }
    ParallelDescriptor::ReduceRealMin(p_min);

    if (verbose > 0) {
        amrex::Print() << std::setprecision(15)
                       << "[DG] t = " << time << "  mass = " << tot[RHO]
                       << "  x-mom = " << tot[MX] << "  y-mom = " << tot[MY]
                       << "  energy = " << tot[ENER] << "  rho in [" << rho_min << ", "
                       << rho_max << "]  p_min = " << p_min << "\n";
    }

    if (!diag_file.empty() && ParallelDescriptor::IOProcessor()) {
        std::ofstream os(diag_file, header ? std::ios::trunc : std::ios::app);
        if (header) {
            os << "step,time,mass,x_momentum,y_momentum,energy,rho_min,rho_max,p_min,finest_level";
            for (int lev = 0; lev <= parent->maxLevel(); ++lev) { os << ",cells_lev" << lev; }
            os << "\n";
        }
        os << std::setprecision(17) << parent->levelSteps(0) << "," << time << ","
           << tot[RHO] << "," << tot[MX] << "," << tot[MY] << "," << tot[ENER] << ","
           << rho_min << "," << rho_max << "," << p_min << "," << parent->finestLevel();
        for (int lev = 0; lev <= parent->maxLevel(); ++lev) {
            os << "," << ((lev < static_cast<int>(ncells.size())) ? ncells[lev] : 0);
        }
        os << "\n";
    }
}

// Refine where the density varies strongly inside a cell (first-order modes
// relative to the mean), where the velocity changes quickly across a cell, or
// inside an optional fixed box (dg.refine_box_lo / dg.refine_box_hi).
void DGLevel::errorEst (TagBoxArray& tags, int /*clearval*/, int /*tagval*/,
                        Real /*time*/, int /*n_error_buf*/, int /*ngrow*/)
{
    if (NP < 2) { return; }
    const MultiFab& S = get_new_data(State_Type);
    const Real thr_rho = refine_dengrad;
    const Real thr_vort = refine_vort;
    const Real sqrt3 = std::sqrt(3.0);
    const bool use_box = (refine_box.size() == 4);
    const Real bxlo = use_box ? refine_box[0] : 0.0, bylo = use_box ? refine_box[1] : 0.0;
    const Real bxhi = use_box ? refine_box[2] : 0.0, byhi = use_box ? refine_box[3] : 0.0;
    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    auto const& tagma = tags.arrays();
    auto const& sma = S.const_arrays();
    amrex::ParallelFor(S, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int) noexcept
    {
        auto const& s = sma[bno];
        const Real rho  = s(i,j,0,comp(RHO,0,0));
        const Real drx  = s(i,j,0,comp(RHO,1,0));
        const Real dry  = s(i,j,0,comp(RHO,0,1));
        // Jump of the linear part across the cell, relative to the mean.
        const Real jump = 2.0*sqrt3*(std::abs(drx) + std::abs(dry))/rho;
        // Velocity difference across the cell (dimensionless rotation measure).
        const Real u = s(i,j,0,comp(MX,0,0))/rho, v = s(i,j,0,comp(MY,0,0))/rho;
        const Real dvx = 2.0*sqrt3*(s(i,j,0,comp(MY,1,0)) - v*drx)/rho;
        const Real duy = 2.0*sqrt3*(s(i,j,0,comp(MX,0,1)) - u*dry)/rho;
        const Real rot = std::abs(dvx - duy);
        const Real xc = problo[0] + (i + 0.5)*dx[0], yc = problo[1] + (j + 0.5)*dx[1];
        const bool in_box = use_box && xc > bxlo && xc < bxhi && yc > bylo && yc < byhi;
        if (jump > thr_rho || rot > thr_vort || in_box) {
            tagma[bno](i,j,0) = TagBox::SET;
        }
    });
    Gpu::streamSynchronize();
}

Real DGLevel::density_wave_error (Real time) const
{
    // Composite error: each level contributes the cells not covered by a finer level.
    const Basis B = basis();
    Real err2 = 0.0;
    for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
        const auto& L = dynamic_cast<const DGLevel&>(parent->getLevel(lev));
        const auto problo = L.geom.ProbLoArray();
        const auto dx = L.geom.CellSizeArray();
        const MultiFab& S = L.get_new_data(State_Type);
        iMultiFab mask(S.boxArray(), S.DistributionMap(), 1, 0);
        mask.setVal(0);
        if (lev < parent->finestLevel()) {
            mask = amrex::makeFineMask(S, parent->boxArray(lev+1), parent->refRatio(lev));
        }
        err2 += amrex::ReduceSum(S, mask, 0,
            [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& s,
                                       Array4<int const> const& m) -> Real
            {
                Real e = 0.0;
                amrex::Loop(bx, [&] (int i, int j, int) {
                    if (m(i,j,0)) { return; }
                    Real a[NVAR][NP][NP];
                    load_cell(s, i, j, a);
                    for (int qy = 0; qy < NQ; ++qy) {
                        for (int qx = 0; qx < NQ; ++qx) {
                            Real U[NVAR], W[4];
                            eval_point(a, B.phi[qx], B.phi[qy], U);
                            const Real x = problo[0] + (i + 0.5 + 0.5*B.xq[qx])*dx[0];
                            const Real y = problo[1] + (j + 0.5 + 0.5*B.xq[qy])*dx[1];
                            kh::density_wave(x, y, time, W);
                            e += B.wq[qx]*B.wq[qy]*dx[0]*dx[1]*(U[RHO] - W[0])*(U[RHO] - W[0]);
                        }
                    }
                });
                return e;
            });
    }
    ParallelDescriptor::ReduceRealSum(err2);
    return std::sqrt(err2);
}
