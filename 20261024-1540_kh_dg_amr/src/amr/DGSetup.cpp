// State description, run-time parameters and derived plot quantities.
#include "DGLevel.H"
#include "DG_Interp.H"

#include <AMReX_ParmParse.H>
#include <AMReX_StateDescriptor.H>

#include <cmath>

using namespace amrex;
using namespace dg;

dg::EulerParm   DGLevel::euler_parm;
dg::LimiterParm DGLevel::limiter_parm;
kh::ProbParm    DGLevel::prob_parm;

int         DGLevel::verbose        = 1;
Real        DGLevel::cfl            = 0.6;
int         DGLevel::rk_order       = 3;
int         DGLevel::do_reflux      = 1;
Real        DGLevel::refine_dengrad = 0.02;
Real        DGLevel::refine_vort    = 1.0e30;
Vector<Real> DGLevel::refine_box;
std::string DGLevel::diag_file      = "diagnostics.csv";
BCRec       DGLevel::phys_bc;

void DGLevel::read_params ()
{
    ParmParse pp("dg");
    pp.query("v", verbose);
    pp.query("cfl", cfl);
    pp.query("rk_order", rk_order);
    if (rk_order != 3 && rk_order != 4) { amrex::Abort("dg.rk_order must be 3 or 4"); }
    pp.query("do_reflux", do_reflux);
    pp.query("gamma", euler_parm.gamma);
    pp.query("small_rho", euler_parm.small_rho);
    pp.query("small_p", euler_parm.small_p);
    pp.query("refine_dengrad", refine_dengrad);
    pp.query("refine_vort", refine_vort);
    {
        Vector<Real> lo, hi;
        if (pp.queryarr("refine_box_lo", lo) && pp.queryarr("refine_box_hi", hi)) {
            refine_box = {lo[0], lo[1], hi[0], hi[1]};
        }
    }
    pp.query("diag_file", diag_file);

    pp.query("do_filter", limiter_parm.do_filter);
    pp.query("sensor_s0", limiter_parm.sensor_s0);
    pp.query("sensor_kappa", limiter_parm.sensor_kappa);
    pp.query("filter_alpha", limiter_parm.filter_alpha);
    pp.query("do_positivity", limiter_parm.do_positivity);


    // Kelvin-Helmholtz setup: doubly periodic box.
    ParmParse pg("geometry");
    Vector<int> periodic(AMREX_SPACEDIM, 0);
    pg.queryarr("is_periodic", periodic);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (!periodic[d]) { amrex::Abort("This solver supports periodic domains only (geometry.is_periodic = 1 1)"); }
        phys_bc.setLo(d, BCType::int_dir);
        phys_bc.setHi(d, BCType::int_dir);
    }

    ParmParse pa("amr");
    Vector<int> rr;
    pa.queryarr("ref_ratio", rr);
    for (int r : rr) {
        if (r != 2 && r != 4) { amrex::Abort("amr.ref_ratio must be 2 or 4 for the DG projection operators"); }
    }
}

// Periodic in every direction: nothing to fill outside the domain.
void dg_bcfill (Box const& /*bx*/, FArrayBox& /*data*/, int /*dcomp*/, int /*numcomp*/,
                Geometry const& /*geom*/, Real /*time*/, const Vector<BCRec>& /*bcr*/,
                int /*bcomp*/, int /*scomp*/)
{
}

namespace {

Box the_same_box (const Box& b) { return b; }

// All derived quantities are evaluated from the full modal state.
// "density" etc. are cell averages (mode 0); vorticity uses the linear modes.
void der_density (const Box& bx, FArrayBox& der, int dcomp, int /*ncomp*/,
                  const FArrayBox& dat, const Geometry& /*geom*/, Real /*time*/,
                  const int* /*bcrec*/, int /*level*/)
{
    auto const d = der.array(dcomp);
    auto const s = dat.const_array();
    amrex::ParallelFor(bx, [=] (int i, int j, int k) noexcept
    {
        d(i,j,k) = s(i,j,k,comp(RHO,0,0));
    });
}

template <int DIR>
void der_velocity (const Box& bx, FArrayBox& der, int dcomp, int /*ncomp*/,
                   const FArrayBox& dat, const Geometry& /*geom*/, Real /*time*/,
                   const int* /*bcrec*/, int /*level*/)
{
    auto const d = der.array(dcomp);
    auto const s = dat.const_array();
    amrex::ParallelFor(bx, [=] (int i, int j, int k) noexcept
    {
        d(i,j,k) = s(i,j,k,comp(MX+DIR,0,0))/s(i,j,k,comp(RHO,0,0));
    });
}

void der_pressure (const Box& bx, FArrayBox& der, int dcomp, int /*ncomp*/,
                   const FArrayBox& dat, const Geometry& /*geom*/, Real /*time*/,
                   const int* /*bcrec*/, int /*level*/)
{
    auto const d = der.array(dcomp);
    auto const s = dat.const_array();
    const auto ep = DGLevel::euler_parm;
    amrex::ParallelFor(bx, [=] (int i, int j, int k) noexcept
    {
        Real U[NVAR];
        for (int v = 0; v < NVAR; ++v) { U[v] = s(i,j,k,comp(v,0,0)); }
        d(i,j,k) = pressure(U, ep);
    });
}

// z-vorticity at the cell centre from the polynomial solution.
void der_vorticity (const Box& bx, FArrayBox& der, int dcomp, int /*ncomp*/,
                    const FArrayBox& dat, const Geometry& geom, Real /*time*/,
                    const int* /*bcrec*/, int /*level*/)
{
    auto const d = der.array(dcomp);
    auto const s = dat.const_array();
    const Real dxinv = geom.InvCellSize(0), dyinv = geom.InvCellSize(1);
    Real p0[NP], dp0[NP];
    for (int k = 0; k < NP; ++k) {
        p0[k]  = legendre_orthonormal(k, 0.0);
        dp0[k] = legendre_orthonormal_deriv(k, 0.0);
    }
    amrex::ParallelFor(bx, [&] (int i, int j, int k) noexcept
    {
        Real val[3] = {}, ddx[3] = {}, ddy[3] = {};   // rho, mx, my
        for (int v = 0; v < 3; ++v) {
            for (int ky = 0; ky < NP; ++ky) {
                for (int kx = 0; kx < NP; ++kx) {
                    const Real a = s(i,j,k,comp(v,kx,ky));
                    val[v] += a*p0[kx]*p0[ky];
                    ddx[v] += a*dp0[kx]*p0[ky]*2.0*dxinv;
                    ddy[v] += a*p0[kx]*dp0[ky]*2.0*dyinv;
                }
            }
        }
        const Real rho = val[0], u = val[1]/rho, v = val[2]/rho;
        const Real dvdx = (ddx[2] - v*ddx[0])/rho;
        const Real dudy = (ddy[1] - u*ddy[0])/rho;
        d(i,j,k) = dvdx - dudy;
    });
}

// Persson-Peraire smoothness indicator log10(S) of the density polynomial.
void der_sensor (const Box& bx, FArrayBox& der, int dcomp, int /*ncomp*/,
                 const FArrayBox& dat, const Geometry& /*geom*/, Real /*time*/,
                 const int* /*bcrec*/, int /*level*/)
{
    auto const d = der.array(dcomp);
    auto const s = dat.const_array();
    amrex::ParallelFor(bx, [=] (int i, int j, int k) noexcept
    {
        Real top = 1.0e-300, all = 1.0e-300;
        for (int ky = 0; ky < NP; ++ky) {
            for (int kx = 0; kx < NP; ++kx) {
                const Real a = s(i,j,k,comp(RHO,kx,ky));
                all += a*a;
                if (kx == P || ky == P) { top += a*a; }
            }
        }
        d(i,j,k) = std::log10(top/all);
    });
}

} // namespace

void DGLevel::variableSetUp ()
{
    read_params();
    (void) basis();          // build the quadrature / projection tables once
    (void) refine(2);

    constexpr bool extrap = false;
    constexpr bool store_in_checkpoint = true;
    desc_lst.addDescriptor(State_Type, IndexType::TheCellType(), StateDescriptor::Point,
                           NUM_GROW, NCOMP, &dg::dg_interp, extrap, store_in_checkpoint);

    const char* vname[NVAR] = {"rho", "xmom", "ymom", "rhoE"};
    Vector<BCRec> bcs(NCOMP, phys_bc);
    Vector<std::string> names(NCOMP);
    for (int v = 0; v < NVAR; ++v) {
        for (int ky = 0; ky < NP; ++ky) {
            for (int kx = 0; kx < NP; ++kx) {
                names[comp(v,kx,ky)] = std::string(vname[v]) + "_" + std::to_string(kx) + std::to_string(ky);
            }
        }
    }
    StateDescriptor::BndryFunc bndryfunc(dg_bcfill);
    desc_lst.setComponent(State_Type, 0, names, bcs, bndryfunc);
    StateDescriptor::setBndryFuncThreadSafety(true);

    auto add_derived = [] (const std::string& name, amrex::DeriveFuncFab f) {
        derive_lst.add(name, IndexType::TheCellType(), 1, f, the_same_box);
        derive_lst.addComponent(name, desc_lst, State_Type, 0, NCOMP);
    };
    add_derived("density",    der_density);
    add_derived("x_velocity", der_velocity<0>);
    add_derived("y_velocity", der_velocity<1>);
    add_derived("pressure",   der_pressure);
    add_derived("vorticity",  der_vorticity);
    add_derived("dg_sensor",  der_sensor);
}

void DGLevel::variableCleanUp ()
{
    desc_lst.clear();
    derive_lst.clear();
}
