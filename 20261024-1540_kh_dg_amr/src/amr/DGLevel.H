#ifndef DG_LEVEL_H_
#define DG_LEVEL_H_

// One AMR level of the discontinuous Galerkin Euler solver.
//
// The structure follows the AMReX CNS tutorial (AmrLevel + LevelBld,
// subcycling in time, flux registers for conservation), but every cell stores
// the full set of Legendre modes of rho, rho*u, rho*v and rho*E instead of a
// single cell average. The State_Type MultiFab therefore has
// dg::NCOMP = 4 * (DG_ORDER)^2 components (64 for 4th order).

#include "DG_Basis.H"
#include "DG_Euler.H"
#include "DG_Kernels.H"
#include "KH_Problem.H"

#include <AMReX_AmrLevel.H>
#include <AMReX_FluxRegister.H>

#include <array>
#include <string>

class DGLevel : public amrex::AmrLevel
{
public:
    DGLevel ();
    DGLevel (amrex::Amr& papa, int lev, const amrex::Geometry& level_geom,
             const amrex::BoxArray& bl, const amrex::DistributionMapping& dm,
             amrex::Real time);
    ~DGLevel () override;

    DGLevel (const DGLevel&) = delete;
    DGLevel (DGLevel&&) = delete;
    DGLevel& operator= (const DGLevel&) = delete;
    DGLevel& operator= (DGLevel&&) = delete;

    void restart (amrex::Amr& papa, std::istream& is, bool bReadSpecial = false) override;

    std::string thePlotFileType () const override { return {"HyperCLaw-V1.1"}; }

    void init (amrex::AmrLevel& old) override;   // regrid: new level from old one
    void init () override;                        // regrid: brand-new level
    void initData () override;                    // t = 0 initial condition

    amrex::Real advance (amrex::Real time, amrex::Real dt, int iteration, int ncycle) override;

    void computeInitialDt (int finest_level, int sub_cycle, amrex::Vector<int>& n_cycle,
                           const amrex::Vector<amrex::IntVect>& ref_ratio,
                           amrex::Vector<amrex::Real>& dt_level, amrex::Real stop_time) override;

    void computeNewDt (int finest_level, int sub_cycle, amrex::Vector<int>& n_cycle,
                       const amrex::Vector<amrex::IntVect>& ref_ratio,
                       amrex::Vector<amrex::Real>& dt_min, amrex::Vector<amrex::Real>& dt_level,
                       amrex::Real stop_time, int post_regrid_flag) override;

    void post_regrid (int lbase, int new_finest) override;
    void post_timestep (int iteration) override;
    void postCoarseTimeStep (amrex::Real time) override;
    void post_init (amrex::Real stop_time) override;
    void post_restart () override;

    void errorEst (amrex::TagBoxArray& tags, int clearval, int tagval,
                   amrex::Real time, int n_error_buf = 0, int ngrow = 0) override;

    static void variableSetUp ();
    static void variableCleanUp ();

    amrex::Real estTimeStep ();

    // L2 error of the density polynomial against the exact density wave
    // (prob.type = 1), evaluated at the Gauss points of this level.
    amrex::Real density_wave_error (amrex::Real time) const;

    // Shared parameters (read from the "dg." and "prob." namespaces).
    static dg::EulerParm   euler_parm;
    static dg::LimiterParm limiter_parm;
    static kh::ProbParm    prob_parm;

    enum StateType { State_Type = 0, NUM_STATE_TYPE };
    static constexpr int NUM_GROW = 1;   // DG only needs the face neighbour

private:
    static void read_params ();

    DGLevel& getLevel (int lev) { return dynamic_cast<DGLevel&>(parent->getLevel(lev)); }

    void define_flux_register ();

    // Ghost-filled copy of the stage solution U at stage time t.
    void fill_stage (amrex::MultiFab& Sborder, const amrex::MultiFab& U, amrex::Real t);

    // dU/dt for all modes; also returns face fluxes for refluxing.
    void compute_rhs (const amrex::MultiFab& Sborder, amrex::MultiFab& dUdt,
                      std::array<amrex::MultiFab,AMREX_SPACEDIM>& flux,
                      std::array<amrex::MultiFab,AMREX_SPACEDIM>* flux_crse_basis);

    void apply_limiter (amrex::MultiFab& S);
    void reflux ();
    void avgDown ();
    void write_diagnostics (amrex::Real time, bool header);

    amrex::FluxRegister flux_reg;   // coarse/fine interface with level-1

    static int  verbose;
    static amrex::Real cfl;
    static int  rk_order;
    static int  do_reflux;
    static amrex::Real refine_dengrad;
    static amrex::Real refine_vort;
    static amrex::Vector<amrex::Real> refine_box;   // optional static region: xlo ylo xhi yhi
    static std::string diag_file;
    static amrex::BCRec phys_bc;
};

void dg_bcfill (amrex::Box const& bx, amrex::FArrayBox& data, int dcomp, int numcomp,
                amrex::Geometry const& geom, amrex::Real time,
                const amrex::Vector<amrex::BCRec>& bcr, int bcomp, int scomp);

#endif
