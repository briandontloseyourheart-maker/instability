
#include <CNS.H>

#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>
#include <AMReX_BCUtil.H>

using namespace amrex;

Real
CNS::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
    BL_PROFILE("CNS::advance()");

    for (int i = 0; i < num_state_data_types; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);
    MultiFab dSdt(grids,dmap,NUM_STATE,0,MFInfo(),Factory());
    MultiFab Sborder(grids,dmap,NUM_STATE,NUM_GROW,MFInfo(),Factory());

    MultiFab& C_new = get_new_data(Cost_Type);
    C_new.setVal(0.0);

    EBFluxRegister* fr_as_crse = nullptr;
    if (do_reflux && level < parent->finestLevel()) {
        CNS& fine_level = getLevel(level+1);
        fr_as_crse = &fine_level.flux_reg;
    }

    EBFluxRegister* fr_as_fine = nullptr;
    if (do_reflux && level > 0) {
        fr_as_fine = &flux_reg;
    }

    if (fr_as_crse) {
        fr_as_crse->reset();
    }

    dSdt.setVal(0.);

    if (rk_scheme == 104) {
        // Ketcheson's low-storage SSPRK(10,4): 10 stages, 4th order, SSP
        // coefficient 6, two registers (q1 = S_new, q2).
        // Every stage carries the same weight 1/10 in the final update, so each
        // stage adds its fluxes to the registers with weight dt/10.
        MultiFab q2(grids,dmap,NUM_STATE,0,MFInfo(),Factory());
        MultiFab::Copy(S_new, S_old, 0, 0, NUM_STATE, 0);
        MultiFab::Copy(q2,    S_old, 0, 0, NUM_STATE, 0);
        const Real c6 = Real(1.0)/Real(6.0);

        auto stage = [&] (Real tstage, Real coef)
        {
            fill_stage(Sborder, S_new, tstage);
            compute_dSdt(Sborder, dSdt, Real(0.1)*dt, fr_as_crse, fr_as_fine);
            MultiFab::Saxpy(S_new, coef*dt, dSdt, 0, 0, NUM_STATE, 0);
            computeTemp(S_new, 0);
        };

        for (int s = 0; s < 5; ++s) { stage(time + s*c6*dt, c6); }
        // q2 = q2/25 + 9 q1/25 ;  q1 = 15 q2 - 5 q1
        MultiFab::LinComb(q2, Real(1.0)/Real(25.0), q2, 0, Real(9.0)/Real(25.0), S_new, 0, 0, NUM_STATE, 0);
        MultiFab::LinComb(S_new, Real(15.0), q2, 0, Real(-5.0), S_new, 0, 0, NUM_STATE, 0);
        computeTemp(S_new, 0);
        for (int s = 5; s < 9; ++s) { stage(time + (s-3)*c6*dt, c6); }   // t = 1/3 .. 5/6
        // u^{n+1} = q2 + 3/5 q1 + dt/10 L(q1)
        fill_stage(Sborder, S_new, time + dt);
        compute_dSdt(Sborder, dSdt, Real(0.1)*dt, fr_as_crse, fr_as_fine);
        MultiFab::LinComb(S_new, Real(1.0), q2, 0, Real(0.6), S_new, 0, 0, NUM_STATE, 0);
        MultiFab::Saxpy(S_new, Real(0.1)*dt, dSdt, 0, 0, NUM_STATE, 0);
        computeTemp(S_new, 0);
        return dt;
    }

    // Original CNS scheme: 2-stage RK2 (Heun)
    // RK2 stage 1
    FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);

    compute_dSdt(Sborder, dSdt, 0.5*dt, fr_as_crse, fr_as_fine);

    // U^* = U^n + dt*dUdt^n
    MultiFab::LinComb(S_new, 1.0, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
    computeTemp(S_new,0);

    // RK2 stage 2
    // After fillpatch Sborder = U^n+dt*dUdt^n
    FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
    compute_dSdt(Sborder, dSdt, 0.5*dt, fr_as_crse, fr_as_fine);

    // S_new = 0.5*(Sborder+S_old) = U^n + 0.5*dt*dUdt^n
    MultiFab::LinComb(S_new, 0.5, Sborder, 0, 0.5, S_old, 0, 0, NUM_STATE, 0);

    // S_new += 0.5*dt*dSdt
    MultiFab::Saxpy(S_new, 0.5*dt, dSdt, 0, 0, NUM_STATE, 0);

    // We now have S_new = U^{n+1} = (U^n+0.5*dt*dUdt^n) + 0.5*dt*dUdt^*
    computeTemp(S_new,0);

    return dt;
}

// Ghost cells for a Runge-Kutta stage at time t: FillPatch supplies the
// coarse/fine ghost cells (coarse data interpolated to t); the valid region,
// same-level ghosts and physical-boundary ghosts are then rebuilt from the
// stage solution U itself.
void
CNS::fill_stage (MultiFab& Sborder, const MultiFab& U, Real t)
{
    FillPatch(*this, Sborder, NUM_GROW, t, State_Type, 0, NUM_STATE);
    MultiFab::Copy(Sborder, U, 0, 0, NUM_STATE, 0);
    Sborder.FillBoundary(geom.periodicity());
    const auto& bcs = desc_lst[State_Type].getBCs();
    amrex::FillDomainBoundary(Sborder, geom, bcs);
}

void
CNS::compute_dSdt (const MultiFab& S, MultiFab& dSdt, Real dt,
                   EBFluxRegister* fr_as_crse, EBFluxRegister* fr_as_fine)
{
    BL_PROFILE("CNS::compute_dSdt()");

    const Real* dx = geom.CellSize();
    const int ncomp = dSdt.nComp();

    int as_crse = (fr_as_crse != nullptr);
    int as_fine = (fr_as_fine != nullptr);

    MultiFab& cost = get_new_data(Cost_Type);

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    MFItInfo mfiinfo;
    if (Gpu::notInLaunchRegion()) {
        mfiinfo.EnableTiling(hydro_tile_size).SetDynamic(true);
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        std::array<FArrayBox,AMREX_SPACEDIM> flux;
        FArrayBox dm_as_fine(Box::TheUnitBox(),ncomp);
        FArrayBox fab_drho_as_crse(Box::TheUnitBox(),ncomp);
        IArrayBox fab_rrflag_as_crse(Box::TheUnitBox());

        for (MFIter mfi(S, mfiinfo); mfi.isValid(); ++mfi)
        {
            auto wt = amrex::second();

            const Box& bx = mfi.tilebox();

            const auto& flag = flags[mfi];

            if (flag.getType(bx) == FabType::covered) {
                dSdt[mfi].setVal<RunOn::Device>(0.0, bx, 0, ncomp);
            } else {

                // flux is used to store centroid flux needed for reflux
                for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                    flux[idim].resize(amrex::surroundingNodes(bx,idim),ncomp);
                    flux[idim].setVal<RunOn::Device>(0.);
                }

                if (flag.getType(amrex::grow(bx,1)) == FabType::regular)
                {
                    Array4<Real const>    s_arr =    S.array(mfi);
                    Array4<Real      > dsdt_arr = dSdt.array(mfi);

                    compute_dSdt_box(bx,s_arr,dsdt_arr,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])});

                    if (fr_as_crse) {
                        fr_as_crse->CrseAdd(mfi,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},dx,dt,RunOn::Device);
                    }

                    if (fr_as_fine) {
                        fr_as_fine->FineAdd(mfi,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},dx,dt,RunOn::Device);
                    }
                }
                else
                {
                    FArrayBox* p_drho_as_crse = (fr_as_crse) ?
                        fr_as_crse->getCrseData(mfi) : &fab_drho_as_crse;
                    const IArrayBox* p_rrflag_as_crse = (fr_as_crse) ?
                        fr_as_crse->getCrseFlag(mfi) : &fab_rrflag_as_crse;

                    if (fr_as_fine) {
                        dm_as_fine.resize(amrex::grow(bx,1),ncomp);
                    }

                    Array4<Real const> const&    s_arr =    S.array(mfi);
                    Array4<Real      > const& dsdt_arr = dSdt.array(mfi);

                    Array4<Real const> vf_arr = (*volfrac).array(mfi);
                    Array4<Real const> bcent_arr = (*bndrycent).array(mfi);

                    AMREX_D_TERM(Array4<Real const> const& apx = areafrac[0]->const_array(mfi);,
                                 Array4<Real const> const& apy = areafrac[1]->const_array(mfi);,
                                 Array4<Real const> const& apz = areafrac[2]->const_array(mfi));
                    AMREX_D_TERM(Array4<Real const> const& fcx = facecent[0]->const_array(mfi);,
                                 Array4<Real const> const& fcy = facecent[1]->const_array(mfi);,
                                 Array4<Real const> const& fcz = facecent[2]->const_array(mfi));

                    compute_dSdt_box_eb(bx,s_arr,dsdt_arr,
                                       {AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},
                                        flags.const_array(mfi), vf_arr,
                                        AMREX_D_DECL(apx,apy,apz),AMREX_D_DECL(fcx,fcy,fcz), bcent_arr,
                                        as_crse, p_drho_as_crse->array(), p_rrflag_as_crse->const_array(),
                                        as_fine, dm_as_fine.array(), level_mask.const_array(mfi), dt);

                    if (fr_as_crse) {
                        fr_as_crse->CrseAdd(mfi,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},
                                            dx,dt,(*volfrac)[mfi],
                                            {AMREX_D_DECL(&((*areafrac[0])[mfi]),
                                                          &((*areafrac[1])[mfi]),
                                                          &((*areafrac[2])[mfi]))},
                                            RunOn::Device);
                    }

                    if (fr_as_fine) {
                        fr_as_fine->FineAdd(mfi,{AMREX_D_DECL(&flux[0],&flux[1],&flux[2])},
                                            dx,dt,(*volfrac)[mfi],
                                            {AMREX_D_DECL(&((*areafrac[0])[mfi]),
                                                          &((*areafrac[1])[mfi]),
                                                          &((*areafrac[2])[mfi]))},
                                            dm_as_fine,
                                            RunOn::Device);
                    }
                }
            }

            Gpu::streamSynchronize();

            wt = (amrex::second() - wt) / bx.d_numPts();
            cost[mfi].plus<RunOn::Device>(wt, bx);
        }
    }
}
