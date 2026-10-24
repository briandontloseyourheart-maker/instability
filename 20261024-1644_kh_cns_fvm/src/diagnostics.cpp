#include "diagnostics.H"
#include "CNS.H"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <fstream>
#include <iomanip>
#include <limits>

void observe (amrex::Amr& amr, bool initial)
{
    using namespace amrex;
    // sums: mass, x-mom, y-mom, energy; extrema: rho_min, rho_max, p_min
    std::array<Real,4> tot{};
    Real rmin = std::numeric_limits<Real>::max(), rmax = -rmin, pmin = rmin;
    const Real gm1 = CNS::h_parm->eos_gamma - 1.0;
    Vector<Long> cells(amr.maxLevel()+1, 0);

    for (int l = 0; l <= amr.finestLevel(); ++l) {
        auto& lev = amr.getLevel(l);
        const MultiFab& s = lev.get_new_data(CNS::conservedStateType());
        const auto dx = lev.Geom().CellSizeArray();
        const Real dv = dx[0]*dx[1];
        iMultiFab mask(s.boxArray(), s.DistributionMap(), 1, 0);
        mask.setVal(1);
        if (l < amr.finestLevel()) {
            // 1 = leaf cell, 0 = covered by the finer level
            mask = makeFineMask(s, amr.boxArray(l+1), amr.refRatio(l), 1, 0);
        }
        cells[l] = s.boxArray().numPts();
        Real t0 = 0, t1 = 0, t2 = 0, t3 = 0;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:t0,t1,t2,t3) reduction(min:rmin,pmin) reduction(max:rmax)
#endif
        for (MFIter mfi(s); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto const a = s.const_array(mfi);
            auto const m = mask.const_array(mfi);
            amrex::Loop(bx, [&] (int i, int j, int k) {
                const Real rho = a(i,j,k,URHO);
                const Real ke  = 0.5*(a(i,j,k,UMX)*a(i,j,k,UMX) + a(i,j,k,UMY)*a(i,j,k,UMY))/rho;
                const Real p   = gm1*(a(i,j,k,UEDEN) - ke);
                rmin = std::min(rmin, rho); rmax = std::max(rmax, rho); pmin = std::min(pmin, p);
                if (m(i,j,k)) {
                    t0 += rho*dv; t1 += a(i,j,k,UMX)*dv;
                    t2 += a(i,j,k,UMY)*dv; t3 += a(i,j,k,UEDEN)*dv;
                }
            });
        }
        tot[0] += t0; tot[1] += t1; tot[2] += t2; tot[3] += t3;
    }
    ParallelDescriptor::ReduceRealSum(tot.data(), 4);
    ParallelDescriptor::ReduceRealMin(rmin);
    ParallelDescriptor::ReduceRealMin(pmin);
    ParallelDescriptor::ReduceRealMax(rmax);

    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out("diagnostics.csv", initial ? std::ios::trunc : std::ios::app);
    if (initial) {
        out << "step,time,mass,x_momentum,y_momentum,energy,rho_min,rho_max,p_min,finest_level";
        for (int l = 0; l <= amr.maxLevel(); ++l) { out << ",cells_lev" << l; }
        out << "\n";
    }
    out << std::setprecision(17) << amr.levelSteps(0) << ',' << amr.cumTime() << ','
        << tot[0] << ',' << tot[1] << ',' << tot[2] << ',' << tot[3] << ','
        << rmin << ',' << rmax << ',' << pmin << ',' << amr.finestLevel();
    for (int l = 0; l <= amr.maxLevel(); ++l) { out << ',' << cells[l]; }
    out << "\n";
}
