#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include "cns_prob_parm.H"
#include "CNS.H"

extern "C" {
    void amrex_probinit (const int* /*init*/,
                         const int* /*name*/,
                         const int* /*namelen*/,
                         const amrex_real* /*problo*/,
                         const amrex_real* /*probhi*/)
    {
        // prob.type, prob.x0, prob.y0 and per-quadrant states
        // prob.rho = r_UR r_UL r_LL r_LR  (same for prob.u, prob.v, prob.p)
        amrex::ParmParse pp("prob");
        ProbParm& p = *CNS::h_prob_parm;
        pp.query("type", p.type);
        pp.query("x0", p.x0);
        pp.query("y0", p.y0);
        amrex::Vector<amrex::Real> tmp;
        if (pp.queryarr("rho", tmp, 0, 4)) { for (int q = 0; q < 4; ++q) { p.rho[q] = tmp[q]; } }
        if (pp.queryarr("u",   tmp, 0, 4)) { for (int q = 0; q < 4; ++q) { p.u[q]   = tmp[q]; } }
        if (pp.queryarr("v",   tmp, 0, 4)) { for (int q = 0; q < 4; ++q) { p.v[q]   = tmp[q]; } }
        if (pp.queryarr("p",   tmp, 0, 4)) { for (int q = 0; q < 4; ++q) { p.p[q]   = tmp[q]; } }

        amrex::Gpu::copy(amrex::Gpu::hostToDevice, CNS::h_prob_parm, CNS::h_prob_parm+1,
                         CNS::d_prob_parm);
    }
}
