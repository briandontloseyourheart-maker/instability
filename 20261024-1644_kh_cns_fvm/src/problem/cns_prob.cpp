#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include <random>
#include "cns_prob_parm.H"
#include "CNS.H"

extern "C" {
    void amrex_probinit (const int* /*init*/,
                         const int* /*name*/,
                         const int* /*namelen*/,
                         const amrex_real* /*problo*/,
                         const amrex_real* /*probhi*/)
    {
        amrex::ParmParse pp("prob");
        ProbParm& p = *CNS::h_prob_parm;
        pp.query("rho_in", p.rho_in);
        pp.query("rho_out", p.rho_out);
        pp.query("v_in", p.v_in);
        pp.query("v_out", p.v_out);
        pp.query("p0", p.p0);
        pp.query("y_lo", p.y_lo);
        pp.query("y_hi", p.y_hi);
        pp.query("delta", p.delta);
        pp.query("w0", p.w0);
        pp.query("sigma", p.sigma);
        pp.query("lambda_x", p.lambda_x);
        pp.query("noise_amp", p.noise_amp);
        pp.query("noise_modes", p.noise_modes);
        if (p.noise_modes < 0 || p.noise_modes > 64) {
            amrex::Abort("prob.noise_modes must be between 0 and 64");
        }
        int seed = 12345;
        pp.query("noise_seed", seed);
        std::mt19937 gen(static_cast<unsigned>(seed));
        std::uniform_real_distribution<amrex::Real> dist(0.0, 2.0*3.14159265358979323846);
        for (int k = 0; k < p.noise_modes; ++k) { p.noise_phase[k] = dist(gen); }

        amrex::Gpu::copy(amrex::Gpu::hostToDevice, CNS::h_prob_parm, CNS::h_prob_parm+1,
                         CNS::d_prob_parm);
    }
}
