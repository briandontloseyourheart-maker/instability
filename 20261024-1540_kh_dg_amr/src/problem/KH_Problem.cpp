#include "KH_Problem.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <random>

namespace kh {

void read_prob_parm (ProbParm& pp)
{
    amrex::ParmParse p("prob");
    p.query("type",     pp.type);
    p.query("rho_in",   pp.rho_in);
    p.query("rho_out",  pp.rho_out);
    p.query("v_in",     pp.v_in);
    p.query("v_out",    pp.v_out);
    p.query("p0",       pp.p0);
    p.query("y_lo",     pp.y_lo);
    p.query("y_hi",     pp.y_hi);
    p.query("delta",    pp.delta);
    p.query("w0",       pp.w0);
    p.query("sigma",    pp.sigma);
    p.query("lambda_x", pp.lambda_x);
    p.query("noise_amp",   pp.noise_amp);
    p.query("noise_modes", pp.noise_modes);
    if (pp.noise_modes < 0 || pp.noise_modes > MAX_NOISE_MODES) {
        amrex::Abort("prob.noise_modes must be between 0 and 64");
    }
    int seed = 12345;
    p.query("noise_seed", seed);
    std::mt19937 gen(static_cast<unsigned>(seed));
    std::uniform_real_distribution<Real> dist(0.0, 2.0*3.14159265358979323846);
    for (int k = 0; k < pp.noise_modes; ++k) { pp.noise_phase[k] = dist(gen); }
}

} // namespace kh

// Called by amrex::Amr::init() before the initial data are built.
#include "DGLevel.H"
#include <AMReX_PROB_AMR_F.H>

extern "C" {
void amrex_probinit (const int* /*init*/, const int* /*name*/, const int* /*namelen*/,
                     const amrex_real* /*problo*/, const amrex_real* /*probhi*/)
{
    kh::read_prob_parm(DGLevel::prob_parm);
}
}
