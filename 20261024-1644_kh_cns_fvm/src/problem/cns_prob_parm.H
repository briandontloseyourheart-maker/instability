#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

// Kelvin-Helmholtz parameters (prob.* in the inputs file).
struct ProbParm
{
    amrex::Real rho_in    = 2.0;     // density of the central band
    amrex::Real rho_out   = 1.0;     // density outside
    amrex::Real v_in      = 0.5;     // x-velocity of the band
    amrex::Real v_out     = -0.5;    // x-velocity outside
    amrex::Real p0        = 2.5;     // uniform pressure
    amrex::Real y_lo      = 0.25;    // lower interface
    amrex::Real y_hi      = 0.75;    // upper interface
    amrex::Real delta     = 0.0035;  // tanh width of the interfaces (0 = sharp)
    amrex::Real w0        = 0.1;     // amplitude of the v_y seed
    amrex::Real sigma     = 0.05/1.4142135623730951;  // vertical extent of the seed
    amrex::Real lambda_x  = 0.5;     // wavelength of the seed
    amrex::Real noise_amp = 0.0;     // extra multi-mode seed
    int         noise_modes = 0;
    amrex::Real noise_phase[64] = {};
};

#endif
