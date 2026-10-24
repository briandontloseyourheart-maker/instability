#ifndef KH_PROBLEM_H_
#define KH_PROBLEM_H_

// Kelvin-Helmholtz initial condition on the periodic unit square.
//
// A dense band (density rho_in, x-velocity v_in) occupies
// y_lo < y < y_hi and slides past the lighter fluid (rho_out, v_out).
// Pressure is uniform. A single-mode vertical velocity perturbation with
// wavelength lambda_x seeds the instability at both interfaces; optional
// multi-mode noise seeds the secondary billows.
// The interfaces can be smoothed with a tanh profile of width delta
// (delta = 0 gives a sharp step).

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>

namespace kh {

using amrex::Real;

constexpr int MAX_NOISE_MODES = 64;

struct ProbParm
{
    int  type     = 0;       // 0: Kelvin-Helmholtz, 1: smooth density wave (accuracy test)
    Real rho_in   = 2.0;
    Real rho_out  = 1.0;
    Real v_in     = 0.5;
    Real v_out    = -0.5;
    Real p0       = 2.5;
    Real y_lo     = 0.25;    // lower interface
    Real y_hi     = 0.75;    // upper interface
    Real delta    = 0.0035;  // tanh smoothing width of the interfaces
    Real w0       = 0.1;     // amplitude of the seed perturbation in v_y
    Real sigma    = 0.05/1.4142135623730951;   // vertical extent of the seed
    Real lambda_x = 0.5;     // wavelength of the seed (two waves in the box)
    Real noise_amp = 0.0;    // amplitude of extra multi-mode seed
    int  noise_modes = 0;
    Real noise_phase[MAX_NOISE_MODES] = {};
};

// Smooth density wave advected diagonally with u = v = 1 and uniform pressure.
// The exact solution at time t is the initial profile shifted by (t, t);
// used to verify the order of accuracy of the DG scheme.
AMREX_GPU_HOST_DEVICE inline
void density_wave (Real x, Real y, Real t, Real W[4]) noexcept
{
    const Real pi = 3.14159265358979323846;
    W[0] = 1.0 + 0.2*std::sin(2.0*pi*(x + y - 2.0*t));
    W[1] = 1.0;
    W[2] = 1.0;
    W[3] = 1.0;
}

// Primitive state (rho, u, v, p) at position (x,y).
AMREX_GPU_HOST_DEVICE inline
void initial_primitive (Real x, Real y, const ProbParm& pp, Real W[4]) noexcept
{
    const Real pi = 3.14159265358979323846;
    if (pp.type == 1) { density_wave(x, y, 0.0, W); return; }
    Real f;   // 1 inside the band, 0 outside
    if (pp.delta > 0.0) {
        f = 0.5*(std::tanh((y - pp.y_lo)/pp.delta) - std::tanh((y - pp.y_hi)/pp.delta));
    } else {
        f = (y > pp.y_lo && y < pp.y_hi) ? 1.0 : 0.0;
    }
    const Real env = std::exp(-(y - pp.y_lo)*(y - pp.y_lo)/(2.0*pp.sigma*pp.sigma))
                   + std::exp(-(y - pp.y_hi)*(y - pp.y_hi)/(2.0*pp.sigma*pp.sigma));
    Real vy = pp.w0*std::sin(2.0*pi*x/pp.lambda_x)*env;
    for (int k = 0; k < pp.noise_modes; ++k) {
        vy += pp.noise_amp/std::sqrt(static_cast<Real>(pp.noise_modes))
            * std::sin(2.0*pi*(k+1)*x + pp.noise_phase[k])*env;
    }
    W[0] = pp.rho_out + (pp.rho_in - pp.rho_out)*f;
    W[1] = pp.v_out   + (pp.v_in   - pp.v_out  )*f;
    W[2] = vy;
    W[3] = pp.p0;
}

// Reads the prob.* parameters (KH_Problem.cpp).
void read_prob_parm (ProbParm& pp);

} // namespace kh

#endif
