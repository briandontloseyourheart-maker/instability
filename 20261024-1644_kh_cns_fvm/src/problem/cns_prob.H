#ifndef CNS_PROB_H_
#define CNS_PROB_H_

// Kelvin-Helmholtz initial condition for the AMReX CNS solver (2D).
// A dense band (rho_in, v_in) between y_lo and y_hi slides past lighter fluid
// (rho_out, v_out) at uniform pressure; a sinusoidal v_y seeds the instability.
// Each cell gets the average of 4x4 sub-samples so thin interfaces are
// represented conservatively on every AMR level.

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <cmath>
#include "CNS_index_macros.H"
#include "CNS_parm.H"
#include "cns_prob_parm.H"

AMREX_GPU_HOST_DEVICE inline
void kh_primitive (amrex::Real x, amrex::Real y, ProbParm const& pp, amrex::Real W[4]) noexcept
{
    using amrex::Real;
    const Real pi = Real(3.14159265358979323846);
    Real f;
    if (pp.delta > Real(0.0)) {
        f = Real(0.5)*(std::tanh((y - pp.y_lo)/pp.delta) - std::tanh((y - pp.y_hi)/pp.delta));
    } else {
        f = (y > pp.y_lo && y < pp.y_hi) ? Real(1.0) : Real(0.0);
    }
    const Real s2  = Real(2.0)*pp.sigma*pp.sigma;
    const Real env = std::exp(-(y - pp.y_lo)*(y - pp.y_lo)/s2) + std::exp(-(y - pp.y_hi)*(y - pp.y_hi)/s2);
    Real vy = pp.w0*std::sin(Real(2.0)*pi*x/pp.lambda_x)*env;
    for (int k = 0; k < pp.noise_modes; ++k) {
        vy += pp.noise_amp/std::sqrt(Real(pp.noise_modes))
            * std::sin(Real(2.0)*pi*Real(k+1)*x + pp.noise_phase[k])*env;
    }
    W[0] = pp.rho_out + (pp.rho_in - pp.rho_out)*f;
    W[1] = pp.v_out   + (pp.v_in   - pp.v_out  )*f;
    W[2] = vy;
    W[3] = pp.p0;
}

AMREX_GPU_DEVICE
inline
void
cns_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm, ProbParm const& pp)
{
    using amrex::Real;
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx      = geomdata.CellSize();

    constexpr int ns = 4;
    Real rho = 0, mx = 0, my = 0, E = 0;
    for (int sy = 0; sy < ns; ++sy) {
        const Real y = prob_lo[1] + (j + (sy + Real(0.5))/ns)*dx[1];
        for (int sx = 0; sx < ns; ++sx) {
            const Real x = prob_lo[0] + (i + (sx + Real(0.5))/ns)*dx[0];
            Real W[4];
            kh_primitive(x, y, pp, W);
            rho += W[0];
            mx  += W[0]*W[1];
            my  += W[0]*W[2];
            E   += W[3]/(parm.eos_gamma - Real(1.0)) + Real(0.5)*W[0]*(W[1]*W[1] + W[2]*W[2]);
        }
    }
    const Real inv = Real(1.0)/(ns*ns);
    rho *= inv; mx *= inv; my *= inv; E *= inv;

    const Real ei = E - Real(0.5)*(mx*mx + my*my)/rho;   // rho * e
    state(i,j,k,URHO ) = rho;
    state(i,j,k,UMX  ) = mx;
    state(i,j,k,UMY  ) = my;
    state(i,j,k,UEDEN) = E;
    state(i,j,k,UEINT) = ei;
    state(i,j,k,UTEMP) = ei/(rho*parm.cv);
}

#endif
