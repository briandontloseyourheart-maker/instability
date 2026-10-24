#ifndef DG_INTERP_H_
#define DG_INTERP_H_

// Moving DG data between AMR levels.
//
// Both directions are exact L2 projections of polynomials:
//  * coarse -> fine (new fine patches, ghost cells at coarse/fine boundaries):
//    the coarse polynomial restricted to a fine cell is still a polynomial of
//    the same degree, so it is represented exactly by the fine modes.
//  * fine -> coarse (average down): the coarse modes are the L2 projection of
//    the piecewise fine polynomial. Mode 0 is the volume average, so mass,
//    momentum and energy are conserved to round-off.
// No limiting or reconstruction is involved.

#include "DG_Basis.H"

#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>

namespace dg {

class DGInterp final : public amrex::Interpolater
{
public:
    amrex::Box CoarseBox (const amrex::Box& fine, int ratio) override;
    amrex::Box CoarseBox (const amrex::Box& fine, const amrex::IntVect& ratio) override;

    void interp (const amrex::FArrayBox& crse, int crse_comp,
                 amrex::FArrayBox& fine, int fine_comp, int ncomp,
                 const amrex::Box& fine_region, const amrex::IntVect& ratio,
                 const amrex::Geometry& crse_geom, const amrex::Geometry& fine_geom,
                 amrex::Vector<amrex::BCRec> const& bcr,
                 int actual_comp, int actual_state, amrex::RunOn runon) override;
};

extern DGInterp dg_interp;

// Project fine-level DG data onto the coarse level where they overlap.
void average_down (const amrex::MultiFab& fine, amrex::MultiFab& crse,
                   const amrex::Geometry& crse_geom, const amrex::IntVect& ratio);

} // namespace dg

#endif
