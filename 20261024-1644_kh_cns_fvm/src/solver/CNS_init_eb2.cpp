#include <AMReX_EB2.H>
void initialize_EB2(const amrex::Geometry& geom, int required, int maximum) {
    amrex::EB2::Build(geom, required, maximum, 4);
}
