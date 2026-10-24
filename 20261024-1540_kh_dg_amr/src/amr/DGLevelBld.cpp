#include "DGLevel.H"

#include <AMReX_LevelBld.H>

using namespace amrex;

// Factory used by amrex::Amr to create DGLevel objects.
class DGLevelBld : public LevelBld
{
    void variableSetUp () override { DGLevel::variableSetUp(); }
    void variableCleanUp () override { DGLevel::variableCleanUp(); }
    AmrLevel* operator() () override { return new DGLevel; }
    AmrLevel* operator() (Amr& papa, int lev, const Geometry& level_geom, const BoxArray& ba,
                          const DistributionMapping& dm, Real time) override
    {
        return new DGLevel(papa, lev, level_geom, ba, dm, time);
    }
};

namespace { DGLevelBld dg_level_bld; }

LevelBld* getLevelBld () { return &dg_level_bld; }
