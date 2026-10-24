// 2D Kelvin-Helmholtz instability with a high-order discontinuous Galerkin
// method and adaptive mesh refinement (AMReX AmrLevel framework).
#include "DGLevel.H"

#include <AMReX.H>
#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>

#include <iomanip>

using namespace amrex;

LevelBld* getLevelBld ();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        const double t_start = amrex::second();

        int  max_step  = -1;
        Real strt_time = 0.0;
        Real stop_time = -1.0;
        {
            ParmParse pp;
            pp.query("max_step", max_step);
            pp.query("strt_time", strt_time);
            pp.query("stop_time", stop_time);
        }
        if (max_step < 0 && stop_time < 0.0) {
            amrex::Abort("Set max_step and/or stop_time in the inputs file");
        }

        amrex::Print() << "Kelvin-Helmholtz DG-AMR: polynomial degree " << dg::P
                       << " (order " << dg::NP << "), " << dg::NCOMP
                       << " modal components per cell\n";

        Amr amr(getLevelBld());
        amr.init(strt_time, stop_time);

        while (amr.okToContinue() &&
               (amr.levelSteps(0) < max_step || max_step < 0) &&
               (amr.cumTime() < stop_time || stop_time < 0.0))
        {
            amr.coarseTimeStep(stop_time);
        }

        if (DGLevel::prob_parm.type == 1) {
            auto& lev0 = dynamic_cast<DGLevel&>(amr.getLevel(0));
            amrex::Print() << std::setprecision(10) << "[DG] density-wave L2 error = "
                           << lev0.density_wave_error(amr.cumTime()) << "\n";
        }

        // Final plotfile / checkpoint if the last step did not write one.
        if (amr.stepOfLastPlotFile() < amr.levelSteps(0)) {
            amr.writePlotFile();
        }
        if (amr.stepOfLastCheckPoint() < amr.levelSteps(0)) {
            ParmParse pa("amr");
            int write_check = 1;
            pa.query("checkpoint_files_output", write_check);
            if (write_check) { amr.checkPoint(); }
        }

        amrex::Print() << "Run time = " << amrex::second() - t_start << " s\n";
    }
    amrex::Finalize();
    return 0;
}
