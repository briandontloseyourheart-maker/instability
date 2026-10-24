#pragma once
#include <AMReX_Amr.H>

// Append one line per coarse step to diagnostics.csv: composite (leaf-cell)
// totals of mass, momentum and energy plus density/pressure extrema.
void observe (amrex::Amr& amr, bool initial);
