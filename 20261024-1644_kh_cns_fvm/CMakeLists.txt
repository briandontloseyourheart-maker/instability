cmake_minimum_required(VERSION 3.20)
project(kh_cns_fvm VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# AMReX: either an installed 2D package (AMReX_DIR) or a source checkout.
set(AMREX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../amrex" CACHE PATH "AMReX source checkout")
option(KH_USE_OMP "Build with OpenMP" ON)
option(KH_USE_MPI "Build with MPI (run with mpiexec)" OFF)

if(AMReX_DIR)
  find_package(AMReX CONFIG REQUIRED COMPONENTS 2D EB AMRLEVEL)
else()
  if(NOT EXISTS "${AMREX_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "AMReX not found. Pass -DAMReX_DIR=<install>/lib/cmake/AMReX "
                        "or -DAMREX_SOURCE_DIR=<path to amrex checkout>")
  endif()
  # The CNS module is the AMReX EB-CNS solver; EB support is compiled in but
  # the Kelvin-Helmholtz box uses an all-regular geometry (no embedded walls).
  set(AMReX_SPACEDIM 2 CACHE STRING "Spatial dimension" FORCE)
  set(AMReX_EB ON CACHE BOOL "Embedded boundaries (needed by CNS)" FORCE)
  set(AMReX_AMRLEVEL ON CACHE BOOL "AmrLevel driver" FORCE)
  set(AMReX_FORTRAN OFF CACHE BOOL "C++ solver" FORCE)
  set(AMReX_MPI ${KH_USE_MPI} CACHE BOOL "MPI" FORCE)
  set(AMReX_OMP ${KH_USE_OMP} CACHE BOOL "OpenMP" FORCE)
  set(AMReX_GPU_BACKEND NONE CACHE STRING "CPU backend" FORCE)
  set(AMReX_PARTICLES OFF CACHE BOOL "No particles" FORCE)
  set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "Explicit solver" FORCE)
  add_subdirectory("${AMREX_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/amrex")
endif()

add_executable(kh_cns
  src/main.cpp src/diagnostics.cpp src/problem/cns_prob.cpp
  src/solver/CNS.cpp src/solver/CNSBld.cpp src/solver/CNS_advance.cpp
  src/solver/CNS_advance_box.cpp src/solver/CNS_advance_box_eb.cpp
  src/solver/CNS_bcfill.cpp src/solver/CNS_derive.cpp
  src/solver/CNS_init_eb2.cpp src/solver/CNS_io.cpp
  src/solver/CNS_parm.cpp src/solver/CNS_setup.cpp)
target_include_directories(kh_cns PRIVATE src src/solver src/problem src/solver/hydro src/solver/diffusion)
target_compile_features(kh_cns PRIVATE cxx_std_17)
target_link_libraries(kh_cns PRIVATE AMReX::amrex_2d)
set_target_properties(kh_cns PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME kh_cns_smoke
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_test.py"
            "$<TARGET_FILE:kh_cns>" "${CMAKE_CURRENT_SOURCE_DIR}/inputs/kh_test.inputs"
            "${CMAKE_BINARY_DIR}/tests/kh_smoke")
  set_tests_properties(kh_cns_smoke PROPERTIES TIMEOUT 600)
endif()
