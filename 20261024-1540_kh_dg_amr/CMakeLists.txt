cmake_minimum_required(VERSION 3.20)
project(kh_dg_amr VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- DG discretisation (compile time) -------------------------------------
set(DG_ORDER 4 CACHE STRING "Order of accuracy of the DG scheme (polynomial degree + 1)")
set(DG_NQUAD "" CACHE STRING "Gauss points per direction (default: DG_ORDER)")
if(NOT DG_NQUAD)
  set(DG_NQUAD ${DG_ORDER})
endif()

# ---- AMReX -------------------------------------------------------------------
# Either point AMReX_DIR at an installed 2D AMReX (with AMRLEVEL), or let this
# project build AMReX from a source checkout (AMREX_SOURCE_DIR).
set(AMREX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../amrex" CACHE PATH "AMReX source checkout")
option(KH_USE_OMP "Build with OpenMP" ON)
option(KH_NATIVE "Optimise for the build machine (-march=native)" OFF)

if(AMReX_DIR)
  find_package(AMReX CONFIG REQUIRED COMPONENTS 2D AMRLEVEL)
else()
  if(NOT EXISTS "${AMREX_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "AMReX not found. Pass -DAMReX_DIR=<install>/lib/cmake/AMReX "
                        "or -DAMREX_SOURCE_DIR=<path to amrex checkout>")
  endif()
  set(AMReX_SPACEDIM 2 CACHE STRING "Spatial dimension" FORCE)
  set(AMReX_AMRLEVEL ON CACHE BOOL "AmrLevel driver" FORCE)
  set(AMReX_EB OFF CACHE BOOL "No embedded boundaries" FORCE)
  set(AMReX_FORTRAN OFF CACHE BOOL "C++ only" FORCE)
  set(AMReX_MPI OFF CACHE BOOL "Serial (OpenMP) build" FORCE)
  set(AMReX_OMP ${KH_USE_OMP} CACHE BOOL "OpenMP" FORCE)
  set(AMReX_GPU_BACKEND NONE CACHE STRING "CPU backend" FORCE)
  set(AMReX_PARTICLES OFF CACHE BOOL "No particles" FORCE)
  set(AMReX_LINEAR_SOLVERS OFF CACHE BOOL "Explicit solver" FORCE)
  add_subdirectory("${AMREX_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/amrex")
endif()

# ---- Executable ---------------------------------------------------------------
add_executable(kh_dg
  src/main.cpp
  src/dg/DG_Basis.cpp
  src/dg/DG_Interp.cpp
  src/amr/DGLevel.cpp
  src/amr/DGAdvance.cpp
  src/amr/DGSetup.cpp
  src/amr/DGLevelBld.cpp
  src/problem/KH_Problem.cpp)
target_include_directories(kh_dg PRIVATE src/dg src/amr src/problem)
target_compile_definitions(kh_dg PRIVATE DG_ORDER=${DG_ORDER} DG_NQUAD=${DG_NQUAD})
target_compile_features(kh_dg PRIVATE cxx_std_17)
target_link_libraries(kh_dg PRIVATE AMReX::amrex_2d)
if(KH_NATIVE)
  target_compile_options(kh_dg PRIVATE -march=native)
endif()
set_target_properties(kh_dg PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---- Tests --------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME kh_smoke
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke_test.py"
            "$<TARGET_FILE:kh_dg>" "${CMAKE_CURRENT_SOURCE_DIR}/inputs/kh_test.inputs"
            "${CMAKE_BINARY_DIR}/tests/kh_smoke")
  set_tests_properties(kh_smoke PROPERTIES TIMEOUT 600)
endif()
