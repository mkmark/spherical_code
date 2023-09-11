## torch
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(spherical_code)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test_cuda archive/test_cuda/example-app.cpp)
target_link_libraries(test_cuda "${TORCH_LIBRARIES}")
set_property(TARGET test_cuda PROPERTY CXX_STANDARD 17)

add_executable(SolverMinPotentialNaive release/v1.0.2/solver/SolverMinPotentialNaive.cpp)
target_link_libraries(SolverMinPotentialNaive)
set_property(TARGET SolverMinPotentialNaive PROPERTY CXX_STANDARD 17)

add_executable(SolverMinPotentialNaiveTorch src/solver/SolverMinPotentialNaiveTorch.cpp)
target_link_libraries(SolverMinPotentialNaiveTorch "${TORCH_LIBRARIES}")
set_property(TARGET SolverMinPotentialNaiveTorch PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET spherical_code
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:spherical_code>)
endif (MSVC)

## spherical code
include_directories(.)
