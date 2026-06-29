# Dev-only build for the bestfit C++ core test suite.
# The shipped R/Python packages compile the vendored core via Makevars / scikit-build;
# this file exists so contributors (and CI) can build and run the C++ tests directly.
cmake_minimum_required(VERSION 3.15)
project(bestfit_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(bestfit_core INTERFACE)
target_include_directories(bestfit_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(BF_TESTS
  test_mersenne_twister
  test_generalized_extreme_value
  test_gev_estimation
)

foreach(t ${BF_TESTS})
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE bestfit_core)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_options(${t} PRIVATE -Wall -Wextra)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
