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
  test_goodness_of_fit
  test_mcmc_extra
  test_multiple_grubbs_beck
  test_matrix_regularization
  test_stratification
  test_numerical_diff
  test_model_parameter
  test_model_base
  test_quantile_prior
  test_univariate_distribution_model
  test_maximum_likelihood
  test_maximum_a_posteriori
  test_bayesian_analysis
  test_data_types
  test_data_frame
  test_plotting_positions_df
  test_threshold_diagnostics
  test_trend_functions
)

# Sobol test takes the direction-number file as its sole argument.
add_executable(test_sobol tests/test_sobol.cpp)
target_link_libraries(test_sobol PRIVATE bestfit_core)
target_include_directories(test_sobol PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(test_sobol PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wextra>)
add_test(NAME test_sobol
         COMMAND test_sobol "${CMAKE_CURRENT_SOURCE_DIR}/data/new-joe-kuo-6.21201")

foreach(t ${BF_TESTS})
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE bestfit_core)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_options(${t} PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wextra>)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# Generic fixture-driven runner: reads the canonical fixtures/ shared by all languages.
add_executable(test_fixtures tests/test_fixtures.cpp)
target_link_libraries(test_fixtures PRIVATE bestfit_core)
target_include_directories(test_fixtures PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_test(NAME test_fixtures
         COMMAND test_fixtures ${CMAKE_CURRENT_SOURCE_DIR}/../fixtures)
