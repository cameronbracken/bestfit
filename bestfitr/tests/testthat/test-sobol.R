# Identity test for the Sobol quasi-random sequence.
# Pins the R binding to the canonical first-10 points for dimension 2.
# Oracle values come from upstream/Numerics/Test_Numerics/Sampling/Test_SobolSequence.cs
# (Test_Sobol), which are themselves verified against R's randtoolbox::sobol().

ns <- asNamespace("bestfitr")

# Locate the direction-numbers file installed with the package.
path <- system.file("extdata", "new-joe-kuo-6.21201", package = "bestfitr")
if (!nzchar(path)) stop("direction-numbers file not found in bestfitr inst/extdata")

test_that("sobol dimension-2 first 10 points match C# oracle", {
  mat <- ns$bf_sobol_generate_(2L, 10L, path)
  expected <- rbind(
    c(0.5000, 0.5000),
    c(0.7500, 0.2500),
    c(0.2500, 0.7500),
    c(0.3750, 0.3750),
    c(0.8750, 0.8750),
    c(0.6250, 0.1250),
    c(0.1250, 0.6250),
    c(0.1875, 0.3125),
    c(0.6875, 0.8125),
    c(0.9375, 0.0625)
  )
  for (i in seq_len(nrow(expected))) {
    for (j in seq_len(ncol(expected))) {
      expect_equal(mat[i, j], expected[i, j], tolerance = 0,
                   label = paste0("point[", i, ",", j, "]"))
    }
  }
})

test_that("sobol dimension-1 first point is 0.5", {
  # Dimension 1 needs no file (unit initialization only).
  mat <- ns$bf_sobol_generate_(1L, 1L, "")
  expect_equal(mat[1, 1], 0.5, tolerance = 0)
})

test_that("sobol skip_to(1) matches first sequential point", {
  first_seq <- ns$bf_sobol_generate_(2L, 1L, path)[1, ]
  first_skip <- ns$bf_sobol_skip_to_(2L, 1L, path)
  expect_equal(first_skip, first_seq, tolerance = 0)
})

test_that("sobol skip_to(5) matches 5th sequential point", {
  fifth_seq <- ns$bf_sobol_generate_(2L, 5L, path)[5, ]
  fifth_skip <- ns$bf_sobol_skip_to_(2L, 5L, path)
  expect_equal(fifth_skip, fifth_seq, tolerance = 0)
})
