# Binding-specific behaviour (vectorisation, return shapes). Oracle-value checks
# live in test-fixtures.R, driven by the shared language-neutral fixtures.

test_that("density, distribution, quantile are vectorised", {
  x <- c(90, 100, 110)
  expect_length(dgev(x, 100, 10, 0), 3L)
  expect_equal(dgev(x, 100, 10, 0)[2], dgev(100, 100, 10, 0))
  expect_length(pgev(x, 100, 10, 0), 3L)
})

test_that("gev_fit returns a named location/scale/shape vector", {
  fit <- gev_fit(c(5, 7, 3, 9, 6, 8, 4, 11, 2, 10), method = "lmom")
  expect_named(fit, c("location", "scale", "shape"))
})
