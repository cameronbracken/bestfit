# Oracle fixtures

Language-neutral validation data — the **single source of truth** for the expected
values that the C++ core, the R package (`bestfitr`), and the Python package
(`bestfitpy`) all check against. Each language has a thin generic *runner* that loads
these JSON files and applies every assertion; no oracle values are duplicated in any
test file. This is what makes "validate identically" structural rather than aspirational.

Oracle values trace to the upstream C# unit tests (and through them to published
references such as Rao & Hamed, *Flood Frequency Analysis*). They are curated from those
tests and then **verified reproducible against the real Numerics library** by
`tools/verify_oracles.py` (which runs the C# `tools/oracle_emitter` project). That gate is
dev-only — it needs `dotnet` and the `upstream/Numerics` submodule — and confirms every
expected value still reproduces to its stated tolerance. A scripted extractor to harvest
the C# test literals *en masse* is planned for the bulk distribution port.

`tools/sync_fixtures.py` copies this directory into each package
(`bestfitr/inst/fixtures`, `bestfitpy/src/bestfitpy/fixtures`) so each ships its own
verbatim copy; a CI `--check` run fails on drift.

## Schema

### `univariate_distribution`

```jsonc
{
  "target":  "GeneralizedExtremeValue",     // which model/component
  "kind":    "univariate_distribution",     // runner family
  "source":  "Numerics/.../Test_*.cs",      // provenance
  "datasets": { "name": [ ...numbers... ] }, // samples referenced by fit cases
  "cases": [
    {
      "name": "default_moments",
      "construct": { "params": [100, 10, 0] },              // OR
      "construct": { "fit": { "dataset": "white_river", "method": "mle" } },
      "assertions": [
        { "method": "mean", "args": [], "expected": 105.77, "mode": "abs", "tol": 1e-6 }
      ]
    }
  ]
}
```

### `special_function`

For internal C++ math utilities (not exposed to R/Python). The R and Python fixture runners
skip files with this kind; only the C++ runner and the dotnet oracle gate process them.

```jsonc
{
  "target":  "Erf.function",               // "<Module>.<method>" dispatch key
  "kind":    "special_function",
  "source":  "Numerics/.../Erf.cs",        // provenance
  "cases": [
    {
      "name": "erf_one",
      "args": [1.0],                        // positional arguments to the function
      "assertions": [
        { "method": "value", "expected": 0.8427007929497149, "mode": "abs", "tol": 1e-12 }
      ]
    }
  ]
}
```

The dispatch key maps to a free function: `"Erf.function"` → `bestfit::numerics::math::special::erf::function(x)`.
Each case has a flat `args` array (not nested in `construct`); the single assertion method is always `"value"`.

Most files dispatch every case through that one file-level `target`. A file may instead group
several related dispatch keys and let individual cases override it with their own `"target"`; a
case without one falls back to the file-level `target`, so single-target files are unaffected. See
`fixtures/special_functions/cholesky.json`, whose cases select among `Cholesky.determinant`,
`Cholesky.log_determinant`, `Cholesky.inverse_element`, `Cholesky.solve_element`, and
`Cholesky.is_positive_definite`.

The `Cholesky.*` targets take a matrix flattened to row-major doubles in `args`, with `n` inferred
from the argument count: `n = sqrt(len(args))` for `determinant`/`log_determinant`/
`is_positive_definite`; `inverse_element` appends two trailing `(i, j)` indices
(`n = sqrt(len(args) - 2)`); `solve_element` appends an `n`-length right-hand-side vector followed
by a trailing solution index `i` (`n` solves `n^2 + n + 1 = len(args)`).

**Comparison modes:** `abs` (|actual−expected| ≤ tol), `rel` (|actual−expected|/|expected| ≤ tol),
`equal` (exact; `expected` may be the strings `"inf"`, `"-inf"`, `"nan"`), `bool`.

**Special values:** in `params` and `equal`-mode `expected`, the strings `"nan"`,
`"inf"`, `"-inf"` denote the corresponding floating-point values (JSON has no literals
for them).

Each runner maps fixture `method` names to its own API (e.g. `"sd"` →
`standard_deviation()`), so a new distribution usually needs only a new fixture file
plus a few dispatch entries per language.
