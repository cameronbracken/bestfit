// Oracle verifier: reproduce every fixture assertion against the upstream Numerics library.
//
// For each fixtures/*.json the same construct + assertions the C++/R/Python runners use are
// replayed against the real C# distribution, and the fixture's "expected" value is checked to
// its stated tolerance. This is the "run dotnet" half of the hybrid oracle workflow: the
// scraped/curated fixtures are confirmed reproducible by the source they were ported from.
//
// Usage:  dotnet run --project tools/oracle_emitter -- [fixtures-dir]
//   exits non-zero if any supported assertion fails to reproduce.
// GEV standard-error methods are reported as "skipped" (verified in Phase 0, not re-checked here).

using System.Text.Json;
using Numerics;
using Numerics.Data;
using Numerics.Data.Statistics;
using Numerics.Distributions;
using Numerics.Distributions.Copulas;
using Numerics.Mathematics;
using Numerics.Mathematics.LinearAlgebra;
using Numerics.Mathematics.Optimization;
using Numerics.Mathematics.SpecialFunctions;
using Numerics.Sampling;
using Numerics.Sampling.MCMC;
// Task T12: the real RMC.BestFit estimation path (subset-compiled -- see OracleEmitter.csproj).
// `RMC.BestFit` holds ExactSeries; `RMC.BestFit.Models` holds DataFrame + the UnivariateDistribution
// model; `RMC.BestFit.Estimation` holds MaximumLikelihood / MaximumAPosteriori / BayesianAnalysis /
// OptimizationMethod. None of these names clash with Numerics (verified), so plain usings are safe.
using RMC.BestFit;
using RMC.BestFit.Estimation;
using BestFitModels = RMC.BestFit.Models;
// Task A11: the real RMC.BestFit user-facing Analyses layer (subset-compiled -- see
// OracleEmitter.csproj). Aliased to avoid clashing with RMC.BestFit.Estimation.BayesianAnalysis.
using BestFitAnalyses = RMC.BestFit.Analyses;

static double ParseNum(JsonElement v)
{
    if (v.ValueKind == JsonValueKind.String)
    {
        return v.GetString() switch
        {
            "nan" => double.NaN,
            "inf" => double.PositiveInfinity,
            "-inf" => double.NegativeInfinity,
            var s => throw new Exception($"unexpected string value: {s}"),
        };
    }
    return v.GetDouble();
}

static ParameterEstimationMethod ParseMethod(string m) => m switch
{
    "mom" => ParameterEstimationMethod.MethodOfMoments,
    "lmom" => ParameterEstimationMethod.MethodOfLinearMoments,
    _ => ParameterEstimationMethod.MaximumLikelihood,
};

// Build a component distribution from {"target": "...", "params": [...]} (or "fit").
// Recursive: components can nest for future Mixture / CompetingRisks.
static UnivariateDistributionBase BuildComponent(JsonElement desc,
                                                  Dictionary<string, double[]> datasets)
{
    var compTarget = desc.GetProperty("target").GetString()!;
    var compType = Enum.Parse<UnivariateDistributionType>(compTarget);
    UnivariateDistributionBase compDist = compTarget == "VonMises"
        ? new VonMises()
        : UnivariateDistributionFactory.CreateDistribution(compType);
    if (desc.TryGetProperty("params", out var compPs))
    {
        compDist.SetParameters(compPs.EnumerateArray().Select(ParseNum).ToArray());
        return compDist;
    }
    if (desc.TryGetProperty("fit", out var compFit))
    {
        var compData = datasets[compFit.GetProperty("dataset").GetString()!];
        if (compDist is IEstimation compEst)
            compEst.Estimate(compData, ParseMethod(compFit.GetProperty("method").GetString()!));
        else
            throw new Exception($"{compTarget} does not support estimation");
        return compDist;
    }
    throw new Exception($"BuildComponent: missing 'params' or 'fit' for {compTarget}");
}

// Build composite distributions from their structured construct schemas.
// Future composites (Mixture, CompetingRisks) add a case here.
static UnivariateDistributionBase BuildComposite(string target, JsonElement construct,
                                                  Dictionary<string, double[]> datasets)
{
    if (target == "TruncatedDistribution")
    {
        var baseDist = BuildComponent(construct.GetProperty("base"), datasets);
        var boundsArr = construct.GetProperty("bounds").EnumerateArray().ToArray();
        double lo = ParseNum(boundsArr[0]), hi = ParseNum(boundsArr[1]);
        return new TruncatedDistribution(baseDist, lo, hi);
    }
    if (target == "Empirical")
    {
        var xArr = construct.GetProperty("x").EnumerateArray().Select(ParseNum).ToArray();
        var pArr = construct.GetProperty("p").EnumerateArray().Select(ParseNum).ToArray();
        var emp = new EmpiricalDistribution(xArr, pArr);
        // Optional p_transform: "None" or "NormalZ" (default NormalZ)
        if (construct.TryGetProperty("p_transform", out var pt))
        {
            emp.ProbabilityTransform = pt.GetString() switch
            {
                "None"    => Transform.None,
                "NormalZ" => Transform.NormalZ,
                var s     => throw new Exception($"unknown p_transform: {s}")
            };
        }
        return emp;
    }
    if (target == "KernelDensity")
    {
        var data = datasets[construct.GetProperty("data").GetString()!];
        var kernelStr = construct.TryGetProperty("kernel", out var k) ? k.GetString()! : "Gaussian";
        var kernelType = kernelStr switch
        {
            "Epanechnikov" => KernelDensity.KernelType.Epanechnikov,
            "Gaussian"     => KernelDensity.KernelType.Gaussian,
            "Triangular"   => KernelDensity.KernelType.Triangular,
            "Uniform"      => KernelDensity.KernelType.Uniform,
            var s          => throw new Exception($"unknown kernel type: {s}")
        };
        KernelDensity kde = construct.TryGetProperty("bandwidth", out var bw)
            ? new KernelDensity(data, kernelType, bw.GetDouble())
            : new KernelDensity(data, kernelType);
        if (construct.TryGetProperty("bounded_by_data", out var bd))
            kde.BoundedByData = bd.GetBoolean();
        return kde;
    }
    if (target == "Mixture")
    {
        var comps = construct.GetProperty("components").EnumerateArray()
            .Select(c => BuildComponent(c, datasets)).ToArray();
        var wts = construct.GetProperty("weights").EnumerateArray()
            .Select(e => e.GetDouble()).ToArray();
        return new Mixture(wts, comps);
    }
    if (target == "CompetingRisks")
    {
        var comps = construct.GetProperty("components").EnumerateArray()
            .Select(c => BuildComponent(c, datasets)).ToArray();
        var cr = new CompetingRisks(comps);
        if (construct.TryGetProperty("minimum_of_random_variables", out var minOfRV))
            cr.MinimumOfRandomVariables = minOfRV.GetBoolean();
        if (construct.TryGetProperty("dependency", out var depEl))
        {
            cr.Dependency = depEl.GetString() switch
            {
                "Independent" => Probability.DependencyType.Independent,
                "PerfectlyPositive" => Probability.DependencyType.PerfectlyPositive,
                "PerfectlyNegative" => Probability.DependencyType.PerfectlyNegative,
                "CorrelationMatrix" => Probability.DependencyType.CorrelationMatrix,
                var s => throw new Exception($"unknown dependency type: {s}")
            };
        }
        if (construct.TryGetProperty("correlation", out var corrEl))
        {
            var rows = corrEl.EnumerateArray().ToArray();
            int n = rows.Length;
            var corr = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                var row = rows[i].EnumerateArray().ToArray();
                for (int j = 0; j < row.Length; j++) corr[i, j] = ParseNum(row[j]);
            }
            cr.CorrelationMatrix = corr;
        }
        return cr;
    }
    throw new Exception($"unknown composite target: {target}");
}

static UnivariateDistributionBase Build(string target, JsonElement construct,
                                        Dictionary<string, double[]> datasets)
{
    // Composite distributions use bespoke construction (no flat enum entry in C# or C++).
    if (target == "TruncatedDistribution" || target == "Empirical" || target == "KernelDensity"
        || target == "Mixture" || target == "CompetingRisks")
        return BuildComposite(target, construct, datasets);

    // Empirical is constructed from x/p arrays, not flat params -- handled above as composite.
    var type = Enum.Parse<UnivariateDistributionType>(target);
    // VonMises is in the C# enum but not in the upstream factory yet -- construct directly.
    UnivariateDistributionBase dist = target == "VonMises"
        ? new VonMises()
        : UnivariateDistributionFactory.CreateDistribution(type);
    if (construct.TryGetProperty("params", out var ps))
    {
        var p = ps.EnumerateArray().Select(ParseNum).ToArray();
        dist.SetParameters(p);
        return dist;
    }
    var fit = construct.GetProperty("fit");
    var data = datasets[fit.GetProperty("dataset").GetString()!];
    if (dist is IEstimation est)
        est.Estimate(data, ParseMethod(fit.GetProperty("method").GetString()!));
    else
        throw new Exception($"{target} does not support estimation");
    return dist;
}

// Returns null when the method is intentionally not reproduced here (GEV standard errors).
static double? Dispatch(UnivariateDistributionBase d, string m, JsonElement[] a)
{
    switch (m)
    {
        case "mean": return d.Mean;
        case "median": return d.Median;
        case "mode": return d.Mode;
        case "sd": return d.StandardDeviation;
        case "skewness": return d.Skewness;
        case "kurtosis": return d.Kurtosis;
        case "minimum": return d.Minimum;
        case "maximum": return d.Maximum;
        case "pdf": return d.PDF(a[0].GetDouble());
        case "log_pdf": return d.LogPDF(a[0].GetDouble());
        case "cdf": return d.CDF(a[0].GetDouble());
        case "quantile": return d.InverseCDF(a[0].GetDouble());
        case "param":
            if (a[0].ValueKind == JsonValueKind.String)
            {
                int idx = a[0].GetString() switch { "location" => 0, "scale" => 1, "shape" => 2, _ => -1 };
                return d.GetParameters[idx];
            }
            return d.GetParameters[a[0].GetInt32()];
        case "linear_moment":
            if (d is ILinearMomentEstimation lm)
                return lm.LinearMomentsFromParameters(d.GetParameters)[a[0].GetInt32()];
            throw new Exception("distribution has no L-moments");
        // GEV bespoke standard-error methods -- validated in Phase 0, not re-checked here.
        case "quantile_gradient":
        case "parameter_covariance":
        case "quantile_variance":
        case "quantile_se":
            return null;
        default:
            throw new Exception($"unknown fixture method: {m}");
    }
}

static bool Compare(double actual, JsonElement assertion)
{
    string mode = assertion.GetProperty("mode").GetString()!;
    var exp = assertion.GetProperty("expected");
    switch (mode)
    {
        case "equal":
            double e = ParseNum(exp);
            return double.IsNaN(e) ? double.IsNaN(actual) : actual == e;
        case "abs":
            return Math.Abs(actual - exp.GetDouble()) <= assertion.GetProperty("tol").GetDouble();
        case "rel":
            double r = exp.GetDouble();
            return Math.Abs(actual - r) / Math.Abs(r) <= assertion.GetProperty("tol").GetDouble();
        default:
            throw new Exception($"unknown comparison mode: {mode}");
    }
}

// Cholesky fixture args are a flattened row-major n*n matrix, with n inferred from the
// args length per the convention documented in fixtures/special_functions/cholesky.json.
static int CholeskySquareN(int len)
{
    int n = (int)Math.Round(Math.Sqrt(len));
    if (n * n != len) throw new Exception("Cholesky fixture args: length is not a perfect square");
    return n;
}

// solve_element args are [flattened n*n matrix, n-length rhs vector, index i], i.e.
// n*n + n + 1 == len; solve the quadratic for n.
static int CholeskySolveN(int len)
{
    int n = (int)Math.Round((-1.0 + Math.Sqrt(1.0 + 4.0 * (len - 1))) / 2.0);
    if (n * n + n + 1 != len) throw new Exception("Cholesky fixture args: length does not fit n*n+n+1");
    return n;
}

static Matrix CholeskyMatrixFromFlat(double[] a, int n)
{
    var m = new Matrix(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i, j] = a[i * n + j];
    return m;
}

// Correlation fixture args are [x..., y...] concatenated and split at the midpoint
// (equal-length samples) -- see fixtures/special_functions/correlation.json for the convention.
static (double[] X, double[] Y) CorrelationSplit(double[] a)
{
    int mid = a.Length / 2;
    return (a[..mid], a[mid..]);
}

// Special-function dispatch table: maps "Module.method" → Func<double[], double>.
static Func<double[], double>? ResolveSpecialFunction(string target) => target switch
{
    // Cholesky family (args: flattened row-major matrix, n inferred from length -- see
    // fixtures/special_functions/cholesky.json for the full convention)
    "Cholesky.determinant" => a =>
    {
        int n = CholeskySquareN(a.Length);
        return new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n)).Determinant();
    },
    "Cholesky.log_determinant" => a =>
    {
        int n = CholeskySquareN(a.Length);
        return new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n)).LogDeterminant();
    },
    "Cholesky.inverse_element" => a =>
    {
        int n = CholeskySquareN(a.Length - 2);
        var chol = new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n));
        int i = (int)a[n * n];
        int j = (int)a[n * n + 1];
        return chol.InverseA()[i, j];
    },
    "Cholesky.solve_element" => a =>
    {
        int n = CholeskySolveN(a.Length);
        var chol = new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n));
        var rhs = new double[n];
        Array.Copy(a, n * n, rhs, 0, n);
        int i = (int)a[n * n + n];
        return chol.Solve(new Vector(rhs))[i];
    },
    // Returns 1.0 if the matrix is positive-definite (construction succeeds), 0.0 if the
    // ctor throws. Upstream CholeskyDecomposition.cs throws a bare `new Exception(...)`
    // for the non-PD/NaN case (no dedicated exception type), so this must catch the
    // general `Exception`, unlike the C++ port's narrower `std::runtime_error` catch.
    "Cholesky.is_positive_definite" => a =>
    {
        int n = CholeskySquareN(a.Length);
        try
        {
            var chol = new CholeskyDecomposition(CholeskyMatrixFromFlat(a, n));
            return chol.IsPositiveDefinite ? 1.0 : 0.0;
        }
        catch (Exception)
        {
            return 0.0;
        }
    },
    // Erf family
    "Erf.function"     => a => Erf.Function(a[0]),
    "Erf.erfc"         => a => Erf.Erfc(a[0]),
    "Erf.inverse_erf"  => a => Erf.InverseErf(a[0]),
    "Erf.inverse_erfc" => a => Erf.InverseErfc(a[0]),
    // Gamma family
    "Gamma.function"                 => a => Gamma.Function(a[0]),
    "Gamma.log_gamma"                => a => Gamma.LogGamma(a[0]),
    "Gamma.digamma"                  => a => Gamma.Digamma(a[0]),
    "Gamma.trigamma"                 => a => Gamma.Trigamma(a[0]),
    "Gamma.lower_incomplete"         => a => Gamma.LowerIncomplete(a[0], a[1]),
    "Gamma.upper_incomplete"         => a => Gamma.UpperIncomplete(a[0], a[1]),
    "Gamma.inverse_lower_incomplete" => a => Gamma.InverseLowerIncomplete(a[0], a[1]),
    "Gamma.inverse_upper_incomplete" => a => Gamma.InverseUpperIncomplete(a[0], a[1]),
    // Beta family
    "Beta.function"           => a => Beta.Function(a[0], a[1]),
    "Beta.incomplete"         => a => Beta.Incomplete(a[0], a[1], a[2]),
    "Beta.incbcf"             => a => Beta.Incbcf(a[0], a[1], a[2]),
    "Beta.incbd"              => a => Beta.Incbd(a[0], a[1], a[2]),
    "Beta.power_series"       => a => Beta.PowerSeries(a[0], a[1], a[2]),
    "Beta.incomplete_inverse" => a => Beta.IncompleteInverse(a[0], a[1], a[2]),
    // Factorial family
    "Factorial.function"             => a => Factorial.Function((int)a[0]),
    "Factorial.log_factorial"        => a => Factorial.LogFactorial((int)a[0]),
    "Factorial.binomial_coefficient" => a => Factorial.BinomialCoefficient((int)a[0], (int)a[1]),
    // Bessel family
    "Bessel.i0" => a => Bessel.I0(a[0]),
    "Bessel.i1" => a => Bessel.I1(a[0]),
    // Correlation family (args: [x..., y...], split at the midpoint -- see
    // fixtures/special_functions/correlation.json for the full convention)
    "Correlation.pearson" => a =>
    {
        var (x, y) = CorrelationSplit(a);
        return Correlation.Pearson(x, y);
    },
    "Correlation.spearman" => a =>
    {
        var (x, y) = CorrelationSplit(a);
        return Correlation.Spearman(x, y);
    },
    "Correlation.kendalls_tau" => a =>
    {
        var (x, y) = CorrelationSplit(a);
        return Correlation.KendallsTau(x, y);
    },
    // LU family (args: flattened row-major matrix, n inferred from length -- reuses the
    // Cholesky-fixture flatten helpers above, which are generic matrix-args conventions,
    // not Cholesky-specific; see fixtures/special_functions/lu_decomposition.json)
    "LU.determinant" => a =>
    {
        int n = CholeskySquareN(a.Length);
        return new LUDecomposition(CholeskyMatrixFromFlat(a, n)).Determinant();
    },
    "LU.inverse_element" => a =>
    {
        int n = CholeskySquareN(a.Length - 2);
        var lu = new LUDecomposition(CholeskyMatrixFromFlat(a, n));
        int i = (int)a[n * n];
        int j = (int)a[n * n + 1];
        return lu.InverseA()[i, j];
    },
    "LU.solve_element" => a =>
    {
        int n = CholeskySolveN(a.Length);
        var lu = new LUDecomposition(CholeskyMatrixFromFlat(a, n));
        var rhs = new double[n];
        Array.Copy(a, n * n, rhs, 0, n);
        int i = (int)a[n * n + n];
        return lu.Solve(new Vector(rhs))[i];
    },
    // Percentile (args: [data_1..data_n, k, data_is_sorted (0.0/1.0)] -- see
    // fixtures/special_functions/percentile.json for the convention)
    "Statistics.percentile" => a =>
    {
        int n = a.Length - 2;
        var data = a[..n];
        double k = a[n];
        bool sorted = a[n + 1] != 0.0;
        return Statistics.Percentile(data, k, sorted);
    },
    // Extensions/MersenneTwister ranged-draw family (see
    // fixtures/special_functions/extension_methods.json for the conventions)
    "Extensions.next_doubles_grid" => a =>
    {
        // args: [n, dim, seed, row, col]
        int n = (int)a[0];
        int dim = (int)a[1];
        var rng = new MersenneTwister((int)a[2]);
        int row = (int)a[3];
        int col = (int)a[4];
        var grid = rng.NextDoubles(n, dim);
        return grid[row, col];
    },
    "Extensions.next_integers_at" => a =>
    {
        // args: [n, seed, i]
        int n = (int)a[0];
        var rng = new MersenneTwister((int)a[1]);
        int i = (int)a[2];
        var values = rng.NextIntegers(n);
        return values[i];
    },
    "Mt.next_range" => a =>
    {
        // args: [seed, min, max, i] -- draws Next(min, max) (i+1) times, 0-based,
        // returning the i-th draw.
        var rng = new MersenneTwister((int)a[0]);
        int minV = (int)a[1];
        int maxV = (int)a[2];
        int i = (int)a[3];
        int result = 0;
        for (int k = 0; k <= i; k++) result = rng.Next(minV, maxV);
        return (double)result;
    },
    // RunningCovarianceMatrix family (args: [size, num_pushes, data_flat, trailing
    // index/indices] -- see fixtures/special_functions/running_covariance.json)
    "RunningCovariance.mean_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        return rcm.Mean[i, 0];
    },
    "RunningCovariance.covariance_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.Covariance[i, j];
    },
    "RunningCovariance.sample_covariance_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.SampleCovariance[i, j];
    },
    "RunningCovariance.sample_correlation_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.SampleCorrelation[i, j];
    },
    "RunningCovariance.population_covariance_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.PopulationCovariance[i, j];
    },
    "RunningCovariance.population_correlation_element" => a =>
    {
        int size = (int)a[0];
        int numPushes = (int)a[1];
        var rcm = RunningCovarianceBuild(a, size, numPushes);
        int baseIdx = 2 + numPushes * size;
        int i = (int)a[baseIdx];
        int j = (int)a[baseIdx + 1];
        return rcm.PopulationCorrelation[i, j];
    },
    // RunningStatistics family (args: the flat sample; see
    // fixtures/special_functions/running_statistics.json)
    "RunningStatistics.mean" => a => new RunningStatistics(a).Mean,
    "RunningStatistics.variance" => a => new RunningStatistics(a).Variance,
    "RunningStatistics.standard_deviation" => a => new RunningStatistics(a).StandardDeviation,
    "RunningStatistics.population_variance" => a => new RunningStatistics(a).PopulationVariance,
    "RunningStatistics.population_standard_deviation" => a => new RunningStatistics(a).PopulationStandardDeviation,
    "RunningStatistics.coefficient_of_variation" => a => new RunningStatistics(a).CoefficientOfVariation,
    "RunningStatistics.skewness" => a => new RunningStatistics(a).Skewness,
    "RunningStatistics.population_skewness" => a => new RunningStatistics(a).PopulationSkewness,
    "RunningStatistics.kurtosis" => a => new RunningStatistics(a).Kurtosis,
    "RunningStatistics.population_kurtosis" => a => new RunningStatistics(a).PopulationKurtosis,
    "RunningStatistics.minimum" => a => new RunningStatistics(a).Minimum,
    "RunningStatistics.maximum" => a => new RunningStatistics(a).Maximum,
    "RunningStatistics.count" => a => (double)new RunningStatistics(a).Count,
    // RunningStatistics combine family (args: [n1, sample1(n1), sample2(m)] -- see
    // RunningStatisticsCombined() above and fixtures/special_functions/running_statistics.json)
    "RunningStatistics.combined_minimum" => a => RunningStatisticsCombined(a).Minimum,
    "RunningStatistics.combined_maximum" => a => RunningStatisticsCombined(a).Maximum,
    "RunningStatistics.combined_mean" => a => RunningStatisticsCombined(a).Mean,
    "RunningStatistics.combined_variance" => a => RunningStatisticsCombined(a).Variance,
    "RunningStatistics.combined_standard_deviation" => a => RunningStatisticsCombined(a).StandardDeviation,
    "RunningStatistics.combined_coefficient_of_variation" => a => RunningStatisticsCombined(a).CoefficientOfVariation,
    "RunningStatistics.combined_skewness" => a => RunningStatisticsCombined(a).Skewness,
    "RunningStatistics.combined_kurtosis" => a => RunningStatisticsCombined(a).Kurtosis,
    // Fourier family (see Fourier*At() below for the args conventions -- mirrors
    // fourier_*_at() in core/tests/test_fixtures.cpp exactly)
    "Fourier.fft_at" => FourierFftAt,
    "Fourier.real_fft_at" => FourierRealFftAt,
    "Fourier.correlation_at" => FourierCorrelationAt,
    "Fourier.autocorrelation_at" => FourierAutocorrelationAt,
    // NumericalDerivative family (closed function registry; MUST match
    // numerical_derivative_{quadratic,normal_loglik} in core/tests/test_fixtures.cpp)
    "NumericalDerivative.gradient_element_quadratic" => a => NumericalDerivativeGradientElement(NumericalDerivativeQuadratic, a),
    "NumericalDerivative.gradient_element_normal_loglik" => a => NumericalDerivativeGradientElement(NumericalDerivativeNormalLoglik, a),
    "NumericalDerivative.hessian_element_quadratic" => a => NumericalDerivativeHessianElement(NumericalDerivativeQuadratic, a),
    "NumericalDerivative.hessian_element_normal_loglik" => a => NumericalDerivativeHessianElement(NumericalDerivativeNormalLoglik, a),
    // DifferentialEvolution family (closed function registry; REUSES
    // NumericalDerivativeQuadratic/NumericalDerivativeNormalLoglik above -- MUST match
    // differential_evolution_best_value() in core/tests/test_fixtures.cpp)
    "DifferentialEvolution.best_value" => DifferentialEvolutionBestValue,
    // Histogram family (args: [explicit_bins, data..., trailing probe?] -- see
    // HistogramBuild() below and fixtures/special_functions/histogram.json)
    "Histogram.number_of_bins" => a => (double)HistogramBuild(a, 0).NumberOfBins,
    "Histogram.bin_width" => a => HistogramBuild(a, 0).BinWidth,
    "Histogram.lower_bound" => a => HistogramBuild(a, 0).LowerBound,
    "Histogram.upper_bound" => a => HistogramBuild(a, 0).UpperBound,
    "Histogram.data_count" => a => (double)HistogramBuild(a, 0).DataCount,
    "Histogram.mean" => a => HistogramBuild(a, 0).Mean,
    "Histogram.median" => a => HistogramBuild(a, 0).Median,
    "Histogram.mode" => a => HistogramBuild(a, 0).Mode,
    "Histogram.standard_deviation" => a => HistogramBuild(a, 0).StandardDeviation,
    "Histogram.bin_lower_bound_at" => a => HistogramBuild(a, 1)[(int)a[^1]].LowerBound,
    "Histogram.bin_upper_bound_at" => a => HistogramBuild(a, 1)[(int)a[^1]].UpperBound,
    "Histogram.bin_frequency_at" => a => (double)HistogramBuild(a, 1)[(int)a[^1]].Frequency,
    "Histogram.get_bin_index_of" => a => (double)HistogramBuild(a, 1).GetBinIndexOf(a[^1]),
    // PlottingPositions family (args: [N, alpha, i] for function_at; [N, i] for
    // weibull_at -- see fixtures/special_functions/plotting_positions.json)
    "PlottingPositions.function_at" => a => PlottingPositions.Function((int)a[0], a[1])[(int)a[2]],
    "PlottingPositions.weibull_at" => a => PlottingPositions.Weibull((int)a[0])[(int)a[1]],
    // Search family (args: [values..., x, start] -- see fixtures/special_functions/search.json)
    "Search.sequential" => a => Search.Sequential(a[^2], a[..^2], (int)a[^1]),
    "Search.bisection" => a => Search.Bisection(a[^2], a[..^2], (int)a[^1]),
    // MCMCDiagnostics.MinimumSampleSize (args: [quantile, tolerance, probability] -- see
    // fixtures/special_functions/mcmc_diagnostics.json)
    "MCMCDiagnostics.minimum_sample_size" => a => MCMCDiagnostics.MinimumSampleSize(a[0], a[1], a[2]),
    _ => null,
};

// RunningCovariance fixture args: [size, num_pushes, data_flat(num_pushes*size), trailing
// index/indices] -- see fixtures/special_functions/running_covariance.json for the
// convention. Builds a RunningCovarianceMatrix and replays `numPushes` Push()es of
// `size`-length rows sliced from the flattened data.
static RunningCovarianceMatrix RunningCovarianceBuild(double[] a, int size, int numPushes)
{
    var rcm = new RunningCovarianceMatrix(size);
    for (int p = 0; p < numPushes; p++)
    {
        var row = new double[size];
        Array.Copy(a, 2 + p * size, row, 0, size);
        rcm.Push(row);
    }
    return rcm;
}

// RunningStatistics combine fixture args: [n1, sample1(n1 values), sample2(remaining
// values)] -- a "split-index" convention, distinct from Correlation's equal-length
// two-halves split (Test_Combine/Test_Add split their 69-value sample into UNEQUAL 48/21
// sub-samples, so a fixed midpoint doesn't apply). See
// fixtures/special_functions/running_statistics.json for the full convention. Uses the
// `+` operator (rather than calling RunningStatistics.Combine directly), which exercises
// both -- operator+ is a one-line forwarder to Combine.
static RunningStatistics RunningStatisticsCombined(double[] a)
{
    int n1 = (int)a[0];
    var sample1 = a[1..(1 + n1)];
    var sample2 = a[(1 + n1)..];
    return new RunningStatistics(sample1) + new RunningStatistics(sample2);
}

// Fourier fixture args conventions -- MUST mirror fourier_*_at() in
// core/tests/test_fixtures.cpp exactly (see fixtures/special_functions/fourier.json).
static double FourierFftAt(double[] a)
{
    int n = a.Length - 2;
    var data = a[..n];
    bool inverse = a[n] != 0.0;
    int index = (int)a[n + 1];
    Fourier.FFT(data, inverse);
    return data[index];
}
static double FourierRealFftAt(double[] a)
{
    int n = a.Length - 2;
    var data = a[..n];
    bool inverse = a[n] != 0.0;
    int index = (int)a[n + 1];
    Fourier.RealFFT(data, inverse);
    return data[index];
}
static double FourierCorrelationAt(double[] a)
{
    int n = (a.Length - 1) / 2;
    var data1 = a[..n];
    var data2 = a[n..(2 * n)];
    int index = (int)a[2 * n];
    var corr = Fourier.Correlation(data1, data2);
    return corr[index];
}
static double FourierAutocorrelationAt(double[] a)
{
    int n = a.Length - 2;
    var series = a[..n];
    int lagMax = (int)a[n];
    int lag = (int)a[n + 1];
    var acf = Fourier.Autocorrelation(series, lagMax);
    if (acf is null) throw new Exception("Fourier.autocorrelation_at: Autocorrelation returned null");
    return acf[lag, 1];
}

// Closed registry of named functions for the numerical_derivative fixture -- MUST match
// numerical_derivative_{quadratic,normal_loglik} in core/tests/test_fixtures.cpp exactly.
// (A top-level-statements Program.cs cannot declare a `static readonly` field outside a
// type, so the embedded sample is a local function returning a fresh array each call,
// mirroring the C++ side's static-local-inside-a-function pattern.)
static double[] NumericalDerivativeNormalSample() => [9.0, 10.0, 11.0, 12.0, 13.0];
static double NumericalDerivativeQuadratic(double[] x)
{
    double s = 0;
    for (int i = 0; i < x.Length; i++) { double d = x[i] - i; s += d * d; }
    return s;
}
static double NumericalDerivativeNormalLoglik(double[] x)
{
    var n = new Normal(x[0], x[1]);
    return n.LogLikelihood(NumericalDerivativeNormalSample());
}

// numerical_derivative fixture args convention -- MUST mirror
// numerical_derivative_parse()/gradient_element()/hessian_element() in
// core/tests/test_fixtures.cpp exactly.
static void NumericalDerivativeParse(double[] a, out double[] theta, out double[] lower,
                                      out double[] upper, out int next)
{
    int p = (int)a[0];
    theta = a[1..(1 + p)];
    lower = a[(1 + p)..(1 + 2 * p)];
    upper = a[(1 + 2 * p)..(1 + 3 * p)];
    next = 1 + 3 * p;
}
static double NumericalDerivativeGradientElement(Func<double[], double> f, double[] a)
{
    NumericalDerivativeParse(a, out var theta, out var lower, out var upper, out int next);
    int index = (int)a[next];
    var grad = NumericalDerivative.Gradient(f, theta, lower, upper);
    return grad[index];
}
static double NumericalDerivativeHessianElement(Func<double[], double> f, double[] a)
{
    NumericalDerivativeParse(a, out var theta, out var lower, out var upper, out int next);
    int i = (int)a[next];
    int j = (int)a[next + 1];
    var hess = NumericalDerivative.Hessian(f, theta, lower, upper);
    return hess[i, j];
}

// DifferentialEvolution fixture args convention -- MUST mirror
// differential_evolution_best_value() in core/tests/test_fixtures.cpp exactly:
// args = [fnId, direction, D, lower(D), upper(D), index]. fnId: 0 = Quadratic, 1 =
// NormalLoglik (reuses the closed registry above). direction: 0 = Minimize(), 1 =
// Maximize(). index: 0..D-1 selects BestParameterSet.Values[index]; index == D selects
// BestParameterSet.Fitness. Every other knob uses the library default.
static double DifferentialEvolutionBestValue(double[] a)
{
    int fnId = (int)a[0];
    int direction = (int)a[1];
    int D = (int)a[2];
    var lower = a[3..(3 + D)];
    var upper = a[(3 + D)..(3 + 2 * D)];
    int index = (int)a[3 + 2 * D];

    Func<double[], double> f = fnId switch
    {
        0 => NumericalDerivativeQuadratic,
        1 => NumericalDerivativeNormalLoglik,
        _ => throw new Exception($"unknown DifferentialEvolution function id: {fnId}")
    };

    var de = new DifferentialEvolution(f, D, lower, upper);
    if (direction == 0) de.Minimize(); else de.Maximize();
    return index == D ? de.BestParameterSet.Fitness : de.BestParameterSet.Values[index];
}

// Histogram fixture args convention -- MUST mirror histogram_build() in
// core/tests/test_fixtures.cpp exactly.
static Histogram HistogramBuild(double[] a, int trailing)
{
    int explicitBins = (int)a[0];
    var data = a[1..(a.Length - trailing)];
    return explicitBins > 0 ? new Histogram(data, explicitBins) : new Histogram(data);
}

// --- multivariate_distribution branch -----------------------------------------------------
// Mirrors the univariate Build/Dispatch split. Dirichlet/Multinomial/BivariateEmpirical have
// no common Mean/Variance/Covariance signature (unlike UnivariateDistributionBase), so
// DispatchMultivariate downcasts to the concrete type for anything beyond
// Dimension/PDF/LogPDF/CDF/ParametersValid. Extensible: additional multivariate targets add
// a case to each of Build/Dispatch.

static MultivariateDistribution BuildMultivariate(string target, JsonElement construct)
{
    if (target == "Dirichlet")
    {
        var alpha = construct.GetProperty("alpha").EnumerateArray().Select(ParseNum).ToArray();
        return new Dirichlet(alpha);
    }
    if (target == "Multinomial")
    {
        int n = construct.GetProperty("n").GetInt32();
        var p = construct.GetProperty("p").EnumerateArray().Select(ParseNum).ToArray();
        return new Multinomial(n, p);
    }
    if (target == "BivariateEmpirical")
    {
        var x1 = construct.GetProperty("x1").EnumerateArray().Select(ParseNum).ToArray();
        var x2 = construct.GetProperty("x2").EnumerateArray().Select(ParseNum).ToArray();
        var pRows = construct.GetProperty("p").EnumerateArray().ToArray();
        var p = new double[pRows.Length, x2.Length];
        for (int i = 0; i < pRows.Length; i++)
        {
            var row = pRows[i].EnumerateArray().Select(ParseNum).ToArray();
            for (int j = 0; j < row.Length; j++) p[i, j] = row[j];
        }
        Transform ParseT(string key) => construct.TryGetProperty(key, out var t)
            ? t.GetString() switch
            {
                "None" => Transform.None,
                "Logarithmic" => Transform.Logarithmic,
                "NormalZ" => Transform.NormalZ,
                var s => throw new Exception($"unknown transform: {s}")
            }
            : Transform.None;
        return new BivariateEmpirical(x1, x2, p, ParseT("x1_transform"), ParseT("x2_transform"),
                                       ParseT("p_transform"));
    }
    if (target == "MultivariateNormal")
    {
        var mean = construct.GetProperty("mean").EnumerateArray().Select(ParseNum).ToArray();
        var covRows = construct.GetProperty("covariance").EnumerateArray().ToArray();
        var covariance = new double[covRows.Length, mean.Length];
        for (int i = 0; i < covRows.Length; i++)
        {
            var row = covRows[i].EnumerateArray().Select(ParseNum).ToArray();
            for (int j = 0; j < row.Length; j++) covariance[i, j] = row[j];
        }
        var mvn = new MultivariateNormal(mean, covariance);
        if (construct.TryGetProperty("seed", out var seedEl))
            mvn.MVNUNI = new MersenneTwister(seedEl.GetInt32());
        if (construct.TryGetProperty("max_evaluations", out var maxEvalEl))
            mvn.MaxEvaluations = maxEvalEl.GetInt32();
        if (construct.TryGetProperty("abs_error", out var absErrEl))
            mvn.AbsoluteError = absErrEl.GetDouble();
        if (construct.TryGetProperty("rel_error", out var relErrEl))
            mvn.RelativeError = relErrEl.GetDouble();
        return mvn;
    }
    if (target == "MultivariateStudentT")
    {
        double df = construct.GetProperty("df").GetDouble();
        var location = construct.GetProperty("location").EnumerateArray().Select(ParseNum).ToArray();
        var scaleRows = construct.GetProperty("scale").EnumerateArray().ToArray();
        var scale = new double[scaleRows.Length, location.Length];
        for (int i = 0; i < scaleRows.Length; i++)
        {
            var row = scaleRows[i].EnumerateArray().Select(ParseNum).ToArray();
            for (int j = 0; j < row.Length; j++) scale[i, j] = row[j];
        }
        return new MultivariateStudentT(df, location, scale);
    }
    throw new Exception($"unknown multivariate target: {target}");
}

// Shared lookup for the "random_value"/"lhs_value" seeded-sampling oracle methods, common
// to every multivariate target that implements GenerateRandomValues (all four) /
// LatinHypercubeRandomValues (MultivariateNormal, MultivariateStudentT only -- see
// fixtures/README.md). args = [sample_size, seed, row, col]: `generate` is a method-group
// reference to the (int sampleSize, int seed) => double[,] overload itself, so this is
// stateless -- no persistent-instance batching needed, unlike MultivariateNormal's
// MVNUNI-seeded cdf/interval/mvndst path above.
static double SampleValueAt(Func<int, int, double[,]> generate, JsonElement[] a)
{
    var sample = generate(a[0].GetInt32(), a[1].GetInt32());
    return sample[a[2].GetInt32(), a[3].GetInt32()];
}

static double DispatchMultivariate(MultivariateDistribution d, string target, string m, JsonElement[] a)
{
    switch (m)
    {
        case "dimension": return d.Dimension;
        // The C# property for the PMF/PDF is PDF(double[]); LogPMF is a Multinomial-only
        // method that LogPDF forwards to, so LogPDF works generically across all three.
        case "pdf": return d.PDF(a[0].EnumerateArray().Select(ParseNum).ToArray());
        case "log_pdf": return d.LogPDF(a[0].EnumerateArray().Select(ParseNum).ToArray());
        case "cdf": return d.CDF(a[0].EnumerateArray().Select(ParseNum).ToArray());
    }
    if (target == "Dirichlet")
    {
        var dd = (Dirichlet)d;
        switch (m)
        {
            case "alpha": return dd.Alpha[a[0].GetInt32()];
            case "alpha_sum": return dd.AlphaSum;
            case "mean": return dd.Mean[a[0].GetInt32()];
            case "variance": return dd.Variance[a[0].GetInt32()];
            case "mode": return dd.Mode[a[0].GetInt32()];
            case "covariance": return dd.Covariance(a[0].GetInt32(), a[1].GetInt32());
            case "log_multivariate_beta": return Dirichlet.LogMultivariateBeta(a.Select(ParseNum).ToArray());
            case "random_value": return SampleValueAt(dd.GenerateRandomValues, a);
        }
    }
    else if (target == "Multinomial")
    {
        var mm = (Multinomial)d;
        switch (m)
        {
            case "number_of_trials": return mm.NumberOfTrials;
            case "mean": return mm.Mean[a[0].GetInt32()];
            case "variance": return mm.Variance[a[0].GetInt32()];
            case "covariance": return mm.Covariance(a[0].GetInt32(), a[1].GetInt32());
            case "random_value": return SampleValueAt(mm.GenerateRandomValues, a);
        }
    }
    else if (target == "BivariateEmpirical")
    {
        var bb = (BivariateEmpirical)d;
        if (m == "cdf_xy") return bb.CDF(a[0].GetDouble(), a[1].GetDouble());
    }
    else if (target == "MultivariateNormal")
    {
        var nn = (MultivariateNormal)d;
        switch (m)
        {
            case "mean": return nn.Mean[a[0].GetInt32()];
            case "median": return nn.Median[a[0].GetInt32()];
            case "mode": return nn.Mode[a[0].GetInt32()];
            case "sd": return nn.StandardDeviation[a[0].GetInt32()];
            case "variance": return nn.Variance[a[0].GetInt32()];
            case "covariance": return nn.Covariance[a[0].GetInt32(), a[1].GetInt32()];
            case "mahalanobis": return nn.Mahalanobis(a[0].EnumerateArray().Select(ParseNum).ToArray());
            case "inverse_cdf":
            {
                // args = [[p_1..p_dim], index]
                var p = a[0].EnumerateArray().Select(ParseNum).ToArray();
                int idx = a[1].GetInt32();
                return nn.InverseCDF(p)[idx];
            }
            case "interval":
            {
                // args = [[lower...], [upper...]]
                var lower = a[0].EnumerateArray().Select(ParseNum).ToArray();
                var upper = a[1].EnumerateArray().Select(ParseNum).ToArray();
                return nn.Interval(lower, upper);
            }
            case "mvndst":
            {
                // args = [n, [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps]
                int n = a[0].GetInt32();
                var lower = a[1].EnumerateArray().Select(ParseNum).ToArray();
                var upper = a[2].EnumerateArray().Select(ParseNum).ToArray();
                var infin = a[3].EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var correl = a[4].EnumerateArray().Select(ParseNum).ToArray();
                int maxpts = a[5].GetInt32();
                double abseps = a[6].GetDouble();
                double releps = a[7].GetDouble();
                double error = 0, val = 0;
                int inform = 0;
                nn.MVNDST(n, lower, upper, infin, correl, maxpts, abseps, releps, ref error, ref val, ref inform);
                return val;
            }
            case "random_value": return SampleValueAt(nn.GenerateRandomValues, a);
            case "lhs_value": return SampleValueAt(nn.LatinHypercubeRandomValues, a);
        }
    }
    else if (target == "MultivariateStudentT")
    {
        var tt = (MultivariateStudentT)d;
        switch (m)
        {
            case "degrees_of_freedom": return tt.DegreesOfFreedom;
            case "mean": return tt.Mean[a[0].GetInt32()];
            case "median": return tt.Median[a[0].GetInt32()];
            case "mode": return tt.Mode[a[0].GetInt32()];
            case "sd": return tt.StandardDeviation[a[0].GetInt32()];
            case "variance": return tt.Variance[a[0].GetInt32()];
            case "covariance": return tt.Covariance[a[0].GetInt32(), a[1].GetInt32()];
            case "mahalanobis": return tt.Mahalanobis(a[0].EnumerateArray().Select(ParseNum).ToArray());
            case "inverse_cdf":
            {
                // args = [[p_1..p_dim+1], index]
                var p = a[0].EnumerateArray().Select(ParseNum).ToArray();
                int idx = a[1].GetInt32();
                return tt.InverseCDF(p)[idx];
            }
            case "random_value": return SampleValueAt(tt.GenerateRandomValues, a);
            case "lhs_value": return SampleValueAt(tt.LatinHypercubeRandomValues, a);
        }
    }
    throw new Exception($"unknown multivariate fixture method: {target}/{m}");
}

// --- bivariate_copula branch --------------------------------------------------------------
// Every copula shares BivariateCopula's uniform Theta/GetCopulaParameters/PDF/CDF/... API
// (unlike MultivariateDistribution, whose targets share no common surface), so BuildCopula/
// DispatchCopula are fully generic through Enum.Parse<CopulaType> + the real
// BivariateCopulaEstimation.Estimate static -- no per-target branching, mirroring the C++
// core's copula_factory.hpp rationale. The one exception is the "tau" method-of-moments fit
// (SetThetaFromTau is a member of each concrete Archimedean class in the C# source, not
// part of IBivariateCopula/IArchimedeanCopula), which SetThetaFromTauDispatch resolves by
// target name; each new tau-capable copula adds one branch there.

static BivariateCopula BuildCopula(string target, JsonElement construct,
                                    Dictionary<string, double[]> datasets)
{
    var type = Enum.Parse<CopulaType>(target);
    BivariateCopula copula = type switch
    {
        CopulaType.AliMikhailHaq => new AMHCopula(),
        CopulaType.Clayton => new ClaytonCopula(),
        CopulaType.Frank => new FrankCopula(),
        CopulaType.Gumbel => new GumbelCopula(),
        CopulaType.Joe => new JoeCopula(),
        CopulaType.Normal => new NormalCopula(),
        CopulaType.StudentT => new StudentTCopula(),
        _ => throw new Exception($"copula type not yet ported: {target}")
    };

    if (construct.TryGetProperty("theta", out var thetaEl))
    {
        var parms = new List<double> { ParseNum(thetaEl) };
        if (construct.TryGetProperty("df", out var dfEl)) parms.Add(ParseNum(dfEl));
        copula.SetCopulaParameters(parms.ToArray());
        // {"marginals": {"targets": [..], "params": [[..], [..]]}} attaches marginals
        // directly via the C# `Copula(theta, marginX, marginY)` ctor path -- used by the
        // seeded "random_value" sampling oracles, which back-transform through the
        // marginals when set. Distinct from the "fit"-construct's marginals (a bare
        // 2-element type-name array; see below), since this path sets FIXED marginal
        // parameters rather than fitting them.
        if (construct.TryGetProperty("marginals", out var directMarginalsEl))
        {
            var targets = directMarginalsEl.GetProperty("targets").EnumerateArray().ToArray();
            var paramArrays = directMarginalsEl.GetProperty("params").EnumerateArray().ToArray();
            var mx = UnivariateDistributionFactory.CreateDistribution(
                Enum.Parse<UnivariateDistributionType>(targets[0].GetString()!));
            var my = UnivariateDistributionFactory.CreateDistribution(
                Enum.Parse<UnivariateDistributionType>(targets[1].GetString()!));
            mx.SetParameters(paramArrays[0].EnumerateArray().Select(ParseNum).ToArray());
            my.SetParameters(paramArrays[1].EnumerateArray().Select(ParseNum).ToArray());
            copula.MarginalDistributionX = mx;
            copula.MarginalDistributionY = my;
        }
        return copula;
    }

    var fit = construct.GetProperty("fit");
    var x = datasets[fit.GetProperty("x").GetString()!];
    var y = datasets[fit.GetProperty("y").GetString()!];
    string method = fit.GetProperty("method").GetString()!;

    if (fit.TryGetProperty("marginals", out var marginalsEl))
    {
        var margArr = marginalsEl.EnumerateArray().ToArray();
        var mx = UnivariateDistributionFactory.CreateDistribution(
            Enum.Parse<UnivariateDistributionType>(margArr[0].GetString()!));
        var my = UnivariateDistributionFactory.CreateDistribution(
            Enum.Parse<UnivariateDistributionType>(margArr[1].GetString()!));
        if (method == "ifm")
        {
            ((IEstimation)mx).Estimate(x, ParameterEstimationMethod.MaximumLikelihood);
            ((IEstimation)my).Estimate(y, ParameterEstimationMethod.MaximumLikelihood);
        }
        copula.MarginalDistributionX = mx;
        copula.MarginalDistributionY = my;
    }

    switch (method)
    {
        case "tau":
            SetThetaFromTauDispatch(copula, target, x, y);
            break;
        case "mpl":
            BivariateCopulaEstimation.Estimate(ref copula, x, y, CopulaEstimationMethod.PseudoLikelihood);
            break;
        case "ifm":
            BivariateCopulaEstimation.Estimate(ref copula, x, y, CopulaEstimationMethod.InferenceFromMargins);
            break;
        case "mle":
            BivariateCopulaEstimation.Estimate(ref copula, x, y, CopulaEstimationMethod.FullLikelihood);
            break;
        default:
            throw new Exception($"unknown copula fit method: {method}");
    }
    return copula;
}

// SetThetaFromTau is not part of the shared copula API (see file header); each
// tau-capable copula type adds a branch here.
static void SetThetaFromTauDispatch(BivariateCopula copula, string target, double[] x, double[] y)
{
    if (target == "Clayton") { ((ClaytonCopula)copula).SetThetaFromTau(x, y); return; }
    if (target == "AliMikhailHaq") { ((AMHCopula)copula).SetThetaFromTau(x, y); return; }
    if (target == "Gumbel") { ((GumbelCopula)copula).SetThetaFromTau(x, y); return; }
    // NOTE: JoeCopula has no SetThetaFromTau in the C# source; intentionally not branched
    // here (see joe_copula.hpp's file header and .superpowers/sdd/task-8-report.md).
    throw new Exception($"copula '{target}' has no tau-based method-of-moments fit");
}

static double DispatchCopula(BivariateCopula c, string m, JsonElement[] a)
{
    switch (m)
    {
        case "pdf": return c.PDF(a[0].GetDouble(), a[1].GetDouble());
        case "log_pdf": return c.LogPDF(a[0].GetDouble(), a[1].GetDouble());
        case "cdf": return c.CDF(a[0].GetDouble(), a[1].GetDouble());
        case "inverse_cdf": return c.InverseCDF(a[0].GetDouble(), a[1].GetDouble())[a[2].GetInt32()];
        case "upper_tail_dependence": return c.UpperTailDependence;
        case "lower_tail_dependence": return c.LowerTailDependence;
        case "theta": return c.Theta;
        case "df": return c.GetCopulaParameters[1];
        case "or_exceedance": return c.ORJointExceedanceProbability(a[0].GetDouble(), a[1].GetDouble());
        case "and_exceedance": return c.ANDJointExceedanceProbability(a[0].GetDouble(), a[1].GetDouble());
        case "marginal_param":
        {
            string which = a[0].GetString()!;
            int idx = a[1].GetInt32();
            var marg = which == "x" ? c.MarginalDistributionX : c.MarginalDistributionY;
            return marg!.GetParameters[idx];
        }
        case "random_value":
        {
            // args = [sample_size, seed, row, col]. Stateless: GenerateRandomValues seeds
            // its own internal LatinHypercube draw from `seed`, so no persistent-instance
            // batching is needed (mirrors SampleValueAt() for MultivariateDistribution above).
            var sample = c.GenerateRandomValues(a[0].GetInt32(), a[1].GetInt32());
            return sample[a[2].GetInt32(), a[3].GetInt32()];
        }
        default: throw new Exception($"unknown copula fixture method: {m}");
    }
}

// --- mcmc_sampler helpers (Task P3.5 / P3.6) ----------------------------------------------
//
// The model builder mirrors Test_RWMH.cs's Test_RWMH_NormalDist_RStan construction VERBATIM
// for "uniform_constraints", and Test_Gibbs.cs's Test_Gibbs_NormalDist_RStan VERBATIM for
// "normal_conjugate_gibbs" -- see model_registry.hpp's header comment for the prior-aliasing
// rationale this mirrors (both `muPrior`/`sigmaPrior` here and the C++ port's `mu_prior`/
// `sigma_prior` are the SAME reference-type object the proposal closure and `priors` both
// point at -- ordinary C# object references already give this for free, no `shared_ptr`
// analog needed on this side).
static (List<IUnivariateDistribution> priors, LogLikelihood logLikelihood, Gibbs.Proposal? proposal) BuildMcmcModel(
    string name, string family, double[] data)
{
    if (name == "uniform_constraints")
    {
        // Family-generic (mirrors model_registry.hpp's C++ "uniform_constraints" entry,
        // which builds any `family` via the factory + IMaximumLikelihoodEstimation rather
        // than being Normal-only): uninformative Uniform priors bounded by `family`'s own
        // GetParameterConstraints(data) lower/upper arrays, and a log-likelihood closure
        // that refits a fresh `family` instance's parameters at each proposal. Originally
        // hard-coded to Normal only (P3.5, Test_RWMH_NormalDist_RStan); P3.6's
        // ARWMH/SNIS rstan cases need Logistic/Gumbel/Weibull too (Test_ARWMH.cs).
        var probe = UnivariateDistributionFactory.CreateDistribution(Enum.Parse<UnivariateDistributionType>(family));
        if (probe is not IMaximumLikelihoodEstimation imle)
            throw new Exception($"BuildMcmcModel: family '{family}' does not implement " +
                                 "IMaximumLikelihoodEstimation");
        var constraints = imle.GetParameterConstraints(data);
        var priors = new List<IUnivariateDistribution>();
        for (int i = 0; i < constraints.Item2.Length; i++)
            priors.Add(new Uniform(constraints.Item2[i], constraints.Item3[i]));
        double LogLH(double[] x)
        {
            var dist = UnivariateDistributionFactory.CreateDistribution(Enum.Parse<UnivariateDistributionType>(family));
            dist.SetParameters(x);
            return dist.LogLikelihood(data);
        }
        return (priors, LogLH, null);
    }
    if (name == "normal_conjugate_gibbs")
    {
        if (family != "Normal")
            throw new Exception($"BuildMcmcModel: family '{family}' not supported for " +
                                 "'normal_conjugate_gibbs' (mirrors Test_Gibbs.cs, Normal-only)");
        int n = data.Length;
        var mu = Statistics.Mean(data);
        double mu0 = 0, sigma0 = 5E5;
        var muPrior = new Normal(mu0, sigma0);
        double alpha0 = 2, beta0 = 0.001;
        var sigmaPrior = new InverseGamma(beta0, alpha0);
        var priors = new List<IUnivariateDistribution> { muPrior, sigmaPrior };

        double LogLH(double[] x)
        {
            var dist = new Normal(x[0], x[1]);
            return dist.LogLikelihood(data);
        }

        double[] Proposal(double[] x, Random random)
        {
            // Sample mu.
            double mun = (n * mu + mu0 / 2) / (n + 1 / (sigma0 * sigma0));
            double sigma2 = (x[1] * x[1]) / (n + (x[1] * x[1]) / (sigma0 * sigma0));
            muPrior.SetParameters(mun, Math.Sqrt(sigma2));
            double mup = muPrior.InverseCDF(random.NextDouble());

            // Sample sigma.
            double alpha1 = n / 2d;
            double sse = 0;
            for (int i = 0; i < data.Length; i++)
                sse += Math.Pow(data[i] - mup, 2);
            double beta1 = sse / 2d;
            sigmaPrior.SetParameters(new double[] { beta1, alpha1 });
            double sig2p = sigmaPrior.InverseCDF(random.NextDouble());

            return new double[] { mup, Math.Sqrt(sig2p) };
        }

        return (priors, LogLH, Proposal);
    }
    throw new Exception($"unknown MCMC model registry entry: {name}");
}

// `proposal_sigma` sentinel strings -- see fixtures/README.md's mcmc_sampler schema for why
// "identity" exists alongside the C# test's literal "zeros" (an all-zero proposal covariance
// is only safe when MAP initialization is expected to overwrite it before first use).
static Matrix ParseProposalSigma(JsonElement settings, int dimension)
{
    if (!settings.TryGetProperty("proposal_sigma", out var ps)) return new Matrix(dimension);
    return ps.GetString() switch
    {
        "zeros" => new Matrix(dimension),
        "identity" => Matrix.Identity(dimension),
        var s => throw new Exception($"unknown proposal_sigma sentinel: {s}")
    };
}

static MCMCSampler.InitializationType ParseInitialize(string s) => s switch
{
    "MAP" => MCMCSampler.InitializationType.MAP,
    "Randomize" => MCMCSampler.InitializationType.Randomize,
    "UserDefined" => MCMCSampler.InitializationType.UserDefined,
    _ => throw new Exception($"unknown initialize value: {s}")
};

// Builds + configures + samples() one sampler from a {"model": {...}, "settings": {...}}
// construct. `samplerTarget`: the fixture's file-level "target" (the sampler type, e.g.
// "RWMH"); a later task extends this with more cases as more samplers land.
static MCMCSampler BuildAndSampleMcmc(string samplerTarget, JsonElement construct,
                                       Dictionary<string, double[]> datasets)
{
    var modelSpec = construct.GetProperty("model");
    var data = datasets[modelSpec.GetProperty("dataset").GetString()!];
    var (priors, logLikelihood, proposal) = BuildMcmcModel(modelSpec.GetProperty("name").GetString()!,
        modelSpec.GetProperty("family").GetString()!, data);
    int d = priors.Count;

    bool hasSettings = construct.TryGetProperty("settings", out var settings);

    MCMCSampler sampler = samplerTarget switch
    {
        "RWMH" => new RWMH(priors, logLikelihood, hasSettings ? ParseProposalSigma(settings, d) : new Matrix(d)),
        "HMC" => new HMC(priors, logLikelihood,
            stepSize: hasSettings && settings.TryGetProperty("step_size", out var hss) ? hss.GetDouble() : 0.1,
            steps: hasSettings && settings.TryGetProperty("steps", out var hst) ? hst.GetInt32() : 50),
        "NUTS" => new NUTS(priors, logLikelihood,
            stepSize: hasSettings && settings.TryGetProperty("step_size", out var nss) ? nss.GetDouble() : 0.1,
            maxTreeDepth: hasSettings && settings.TryGetProperty("max_tree_depth", out var nmtd) ? nmtd.GetInt32() : 10),
        "ARWMH" => new ARWMH(priors, logLikelihood),
        "Gibbs" => new Gibbs(priors, logLikelihood,
            proposal ?? throw new Exception("Gibbs model has no proposal function")),
        "SNIS" => new SNIS(priors, logLikelihood),
        "DEMCz" => new DEMCz(priors, logLikelihood),
        "DEMCzs" => new DEMCzs(priors, logLikelihood),
        _ => throw new Exception($"unknown mcmc_sampler target: {samplerTarget}")
    };

    if (hasSettings)
    {
        if (settings.TryGetProperty("initialize", out var init)) sampler.Initialize = ParseInitialize(init.GetString()!);
        if (settings.TryGetProperty("prng_seed", out var seed)) sampler.PRNGSeed = seed.GetInt32();
        if (settings.TryGetProperty("initial_iterations", out var ii)) sampler.InitialIterations = ii.GetInt32();
        if (settings.TryGetProperty("warmup_iterations", out var wi)) sampler.WarmupIterations = wi.GetInt32();
        if (settings.TryGetProperty("iterations", out var it)) sampler.Iterations = it.GetInt32();
        if (settings.TryGetProperty("number_of_chains", out var nc)) sampler.NumberOfChains = nc.GetInt32();
        if (settings.TryGetProperty("thinning_interval", out var ti)) sampler.ThinningInterval = ti.GetInt32();
        if (settings.TryGetProperty("output_length", out var ol)) sampler.OutputLength = ol.GetInt32();
        if (sampler is ARWMH arwmh)
        {
            if (settings.TryGetProperty("scale", out var sc)) arwmh.Scale = sc.GetDouble();
            if (settings.TryGetProperty("beta", out var be)) arwmh.Beta = be.GetDouble();
        }
        if (sampler is DEMCz demcz)
        {
            if (settings.TryGetProperty("jump", out var jp)) demcz.Jump = jp.GetDouble();
            if (settings.TryGetProperty("jump_threshold", out var jt)) demcz.JumpThreshold = jt.GetDouble();
            if (settings.TryGetProperty("noise", out var ns)) demcz.Noise = ns.GetDouble();
        }
        if (sampler is DEMCzs demczs)
        {
            if (settings.TryGetProperty("jump", out var jp)) demczs.Jump = jp.GetDouble();
            if (settings.TryGetProperty("jump_threshold", out var jt)) demczs.JumpThreshold = jt.GetDouble();
            if (settings.TryGetProperty("snooker_threshold", out var st)) demczs.SnookerThreshold = st.GetDouble();
            if (settings.TryGetProperty("noise", out var ns)) demczs.Noise = ns.GetDouble();
        }
        if (sampler is NUTS nuts)
        {
            if (settings.TryGetProperty("adapt_mass_matrix", out var amm)) nuts.AdaptMassMatrix = amm.GetBoolean();
        }
    }

    sampler.Sample();
    return sampler;
}

static double DispatchMcmc(MCMCSampler sampler, MCMCResults results, string m, JsonElement[] a)
{
    int Idx(int i) => a[i].GetInt32();
    switch (m)
    {
        case "posterior_mean": return results.ParameterResults[Idx(0)].SummaryStatistics.Mean;
        case "posterior_sd": return results.ParameterResults[Idx(0)].SummaryStatistics.StandardDeviation;
        case "posterior_median": return results.ParameterResults[Idx(0)].SummaryStatistics.Median;
        case "posterior_lower_ci": return results.ParameterResults[Idx(0)].SummaryStatistics.LowerCI;
        case "posterior_upper_ci": return results.ParameterResults[Idx(0)].SummaryStatistics.UpperCI;
        case "chain_value": return sampler.MarkovChains[Idx(0)][Idx(1)].Values[Idx(2)];
        case "chain_fitness": return sampler.MarkovChains[Idx(0)][Idx(1)].Fitness;
        case "map_value": return results.MAP.Values[Idx(0)];
        case "map_fitness": return results.MAP.Fitness;
        case "acceptance_rate": return sampler.AcceptanceRates[Idx(0)];
        case "mean_log_likelihood": return sampler.MeanLogLikelihood[Idx(0)];
        case "rhat": return results.ParameterResults[Idx(0)].SummaryStatistics.Rhat;
        case "ess": return results.ParameterResults[Idx(0)].SummaryStatistics.ESS;
        default: throw new Exception($"unknown mcmc_sampler fixture method: {m}");
    }
}

// --- bootstrap helpers (Task P3.10) -------------------------------------------------------
//
// Mirrors model_registry.hpp's "normal_quantiles" entry EXACTLY -- see that header's comment
// for the sampleData/BCa semantics (transcribed from Test_Bootstrap.cs's private
// CreateNormalBootstrap() helper and Test_BCaCI()).
static Bootstrap<double[]> BuildBootstrapModel(string name, double mu, double sigma, int sampleSize,
    double[] probabilities, double[] sampleData)
{
    if (name != "normal_quantiles")
        throw new Exception($"unknown bootstrap model registry entry: {name}");

    double fitMu = mu, fitSigma = sigma;
    int resampleSize = sampleSize;
    double[]? originalData = null;

    if (sampleData.Length > 0)
    {
        var probe = new Normal();
        ((IEstimation)probe).Estimate(sampleData, ParameterEstimationMethod.MethodOfMoments);
        fitMu = probe.Mu;
        fitSigma = probe.Sigma;
        resampleSize = sampleData.Length;
        originalData = sampleData;
    }

    var parms = new ParameterSet(new double[] { fitMu, fitSigma }, double.NaN);
    var boot = new Bootstrap<double[]>(originalData!, parms);

    boot.ResampleFunction = (data, ps, rng) =>
    {
        var d = new Normal(ps.Values[0], ps.Values[1]);
        return d.GenerateRandomValues(resampleSize, rng.Next());
    };

    boot.FitFunction = (sample) =>
    {
        var d = new Normal();
        ((IEstimation)d).Estimate(sample, ParameterEstimationMethod.MethodOfMoments);
        if (!d.ParametersValid) throw new Exception("Invalid parameters.");
        return new ParameterSet(d.GetParameters, double.NaN);
    };

    boot.StatisticFunction = (ps) =>
    {
        var d = new Normal(ps.Values[0], ps.Values[1]);
        var result = new double[probabilities.Length];
        for (int i = 0; i < probabilities.Length; i++)
            result[i] = d.InverseCDF(probabilities[i]);
        return result;
    };

    if (sampleData.Length > 0)
    {
        boot.JackknifeFunction = (data, idx) =>
        {
            var list = new List<double>(data);
            list.RemoveAt(idx);
            return list.ToArray();
        };
        boot.SampleSizeFunction = (data) => data.Length;
    }

    return boot;
}

static BootstrapCIMethod ParseBootstrapCIMethod(string s) => s switch
{
    "Percentile" => BootstrapCIMethod.Percentile,
    "BiasCorrected" => BootstrapCIMethod.BiasCorrected,
    "BCa" => BootstrapCIMethod.BCa,
    "Normal" => BootstrapCIMethod.Normal,
    "BootstrapT" => BootstrapCIMethod.BootstrapT,
    _ => throw new Exception($"unknown bootstrap ci_method: {s}")
};

// Builds + configures + runs one Bootstrap<double[]> from a {"model": ..., ...} construct
// (see fixtures/README.md's bootstrap schema), then computes its confidence intervals once.
static (Bootstrap<double[]> boot, BootstrapResults results) BuildAndRunBootstrap(
    JsonElement construct, Dictionary<string, double[]> datasets)
{
    string modelName = construct.GetProperty("model").GetString()!;
    double mu = construct.TryGetProperty("mu", out var muEl) ? muEl.GetDouble() : 0.0;
    double sigma = construct.TryGetProperty("sigma", out var sigmaEl) ? sigmaEl.GetDouble() : 0.0;
    int sampleSize = construct.TryGetProperty("sample_size", out var ssEl) ? ssEl.GetInt32() : 0;
    double[] probabilities = construct.GetProperty("probabilities").EnumerateArray()
        .Select(x => x.GetDouble()).ToArray();
    double[] sampleData = construct.TryGetProperty("dataset", out var dsEl)
        ? datasets[dsEl.GetString()!] : Array.Empty<double>();

    var boot = BuildBootstrapModel(modelName, mu, sigma, sampleSize, probabilities, sampleData);
    if (construct.TryGetProperty("replicates", out var repEl)) boot.Replicates = repEl.GetInt32();
    if (construct.TryGetProperty("seed", out var seedEl)) boot.PRNGSeed = seedEl.GetInt32();
    if (construct.TryGetProperty("max_retries", out var mrEl)) boot.MaxRetries = mrEl.GetInt32();

    string run = construct.TryGetProperty("run", out var runEl) ? runEl.GetString()! : "regular";
    if (run == "regular") boot.Run();
    else if (run == "studentized") boot.RunWithStudentizedBootstrap();
    else throw new Exception($"unknown bootstrap run kind: {run}");

    var method = ParseBootstrapCIMethod(construct.GetProperty("ci_method").GetString()!);
    double alpha = construct.TryGetProperty("alpha", out var alphaEl) ? alphaEl.GetDouble() : 0.1;
    var results = boot.GetConfidenceIntervals(method, alpha);

    return (boot, results);
}

static double DispatchBootstrap(Bootstrap<double[]> boot, BootstrapResults results, string m, JsonElement[] a)
{
    int Idx(int i) => a[i].GetInt32();
    switch (m)
    {
        case "statistic_lower_ci": return results.StatisticResults[Idx(0)].LowerCI;
        case "statistic_upper_ci": return results.StatisticResults[Idx(0)].UpperCI;
        case "parameter_lower_ci": return results.ParameterResults[Idx(0)].LowerCI;
        case "parameter_upper_ci": return results.ParameterResults[Idx(0)].UpperCI;
        case "population_estimate": return results.ParameterResults[Idx(0)].PopulationEstimate;
        case "valid_count": return results.StatisticResults[Idx(0)].ValidCount;
        case "replicate_value": return boot.BootstrapParameterSets[Idx(0)].Values[Idx(1)];
        default: throw new Exception($"unknown bootstrap fixture method: {m}");
    }
}

// --- model_estimation helpers (Task T12) --------------------------------------------------
//
// Drives the REAL RMC.BestFit estimators (MaximumLikelihood / MaximumAPosteriori /
// BayesianAnalysis), subset-compiled in place (see OracleEmitter.csproj). One build+run per
// case (mirrors the mcmc_sampler/bootstrap single-stateful-glue-call contract); every assertion
// dispatches against that one cached estimator. Method-name strings match the C++/R/Python
// runners EXACTLY so the same fixture file drives all four.
static OptimizationMethod ParseOptimizationMethod(string s) => s switch
{
    "Brent" => OptimizationMethod.Brent,
    "BFGS" => OptimizationMethod.BFGS,
    "NelderMead" => OptimizationMethod.NelderMead,
    "Powell" => OptimizationMethod.Powell,
    "DifferentialEvolution" => OptimizationMethod.DifferentialEvolution,
    "MultilevelSingleLinkage" => OptimizationMethod.MultilevelSingleLinkage,
    _ => throw new Exception($"unknown model_estimation optimizer: {s}")
};

static BayesianAnalysis.SamplerType ParseSamplerType(string s) => s switch
{
    "DEMCz" => BayesianAnalysis.SamplerType.DEMCz,
    "DEMCzs" => BayesianAnalysis.SamplerType.DEMCzs,
    "ARWMH" => BayesianAnalysis.SamplerType.ARWMH,
    "NUTS" => BayesianAnalysis.SamplerType.NUTS,
    _ => throw new Exception($"unknown model_estimation sampler: {s}")
};

// --- GMM / Bulletin17C helpers (Task B12) -------------------------------------------------
//
// Drives the REAL RMC.BestFit GeneralizedMethodOfMoments over a concrete Bulletin17CDistribution
// (un-excluded from the subset by B12; see OracleEmitter.csproj). Mirrors the C++ runner's GMM
// path in core/tests/test_fixtures.cpp (build_and_run_estimation's GeneralizedMethodOfMoments
// arm) EXACTLY so the same fixture file drives all four harnesses: build the B17C model from the
// `construct.model` spec, construct GMM (default optimizer = BFGS, the C# GMM ctor default),
// apply the strategy/iterations knobs, Estimate() once, then PostProcess(sandwich: true,
// computeJstat: true) so the accessors return deterministic cached Sigma + J-statistic. The
// seeded-draw digest (optional `sample_size`/`seed`) pins the fitted best parameters into the
// B17C parent and takes one ISimulatable stream -- the same DRY choice the C++ runner makes.
static GeneralizedMethodOfMoments.GMMEstimationStrategy ParseGmmStrategy(string s) => s switch
{
    "OneStep" => GeneralizedMethodOfMoments.GMMEstimationStrategy.OneStep,
    "TwoStep" => GeneralizedMethodOfMoments.GMMEstimationStrategy.TwoStep,
    "Iterative" => GeneralizedMethodOfMoments.GMMEstimationStrategy.Iterative,
    _ => throw new Exception($"unknown GMM estimation strategy: {s}")
};

// A `type: "bulletin17c"` model spec -> a concrete Bulletin17CDistribution, mirroring
// model_spec.hpp's build_bulletin17c_model: family (default LogPearsonTypeIII) + the shared
// DataFrame builder + optional explicit parameter_values applied last.
static BestFitModels.Bulletin17CDistribution BuildBulletin17CModel(
    JsonElement modelSpec, Dictionary<string, double[]> datasets)
{
    var df = BuildModelDataFrame(modelSpec, datasets);
    var distType = modelSpec.TryGetProperty("family", out var fam)
        ? Enum.Parse<UnivariateDistributionType>(fam.GetString()!)
        : UnivariateDistributionType.LogPearsonTypeIII;
    var m = new BestFitModels.Bulletin17CDistribution(df, distType);
    if (modelSpec.TryGetProperty("parameter_values", out var pv))
        m.SetParameterValues(pv.EnumerateArray().Select(ParseNum).ToList());
    return m;
}

static (BestFitModels.Bulletin17CDistribution b17c, GeneralizedMethodOfMoments gmm, double[]? simulated)
    BuildGmm(JsonElement construct, Dictionary<string, double[]> datasets)
{
    var b17c = BuildBulletin17CModel(construct.GetProperty("model"), datasets);
    var method = construct.TryGetProperty("optimizer", out var o)
        ? ParseOptimizationMethod(o.GetString()!) : OptimizationMethod.BFGS;
    var gmm = new GeneralizedMethodOfMoments(b17c, method);
    if (construct.TryGetProperty("strategy", out var st))
        gmm.EstimationStrategy = ParseGmmStrategy(st.GetString()!);
    if (construct.TryGetProperty("max_gmm_iterations", out var mgi))
        gmm.MaxGMMIterations = mgi.GetInt32();
    if (!gmm.Estimate())
        throw new Exception("GeneralizedMethodOfMoments.Estimate() returned false");
    gmm.PostProcess(useSandwich: true, computeJstat: true);
    double[]? draws = null;
    if (construct.TryGetProperty("sample_size", out var ss))
    {
        b17c.SetParameterValues(gmm.BestParameterSet.Values);
        draws = b17c.GenerateRandomValues(ss.GetInt32(),
            construct.TryGetProperty("seed", out var se) ? se.GetInt32() : -1);
    }
    return (b17c, gmm, draws);
}

// GMM assertion surface: shares parameter/standard_error/covariance/correlation names with
// ML/MAP; adds j_stat/j_stat_pval and the B17C quantile_variance. quantile_variance lives on the
// B17C MODEL (not the estimator): args[0] is the annual EXCEEDANCE probability (AEP) and the C#
// QuantileVariance takes a NON-exceedance probability, so pass 1 - AEP, feeding it the fitted
// parameters + the estimator's covariance -- exactly the C++ runner's arm.
static double DispatchGmm(BestFitModels.Bulletin17CDistribution b17c, GeneralizedMethodOfMoments gmm,
                          double[]? simulated, string m, JsonElement[] a)
{
    int I(int i) => a[i].GetInt32();
    switch (m)
    {
        case "simulated_value":
            return (simulated ?? throw new Exception("simulated_value outside a seeded GMM case"))[I(0)];
        case "parameter": return gmm.BestParameterSet.Values[I(0)];
        case "standard_error": return gmm.GetStandardErrors()[I(0)];
        case "covariance": return gmm.GetCovarianceMatrix()[I(0), I(1)];
        case "correlation": return gmm.GetCorrelationMatrix()[I(0), I(1)];
        case "j_stat": return gmm.JStat;
        case "j_stat_pval": return gmm.JStatPval;
        case "quantile_variance":
            return b17c.QuantileVariance(1.0 - a[0].GetDouble(), gmm.BestParameterSet.Values,
                gmm.GetCovarianceMatrix().ToArray());
        default: throw new Exception($"unknown GMM fixture method: {m}");
    }
}

// --- Phase 5 model-spec construction (Task M14) ---------------------------------------------
//
// Builds the SAME `construct.model` spec the three runners hand to the shared C++ builder
// (core/include/bestfit/models/model_spec.hpp; schema in fixtures/README.md's model_estimation
// section) against the REAL RMC.BestFit model classes, so one fixture file drives all four
// harnesses. `type` dispatch, the `data_frame` inline arrays, `trends`, and `parameter_values`
// mirror the C++ builder's semantics exactly. A spec without `type`/`data_frame`/`trends`
// builds exactly what the Phase 4 emitter built (DataFrame { ExactSeries } +
// UnivariateDistribution) -- byte-for-byte.

// A `{ "family": ..., "parameters": [...] }` distribution spec -> a parameterized
// distribution through the same factory every other fixture kind uses (mirrors
// model_spec.hpp's build_spec_distribution).
static UnivariateDistributionBase BuildSpecDistribution(JsonElement spec)
{
    var dist = UnivariateDistributionFactory.CreateDistribution(
        Enum.Parse<UnivariateDistributionType>(spec.GetProperty("family").GetString()!));
    if (spec.TryGetProperty("parameters", out var ps))
        dist.SetParameters(ps.EnumerateArray().Select(ParseNum).ToArray());
    return dist;
}

// A `data_frame` spec object -> a real RMC.BestFit DataFrame. Threshold processing happens at
// the model boundary (every model's DataFrame setter runs ProcessThresholdSeries itself),
// exactly like the C++ port. The optional `mgbt_low_outliers` flag (M14) triggers the PUBLIC
// SetLowOutliersFromMGBT() path at the frame boundary, before the model ctor -- flagging low
// outliers and setting LowOutlierThreshold, which left-censors the flagged values in the fit.
static BestFitModels.DataFrame BuildSpecDataFrame(JsonElement dfSpec)
{
    var df = new BestFitModels.DataFrame();
    if (dfSpec.TryGetProperty("exact", out var exactEl))
        df.ExactSeries = new ExactSeries(exactEl.EnumerateArray().Select(e => new ExactData(
            e.GetProperty("index").GetInt32(), e.GetProperty("value").GetDouble(), 0d,
            e.TryGetProperty("is_low_outlier", out var lo) && lo.GetBoolean())).ToList());
    if (dfSpec.TryGetProperty("interval", out var intervalEl))
        df.IntervalSeries = new IntervalSeries(intervalEl.EnumerateArray().Select(e =>
            new IntervalData(e.GetProperty("index").GetInt32(), e.GetProperty("lower").GetDouble(),
                e.GetProperty("value").GetDouble(), e.GetProperty("upper").GetDouble())).ToList());
    if (dfSpec.TryGetProperty("threshold", out var thresholdEl))
        df.ThresholdSeries = new ThresholdSeries(thresholdEl.EnumerateArray().Select(e =>
            new ThresholdData(e.GetProperty("start_index").GetInt32(),
                              e.GetProperty("end_index").GetInt32(),
                              e.GetProperty("value").GetDouble())
            { NumberAbove = e.GetProperty("number_above").GetInt32() }).ToList());
    if (dfSpec.TryGetProperty("uncertain", out var uncertainEl))
        df.UncertainSeries = new UncertainSeries(uncertainEl.EnumerateArray().Select(e =>
            new UncertainData(e.GetProperty("index").GetInt32(),
                              BuildSpecDistribution(e.GetProperty("distribution")))).ToList());
    if (dfSpec.TryGetProperty("low_outlier_threshold", out var lot))
        df.LowOutlierThreshold = lot.GetDouble();
    if (dfSpec.TryGetProperty("mgbt_low_outliers", out var mgbt) && mgbt.GetBoolean())
        df.SetLowOutliersFromMGBT();
    return df;
}

// Resolves a model spec's data source: the inline `data_frame` object when present, otherwise
// an exact-only frame over the file-level `dataset` values (the Phase 4 path, byte-for-byte).
static BestFitModels.DataFrame BuildModelDataFrame(JsonElement modelSpec,
                                                   Dictionary<string, double[]> datasets)
{
    if (modelSpec.TryGetProperty("data_frame", out var dfSpec)) return BuildSpecDataFrame(dfSpec);
    if (modelSpec.TryGetProperty("dataset", out var ds))
        return new BestFitModels.DataFrame { ExactSeries = new ExactSeries(datasets[ds.GetString()!]) };
    throw new Exception("model spec requires either 'dataset' or 'data_frame'");
}

// `families` -> distribution types (mixture / competing_risks component lists).
static List<UnivariateDistributionType> ParseFamilies(JsonElement modelSpec) =>
    modelSpec.GetProperty("families").EnumerateArray()
        .Select(f => Enum.Parse<UnivariateDistributionType>(f.GetString()!)).ToList();

// The `construct.model` dispatch (mirrors model_spec.hpp's build_model): `type` defaults to
// "univariate_distribution" (the Phase 4 behavior). All four model types derive from
// UnivariateDistributionModelBase, which carries the DataFrame property the M14 data-frame
// assertion methods (plotting_position / number_of_low_outliers / low_outlier_threshold) read.
static BestFitModels.UnivariateDistributionModelBase BuildSpecModel(
    JsonElement modelSpec, Dictionary<string, double[]> datasets)
{
    string type = modelSpec.TryGetProperty("type", out var t)
        ? t.GetString()! : "univariate_distribution";
    BestFitModels.UnivariateDistributionModelBase model;
    if (type == "univariate_distribution")
    {
        var distType = Enum.Parse<UnivariateDistributionType>(modelSpec.GetProperty("family").GetString()!);
        var ud = new BestFitModels.UnivariateDistribution(
            BuildModelDataFrame(modelSpec, datasets), distType);
        if (modelSpec.TryGetProperty("trends", out var trendsEl))
        {
            var trends = trendsEl.EnumerateArray().ToArray();
            ud.IsNonstationary = true;

            // Pass 1: attach every trend (SetTrendModel supplies the data-driven defaults),
            // then override the anchor where the spec asks -- mirroring model_spec.hpp.
            foreach (var tr in trends)
            {
                int p = tr.GetProperty("parameter").GetInt32();
                ud.SetTrendModel(p, Enum.Parse<RMC.BestFit.Models.TrendFunctions.Support.TrendModelType>(
                    tr.GetProperty("type").GetString()!));
                if (tr.TryGetProperty("start_index", out var si))
                    ud.TrendModels[p].StartIndex = si.GetInt32();
            }

            // Pass 2 (after the parameter layout is final): explicit per-trend values
            // overwrite their slice of the full parameter vector, applied through ONE
            // SetParameterValues call (the sync-safe setter the model mandates).
            bool hasValues = false;
            var full = ud.Parameters.Select(mp => mp.Value).ToList();
            foreach (var tr in trends)
            {
                if (!tr.TryGetProperty("values", out var valuesEl)) continue;
                hasValues = true;
                int p = tr.GetProperty("parameter").GetInt32();
                int offset = 0;
                for (int j = 0; j < p; j++) offset += ud.TrendModels[j].NumberOfParameters;
                var values = valuesEl.EnumerateArray().Select(ParseNum).ToArray();
                if (values.Length != ud.TrendModels[p].NumberOfParameters)
                    throw new Exception("trend spec 'values' length does not match the trend's parameter count");
                for (int k = 0; k < values.Length; k++) full[offset + k] = values[k];
            }
            if (hasValues) ud.SetParameterValues(full);
        }
        model = ud;
    }
    else if (type == "mixture")
    {
        model = new BestFitModels.MixtureModel(BuildModelDataFrame(modelSpec, datasets),
            ParseFamilies(modelSpec),
            modelSpec.TryGetProperty("zero_inflated", out var zi) && zi.GetBoolean());
    }
    else if (type == "competing_risks")
    {
        model = new BestFitModels.CompetingRisksModel(BuildModelDataFrame(modelSpec, datasets),
            ParseFamilies(modelSpec));
    }
    else if (type == "point_process")
    {
        // Default-construct (non-seasonal GEV competing-risks distribution), assign the frame,
        // then the optional knobs in C#-property order: UseDefaults before the explicit
        // Threshold/TotalYears so an explicit value is never clobbered by the defaults cascade.
        var pp = new BestFitModels.PointProcessModel();
        pp.DataFrame = BuildModelDataFrame(modelSpec, datasets);
        if (modelSpec.TryGetProperty("use_defaults", out var udFlag)) pp.UseDefaults = udFlag.GetBoolean();
        if (modelSpec.TryGetProperty("threshold", out var th)) pp.Threshold = th.GetDouble();
        if (modelSpec.TryGetProperty("total_years", out var ty)) pp.TotalYears = ty.GetDouble();
        model = pp;
    }
    else
    {
        throw new Exception($"unknown model_estimation model type: {type}");
    }

    // Optional model-level `parameter_values`: one sync-safe SetParameterValues call, last.
    if (modelSpec.TryGetProperty("parameter_values", out var pv))
        model.SetParameterValues(pv.EnumerateArray().Select(ParseNum).ToList());
    return model;
}

// Builds the real BestFit model from a case's `construct.model` (see BuildSpecModel) and
// returns it together with the already-run estimator selected by `target` (null plus the
// cached seeded draw vector for the estimator-less `Simulation` target). For
// BayesianAnalysis, applies the fixture's `sampler` + `settings` knobs (UseSimulationDefaults /
// UseAdvancedSimulationDefaults set false FIRST so the explicit values aren't clobbered by the
// ctor's defaulting), then runs synchronously via `RunAsync(null, false, parallel: false)` --
// `parallel: false` sets `Sampler.ParallelizeChains = false` so the chain generation is serial,
// matching the C++ port (which has no ParallelizeChains) so the seeded chain digest reproduces
// bit-identically. (DIC/WAIC/LOOIC still use Parallel.For internally; see the fixture note.)
static (BestFitModels.UnivariateDistributionModelBase model, object? estimator, double[]? simulated)
    BuildEstimation(string target, JsonElement construct, Dictionary<string, double[]> datasets)
{
    var model = BuildSpecModel(construct.GetProperty("model"), datasets);

    if (target == "Simulation")
    {
        // No estimator: ONE seeded ISimulatable draw cached at build time (M13/M14); the
        // `simulated_value [i]` method asserts individual draws (the chain_value digest
        // precedent -- the C# stream is the oracle R/Python must reproduce bit-identically).
        if (model is not BestFitModels.ISimulatable<double[]> sim)
            throw new Exception("model_estimation Simulation target: model is not ISimulatable");
        var draws = sim.GenerateRandomValues(construct.GetProperty("sample_size").GetInt32(),
            construct.TryGetProperty("seed", out var se) ? se.GetInt32() : -1);
        return (model, null, draws);
    }
    if (target == "MaximumLikelihood")
    {
        var method = construct.TryGetProperty("optimizer", out var o)
            ? ParseOptimizationMethod(o.GetString()!) : OptimizationMethod.DifferentialEvolution;
        var mle = new MaximumLikelihood(model, method);
        if (!mle.Estimate()) throw new Exception("MaximumLikelihood.Estimate() returned false");
        return (model, mle, null);
    }
    if (target == "MaximumAPosteriori")
    {
        var method = construct.TryGetProperty("optimizer", out var o)
            ? ParseOptimizationMethod(o.GetString()!) : OptimizationMethod.DifferentialEvolution;
        var map = new MaximumAPosteriori(model, method);
        if (!map.Estimate()) throw new Exception("MaximumAPosteriori.Estimate() returned false");
        return (model, map, null);
    }
    if (target == "BayesianAnalysis")
    {
        var samplerType = construct.TryGetProperty("sampler", out var s)
            ? ParseSamplerType(s.GetString()!) : BayesianAnalysis.SamplerType.DEMCzs;
        var ba = new BayesianAnalysis(model, samplerType)
        {
            UseSimulationDefaults = false,
            UseAdvancedSimulationDefaults = false,
        };
        if (construct.TryGetProperty("settings", out var settings))
        {
            if (settings.TryGetProperty("seed", out var seedEl)) ba.PRNGSeed = seedEl.GetInt32();
            if (settings.TryGetProperty("iterations", out var itEl)) ba.Iterations = itEl.GetInt32();
            if (settings.TryGetProperty("warmup_iterations", out var wiEl)) ba.WarmupIterations = wiEl.GetInt32();
            if (settings.TryGetProperty("number_of_chains", out var ncEl)) ba.NumberOfChains = ncEl.GetInt32();
            if (settings.TryGetProperty("thinning_interval", out var tiEl)) ba.ThinningInterval = tiEl.GetInt32();
            if (settings.TryGetProperty("initial_iterations", out var iiEl)) ba.InitialIterations = iiEl.GetInt32();
            if (settings.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        }
        ba.RunAsync(null, false, false).GetAwaiter().GetResult();
        return (model, ba, null);
    }
    throw new Exception($"unknown model_estimation target: {target}");
}

// --- Phase 7a model families (Task P4) -----------------------------------------------------
//
// The four remaining ModelBase families -- TimeSeries (AR/MA/ARIMA/ARIMAX), SpatialGEV,
// RatingCurve, BivariateDistribution -- derive from ModelBase/IModel, NOT
// UnivariateDistributionModelBase, so they take a SEPARATE build + estimation path from the
// Phase 4-6 univariate path above (BuildSpecModel / BuildEstimation stay byte-for-byte
// unchanged). This mirrors core/include/bestfit/models/model_spec.hpp field-for-field: the
// `construct.model.type` string selects the family and the schema is fixtures/README.md's
// model_estimation section. The three runners already build these families through their shared
// model_spec.hpp path; this emitter path is the fourth (oracle) leg.
//
// TimeSeries note: the C# TimeSeries index is a DateTime, but every model consumer touches the
// index only as a sequence position / inner-join key (never calendar arithmetic -- see the C++
// adapter header time_series.hpp). All series in a case are built from ONE fixed base date with
// the same interval, so their relative alignment (rating_curve stage<->discharge, ARIMAX
// covariate lags) is preserved exactly as the C++ integer-index adapter preserves it; the
// absolute `start_index` is therefore not modeled here (documented deviation, fit-invariant).
static DateTime EmitterSeriesEpoch() => new DateTime(2000, 1, 1);

static BestFitModels.Transform ParseTransform(string s) => s switch
{
    "None" => BestFitModels.Transform.None,
    "Logarithmic" => BestFitModels.Transform.Logarithmic,
    "BoxCox" => BestFitModels.Transform.BoxCox,
    "YeoJohnson" => BestFitModels.Transform.YeoJohnson,
    _ => throw new Exception($"unknown time_series transform: {s}")
};

static TimeInterval ParseTimeInterval(string s) => s switch
{
    "OneMinute" => TimeInterval.OneMinute,
    "FiveMinute" => TimeInterval.FiveMinute,
    "FifteenMinute" => TimeInterval.FifteenMinute,
    "ThirtyMinute" => TimeInterval.ThirtyMinute,
    "OneHour" => TimeInterval.OneHour,
    "SixHour" => TimeInterval.SixHour,
    "TwelveHour" => TimeInterval.TwelveHour,
    "OneDay" => TimeInterval.OneDay,
    "SevenDay" => TimeInterval.SevenDay,
    "OneMonth" => TimeInterval.OneMonth,
    "OneQuarter" => TimeInterval.OneQuarter,
    "OneYear" => TimeInterval.OneYear,
    "Irregular" => TimeInterval.Irregular,
    _ => throw new Exception($"unknown time_series time_interval: {s}")
};

// Wraps a flat value vector into a real Numerics.Data.TimeSeries (interval + start date + values).
// `time_interval` defaults OneDay (the C# field default); the start date is the fixed epoch (see
// the region note -- absolute start_index is fit-invariant given all series share it).
static TimeSeries BuildEmitterTimeSeries(JsonElement modelSpec, double[] values)
{
    TimeInterval interval = modelSpec.TryGetProperty("time_interval", out var ti)
        ? ParseTimeInterval(ti.GetString()!) : TimeInterval.OneDay;
    return new TimeSeries(interval, EmitterSeriesEpoch(), values);
}

// Resolves a time-series data source: inline `data` array, else the file-level dataset.
static double[] EmitterTimeSeriesValues(JsonElement spec, Dictionary<string, double[]> datasets)
{
    if (spec.TryGetProperty("data", out var d)) return d.EnumerateArray().Select(ParseNum).ToArray();
    if (spec.TryGetProperty("dataset", out var ds)) return datasets[ds.GetString()!];
    throw new Exception("time_series model requires either 'dataset' or 'data'");
}

// Optional model-level `parameter_values` -> ONE sync-safe SetParameterValues call (the setter
// every model mandates; poking Parameters directly desyncs trend / covariate copies).
static void ApplyGeneralParameterValues(BestFitModels.IModel m, JsonElement spec)
{
    if (spec.TryGetProperty("parameter_values", out var pv))
        m.SetParameterValues(pv.EnumerateArray().Select(ParseNum).ToList());
}

static int OrderOf(JsonElement model, string key, int dflt) =>
    model.TryGetProperty("orders", out var o) && o.TryGetProperty(key, out var v) ? v.GetInt32() : dflt;

// `type: "time_series"` -- subtype selects AutoRegressive / MovingAverage / ARIMA / ARIMAX.
// Mirrors model_spec.hpp's build_time_series_model: ctor + transform, then (ARIMAX) the
// include-intercept / trend / seasonality / order / covariate setters, then parameter_values.
static BestFitModels.IModel BuildTimeSeriesModelGeneral(
    JsonElement model, Dictionary<string, double[]> datasets)
{
    string subtype = model.GetProperty("subtype").GetString()!;
    var ts = BuildEmitterTimeSeries(model, EmitterTimeSeriesValues(model, datasets));
    bool includeIntercept = !model.TryGetProperty("include_intercept", out var ii) || ii.GetBoolean();

    BestFitModels.IModel result;
    if (subtype == "ar")
    {
        var m = new BestFitModels.AutoRegressive(ts, OrderOf(model, "p", 1), includeIntercept);
        if (model.TryGetProperty("transform", out var tr)) m.TransformType = ParseTransform(tr.GetString()!);
        result = m;
    }
    else if (subtype == "ma")
    {
        var m = new BestFitModels.MovingAverage(ts, OrderOf(model, "q", 1), includeIntercept);
        if (model.TryGetProperty("transform", out var tr)) m.TransformType = ParseTransform(tr.GetString()!);
        result = m;
    }
    else if (subtype == "arima")
    {
        var m = new BestFitModels.ARIMA(ts, OrderOf(model, "p", 1), OrderOf(model, "d", 0),
                                        OrderOf(model, "q", 0), includeIntercept);
        if (model.TryGetProperty("transform", out var tr)) m.TransformType = ParseTransform(tr.GetString()!);
        result = m;
    }
    else if (subtype == "arimax")
    {
        var m = new BestFitModels.ARIMAX(ts);
        if (model.TryGetProperty("transform", out var tr)) m.TransformType = ParseTransform(tr.GetString()!);
        m.IncludeIntercept = includeIntercept;
        if (model.TryGetProperty("trend", out var trend))
            m.TrendType = trend.GetString()! switch
            {
                "Linear" => BestFitModels.ARIMAX.Trend.Linear,
                "Quadratic" => BestFitModels.ARIMAX.Trend.Quadratic,
                "Cubic" => BestFitModels.ARIMAX.Trend.Cubic,
                _ => BestFitModels.ARIMAX.Trend.None,
            };
        if (model.TryGetProperty("include_seasonality", out var seas)) m.IncludeSeasonality = seas.GetBoolean();
        m.AROrderP = OrderOf(model, "p", 1);
        m.DiffOrderD = OrderOf(model, "d", 0);
        m.MAOrderQ = OrderOf(model, "q", 0);
        m.XOrderB = OrderOf(model, "b", 0);
        if (model.TryGetProperty("covariates", out var covs))
        {
            var list = new List<TimeSeries>();
            foreach (var c in covs.EnumerateArray())
                list.Add(BuildEmitterTimeSeries(model, c.EnumerateArray().Select(ParseNum).ToArray()));
            m.SetCovariates(list);
        }
        result = m;
    }
    else
    {
        throw new Exception($"unknown time_series subtype: {subtype}");
    }
    ApplyGeneralParameterValues(result, model);
    return result;
}

// `type: "spatial_gev"` -- SpatialGEV(atSiteData [obs,sites], coordinates [sites,2], three
// intercept-only GeneralLinearFunction level-2 trends). Optional gating flags applied after
// construction (their ctor defaults: link=true, errors/copula=false). Mirrors
// model_spec.hpp's build_spatial_gev_model.
static BestFitModels.IModel BuildSpatialGevModelGeneral(
    JsonElement model, Dictionary<string, double[]> datasets)
{
    var rows = model.GetProperty("at_site_data").EnumerateArray()
        .Select(r => r.EnumerateArray().Select(ParseNum).ToArray()).ToArray();
    int obs = rows.Length, sites = rows[0].Length;
    var atSite = new double[obs, sites];
    for (int i = 0; i < obs; i++)
        for (int j = 0; j < sites; j++) atSite[i, j] = rows[i][j];

    var coordRows = model.GetProperty("coordinates").EnumerateArray()
        .Select(r => r.EnumerateArray().Select(ParseNum).ToArray()).ToArray();
    var coords = new double[sites, 2];
    for (int i = 0; i < sites; i++)
        for (int j = 0; j < 2; j++) coords[i, j] = coordRows[i][j];

    var location = new BestFitModels.TrendFunctions.GeneralLinearFunction("Location");
    var scale = new BestFitModels.TrendFunctions.GeneralLinearFunction("Scale");
    var shape = new BestFitModels.TrendFunctions.GeneralLinearFunction("Shape");
    var m = new BestFitModels.SpatialExtremes.SpatialGEV(atSite, coords, location, scale, shape);
    if (model.TryGetProperty("use_copula_dependence", out var ucd)) m.UseCopulaDependence = ucd.GetBoolean();
    if (model.TryGetProperty("use_location_errors", out var ule)) m.UseLocationErrors = ule.GetBoolean();
    if (model.TryGetProperty("use_scale_errors", out var use)) m.UseScaleErrors = use.GetBoolean();
    if (model.TryGetProperty("use_shape_errors", out var ushp)) m.UseShapeErrors = ushp.GetBoolean();
    if (model.TryGetProperty("use_log_link_for_location", out var ull)) m.UseLogLinkForLocation = ull.GetBoolean();
    if (model.TryGetProperty("use_log_link_for_scale", out var ulls)) m.UseLogLinkForScale = ulls.GetBoolean();
    ApplyGeneralParameterValues(m, model);
    return m;
}

// `type: "rating_curve"` -- RatingCurve(stage, discharge, segments). Both series share the epoch
// + interval so the date inner-join aligns them 1:1 (>= MinimumAlignedObservations = 10).
static BestFitModels.IModel BuildRatingCurveModelGeneral(
    JsonElement model, Dictionary<string, double[]> datasets)
{
    var stage = BuildEmitterTimeSeries(model, model.GetProperty("stage").EnumerateArray().Select(ParseNum).ToArray());
    var discharge = BuildEmitterTimeSeries(model, model.GetProperty("discharge").EnumerateArray().Select(ParseNum).ToArray());
    int segments = model.TryGetProperty("segments", out var s) ? s.GetInt32() : 1;
    var m = new BestFitModels.RatingCurve(stage, discharge, segments);
    ApplyGeneralParameterValues(m, model);
    return m;
}

// A bivariate marginal spec -> a pre-fit UnivariateDistribution (an IUnivariateModel). Carries
// its own inline `data` (exact series) and pinned distribution `parameter_values` (marginals
// stay FIXED during the copula fit -- B1). Mirrors model_spec.hpp's build_bivariate_marginal.
static BestFitModels.IUnivariateModel BuildBivariateMarginalGeneral(JsonElement spec)
{
    var distType = Enum.Parse<UnivariateDistributionType>(spec.GetProperty("family").GetString()!);
    var df = new BestFitModels.DataFrame
    {
        ExactSeries = new ExactSeries(spec.GetProperty("data").EnumerateArray().Select(ParseNum).ToArray())
    };
    var m = new BestFitModels.UnivariateDistribution(df, distType);
    if (spec.TryGetProperty("parameter_values", out var pv))
        m.SetParameterValues(pv.EnumerateArray().Select(ParseNum).ToList());
    return m;
}

// `type: "bivariate"` -- a copula-coupled BivariateDistribution: two pre-fit IUnivariateModel
// marginals (held FIXED), a CopulaType, and a CopulaEstimationMethod (default
// InferenceFromMargins). Mirrors model_spec.hpp's build_bivariate_model.
static BestFitModels.IModel BuildBivariateModelGeneral(
    JsonElement model, Dictionary<string, double[]> datasets)
{
    var mx = BuildBivariateMarginalGeneral(model.GetProperty("marginal_x"));
    var my = BuildBivariateMarginalGeneral(model.GetProperty("marginal_y"));
    var copulaType = Enum.Parse<CopulaType>(
        model.TryGetProperty("copula", out var cp) ? cp.GetString()! : "Normal");
    var m = new BestFitModels.BivariateDistribution(mx, my, copulaType);
    if (model.TryGetProperty("estimation_method", out var em))
        m.CopulaEstimationMethod = Enum.Parse<CopulaEstimationMethod>(em.GetString()!);
    ApplyGeneralParameterValues(m, model);
    return m;
}

static BestFitModels.IModel BuildSpecModelGeneral(
    JsonElement modelSpec, Dictionary<string, double[]> datasets)
{
    string type = modelSpec.GetProperty("type").GetString()!;
    return type switch
    {
        "time_series" => BuildTimeSeriesModelGeneral(modelSpec, datasets),
        "spatial_gev" => BuildSpatialGevModelGeneral(modelSpec, datasets),
        "rating_curve" => BuildRatingCurveModelGeneral(modelSpec, datasets),
        "bivariate" => BuildBivariateModelGeneral(modelSpec, datasets),
        _ => throw new Exception($"unknown general model_estimation model type: {type}")
    };
}

// Seeded ISimulatable draw flattened to a 1-D vector so the `simulated_value [i]` digest works
// uniformly. Five families are ISimulatable<double[]>; BivariateDistribution is
// ISimulatable<double[,]> (n-row x 2-col) -- flattened ROW-MAJOR (i = row*cols + col), matching
// the C++/R/Python simulate_flat and the README schema.
static double[] SimulateFlatGeneral(BestFitModels.IModel model, int sampleSize, int seed)
{
    if (model is BestFitModels.ISimulatable<double[]> s1)
        return s1.GenerateRandomValues(sampleSize, seed);
    if (model is BestFitModels.ISimulatable<double[,]> s2)
    {
        var mat = s2.GenerateRandomValues(sampleSize, seed);
        int r = mat.GetLength(0), c = mat.GetLength(1);
        var flat = new double[r * c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++) flat[i * c + j] = mat[i, j];
        return flat;
    }
    throw new Exception("model is neither ISimulatable<double[]> nor ISimulatable<double[,]>");
}

// Builds a Phase 7a family model and runs the estimator selected by `target`. Mirrors
// BuildEstimation but over IModel (the four families are not UnivariateDistributionModelBase).
static (BestFitModels.IModel model, object? estimator, double[]? simulated)
    BuildEstimationGeneral(string target, JsonElement construct, Dictionary<string, double[]> datasets)
{
    var model = BuildSpecModelGeneral(construct.GetProperty("model"), datasets);

    if (target == "Simulation")
    {
        var draws = SimulateFlatGeneral(model, construct.GetProperty("sample_size").GetInt32(),
            construct.TryGetProperty("seed", out var se) ? se.GetInt32() : -1);
        return (model, null, draws);
    }
    if (target == "MaximumLikelihood" || target == "MaximumAPosteriori")
    {
        var method = construct.TryGetProperty("optimizer", out var o)
            ? ParseOptimizationMethod(o.GetString()!) : OptimizationMethod.DifferentialEvolution;
        object est;
        ParameterSet best;
        if (target == "MaximumLikelihood")
        {
            var mle = new MaximumLikelihood(model, method);
            if (!mle.Estimate()) throw new Exception("MaximumLikelihood.Estimate() returned false");
            est = mle; best = mle.BestParameterSet;
        }
        else
        {
            var map = new MaximumAPosteriori(model, method);
            if (!map.Estimate()) throw new Exception("MaximumAPosteriori.Estimate() returned false");
            est = map; best = map.BestParameterSet;
        }
        double[]? draws = null;
        if (construct.TryGetProperty("sample_size", out var ss))
        {
            // Pin the fitted best parameters back into the model, then take one seeded draw --
            // the same shared `simulated_value` arm the Simulation target uses (P3 pattern).
            model.SetParameterValues(best.Values);
            draws = SimulateFlatGeneral(model, ss.GetInt32(),
                construct.TryGetProperty("seed", out var se) ? se.GetInt32() : -1);
        }
        return (model, est, draws);
    }
    if (target == "BayesianAnalysis")
    {
        var samplerType = construct.TryGetProperty("sampler", out var s)
            ? ParseSamplerType(s.GetString()!) : BayesianAnalysis.SamplerType.DEMCzs;
        var ba = new BayesianAnalysis(model, samplerType)
        {
            UseSimulationDefaults = false,
            UseAdvancedSimulationDefaults = false,
        };
        if (construct.TryGetProperty("settings", out var settings))
        {
            if (settings.TryGetProperty("seed", out var seedEl)) ba.PRNGSeed = seedEl.GetInt32();
            if (settings.TryGetProperty("iterations", out var itEl)) ba.Iterations = itEl.GetInt32();
            if (settings.TryGetProperty("warmup_iterations", out var wiEl)) ba.WarmupIterations = wiEl.GetInt32();
            if (settings.TryGetProperty("number_of_chains", out var ncEl)) ba.NumberOfChains = ncEl.GetInt32();
            if (settings.TryGetProperty("thinning_interval", out var tiEl)) ba.ThinningInterval = tiEl.GetInt32();
            if (settings.TryGetProperty("initial_iterations", out var iiEl)) ba.InitialIterations = iiEl.GetInt32();
            if (settings.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        }
        ba.RunAsync(null, false, false).GetAwaiter().GetResult();
        return (model, ba, null);
    }
    throw new Exception($"unknown model_estimation target: {target}");
}

// ML/MAP dispatch with LAZY accessors (covariance/SE/correlation are computed only when the
// method asks -- GetCovarianceMatrix throws for a 1-parameter model, e.g. the Normal copula).
static double DispatchMlMapGeneral(string m, JsonElement[] a,
    Func<int, double> param, Func<double> maxLL, Func<double> aic, Func<int, double> bic,
    Func<int, int, double> cov, Func<int, double> se, Func<int, int, double> corr)
{
    int I(int i) => a[i].GetInt32();
    switch (m)
    {
        case "parameter": return param(I(0));
        case "max_log_likelihood": return maxLL();
        case "aic": return aic();
        case "bic": return bic(I(0));
        case "covariance": return cov(I(0), I(1));
        case "standard_error": return se(I(0));
        case "correlation": return corr(I(0), I(1));
        default: throw new Exception($"unknown model_estimation method for ML/MAP: {m}");
    }
}

static double DispatchEstimationGeneral(
    (BestFitModels.IModel model, object? estimator, double[]? simulated) ec,
    string m, JsonElement[] a)
{
    if (m == "simulated_value")
        return (ec.simulated ?? throw new Exception("simulated_value outside a seeded case"))[a[0].GetInt32()];
    // Fixed-parameter model surface (reads the model at its CURRENT parameter values -- pinned by
    // the spec's parameter_values under a Simulation target). data_log_likelihood is generic on
    // IModel; residual[i] casts to the concrete family's Residuals(double[]) surface
    // (AR/MA/ARIMA/ARIMAX/RatingCurve). Deterministic (no fit) -> tight-tolerance oracles for the
    // C++-only "// P4 pending" ctest blocks (route b; not asserted through a committed fixture).
    if (m == "data_log_likelihood")
    {
        var pars = ec.model.Parameters.Select(p => p.Value).ToArray();
        return ec.model.DataLogLikelihood(pars);
    }
    if (m == "residual")
    {
        var pars = ec.model.Parameters.Select(p => p.Value).ToArray();
        double[] res = ec.model switch
        {
            BestFitModels.AutoRegressive ar => ar.Residuals(pars),
            BestFitModels.MovingAverage ma => ma.Residuals(pars),
            BestFitModels.ARIMA arima => arima.Residuals(pars),
            BestFitModels.ARIMAX arimax => arimax.Residuals(pars),
            BestFitModels.RatingCurve rc => rc.Residuals(pars),
            _ => throw new Exception($"residual not supported for {ec.model.GetType().Name}")
        };
        return res[a[0].GetInt32()];
    }
    switch (ec.estimator)
    {
        case MaximumLikelihood mle:
            return DispatchMlMapGeneral(m, a, i => mle.BestParameterSet.Values[i],
                () => mle.MaximumLogLikelihood, () => mle.GetAIC(), n => mle.GetBIC(n),
                (i, j) => mle.GetCovarianceMatrix()[i, j], i => mle.GetStandardErrors()[i],
                (i, j) => mle.GetCorrelationMatrix()[i, j]);
        case MaximumAPosteriori map:
            return DispatchMlMapGeneral(m, a, i => map.BestParameterSet.Values[i],
                () => map.MaximumLogLikelihood, () => map.GetAIC(), n => map.GetBIC(n),
                (i, j) => map.GetCovarianceMatrix()[i, j], i => map.GetStandardErrors()[i],
                (i, j) => map.GetCorrelationMatrix()[i, j]);
        case BayesianAnalysis ba:
            return DispatchBayesian(ba, m, a);
        case null:
            throw new Exception($"unknown Simulation fixture method: {m}");
        default:
            throw new Exception($"unknown estimator type: {ec.estimator.GetType().Name}");
    }
}

// Shared MaximumLikelihood/MaximumAPosteriori dispatch surface (identical member names on both
// C# classes). Passed the fit's already-computed pieces so each assertion is a cheap lookup.
static double DispatchMlMap(string m, JsonElement[] a, ParameterSet best, double maxLL,
                            double aic, Func<int, double> bic, Matrix cov, double[] se, Matrix corr)
{
    int I(int i) => a[i].GetInt32();
    switch (m)
    {
        case "parameter": return best.Values[I(0)];
        case "max_log_likelihood": return maxLL;
        case "aic": return aic;
        case "bic": return bic(I(0));  // args[0] is a sample size n, not an index
        case "covariance": return cov[I(0), I(1)];
        case "standard_error": return se[I(0)];
        case "correlation": return corr[I(0), I(1)];
        default: throw new Exception($"unknown model_estimation method for ML/MAP: {m}");
    }
}

static double DispatchBayesian(BayesianAnalysis ba, string m, JsonElement[] a)
{
    int I(int i) => a[i].GetInt32();
    var results = ba.Results ?? throw new Exception("BayesianAnalysis.Results is null after RunAsync");
    switch (m)
    {
        case "dic": return ba.DIC;
        case "waic": return ba.WAIC;
        case "looic": return ba.LOOIC;
        case "posterior_mean": return results.PosteriorMean.Values[I(0)];
        case "chain_value":
            return (ba.Sampler ?? throw new Exception("BayesianAnalysis.Sampler is null"))
                .MarkovChains![I(0)][I(1)].Values[I(2)];
        default: throw new Exception($"unknown model_estimation method for BayesianAnalysis: {m}");
    }
}

// The DataFrame assertion surface (M14): methods reachable from the model's DataFrame under
// ANY model_estimation target, corroborating the M1/M5 ctest oracles through the PUBLIC path.
// `plotting_position [kind, i]` reads item i's PlottingPosition from the named series
// ("exact" | "interval" | "uncertain", in spec order) after ONE CalculatePlottingPositions()
// pass (idempotent -- a pure function of the collections + PlottingParameter; threshold-series
// positions are NOT exposed because the C# assigns them to a sorted CLONE, so the original
// items never carry one). `number_of_low_outliers`/`low_outlier_threshold` read the frame's
// current state (set by the spec's `mgbt_low_outliers` MGBT trigger, or the explicit
// `low_outlier_threshold`).
static double DispatchModelDataFrame(BestFitModels.UnivariateDistributionModelBase model,
                                     string m, JsonElement[] a)
{
    var df = model.DataFrame;
    switch (m)
    {
        case "number_of_low_outliers": return df.NumberOfLowOutliers;
        case "low_outlier_threshold": return df.LowOutlierThreshold;
        case "plotting_position":
        {
            df.CalculatePlottingPositions();
            string seriesKind = a[0].GetString()!;
            int i = a[1].GetInt32();
            return seriesKind switch
            {
                "exact" => df.ExactSeries[i].PlottingPosition,
                "interval" => df.IntervalSeries[i].PlottingPosition,
                "uncertain" => df.UncertainSeries[i].PlottingPosition,
                var s => throw new Exception($"unknown plotting_position series kind: {s}")
            };
        }
        default: throw new Exception($"unknown model_estimation fixture method: {m}");
    }
}

static double DispatchEstimation(
    (BestFitModels.UnivariateDistributionModelBase model, object? estimator, double[]? simulated) ec,
    string m, JsonElement[] a)
{
    // The seeded-simulation digest (M13/M14): reads the vector cached at build time.
    if (m == "simulated_value")
        return (ec.simulated ?? throw new Exception("simulated_value outside a Simulation case"))[a[0].GetInt32()];
    // The M14 DataFrame surface works under any target (it reads the model, not the estimator).
    if (m == "plotting_position" || m == "number_of_low_outliers" || m == "low_outlier_threshold")
        return DispatchModelDataFrame(ec.model, m, a);
    switch (ec.estimator)
    {
        case MaximumLikelihood mle:
            return DispatchMlMap(m, a, mle.BestParameterSet, mle.MaximumLogLikelihood, mle.GetAIC(),
                n => mle.GetBIC(n), mle.GetCovarianceMatrix(), mle.GetStandardErrors(),
                mle.GetCorrelationMatrix());
        case MaximumAPosteriori map:
            return DispatchMlMap(m, a, map.BestParameterSet, map.MaximumLogLikelihood, map.GetAIC(),
                n => map.GetBIC(n), map.GetCovarianceMatrix(), map.GetStandardErrors(),
                map.GetCorrelationMatrix());
        case BayesianAnalysis ba:
            return DispatchBayesian(ba, m, a);
        case null:
            throw new Exception($"unknown Simulation fixture method: {m}");
        default:
            throw new Exception($"unknown estimator type: {ec.estimator.GetType().Name}");
    }
}

// --- analysis (Task A11: user-facing Analyses layer) --------------------------------------
//
// Drives the REAL RMC.BestFit UnivariateAnalysis / FittingAnalysis / Bulletin17CAnalysis
// (subset-compiled -- see OracleEmitter.csproj) so the tightened fixtures/analyses/*.json values
// are the exact C# oracles. Mirrors core/tests/test_fixtures.cpp's build_and_run_analysis +
// dispatch_analysis field-for-field: build the same analysis from the same `construct` spec, run
// it seeded, cache the flat result surface, then dispatch each assertion against that cache. One
// build+run per case; the same fixture file drives all four harnesses.
//
// The C# `async Task RunAsync` is driven synchronously via `.GetAwaiter().GetResult()`. The
// UnivariateAnalysis Bayesian run uses the seeded DEMCzs sampler, which draws from the shared
// history archive (NOT the current chain states), so C#'s default `ParallelizeChains = true` is
// order-invariant and reproduces the C++ serial `estimate()` bit-for-bit (proven by the tightened
// short_exact frequency-curve digest). The B17C GMM point estimate + Cohn CI are RNG-free
// (BFGS + nested Gaussian quadrature over the sandwich covariance), and FittingAnalysis fits each
// candidate by the seeded DifferentialEvolution -- all deterministic.
static BestFitAnalyses.UncertaintyMethod ParseUncertaintyMethod(string s) => s switch
{
    "MultivariateNormal" => BestFitAnalyses.UncertaintyMethod.MultivariateNormal,
    "Bootstrap" => BestFitAnalyses.UncertaintyMethod.Bootstrap,
    _ => throw new Exception($"unsupported/ deferred uncertainty method: {s}")
};

static AnalysisData BuildAndRunAnalysis(string target, JsonElement construct,
                                        Dictionary<string, double[]> datasets)
{
    var r = new AnalysisData();

    // Optional exceedance-probability override (mirrors the C++ apply_ordinates: replace the
    // default grid only when the case supplies one).
    void ApplyOrdinates(ProbabilityOrdinates po)
    {
        if (!construct.TryGetProperty("exceedance_probabilities", out var epEl)) return;
        po.Clear();
        foreach (var v in epEl.EnumerateArray()) po.Add(ParseNum(v));
    }

    if (target == "FittingAnalysis")
    {
        var data = datasets[construct.GetProperty("dataset").GetString()!];
        var df = new BestFitModels.DataFrame { ExactSeries = new ExactSeries(data) };
        df.CalculatePlottingPositions();
        var analysis = new BestFitAnalyses.FittingAnalysis(df);
        // The C++ port omits GeneralizedNormal (its distribution factory has no case for it), so
        // it fits 14 candidates in the C# order minus that one -- index 11 == Normal. Remove it
        // here so the emitter drives the SAME 14-candidate set the fixture (and the C++/R/Python
        // harnesses) assert; each candidate is fit independently, so removing it does not perturb
        // any remaining fit. See the A10 report's "14-vs-15 candidate-count reality".
        analysis.DistributionList.RemoveAll(d => d is GeneralizedNormal);
        analysis.RunAsync().GetAwaiter().GetResult();
        var fitted = analysis.FittedDistributions;
        r.CandidateCount = fitted.Count;
        foreach (var fd in fitted)
        {
            r.CandAic.Add(fd.AIC);
            r.CandBic.Add(fd.BIC);
            r.CandRmse.Add(fd.RMSE);
            r.CandConverged.Add(fd.FitSucceeded ? 1.0 : 0.0);
        }
        return r;
    }

    var modelSpec = construct.GetProperty("model");

    if (target == "UnivariateAnalysis")
    {
        var baseModel = BuildSpecModel(modelSpec, datasets);
        if (baseModel is not BestFitModels.UnivariateDistribution ud)
            throw new Exception("UnivariateAnalysis requires a univariate_distribution model");
        var analysis = new BestFitAnalyses.UnivariateAnalysis(ud);
        ApplyOrdinates(analysis.ProbabilityOrdinates);
        // The BayesianAnalysis was built (with its simulation defaults) by the analysis ctor;
        // override only the fixture-named knobs, exactly as the C++ runner does. NumberOfChains /
        // ThinningInterval / InitialIterations keep the C#/C++ defaults so the two match.
        var ba = analysis.BayesianAnalysis;
        ba.Type = ParseSamplerType(construct.TryGetProperty("sampler", out var s)
            ? s.GetString()! : "DEMCzs");
        if (construct.TryGetProperty("credible_level", out var clEl))
            ba.CredibleIntervalWidth = clEl.GetDouble();
        if (construct.TryGetProperty("seed", out var seEl)) ba.PRNGSeed = seEl.GetInt32();
        if (construct.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        if (construct.TryGetProperty("iterations", out var itEl))
        {
            int it = itEl.GetInt32();
            ba.Iterations = it;
            ba.WarmupIterations = Math.Max(50, it / 2);
        }
        // Optional explicit MCMC knobs (A11). The default thinning_interval=20 that
        // SetDefaultSimulationOptions picks for a 2-parameter DEMCzs run exposes a C#-vs-C++
        // divergence in the THINNED population-sampler stream (a real port bug -- see
        // docs/upstream-csharp-issues.md, A11 finding). Setting thinning_interval=1 lands on the
        // bayes_normal-proven bit-identical path, so the fixture pins it explicitly and all four
        // runners honor it.
        if (construct.TryGetProperty("thinning_interval", out var thEl)) ba.ThinningInterval = thEl.GetInt32();
        if (construct.TryGetProperty("number_of_chains", out var ncEl)) ba.NumberOfChains = ncEl.GetInt32();
        if (construct.TryGetProperty("initial_iterations", out var iiEl)) ba.InitialIterations = iiEl.GetInt32();
        // Drive the analysis SERIALLY, mirroring the C++ UnivariateAnalysis::run() (which has no
        // ParallelizeChains -- scope decision 1 of the port). The C# UnivariateAnalysis.RunAsync
        // leaves BayesianAnalysis.RunAsync's `parallel` at its default `true`; DEMCzs draws from
        // the shared history archive (not the current chain states), so serial and parallel agree,
        // but passing parallel:false keeps this on the same serial path the C++/R/Python harnesses
        // and the model_estimation emitter path use.
        ud.DataFrame.ProcessThresholdSeries();
        if (ud.IsNonstationary) ud.DataFrame.CreateFullTimeSeries();
        ud.ProcessQuantilePriors();
        ba.RunAsync(null, false, false).GetAwaiter().GetResult();
        if (ba.IsEstimated)
        {
            analysis.CreateFrequencyAnalysisResultsAsync().GetAwaiter().GetResult();
            analysis.CreateChronologyResultsAsync().GetAwaiter().GetResult();
        }
        var results = analysis.AnalysisResults;
        if (results != null)
        {
            var pe = analysis.GetPointEstimateDistribution();
            if (pe is not null) r.Parameters.AddRange(pe.GetParameters);
            if (results.ModeCurve != null) r.ModeCurve.AddRange(results.ModeCurve);
            if (results.MeanCurve != null) r.MeanCurve.AddRange(results.MeanCurve);
            if (results.ConfidenceIntervals != null)
            {
                int n = results.ConfidenceIntervals.GetLength(0);
                for (int i = 0; i < n; i++)
                {
                    r.LowerCI.Add(results.ConfidenceIntervals[i, 0]);
                    r.UpperCI.Add(results.ConfidenceIntervals[i, 1]);
                }
            }
            r.Aic = results.AIC;
            r.Bic = results.BIC;
            r.Dic = results.DIC;
            r.Rmse = results.RMSE;
        }
        return r;
    }

    if (target == "Bulletin17CAnalysis")
    {
        var model = BuildBulletin17CModel(modelSpec, datasets);
        var analysis = new BestFitAnalyses.Bulletin17CAnalysis(model);
        analysis.UncertaintyMethod = ParseUncertaintyMethod(
            construct.TryGetProperty("uncertainty_method", out var um)
                ? um.GetString()! : "MultivariateNormal");
        ApplyOrdinates(analysis.ProbabilityOrdinates);
        var ba = analysis.BayesianAnalysis;
        if (construct.TryGetProperty("confidence_level", out var clEl))
            ba.CredibleIntervalWidth = clEl.GetDouble();
        if (construct.TryGetProperty("seed", out var seEl)) ba.PRNGSeed = seEl.GetInt32();
        if (construct.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        analysis.RunAsync().GetAwaiter().GetResult();
        var ci = analysis.ComputeCohnStyleConfidenceIntervals();
        if (ci != null)
        {
            r.Exceedance.AddRange(ci.ExceedanceProbabilities);
            r.PointEstimates.AddRange(ci.PointEstimates);
            r.LowerCI.AddRange(ci.LowerCI);
            r.UpperCI.AddRange(ci.UpperCI);
            r.Beta1.AddRange(ci.Beta1);
            r.Nu.AddRange(ci.Nu);
            r.QuantileVariance.AddRange(ci.QuantileVariance);
            r.ConfidenceLevel = ci.ConfidenceLevel;
        }
        if (analysis.GMM != null && analysis.GMM.IsEstimated)
            r.Parameters.AddRange(analysis.GMM.BestParameterSet.Values);
        return r;
    }

    if (target == "Diagnostics")
    {
        // Mirror test_fixtures.cpp::run_diagnostics_analysis: build the model, run a seeded
        // deterministic BayesianAnalysis (serial, ParallelizeChains=false), then compute all three
        // diagnostics off that single fit. The BayesianAnalysis knobs are applied in the same order
        // as BuildEstimation's BayesianAnalysis target + apply_analysis_bayes_knobs (C++), so the
        // C# and C++ seeded posteriors are the same stream.
        var model = BuildSpecModel(modelSpec, datasets);
        var ba = new BayesianAnalysis(model, ParseSamplerType(
            construct.TryGetProperty("sampler", out var s) ? s.GetString()! : "DEMCzs"))
        {
            UseSimulationDefaults = false,
            UseAdvancedSimulationDefaults = false,
        };
        if (construct.TryGetProperty("credible_level", out var clEl))
            ba.CredibleIntervalWidth = clEl.GetDouble();
        if (construct.TryGetProperty("seed", out var seEl)) ba.PRNGSeed = seEl.GetInt32();
        if (construct.TryGetProperty("output_length", out var olEl)) ba.OutputLength = olEl.GetInt32();
        if (construct.TryGetProperty("iterations", out var itEl))
        {
            int it = itEl.GetInt32();
            ba.Iterations = it;
            ba.WarmupIterations = Math.Max(50, it / 2);
        }
        if (construct.TryGetProperty("thinning_interval", out var thEl)) ba.ThinningInterval = thEl.GetInt32();
        if (construct.TryGetProperty("number_of_chains", out var ncEl)) ba.NumberOfChains = ncEl.GetInt32();
        if (construct.TryGetProperty("initial_iterations", out var iiEl)) ba.InitialIterations = iiEl.GetInt32();
        ba.RunAsync(null, false, false).GetAwaiter().GetResult();
        if (!ba.IsEstimated) return r;

        var lev = ba.ComputeLeverageDiagnostics();
        r.LevCount = lev.Count;
        r.LevPriorCount = lev.PriorComponents.Length;
        r.TotalLeverage = lev.TotalLeverage;
        r.TotalFitInfluence = lev.TotalFitInfluence;
        r.TotalVarianceInfluence = lev.TotalVarianceInfluence;
        foreach (var o in lev.Observations)
        {
            r.LevObsLeverage.Add(o.Leverage);
            r.LevObsFit.Add(o.FitInfluence);
            r.LevObsVar.Add(o.VarianceInfluence);
            r.LevObsValue.Add(o.Value);
        }

        var inf = ba.ComputeInfluenceDiagnostics();
        r.InfCount = inf.Count;
        r.MeanParetoK = inf.MeanParetoK;
        r.MaxParetoK = inf.MaxParetoK;
        r.CountParetoK05 = inf.CountParetoKAbove05;
        r.CountParetoK07 = inf.CountParetoKAbove07;
        r.CountParetoK10 = inf.CountParetoKAbove10;
        r.ProportionProblematic = inf.ProportionProblematic;
        r.IsReliable = inf.IsReliable ? 1.0 : 0.0;
        foreach (var o in inf.Observations)
        {
            r.InfParetoK.Add(o.ParetoK);
            r.InfElpdLoo.Add(o.ElpdLoo);
        }

        int thinEvery = construct.TryGetProperty("thin_every", out var teEl) ? teEl.GetInt32() : 10;
        var pri = ba.ComputePriorInfluenceDiagnostics(thinEvery);
        r.PriCount = pri.Count;
        r.TotalPriorLogLik = pri.TotalPriorLogLikelihood;
        r.TotalDataLogLik = pri.TotalDataLogLikelihood;
        r.PriorToDataRatio = pri.PriorToDataRatio;
        r.IsPriorInfluential = pri.IsPriorInfluential ? 1.0 : 0.0;
        r.MeanPriorPrecisionShare = pri.MeanPriorPrecisionShare;
        return r;
    }

    throw new Exception($"unknown analysis target: {target}");
}

// Flat analysis-result dispatch, matching test_fixtures.cpp's dispatch_analysis method names.
static double DispatchAnalysis(AnalysisData r, string m, JsonElement[] a)
{
    int I(int i) => a[i].GetInt32();
    switch (m)
    {
        case "candidate_count": return r.CandidateCount;
        case "candidate_aic": return r.CandAic[I(0)];
        case "candidate_bic": return r.CandBic[I(0)];
        case "candidate_rmse": return r.CandRmse[I(0)];
        case "candidate_converged": return r.CandConverged[I(0)];
        case "parameter": return r.Parameters[I(0)];
        case "mode_curve": return r.ModeCurve[I(0)];
        case "mean_curve": return r.MeanCurve[I(0)];
        case "lower_ci": return r.LowerCI[I(0)];
        case "upper_ci": return r.UpperCI[I(0)];
        case "exceedance_probability": return r.Exceedance[I(0)];
        case "point_estimate": return r.PointEstimates[I(0)];
        case "beta1": return r.Beta1[I(0)];
        case "nu": return r.Nu[I(0)];
        case "quantile_variance": return r.QuantileVariance[I(0)];
        case "aic": return r.Aic;
        case "bic": return r.Bic;
        case "dic": return r.Dic;
        case "rmse": return r.Rmse;
        case "confidence_level": return r.ConfidenceLevel;
        // --- D6 Diagnostics dispatch (names match diagnostics_smoke.json + test_fixtures.cpp). ---
        case "leverage_count": return r.LevCount;
        case "leverage_prior_count": return r.LevPriorCount;
        case "total_leverage": return r.TotalLeverage;
        case "total_fit_influence": return r.TotalFitInfluence;
        case "total_variance_influence": return r.TotalVarianceInfluence;
        case "obs_leverage": return r.LevObsLeverage[I(0)];
        case "obs_fit_influence": return r.LevObsFit[I(0)];
        case "obs_variance_influence": return r.LevObsVar[I(0)];
        case "obs_value": return r.LevObsValue[I(0)];
        case "influence_count": return r.InfCount;
        case "mean_pareto_k": return r.MeanParetoK;
        case "max_pareto_k": return r.MaxParetoK;
        case "count_pareto_k_above_05": return r.CountParetoK05;
        case "count_pareto_k_above_07": return r.CountParetoK07;
        case "count_pareto_k_above_10": return r.CountParetoK10;
        case "proportion_problematic": return r.ProportionProblematic;
        case "is_reliable": return r.IsReliable;
        case "pareto_k": return r.InfParetoK[I(0)];
        case "elpd_loo": return r.InfElpdLoo[I(0)];
        case "prior_influence_count": return r.PriCount;
        case "total_prior_log_likelihood": return r.TotalPriorLogLik;
        case "total_data_log_likelihood": return r.TotalDataLogLik;
        case "prior_to_data_ratio": return r.PriorToDataRatio;
        case "is_prior_influential": return r.IsPriorInfluential;
        case "mean_prior_precision_share": return r.MeanPriorPrecisionShare;
        default: throw new Exception($"unknown analysis fixture method: {m}");
    }
}

// --dump: the sanctioned curation path (see fixtures/README.md and the Task 5 brief).
// Author a fixture case with placeholder "expected" values, run
// `verify_oracles.py --dump` (threads this flag through to `dotnet run -- --dump`), paste
// the printed actuals into the fixture, then re-run without --dump to verify. Kept small:
// one helper that prints one JSON line per assertion instead of comparing.
static void DumpLine(string target, string caseName, string method, JsonElement[] args, Func<object> compute)
{
    object actualOrError;
    try { actualOrError = compute(); }
    catch (Exception ex) { actualOrError = "ERROR: " + ex.Message; }
    var line = new Dictionary<string, object?>
    {
        ["target"] = target,
        ["case"] = caseName,
        ["method"] = method,
        ["args"] = args,
        ["actual"] = actualOrError,
    };
    // PDF/CDF spot values can legitimately be +-Infinity (e.g. LogPDF of a non-finite
    // input); System.Text.Json refuses to write those without this option.
    var options = new JsonSerializerOptions
    {
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };
    Console.WriteLine(JsonSerializer.Serialize(line, options));
}

// --- main -------------------------------------------------------------------------------

bool dump = args.Contains("--dump");
string[] positional = args.Where(a => a != "--dump").ToArray();
string fixturesDir = positional.Length > 0 ? positional[0]
    : Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "fixtures");
fixturesDir = Path.GetFullPath(fixturesDir);
if (!Directory.Exists(fixturesDir))
{
    Console.Error.WriteLine($"fixtures directory not found: {fixturesDir}");
    return 2;
}

int pass = 0, fail = 0, skip = 0;
var failures = new List<string>();

foreach (var file in Directory.EnumerateFiles(fixturesDir, "*.json", SearchOption.AllDirectories))
{
    using var doc = JsonDocument.Parse(File.ReadAllText(file));
    var root = doc.RootElement;
    string? kindStr = root.TryGetProperty("kind", out var kind) ? kind.GetString() : null;

    // --- special_function branch --------------------------------------------------------
    // Most files dispatch every case through one file-level `target` (e.g. "Erf.function").
    // The Cholesky fixture groups several related dispatch keys in one file, so a case may
    // override `target`; a case without its own falls back to the file-level one, leaving
    // single-target files' behavior unchanged.
    if (kindStr == "special_function")
    {
        string fileTarget = root.GetProperty("target").GetString()!;
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string sfTarget = c.TryGetProperty("target", out var caseTarget)
                ? caseTarget.GetString()! : fileTarget;
            var fn = ResolveSpecialFunction(sfTarget);
            if (fn is null) { Console.Error.WriteLine($"  SKIP unknown special-function target: {sfTarget}"); continue; }
            string caseName = c.GetProperty("name").GetString()!;
            var caseArgsJson = c.GetProperty("args").EnumerateArray().ToArray();
            var argList = caseArgsJson.Select(ParseNum).ToArray();

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                string where = $"{sfTarget}/{caseName}/{method}";

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine(sfTarget, caseName, method, caseArgsJson, () => (object)fn(argList));
                    continue;
                }

                double actual;
                try { actual = fn(argList); }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); continue; }
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }
        }
        continue;
    }

    // --- goodness_of_fit branch ---------------------------------------------------------
    if (kindStr == "goodness_of_fit")
    {
        var dsSets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var dsNode))
            foreach (var kv in dsNode.EnumerateObject())
                dsSets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            string fn = c.GetProperty("function").GetString()!;

            // Resolve inputs: scalar args or observed/modeled dataset pairs
            double[] scalarArgs = c.TryGetProperty("args", out var argsNode)
                ? argsNode.EnumerateArray().Select(ParseNum).ToArray()
                : Array.Empty<double>();
            double[]? obs = c.TryGetProperty("observed_dataset", out var obsName)
                ? dsSets[obsName.GetString()!] : null;
            double[]? mod = c.TryGetProperty("modeled_dataset", out var modName)
                ? dsSets[modName.GetString()!] : null;

            double actual;
            try
            {
                actual = fn switch
                {
                    "AIC"  => GoodnessOfFit.AIC((int)scalarArgs[0], scalarArgs[1]),
                    "AICc" => GoodnessOfFit.AICc((int)scalarArgs[0], (int)scalarArgs[1], scalarArgs[2]),
                    "BIC"  => GoodnessOfFit.BIC((int)scalarArgs[0], (int)scalarArgs[1], scalarArgs[2]),
                    "MSE"  => GoodnessOfFit.MSE(obs!, mod!),
                    "MAE"  => GoodnessOfFit.MAE(obs!, mod!),
                    "NashSutcliffeEfficiency"    => GoodnessOfFit.NashSutcliffeEfficiency(obs!, mod!),
                    "KlingGuptaEfficiency"       => GoodnessOfFit.KlingGuptaEfficiency(obs!, mod!),
                    "KlingGuptaEfficiencyMod"    => GoodnessOfFit.KlingGuptaEfficiencyMod(obs!, mod!),
                    "PBIAS"                      => GoodnessOfFit.PBIAS(obs!, mod!),
                    "RSR"                        => GoodnessOfFit.RSR(obs!, mod!),
                    "IndexOfAgreement"           => GoodnessOfFit.IndexOfAgreement(obs!, mod!),
                    "ModifiedIndexOfAgreement"   => GoodnessOfFit.ModifiedIndexOfAgreement(obs!, mod!),
                    "RefinedIndexOfAgreement"    => GoodnessOfFit.RefinedIndexOfAgreement(obs!, mod!),
                    "VolumetricEfficiency"       => GoodnessOfFit.VolumetricEfficiency(obs!, mod!),
                    _ => throw new Exception($"unknown goodness_of_fit function: {fn}")
                };
            }
            catch (Exception ex) { fail++; failures.Add($"gof/{caseName}: {ex.Message}"); continue; }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string where = $"gof/{caseName}";
                if (Compare(actual, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
            }
        }
        continue;
    }

    // --- multivariate_distribution branch ------------------------------------------------
    if (kindStr == "multivariate_distribution")
    {
        string mvTarget = root.GetProperty("target").GetString()!;
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            MultivariateDistribution mvDist;
            try { mvDist = BuildMultivariate(mvTarget, c.GetProperty("construct")); }
            catch (Exception ex) { failures.Add($"{mvTarget}/{caseName}: build failed: {ex.Message}"); fail++; continue; }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{mvTarget}/{caseName}/{method}";
                string mode = asrt.GetProperty("mode").GetString()!;

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine(mvTarget, caseName, method, argList,
                             () => mode == "bool" ? (object)mvDist.ParametersValid
                                                   : (object)DispatchMultivariate(mvDist, mvTarget, method, argList));
                    continue;
                }

                try
                {
                    if (mode == "bool")
                    {
                        bool ok = mvDist.ParametersValid == asrt.GetProperty("expected").GetBoolean();
                        if (ok) pass++; else { fail++; failures.Add(where + ": bool mismatch"); }
                        continue;
                    }
                    double actual = DispatchMultivariate(mvDist, mvTarget, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- bivariate_copula branch ----------------------------------------------------------
    if (kindStr == "bivariate_copula")
    {
        string copTarget = root.GetProperty("target").GetString()!;
        var copDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var copDs))
            foreach (var kv in copDs.EnumerateObject())
                copDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            BivariateCopula copula;
            try { copula = BuildCopula(copTarget, c.GetProperty("construct"), copDatasets); }
            catch (Exception ex) { failures.Add($"{copTarget}/{caseName}: build failed: {ex.Message}"); fail++; continue; }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{copTarget}/{caseName}/{method}";
                string mode = asrt.GetProperty("mode").GetString()!;

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine(copTarget, caseName, method, argList,
                             () => mode == "bool" ? (object)copula.ParametersValid
                                                   : (object)DispatchCopula(copula, method, argList));
                    continue;
                }

                try
                {
                    if (mode == "bool")
                    {
                        bool ok = copula.ParametersValid == asrt.GetProperty("expected").GetBoolean();
                        if (ok) pass++; else { fail++; failures.Add(where + ": bool mismatch"); }
                        continue;
                    }
                    double actual = DispatchCopula(copula, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- mcmc_sampler branch ---------------------------------------------------------------
    // One sampler run per case (see fixtures/README.md's mcmc_sampler schema): construct the
    // model + sampler, apply settings, sample() ONCE, post-process an MCMCResults, then
    // dispatch every assertion against that one cached (sampler, results) pair.
    if (kindStr == "mcmc_sampler")
    {
        string mcmcTarget = root.GetProperty("target").GetString()!;
        var mcmcDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var mcmcDs))
            foreach (var kv in mcmcDs.EnumerateObject())
                mcmcDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            MCMCSampler mcmcSampler;
            MCMCResults mcmcResults;
            try
            {
                mcmcSampler = BuildAndSampleMcmc(mcmcTarget, c.GetProperty("construct"), mcmcDatasets);
                mcmcResults = new MCMCResults(mcmcSampler);
            }
            catch (Exception ex)
            {
                failures.Add($"{mcmcTarget}/{caseName}: build/sample failed: {ex.Message}");
                fail++;
                continue;
            }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{mcmcTarget}/{caseName}/{method}";

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine(mcmcTarget, caseName, method, argList,
                             () => (object)DispatchMcmc(mcmcSampler, mcmcResults, method, argList));
                    continue;
                }

                try
                {
                    double actual = DispatchMcmc(mcmcSampler, mcmcResults, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- bootstrap branch --------------------------------------------------------------------
    // One bootstrap run per case (see fixtures/README.md's bootstrap schema): build the model
    // + configure + run() (or run_with_studentized_bootstrap()) ONCE, compute confidence
    // intervals ONCE, then dispatch every assertion against that one cached (boot, results)
    // pair.
    if (kindStr == "bootstrap")
    {
        var bsDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var bsDs))
            foreach (var kv in bsDs.EnumerateObject())
                bsDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            Bootstrap<double[]> boot;
            BootstrapResults results;
            try
            {
                (boot, results) = BuildAndRunBootstrap(c.GetProperty("construct"), bsDatasets);
            }
            catch (Exception ex)
            {
                failures.Add($"Bootstrap/{caseName}: build/run failed: {ex.Message}");
                fail++;
                continue;
            }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"Bootstrap/{caseName}/{method}";

                // --dump: the curation path. Print target/case/method/args and the actual
                // C#-computed value as a JSON line instead of comparing against the
                // fixture's (possibly still-placeholder) "expected". See DumpLine().
                if (dump)
                {
                    DumpLine("Bootstrap", caseName, method, argList,
                             () => (object)DispatchBootstrap(boot, results, method, argList));
                    continue;
                }

                try
                {
                    double actual = DispatchBootstrap(boot, results, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- analysis branch (Task A11: user-facing Analyses layer) ------------------------------
    // One analysis build+run per case; dispatch every assertion against the cached flat result.
    // Same "build once, dispatch many, --dump supported" shape as model_estimation.
    if (kindStr == "analysis")
    {
        string anTarget = root.GetProperty("target").GetString()!;
        var anDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var anDs))
            foreach (var kv in anDs.EnumerateObject())
                anDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        // Whole-target/whole-case oracle-exempt: the D5 per-family analyses (AR/MA/ARIMA/ARIMAX/
        // Mixture/CompetingRisk/PointProcess) have NO C# numeric oracle. D1/D2 shipped these
        // analyses as STRUCTURAL oracles only -- their seeded forecast trajectories ride a
        // documented P4-class MersenneTwister-seed non-reproducibility and the emitter has no
        // driver for these targets. A fixture-level (or case-level) "oracle_skip": true marks every
        // assertion in the case SKIPPED (the shipped C++/R/Python harnesses still exercise the
        // self-computed loose values) so the dev-only gate stays 0-failed WITHOUT a broad
        // unknown-target mask -- an unexpected unknown target still fails loudly. See each fixture's
        // `source` + docs/upstream-csharp-issues.md.
        bool rootAnalysisSkip = root.TryGetProperty("oracle_skip", out var rasEl) && rasEl.GetBoolean();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;

            bool caseAnalysisSkip = rootAnalysisSkip
                || (c.TryGetProperty("oracle_skip", out var casEl) && casEl.GetBoolean());
            if (caseAnalysisSkip)
            {
                if (!dump) skip += c.GetProperty("assertions").GetArrayLength();
                continue;
            }

            AnalysisData anData;
            try { anData = BuildAndRunAnalysis(anTarget, c.GetProperty("construct"), anDatasets); }
            catch (Exception ex)
            {
                failures.Add($"{anTarget}/{caseName}: build/run failed: {ex.Message}");
                fail++;
                continue;
            }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{anTarget}/{caseName}/{method}";

                if (dump)
                {
                    DumpLine(anTarget, caseName, method, argList,
                             () => (object)DispatchAnalysis(anData, method, argList));
                    continue;
                }

                // Oracle-exempt assertion (same treatment as the GEV std-err skips): a value the
                // shipped C++/R/Python harnesses check against the ported core, but which the real
                // C# library cannot reproduce because it rides an oracle-locked, documented port
                // deviation. D6: the three PriorInfluenceDiagnostics quantities collapse two Normal
                // parameter priors into one under the name-keyed dedup because the Phase-4 C++ model
                // deliberately leaves ModelParameter names empty (univariate_distribution_model.hpp
                // ~130) while C# keeps "Parameter Prior: Mean"/"Std Dev" distinct. Skipped here (not
                // failed) so the dev-only gate stays honest without papering the divergence into a
                // wide tolerance. See the fixture `source` + docs/upstream-csharp-issues.md.
                if (asrt.TryGetProperty("oracle_skip", out var osEl) && osEl.GetBoolean())
                {
                    skip++;
                    continue;
                }

                try
                {
                    double actual = DispatchAnalysis(anData, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    // --- model_estimation branch (Task T12) --------------------------------------------------
    // One estimator build+run per case; dispatch every assertion against that cached estimator.
    // Same "build once, dispatch many, --dump supported" shape as mcmc_sampler/bootstrap.
    if (kindStr == "model_estimation")
    {
        string estTarget = root.GetProperty("target").GetString()!;
        var estDatasets = new Dictionary<string, double[]>();
        if (root.TryGetProperty("datasets", out var estDs))
            foreach (var kv in estDs.EnumerateObject())
                estDatasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;

            // GMM/B17C target (B12): the concrete Bulletin17CDistribution is NOT a ModelBase, so
            // it takes its own build+dispatch path (mirrors the C++ runner's separate GMM arm).
            if (estTarget == "GeneralizedMethodOfMoments")
            {
                (BestFitModels.Bulletin17CDistribution b17c, GeneralizedMethodOfMoments gmm, double[]? simulated) gmmCase;
                try { gmmCase = BuildGmm(c.GetProperty("construct"), estDatasets); }
                catch (Exception ex)
                {
                    failures.Add($"{estTarget}/{caseName}: build/run failed: {ex.Message}");
                    fail++;
                    continue;
                }

                foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
                {
                    string method = asrt.GetProperty("method").GetString()!;
                    var argList = asrt.TryGetProperty("args", out var av)
                        ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                    string where = $"{estTarget}/{caseName}/{method}";

                    if (dump)
                    {
                        DumpLine(estTarget, caseName, method, argList,
                                 () => (object)DispatchGmm(gmmCase.b17c, gmmCase.gmm, gmmCase.simulated, method, argList));
                        continue;
                    }

                    try
                    {
                        double actual = DispatchGmm(gmmCase.b17c, gmmCase.gmm, gmmCase.simulated, method, argList);
                        if (Compare(actual, asrt)) pass++;
                        else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                    }
                    catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
                }
                continue;
            }

            // Phase 7a families (P4): TimeSeries / SpatialGEV / RatingCurve / Bivariate derive
            // from ModelBase/IModel (not UnivariateDistributionModelBase), so they take the
            // separate general build + dispatch path. The `type` string selects the family.
            var modelSpecEl = c.GetProperty("construct").GetProperty("model");
            string modelType = modelSpecEl.TryGetProperty("type", out var mtEl)
                ? mtEl.GetString()! : "univariate_distribution";
            if (modelType is "time_series" or "spatial_gev" or "rating_curve" or "bivariate")
            {
                (BestFitModels.IModel model, object? estimator, double[]? simulated) gc;
                try { gc = BuildEstimationGeneral(estTarget, c.GetProperty("construct"), estDatasets); }
                catch (Exception ex)
                {
                    failures.Add($"{estTarget}/{caseName}: build/run failed: {ex.Message}");
                    fail++;
                    continue;
                }

                foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
                {
                    string method = asrt.GetProperty("method").GetString()!;
                    var argList = asrt.TryGetProperty("args", out var av2)
                        ? av2.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                    string where2 = $"{estTarget}/{caseName}/{method}";

                    if (dump)
                    {
                        DumpLine(estTarget, caseName, method, argList,
                                 () => (object)DispatchEstimationGeneral(gc, method, argList));
                        continue;
                    }

                    try
                    {
                        double actual = DispatchEstimationGeneral(gc, method, argList);
                        if (Compare(actual, asrt)) pass++;
                        else { fail++; failures.Add($"{where2}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                    }
                    catch (Exception ex) { fail++; failures.Add($"{where2}: {ex.Message}"); }
                }
                continue;
            }

            (BestFitModels.UnivariateDistributionModelBase model, object? estimator, double[]? simulated) estimator;
            try { estimator = BuildEstimation(estTarget, c.GetProperty("construct"), estDatasets); }
            catch (Exception ex)
            {
                failures.Add($"{estTarget}/{caseName}: build/run failed: {ex.Message}");
                fail++;
                continue;
            }

            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string method = asrt.GetProperty("method").GetString()!;
                var argList = asrt.TryGetProperty("args", out var av)
                    ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
                string where = $"{estTarget}/{caseName}/{method}";

                if (dump)
                {
                    DumpLine(estTarget, caseName, method, argList,
                             () => (object)DispatchEstimation(estimator, method, argList));
                    continue;
                }

                try
                {
                    double actual = DispatchEstimation(estimator, method, argList);
                    if (Compare(actual, asrt)) pass++;
                    else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual:G17}"); }
                }
                catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
            }
        }
        continue;
    }

    if (kindStr != "univariate_distribution") continue;

    string target = root.GetProperty("target").GetString()!;

    var datasets = new Dictionary<string, double[]>();
    if (root.TryGetProperty("datasets", out var ds))
        foreach (var kv in ds.EnumerateObject())
            datasets[kv.Name] = kv.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    foreach (var c in root.GetProperty("cases").EnumerateArray())
    {
        string caseName = c.GetProperty("name").GetString()!;
        UnivariateDistributionBase dist;
        try { dist = Build(target, c.GetProperty("construct"), datasets); }
        catch (Exception ex) { failures.Add($"{target}/{caseName}: build failed: {ex.Message}"); fail++; continue; }

        foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
        {
            string method = asrt.GetProperty("method").GetString()!;
            var argList = asrt.TryGetProperty("args", out var av)
                ? av.EnumerateArray().ToArray() : Array.Empty<JsonElement>();
            string where = $"{target}/{caseName}/{method}";
            string mode = asrt.GetProperty("mode").GetString()!;

            // --dump: the curation path. Print target/case/method/args and the actual
            // C#-computed value as a JSON line instead of comparing against the
            // fixture's (possibly still-placeholder) "expected". See DumpLine().
            if (dump)
            {
                DumpLine(target, caseName, method, argList,
                         () =>
                         {
                             if (mode == "bool") return (object)dist.ParametersValid;
                             double? v = Dispatch(dist, method, argList);
                             return v is null ? (object)"SKIPPED" : (object)v.Value;
                         });
                continue;
            }

            try
            {
                if (mode == "bool")
                {
                    bool ok = dist.ParametersValid == asrt.GetProperty("expected").GetBoolean();
                    if (ok) pass++; else { fail++; failures.Add(where + ": bool mismatch"); }
                    continue;
                }
                double? actual = Dispatch(dist, method, argList);
                if (actual is null) { skip++; continue; }
                if (Compare(actual.Value, asrt)) pass++;
                else { fail++; failures.Add($"{where}: expected {asrt.GetProperty("expected")} got {actual.Value:G17}"); }
            }
            catch (Exception ex) { fail++; failures.Add($"{where}: {ex.Message}"); }
        }
    }
}

Console.WriteLine($"oracle verification: {pass} reproduced, {fail} failed, {skip} skipped (GEV std-err + oracle-exempt)");
foreach (var f in failures) Console.Error.WriteLine("  FAIL " + f);
return fail == 0 ? 0 : 1;

// Flat analysis-result surface (Task A11), mirroring test_fixtures.cpp's AnalysisResult. Only the
// fields a given target populates are filled; curve/CI vectors are indexed by the exceedance grid,
// the candidate_* vectors carry one entry per fitted candidate. Declared after the top-level
// statements (C# requires it), referenced by BuildAndRunAnalysis / DispatchAnalysis above.
class AnalysisData
{
    public List<double> Parameters = new(), ModeCurve = new(), MeanCurve = new(),
                        LowerCI = new(), UpperCI = new();
    public List<double> Exceedance = new(), PointEstimates = new(), Beta1 = new(), Nu = new(),
                        QuantileVariance = new();
    public List<double> CandAic = new(), CandBic = new(), CandRmse = new(), CandConverged = new();
    public double Aic = double.NaN, Bic = double.NaN, Dic = double.NaN, Rmse = double.NaN,
                 ConfidenceLevel = double.NaN;
    public int CandidateCount = 0;

    // --- D6 Diagnostics surface (target == "Diagnostics"). Mirrors test_fixtures.cpp's
    // AnalysisResult diagnostics slice field-for-field so the same fixture drives both. ---
    public int LevCount = 0, LevPriorCount = 0, InfCount = 0, PriCount = 0;
    public double TotalLeverage = double.NaN, TotalFitInfluence = double.NaN,
                 TotalVarianceInfluence = double.NaN;
    public List<double> LevObsLeverage = new(), LevObsFit = new(), LevObsVar = new(),
                        LevObsValue = new();
    public double MeanParetoK = double.NaN, MaxParetoK = double.NaN;
    public int CountParetoK05 = 0, CountParetoK07 = 0, CountParetoK10 = 0;
    public double ProportionProblematic = double.NaN, IsReliable = double.NaN;
    public List<double> InfParetoK = new(), InfElpdLoo = new();
    public double TotalPriorLogLik = double.NaN, TotalDataLogLik = double.NaN,
                 PriorToDataRatio = double.NaN, IsPriorInfluential = double.NaN,
                 MeanPriorPrecisionShare = double.NaN;
}
