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

Console.WriteLine($"oracle verification: {pass} reproduced, {fail} failed, {skip} skipped (GEV std-err)");
foreach (var f in failures) Console.Error.WriteLine("  FAIL " + f);
return fail == 0 ? 0 : 1;
