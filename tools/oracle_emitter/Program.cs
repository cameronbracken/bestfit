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
using Numerics.Data;
using Numerics.Data.Statistics;
using Numerics.Distributions;
using Numerics.Mathematics.LinearAlgebra;
using Numerics.Mathematics.SpecialFunctions;

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
    _ => null,
};

// --- main -------------------------------------------------------------------------------

string fixturesDir = args.Length > 0 ? args[0]
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
            var argList = c.GetProperty("args").EnumerateArray().Select(ParseNum).ToArray();
            double actual;
            try { actual = fn(argList); }
            catch (Exception ex) { fail++; failures.Add($"{sfTarget}/{caseName}: {ex.Message}"); continue; }
            foreach (var asrt in c.GetProperty("assertions").EnumerateArray())
            {
                string where = $"{sfTarget}/{caseName}/{asrt.GetProperty("method").GetString()}";
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
