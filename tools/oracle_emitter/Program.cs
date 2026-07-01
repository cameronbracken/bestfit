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
using Numerics.Distributions;
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

static UnivariateDistributionBase Build(string target, JsonElement construct,
                                        Dictionary<string, double[]> datasets)
{
    var type = Enum.Parse<UnivariateDistributionType>(target);
    var dist = UnivariateDistributionFactory.CreateDistribution(type);
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

// Special-function dispatch table: maps "Module.method" → Func<double[], double>.
static Func<double[], double>? ResolveSpecialFunction(string target) => target switch
{
    "Erf.function"     => a => Erf.Function(a[0]),
    "Erf.erfc"         => a => Erf.Erfc(a[0]),
    "Erf.inverse_erf"  => a => Erf.InverseErf(a[0]),
    "Erf.inverse_erfc" => a => Erf.InverseErfc(a[0]),
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
    if (kindStr == "special_function")
    {
        string sfTarget = root.GetProperty("target").GetString()!;
        var fn = ResolveSpecialFunction(sfTarget);
        if (fn is null) { Console.Error.WriteLine($"  SKIP unknown special-function target: {sfTarget}"); continue; }
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            string caseName = c.GetProperty("name").GetString()!;
            var argList = c.GetProperty("args").EnumerateArray().Select(v => v.GetDouble()).ToArray();
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
