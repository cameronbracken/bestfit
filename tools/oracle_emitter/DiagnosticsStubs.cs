// Compile-only stubs for the RMC.BestFit.Diagnostics types that MaximumAPosteriori.cs,
// BayesianAnalysis.cs, and (Task B12) GeneralizedMethodOfMoments.cs reference from their gated
// diagnostic methods (ComputeLeverageDiagnostics / ComputeInfluenceDiagnostics /
// ComputePriorInfluenceDiagnostics, and GMM's GetObservationInfluence / GetCooksDistance /
// GetInfluenceDiagnostics / GetLeverageDiagnostics region).
//
// Task T12 subset-compiles the real estimation path, but the real Diagnostics/ folder is not in
// the compiled subset (it pulls in more of the Analyses layer, which has an unbuildable CS0104
// ambiguity). Task B12 un-excludes the GMM estimator + IGMMModel + Bulletin17CDistribution so
// their exact fit oracles can be dumped, but the GMM Influence/Leverage-Diagnostics region
// (C# lines ~1382-2061) is DEFERRED and on NO oracle-value path this emitter dumps (params,
// standard errors, covariance, J-stat, quantile variance, seeded draws, DIC/WAIC/LOOIC,
// posterior_mean, chain values). So throwing/empty stub bodies are oracle-faithful: they satisfy
// the compiler, are never executed, and cannot corrupt any dumped value. The stub signatures are
// transcribed from the REAL Diagnostics/InfluenceDiagnostics.cs and Diagnostics/LeverageDiagnostics.cs
// so the GMM references bind exactly. Author-in-emitter-project ONLY -- nothing under upstream/ is
// touched. Must be explicitly <Compile Include>d in the csproj (EnableDefaultCompileItems is off).
using Numerics.Mathematics.LinearAlgebra;
using RMC.BestFit.Models;

namespace RMC.BestFit.Diagnostics
{
    // Per-observation influence record (Diagnostics/InfluenceDiagnostics.cs). GMM's
    // GetInfluenceDiagnostics constructs these; never reached by any dumped oracle.
    public class ObservationInfluence
    {
        public ObservationInfluence(int index, double paretoK, double elpdLoo, double value = double.NaN,
            DataComponentType dataType = DataComponentType.Exact, int count = 1, string? name = null) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (B12 stub).");
    }

    public class InfluenceDiagnostics
    {
        public InfluenceDiagnostics() { }

        // B12: GMM's GetInfluenceDiagnostics() builds an ObservationInfluence[] result.
        public InfluenceDiagnostics(ObservationInfluence[] observations) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (B12 stub).");

        public InfluenceDiagnostics(double[] paretoK, double[] elpdLoo,
                                    System.Collections.Generic.List<DataComponent>? dataComponents) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (Task T12 stub).");
    }

    public class LeverageDiagnostics
    {
        // B12: GMM's GetLeverageDiagnostics returns an empty instance on the regularization-failure
        // fallback path, and a fully-populated one on success.
        public LeverageDiagnostics() { }

        public LeverageDiagnostics(IModel model, double[] values) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (Task T12 stub).");

        public LeverageDiagnostics(ObservationLeverage[] observations,
                                   PriorComponentLeverage[] priorComponents, int numberOfParameters) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (B12 stub).");

        // Static helpers GMM's GetLeverageDiagnostics calls (public accessors in the real class).
        public static Matrix ComputeNumericalHessianPublic(System.Func<double[], double> function,
                                                           double[] parameters, int p) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (B12 stub).");

        public static double ComputeGenVarPublic(System.Func<double[], double> logLikelihoodWithout,
                                                 Matrix sigma, double[] mapValues, int p) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (B12 stub).");

        // Nested records the GMM leverage path constructs (Diagnostics/LeverageDiagnostics.cs).
        public class ObservationLeverage
        {
            public ObservationLeverage(int index, double leverage, double percentOfTotal,
                double fitInfluence, double varianceInfluence,
                double percentFitOfTotal, double percentVarianceOfTotal,
                double value, DataComponentType dataType, int count = 1, string? name = null) =>
                throw new System.NotImplementedException(
                    "Diagnostics layer is not compiled into the oracle emitter (B12 stub).");
        }

        public class PriorComponentLeverage
        {
            public PriorComponentLeverage(string name, PriorComponentType type,
                double leverage, double percentOfTotal,
                double fitInfluence, double varianceInfluence,
                double percentFitOfTotal, double percentVarianceOfTotal) =>
                throw new System.NotImplementedException(
                    "Diagnostics layer is not compiled into the oracle emitter (B12 stub).");
        }
    }

    public class PriorInfluenceDiagnostics
    {
        public PriorInfluenceDiagnostics(IModel model,
                                         Numerics.Sampling.MCMC.MCMCResults results, int thinEvery) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (Task T12 stub).");
    }
}
