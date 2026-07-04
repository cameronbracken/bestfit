// Compile-only stubs for the RMC.BestFit.Diagnostics types that MaximumAPosteriori.cs and
// BayesianAnalysis.cs reference from their gated diagnostic methods
// (ComputeLeverageDiagnostics / ComputeInfluenceDiagnostics / ComputePriorInfluenceDiagnostics).
//
// Task T12 subset-compiles the real estimation path, but the real Diagnostics/ folder is not in
// the compiled subset (it pulls in more of the Analyses layer, which has an unbuildable CS0104
// ambiguity). None of the three gated methods is on the path of any oracle value this emitter
// dumps (params, log-likelihood, covariance, DIC/WAIC/LOOIC, posterior_mean, chain values), so a
// throwing/empty stub body is oracle-faithful: it satisfies the compiler, is never executed, and
// cannot corrupt any dumped value. Author-in-emitter-project ONLY -- nothing under upstream/ is
// touched. Must be explicitly <Compile Include>d in the csproj (EnableDefaultCompileItems is off).
namespace RMC.BestFit.Diagnostics
{
    public class LeverageDiagnostics
    {
        public LeverageDiagnostics(RMC.BestFit.Models.IModel model, double[] values) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (Task T12 stub).");
    }

    public class InfluenceDiagnostics
    {
        public InfluenceDiagnostics() { }

        public InfluenceDiagnostics(double[] paretoK, double[] elpdLoo,
                                    System.Collections.Generic.List<RMC.BestFit.Models.DataComponent>? dataComponents) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (Task T12 stub).");
    }

    public class PriorInfluenceDiagnostics
    {
        public PriorInfluenceDiagnostics(RMC.BestFit.Models.IModel model,
                                         Numerics.Sampling.MCMC.MCMCResults results, int thinEvery) =>
            throw new System.NotImplementedException(
                "Diagnostics layer is not compiled into the oracle emitter (Task T12 stub).");
    }
}
