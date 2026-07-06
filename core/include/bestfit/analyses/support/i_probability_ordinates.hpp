// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Support/IProbabilityOrdinates.cs @ fc28c0c
//
// Capability interface exposing the exceedance-probability grid an analysis tabulates its
// frequency curve on. A pure getter -- no INotifyPropertyChanged baggage survives (the C# type had
// none of its own; the ProbabilityOrdinates collection's INPC/INCC plumbing was already dropped in
// its A1 port).
//
// Return type: a NON-const `ProbabilityOrdinates&`. The C# get-only property returns the live,
// mutable collection object; downstream analyses (A5/A6/A7) flip its ordinates exceedance <->
// non-exceedance in place while driving distributions, so the reference must permit mutation.
#pragma once

#include "bestfit/numerics/data/probability_ordinates.hpp"

namespace bestfit::analyses {

class IProbabilityOrdinates {
   public:
    virtual ~IProbabilityOrdinates() = default;

    // The exceedance-probability grid used for plotting the distribution
    // (C# `ProbabilityOrdinates ProbabilityOrdinates { get; }`).
    virtual bestfit::numerics::data::ProbabilityOrdinates& probability_ordinates() = 0;
};

}  // namespace bestfit::analyses
