// ported from: Numerics/Sampling/StratificationBin.cs @ 2a0357a
//
// A stratification bin: [lower_bound, upper_bound) with an associated weight
// (defaults to the bin width) and midpoint. Feeds stratified/Latin-hypercube sampling
// and copula parameter binning.
//
// Omitted: the XElement-deserializing constructor and SaveToXElement(s)/
// XElementToStratificationBinList (desktop-app XML persistence; not needed by this
// port). GetHashCode is omitted too -- nothing here stores bins in a hash-based
// container; operator== below mirrors C# Equals (LowerBound/UpperBound only, weight
// intentionally excluded, matching the C# doc comment on GetHashCode).
#pragma once
#include <stdexcept>

namespace corehydro::numerics::sampling {

class StratificationBin {
   public:
    // weight < 0 defaults to (upper_bound - lower_bound), matching the C# default.
    StratificationBin(double lower_bound, double upper_bound, double weight = -1.0)
        : lower_bound_(lower_bound), upper_bound_(upper_bound) {
        if (lower_bound > upper_bound)
            throw std::invalid_argument(
                "The upper bound must be greater than or equal to the lower bound.");
        this->weight = weight < 0 ? upper_bound - lower_bound : weight;
    }

    // The lower bound of the bin.
    double lower_bound() const { return lower_bound_; }

    // The upper bound of the bin.
    double upper_bound() const { return upper_bound_; }

    // The midpoint of the bin.
    double midpoint() const { return (upper_bound_ + lower_bound_) / 2.0; }

    // The weight given to the stratification bin (public get/set in C#; mutable here
    // for the same reason nelder_mead.hpp's tunables are -- this mirrors a stateful C#
    // model object, see the repo's corehydro/CLAUDE.md on the relaxed mutation rule).
    // Often the same value as the bin width, but end bins can be assigned different
    // weights to ensure unity.
    double weight = 0.0;

    // True if x is within the bin (inclusive lower, exclusive upper).
    bool contains(double x) const { return x >= lower_bound_ && x < upper_bound_; }

    // Compares two (non-overlapping) bins: 0 if the bounds are bit-for-bit equal, +1 if
    // this bin is lower than `other`, -1 otherwise (mirrors C# CompareTo).
    int compare_to(const StratificationBin& other) const {
        if (upper_bound_ > other.lower_bound_ && lower_bound_ < other.upper_bound_)
            throw std::invalid_argument("The bins cannot be overlapping.");
        if (upper_bound_ == other.upper_bound_ && lower_bound_ == other.lower_bound_) return 0;
        return other.upper_bound_ <= lower_bound_ ? 1 : -1;
    }

    // Returns a copy of this bin.
    StratificationBin clone() const { return StratificationBin(lower_bound_, upper_bound_, weight); }

    // Mirrors C# Equals: compares LowerBound/UpperBound only (weight excluded).
    bool operator==(const StratificationBin& other) const {
        return lower_bound_ == other.lower_bound_ && upper_bound_ == other.upper_bound_;
    }

   private:
    double lower_bound_;
    double upper_bound_;
};

}  // namespace corehydro::numerics::sampling
