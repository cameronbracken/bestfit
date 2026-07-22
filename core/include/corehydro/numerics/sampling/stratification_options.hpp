// ported from: Numerics/Sampling/StratificationOptions.cs @ 2a0357a
//
// Options describing a stratification of the x-axis (or [0,1] probability axis) into
// `number_of_bins` equal-width strata between `lower_bound` and `upper_bound`. Feeds
// Stratify::XValues (and, in later phases, Stratify's probability-stratification methods).
//
// Omitted: the XElement-deserializing constructor, SaveToXElement/XElementToStratificationOptions
// (desktop-app XML persistence), Clone/Clone(list) (nothing here needs a deep-copy helper beyond
// ordinary copy construction), and Equals/GetHashCode (not needed -- these options are not stored
// in a hash-based container and are compared, if ever, member-by-member by the caller). All are
// desktop-app / serialization concerns, not needed by the estimators this port targets.
#pragma once
#include <string>
#include <vector>

namespace corehydro::numerics::sampling {

class StratificationOptions {
   public:
    // Constructs a stratification options object and validates it immediately (mirrors the C#
    // ctor, which sets the fields then calls Validate()).
    StratificationOptions(double lower_bound, double upper_bound, int number_of_bins,
                           bool is_probability = false)
        : lower_bound_(lower_bound),
          upper_bound_(upper_bound),
          number_of_bins_(number_of_bins),
          is_probability_(is_probability) {
        validate();
    }

    // The lower bound of the starting bin.
    double lower_bound() const { return lower_bound_; }

    // The upper bound of the last bin.
    double upper_bound() const { return upper_bound_; }

    // The number of bins. Must be greater than 1 for the options to be valid.
    int number_of_bins() const { return number_of_bins_; }

    // Determines if the values to be stratified are probabilities [0,1].
    bool is_probability() const { return is_probability_; }

    // Determines if the stratification options are valid.
    bool is_valid() const { return is_valid_; }

    // Gets a list of errors for the stratification options.
    //
    // IMPORTANT ordering quirk mirrored from C#: this early-returns empty once IsValid is
    // already true. On construction IsValid starts false (default-initialized), so validate()'s
    // call to get_errors() computes the real error list exactly once.
    std::vector<std::string> get_errors() const {
        std::vector<std::string> result;
        if (is_valid_) return result;

        if (lower_bound_ >= upper_bound_) {
            result.push_back("The upper bound '" + std::to_string(upper_bound_) +
                              "' must be greater than the lower bound '" +
                              std::to_string(lower_bound_) + "'.");
        }
        if (number_of_bins_ < 2) {
            result.push_back("The number of bins '" + std::to_string(number_of_bins_) +
                              "' must be greater than 1.");
        }
        if (is_probability_ && lower_bound_ < 0.0) {
            result.push_back("The lower bound must be greater than or equal to 0.");
        }
        if (is_probability_ && upper_bound_ > 1.0) {
            result.push_back("The upper bound must be less than or equal to 1.");
        }
        return result;
    }

   private:
    // Validates the options and sets the is_valid_ flag.
    void validate() { is_valid_ = get_errors().empty(); }

    double lower_bound_;
    double upper_bound_;
    int number_of_bins_;
    bool is_probability_;
    bool is_valid_ = false;
};

}  // namespace corehydro::numerics::sampling
