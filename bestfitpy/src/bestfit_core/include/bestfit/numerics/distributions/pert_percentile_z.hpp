// ported from: Numerics/Distributions/Univariate/PertPercentileZ.cs @ a2c4dbf
//
// PERT distribution parameterized by its 5th, 50th, and 95th percentiles expressed as
// probabilities in [0,1]. Internally transforms to Normal Z-space: solves via NelderMead
// for a PERT(min_z, mode_z, max_z) whose 5th/50th/95th percentiles match the Z-converted
// inputs. PDF/CDF/InverseCDF are expressed in probability space. Mean/Median/Mode are
// also in probability space (Φ of the underlying beta's moments). get_parameters() returns
// the three stored probability values (flat-param). No estimation interfaces.
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/generalized_beta.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/pert.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

class PertPercentileZ : public UnivariateDistributionBase {
   public:
    // Constructs a PERT-Percentile-Z with 5th = 0.05 (z=-1.64), 50th = 0.5, 95th = 0.95.
    PertPercentileZ() { set_parameters(0.05, 0.5, 0.95); }
    // Constructs a PERT-Percentile-Z with specified 5th, 50th, 95th probability values.
    PertPercentileZ(double fifth, double fiftieth, double ninety_fifth) {
        set_parameters(fifth, fiftieth, ninety_fifth);
    }

    double percentile_5th() const { return fifth_; }
    double percentile_50th() const { return fiftieth_; }
    double percentile_95th() const { return ninety_fifth_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::PertPercentileZ;
    }
    int number_of_parameters() const override { return 3; }
    // Returns the three stored probability values (the flat parameters).
    std::vector<double> get_parameters() const override {
        return {fifth_, fiftieth_, ninety_fifth_};
    }

    void set_parameters(double fifth, double fiftieth, double ninety_fifth) {
        fifth_ = fifth;
        fiftieth_ = fiftieth;
        ninety_fifth_ = ninety_fifth;
        parameters_valid_ = validate(fifth, fiftieth, ninety_fifth);
        parameters_solved_ = false;
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // --- Moments / support (all in probability space) ---
    double mean() const override {
        solve_parameters();
        if (almost_equals(beta_.min_val(), beta_.max_val()))
            return Normal::standard_cdf(beta_.min_val());
        return Normal::standard_cdf(beta_.mean());
    }
    double median() const override {
        solve_parameters();
        if (almost_equals(beta_.min_val(), beta_.max_val()))
            return Normal::standard_cdf(beta_.min_val());
        return Normal::standard_cdf(beta_.median());
    }
    double mode() const override {
        solve_parameters();
        if (almost_equals(beta_.min_val(), beta_.max_val()))
            return Normal::standard_cdf(beta_.min_val());
        return Normal::standard_cdf(beta_.mode());
    }
    // Standard deviation in Z-space (same as underlying PERT's stddev).
    double standard_deviation() const override {
        solve_parameters();
        return beta_.standard_deviation();
    }
    double skewness() const override {
        solve_parameters();
        return beta_.skewness();
    }
    double kurtosis() const override {
        solve_parameters();
        return beta_.kurtosis();
    }
    // Minimum/Maximum in probability space.
    double minimum() const override {
        solve_parameters();
        return Normal::standard_cdf(beta_.minimum());
    }
    double maximum() const override {
        solve_parameters();
        return Normal::standard_cdf(beta_.maximum());
    }

    // --- Distribution functions (x is a probability in [0,1]) ---
    double pdf(double x) const override {
        if (x < 0.0 || x > 1.0)
            throw std::out_of_range("PertPercentileZ: x must be between 0 and 1");
        if (!parameters_valid_) throw std::invalid_argument("PertPercentileZ: invalid parameters");
        solve_parameters();
        if (almost_equals(beta_.min_val(), beta_.max_val()) &&
            almost_equals(beta_.min_val(), beta_.mode()))
            return 0.0;
        if (std::isnan(beta_.mode())) return 0.0;
        return beta_.pdf(Normal::standard_z(x));
    }

    double cdf(double x) const override {
        if (x < 0.0 || x > 1.0)
            throw std::out_of_range("PertPercentileZ: x must be between 0 and 1");
        if (!parameters_valid_) throw std::invalid_argument("PertPercentileZ: invalid parameters");
        solve_parameters();
        if (almost_equals(beta_.min_val(), beta_.max_val()) &&
            almost_equals(beta_.min_val(), beta_.mode()))
            return 1.0;
        if (std::isnan(beta_.mode())) return 1.0;
        return beta_.cdf(Normal::standard_z(x));
    }

    double inverse_cdf(double probability) const override {
        if (!parameters_valid_) throw std::invalid_argument("PertPercentileZ: invalid parameters");
        solve_parameters();
        if (almost_equals(beta_.min_val(), beta_.max_val()) &&
            almost_equals(beta_.min_val(), beta_.mode()))
            return Normal::standard_cdf(beta_.min_val());
        if (std::isnan(beta_.mode())) return Normal::standard_cdf(beta_.min_val());
        return Normal::standard_cdf(beta_.inverse_cdf(probability));
    }

    // --- Parameter display names (X1; C# PertPercentileZ.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"5%", "50%", "95%"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"5%", "50%", "95%"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<PertPercentileZ>(fifth_, fiftieth_, ninety_fifth_);
    }

   private:
    // Solve for the underlying PERT in Z-space via NelderMead, mirroring SolveParameters().
    void solve_parameters() const {
        if (!parameters_valid_ || parameters_solved_) return;

        double fifth = fifth_, fiftieth = fiftieth_, ninety_fifth = ninety_fifth_;

        // Degenerate: all percentiles equal — use trivial PERT in Z-space directly.
        if (fifth == fiftieth && fiftieth == ninety_fifth) {
            beta_ = GeneralizedBeta::pert(Normal::standard_z(fifth),
                                          Normal::standard_z(fiftieth),
                                          Normal::standard_z(ninety_fifth));
            parameters_solved_ = true;
            return;
        }

        // Convert to Z-space for solving.
        double fifth_z = Normal::standard_z(fifth);
        double fiftieth_z = Normal::standard_z(fiftieth);
        double ninety_fifth_z = Normal::standard_z(ninety_fifth);

        // Initial guess: treat Z-values as PERT (min, mode, max).
        beta_ = GeneralizedBeta::pert(fifth_z, fiftieth_z, ninety_fifth_z);

        double range_z = ninety_fifth_z - fifth_z;
        double z_floor = Normal::standard_z(1E-16);
        double z_ceil = Normal::standard_z(1.0 - 1E-16);
        double min_bound = std::max(z_floor, fifth_z - range_z * 2.0);
        double max_bound = std::min(z_ceil, ninety_fifth_z + range_z * 2.0);
        std::vector<double> initials = {beta_.min_val(), beta_.mode(), beta_.max_val()};
        std::vector<double> lowers = {min_bound, min_bound, min_bound};
        std::vector<double> uppers = {max_bound, max_bound, max_bound};

        // Minimize SSE of (5th, 50th, 95th) Z-percentile errors vs. a PERT(x[0], x[1], x[2]).
        auto sse = [fifth_z, fiftieth_z, ninety_fifth_z](const std::vector<double>& x) -> double {
            try {
                Pert dist(x[0], x[1], x[2]);
                if (!dist.parameters_valid()) return std::numeric_limits<double>::max();
                double s = 0.0;
                s += std::pow(fifth_z - dist.inverse_cdf(0.05), 2.0);
                s += std::pow(fiftieth_z - dist.inverse_cdf(0.5), 2.0);
                s += std::pow(ninety_fifth_z - dist.inverse_cdf(0.95), 2.0);
                return s;
            } catch (...) {
                return std::numeric_limits<double>::max();
            }
        };

        math::optimization::NelderMead solver(sse, 3, initials, lowers, uppers);
        solver.relative_tolerance = 1E-8;
        solver.absolute_tolerance = 1E-8;
        solver.minimize();
        auto sol = solver.best_parameters();
        beta_ = GeneralizedBeta::pert(sol[0], sol[1], sol[2]);
        parameters_solved_ = true;
    }

    static bool almost_equals(double a, double b) {
        return std::fabs(a - b) <=
               kDoubleMachineEpsilon * std::fmax(std::fabs(a), std::fabs(b));
    }

    static bool validate(double fifth, double fiftieth, double ninety_fifth) {
        // Order check.
        if (std::isnan(fifth) || std::isinf(fifth) || std::isnan(ninety_fifth) ||
            std::isinf(ninety_fifth) || fifth > ninety_fifth)
            return false;
        if (std::isnan(fiftieth) || std::isinf(fiftieth) || fiftieth < fifth ||
            fiftieth > ninety_fifth)
            return false;
        // All must be probabilities in [0, 1].
        if (fifth < 0.0 || fifth > 1.0) return false;
        if (fiftieth < 0.0 || fiftieth > 1.0) return false;
        if (ninety_fifth < 0.0 || ninety_fifth > 1.0) return false;
        return true;
    }

    mutable GeneralizedBeta beta_;
    double fifth_ = 0.0;
    double fiftieth_ = 0.0;
    double ninety_fifth_ = 0.0;
    mutable bool parameters_solved_ = false;
};

}  // namespace bestfit::numerics::distributions
