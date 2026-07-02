// cpp11 glue exposing the polymorphic bivariate-copula surface of the shared C++ core to
// R. Unlike mvd.cpp (Dirichlet/Multinomial/BivariateEmpirical/... need bespoke per-target
// entry points because they share no common parameter API), every copula shares
// BivariateCopula's uniform theta/get_copula_parameters/pdf/cdf/... surface -- see
// copula_factory.hpp's header comment -- so a single generic bf_cop_val_ (construct +
// evaluate) and bf_cop_fit_ (fit + return fitted params) cover every copula target with no
// per-type branching beyond the factory itself. The one exception is the "tau"
// method-of-moments fit, which in the C# source is a member of each concrete Archimedean
// class (not part of IBivariateCopula/IArchimedeanCopula), so bf_cop_fit_ dynamic_casts by
// type name for that one method; each new tau-capable copula adds one branch there.
// Core headers are vendored under src/bestfit_core/include (see tools/sync_core.py).
#include <cpp11.hpp>

#include <memory>
#include <string>
#include <vector>

#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bestfit/numerics/distributions/copulas/amh_copula.hpp"
#include "bestfit/numerics/distributions/copulas/base/bivariate_copula_estimation.hpp"
#include "bestfit/numerics/distributions/copulas/base/copula_factory.hpp"
#include "bestfit/numerics/distributions/copulas/clayton_copula.hpp"
#include "bestfit/numerics/distributions/copulas/gumbel_copula.hpp"

namespace dist = bestfit::numerics::distributions;
namespace cop = bestfit::numerics::distributions::copulas;
using namespace cpp11;

static std::unique_ptr<cop::BivariateCopula> make_copula(const std::string& type, doubles params) {
    auto c = cop::create_copula(type);
    c->set_copula_parameters(std::vector<double>(params.begin(), params.end()));
    return c;
}

// method + flat numeric args in, double out. Methods: pdf/log_pdf/cdf (args=[u,v]),
// inverse_cdf (args=[u,v,index]), upper_tail_dependence, lower_tail_dependence, theta, df
// (2-parameter copulas only; get_copula_parameters()[1]), or_exceedance/and_exceedance
// (args=[u,v]), parameters_valid, random_value (args=[sample_size, seed, row, col];
// stateless -- GenerateRandomValues seeds its own LatinHypercube draw from `seed`).
// marg_x_target/marg_y_target optionally attach marginals directly (the C# `Copula(theta,
// marginX, marginY)` ctor path), mirroring bf_cop_fit_'s fitted-marginals convention: ""
// means no marginal for that side.
[[cpp11::register]]
double bf_cop_val_(std::string type, doubles params, std::string method, doubles args,
                    std::string marg_x_target, doubles marg_x_params, std::string marg_y_target,
                    doubles marg_y_params) {
    auto c = make_copula(type, params);
    if (!marg_x_target.empty()) {
        auto mx = dist::create_distribution(marg_x_target);
        mx->set_parameters(std::vector<double>(marg_x_params.begin(), marg_x_params.end()));
        c->marginal_distribution_x = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(mx));
    }
    if (!marg_y_target.empty()) {
        auto my = dist::create_distribution(marg_y_target);
        my->set_parameters(std::vector<double>(marg_y_params.begin(), marg_y_params.end()));
        c->marginal_distribution_y = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(my));
    }
    std::vector<double> a(args.begin(), args.end());

    if (method == "pdf") return c->pdf(a[0], a[1]);
    if (method == "log_pdf") return c->log_pdf(a[0], a[1]);
    if (method == "cdf") return c->cdf(a[0], a[1]);
    if (method == "inverse_cdf")
        return c->inverse_cdf(a[0], a[1])[static_cast<std::size_t>(a[2])];
    if (method == "upper_tail_dependence") return c->upper_tail_dependence();
    if (method == "lower_tail_dependence") return c->lower_tail_dependence();
    if (method == "theta") return c->theta();
    if (method == "df") return c->get_copula_parameters()[1];
    if (method == "or_exceedance") return c->or_joint_exceedance_probability(a[0], a[1]);
    if (method == "and_exceedance") return c->and_joint_exceedance_probability(a[0], a[1]);
    if (method == "parameters_valid") return c->parameters_valid() ? 1.0 : 0.0;
    if (method == "random_value") {
        auto sample = c->generate_random_values(static_cast<int>(a[0]), static_cast<int>(a[1]));
        return sample[static_cast<std::size_t>(a[2])][static_cast<std::size_t>(a[3])];
    }
    stop("unknown copula fixture method '%s'", method.c_str());
}

// The tau-based method of moments is not part of the shared copula API (see file header);
// each tau-capable copula type adds a branch here.
static void set_theta_from_tau_dispatch(cop::BivariateCopula& c, const std::string& type,
                                         const std::vector<double>& x, const std::vector<double>& y) {
    if (type == "Clayton") {
        dynamic_cast<cop::ClaytonCopula&>(c).set_theta_from_tau(x, y);
        return;
    }
    if (type == "AliMikhailHaq") {
        dynamic_cast<cop::AMHCopula&>(c).set_theta_from_tau(x, y);
        return;
    }
    if (type == "Gumbel") {
        dynamic_cast<cop::GumbelCopula&>(c).set_theta_from_tau(x, y);
        return;
    }
    // NOTE: JoeCopula has no SetThetaFromTau in the C# source -- see joe_copula.hpp's file
    // header; intentionally not branched here despite the Phase 2 plan/README listing it.
    stop("copula '%s' has no tau-based method-of-moments fit", type.c_str());
}

// Fits a copula (+ optionally its marginals) and returns the fitted copula theta
// parameters plus each marginal's fitted parameters. `x`/`y` are the sample data for
// "tau"/"ifm"/"mle" or the precomputed plotting positions for "mpl" (mirroring the C#
// test flow, see fixtures/README.md). marg_x/marg_y are the marginal distribution type
// names ("Normal", ...) or "" when the fit has no parametric marginals ("tau"/"mpl").
[[cpp11::register]]
list bf_cop_fit_(std::string type, doubles x, doubles y, std::string method, std::string marg_x,
                  std::string marg_y) {
    std::vector<double> xv(x.begin(), x.end());
    std::vector<double> yv(y.begin(), y.end());
    auto c = cop::create_copula(type);

    if (!marg_x.empty() && !marg_y.empty()) {
        auto mx = dist::create_distribution(marg_x);
        auto my = dist::create_distribution(marg_y);
        if (method == "ifm") {
            auto* ex = dynamic_cast<dist::IEstimation*>(mx.get());
            auto* ey = dynamic_cast<dist::IEstimation*>(my.get());
            if (ex == nullptr || ey == nullptr)
                stop("marginal '%s'/'%s' does not support estimation", marg_x.c_str(), marg_y.c_str());
            ex->estimate(xv, dist::ParameterEstimationMethod::MaximumLikelihood);
            ey->estimate(yv, dist::ParameterEstimationMethod::MaximumLikelihood);
        }
        c->marginal_distribution_x = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(mx));
        c->marginal_distribution_y = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(my));
    }

    if (method == "tau") {
        set_theta_from_tau_dispatch(*c, type, xv, yv);
    } else if (method == "mpl") {
        cop::estimate(*c, xv, yv, cop::CopulaEstimationMethod::PseudoLikelihood);
    } else if (method == "ifm") {
        cop::estimate(*c, xv, yv, cop::CopulaEstimationMethod::InferenceFromMargins);
    } else if (method == "mle") {
        cop::estimate(*c, xv, yv, cop::CopulaEstimationMethod::FullLikelihood);
    } else {
        stop("unknown copula fit method '%s'", method.c_str());
    }

    writable::doubles marg_x_params, marg_y_params;
    if (c->marginal_distribution_x) marg_x_params = writable::doubles(c->marginal_distribution_x->get_parameters());
    if (c->marginal_distribution_y) marg_y_params = writable::doubles(c->marginal_distribution_y->get_parameters());

    return writable::list({"params"_nm = writable::doubles(c->get_copula_parameters()),
                           "marg_x_params"_nm = marg_x_params, "marg_y_params"_nm = marg_y_params});
}
