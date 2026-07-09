// pybind11 glue exposing the polymorphic bivariate-copula surface of the shared C++ core to
// Python. Unlike mvd.cpp (Dirichlet/Multinomial/BivariateEmpirical/... need bespoke
// per-target module functions because they share no common parameter API), every copula
// shares BivariateCopula's uniform theta/get_copula_parameters/pdf/cdf/... surface -- see
// copula_factory.hpp's header comment -- so a single generic cop_val (construct + evaluate)
// and cop_fit (fit + return fitted params) cover every copula target with no per-type
// branching beyond the factory itself. The one exception is the "tau" method-of-moments
// fit, which in the C# source is a member of each concrete Archimedean class (not part of
// IBivariateCopula/IArchimedeanCopula), so cop_fit dynamic_casts by type name for that one
// method; each new tau-capable copula adds one branch there.
// Core headers are vendored under ../bestfit_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <map>
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
#include "bindings.hpp"

namespace py = pybind11;
namespace dist = bestfit::numerics::distributions;
namespace cop = bestfit::numerics::distributions::copulas;

static std::unique_ptr<cop::BivariateCopula> make_copula(const std::string& type,
                                                           const std::vector<double>& params) {
    auto c = cop::create_copula(type);
    c->set_copula_parameters(params);
    return c;
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
    throw py::value_error("copula '" + type + "' has no tau-based method-of-moments fit");
}

void register_copulas(py::module_& m) {
    // method + flat numeric args in, double out. Methods: pdf/log_pdf/cdf (args=[u,v]),
    // inverse_cdf (args=[u,v,index]), upper_tail_dependence, lower_tail_dependence, theta,
    // df (2-parameter copulas only; get_copula_parameters()[1]), or_exceedance/
    // and_exceedance (args=[u,v]), parameters_valid, random_value (args=[sample_size, seed,
    // row, col]; stateless -- GenerateRandomValues seeds its own LatinHypercube draw from
    // `seed`). marg_x_target/marg_y_target optionally attach marginals directly (the C#
    // `Copula(theta, marginX, marginY)` ctor path), mirroring cop_fit's fitted-marginals
    // convention: "" means no marginal for that side.
    m.def(
        "cop_val",
        [](const std::string& type, const std::vector<double>& params, const std::string& method,
           const std::vector<double>& args, const std::string& marg_x_target,
           const std::vector<double>& marg_x_params, const std::string& marg_y_target,
           const std::vector<double>& marg_y_params) {
            auto c = make_copula(type, params);
            if (!marg_x_target.empty()) {
                auto mx = dist::create_distribution(marg_x_target);
                mx->set_parameters(marg_x_params);
                c->marginal_distribution_x = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(mx));
            }
            if (!marg_y_target.empty()) {
                auto my = dist::create_distribution(marg_y_target);
                my->set_parameters(marg_y_params);
                c->marginal_distribution_y = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(my));
            }
            if (method == "pdf") return c->pdf(args[0], args[1]);
            if (method == "log_pdf") return c->log_pdf(args[0], args[1]);
            if (method == "cdf") return c->cdf(args[0], args[1]);
            if (method == "inverse_cdf")
                return c->inverse_cdf(args[0], args[1])[static_cast<std::size_t>(args[2])];
            if (method == "upper_tail_dependence") return c->upper_tail_dependence();
            if (method == "lower_tail_dependence") return c->lower_tail_dependence();
            if (method == "theta") return c->theta();
            if (method == "df") return c->get_copula_parameters()[1];
            if (method == "or_exceedance") return c->or_joint_exceedance_probability(args[0], args[1]);
            if (method == "and_exceedance") return c->and_joint_exceedance_probability(args[0], args[1]);
            if (method == "parameters_valid") return c->parameters_valid() ? 1.0 : 0.0;
            if (method == "random_value") {
                auto sample = c->generate_random_values(static_cast<int>(args[0]), static_cast<int>(args[1]));
                return sample[static_cast<std::size_t>(args[2])][static_cast<std::size_t>(args[3])];
            }
            throw py::value_error("unknown copula fixture method: " + method);
        },
        py::arg("type"), py::arg("params"), py::arg("method"), py::arg("args"),
        py::arg("marg_x_target") = "", py::arg("marg_x_params") = std::vector<double>{},
        py::arg("marg_y_target") = "", py::arg("marg_y_params") = std::vector<double>{});

    // Fits a copula (+ optionally its marginals) and returns {"params": [...],
    // "marg_x_params": [...], "marg_y_params": [...]} (empty lists when the fit has no
    // parametric marginals). x/y are the sample data for "tau"/"ifm"/"mle" or the
    // precomputed plotting positions for "mpl" (mirroring the C# test flow, see
    // fixtures/README.md). marg_x/marg_y are marginal distribution type names ("Normal",
    // ...) or "" when unused.
    m.def("cop_fit", [](const std::string& type, const std::vector<double>& x,
                         const std::vector<double>& y, const std::string& method,
                         const std::string& marg_x, const std::string& marg_y) {
        auto c = cop::create_copula(type);

        if (!marg_x.empty() && !marg_y.empty()) {
            auto mx = dist::create_distribution(marg_x);
            auto my = dist::create_distribution(marg_y);
            if (method == "ifm") {
                auto* ex = dynamic_cast<dist::IEstimation*>(mx.get());
                auto* ey = dynamic_cast<dist::IEstimation*>(my.get());
                if (ex == nullptr || ey == nullptr)
                    throw py::value_error("marginal '" + marg_x + "'/'" + marg_y +
                                           "' does not support estimation");
                ex->estimate(x, dist::ParameterEstimationMethod::MaximumLikelihood);
                ey->estimate(y, dist::ParameterEstimationMethod::MaximumLikelihood);
            }
            c->marginal_distribution_x = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(mx));
            c->marginal_distribution_y = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(my));
        }

        if (method == "tau") {
            set_theta_from_tau_dispatch(*c, type, x, y);
        } else if (method == "mpl") {
            cop::estimate(*c, x, y, cop::CopulaEstimationMethod::PseudoLikelihood);
        } else if (method == "ifm") {
            cop::estimate(*c, x, y, cop::CopulaEstimationMethod::InferenceFromMargins);
        } else if (method == "mle") {
            cop::estimate(*c, x, y, cop::CopulaEstimationMethod::FullLikelihood);
        } else {
            throw py::value_error("unknown copula fit method: " + method);
        }

        std::map<std::string, std::vector<double>> out;
        out["params"] = c->get_copula_parameters();
        out["marg_x_params"] =
            c->marginal_distribution_x ? c->marginal_distribution_x->get_parameters() : std::vector<double>{};
        out["marg_y_params"] =
            c->marginal_distribution_y ? c->marginal_distribution_y->get_parameters() : std::vector<double>{};
        return out;
    });
}
