// ported from: Numerics/Functions/Link Functions/LinkController.cs @ a2c4dbf
//
// Controller for managing independent link functions applied to each element of a
// parameter vector. Each parameter index can have its own link function, or null for no
// transformation (identity) -- the C# `ILinkFunction?[] Links` becomes a
// std::vector<std::unique_ptr<ILinkFunction>> where a nullptr entry preserves the exact
// null-means-identity semantics (no IdentityLink objects are substituted, matching the
// C#). The Jacobian and log-determinant methods support the change-of-variables formula
// for transformed-space probability calculations: p(phi) = p(theta) |dtheta/dphi|.
// The XElement constructor and ToXElement are dropped (serialization is a desktop
// concern). Owning unique_ptr storage makes the controller move-only, unlike the
// shared-reference C# array; the C# `params` constructor becomes the vector constructor.
#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "bestfit/numerics/functions/i_link_function.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"

namespace bestfit::numerics::functions {

class LinkController {
   public:
    // Initializes a new instance with no link functions (all parameters untransformed).
    LinkController() = default;

    // Initializes a new instance with the specified link functions, one per parameter
    // index. Null entries indicate identity (no transformation).
    explicit LinkController(std::vector<std::unique_ptr<ILinkFunction>> links)
        : links_(std::move(links)) {}

    // The array of link functions, one per parameter index. Null entries indicate
    // identity (mirrors the C# `Links` get-only property; elements stay mutable through
    // the non-const overload, like the C# array reference).
    const std::vector<std::unique_ptr<ILinkFunction>>& links() const { return links_; }
    std::vector<std::unique_ptr<ILinkFunction>>& links() { return links_; }

    // Gets the number of link functions registered.
    int count() const { return static_cast<int>(links_.size()); }

    // Gets the link function at the specified parameter index, or null if the index is
    // out of range or no link is assigned.
    const ILinkFunction* operator[](int index) const {
        return index >= 0 && index < count() ? links_[static_cast<std::size_t>(index)].get()
                                             : nullptr;
    }

    // Applies the link functions element-wise: eta[i] = h_i(x[i]).
    std::vector<double> link(const std::vector<double>& x) const {
        std::vector<double> eta = x;
        std::size_t n = std::min(x.size(), links_.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (links_[i] != nullptr) eta[i] = links_[i]->link(x[i]);
        }
        return eta;
    }

    // Applies the inverse link functions element-wise: x[i] = h_i^-1(eta[i]).
    std::vector<double> inverse_link(const std::vector<double>& eta) const {
        std::vector<double> x = eta;
        std::size_t n = std::min(eta.size(), links_.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (links_[i] != nullptr) x[i] = links_[i]->inverse_link(eta[i]);
        }
        return x;
    }

    // Computes the diagonal Jacobian matrix of the link transformation: a diagonal
    // matrix with elements deta_i/dtheta_i for each parameter (each parameter has an
    // independent link function, so off-diagonal elements are zero).
    math::linalg::Matrix link_jacobian(const std::vector<double>& x) const {
        int p = static_cast<int>(x.size());
        auto G = math::linalg::Matrix::identity(p);
        std::size_t n = std::min(x.size(), links_.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (links_[i] != nullptr) {
                int idx = static_cast<int>(i);
                G(idx, idx) = links_[i]->d_link(x[i]);
            }
        }
        return G;
    }

    // Computes the log-determinant of the inverse Jacobian |dtheta/dphi| for the
    // change-of-variables formula, given phi in link-space (transformed coordinates).
    // For diagonal link Jacobians: log|det| = -sum log|deta_j/dtheta_j|.
    double log_det_jacobian(const std::vector<double>& phi) const {
        const double tiny = 1e-16;
        double sum = 0.0;
        std::size_t n = std::min(phi.size(), links_.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (links_[i] != nullptr) {
                double theta_i = links_[i]->inverse_link(phi[i]);
                double d_eta_d_theta = links_[i]->d_link(theta_i);
                // Use fabs to handle both increasing and decreasing link functions.
                sum -= std::log(std::max(std::fabs(d_eta_d_theta), tiny));
            }
        }
        return sum;
    }

    // Creates a LinkController for the standard 3-parameter (location, scale, shape)
    // case. Null arguments indicate identity, exactly as in the C#.
    static LinkController for_location_scale_shape(
        std::unique_ptr<ILinkFunction> location_link = nullptr,
        std::unique_ptr<ILinkFunction> scale_link = nullptr,
        std::unique_ptr<ILinkFunction> shape_link = nullptr) {
        std::vector<std::unique_ptr<ILinkFunction>> links;
        links.reserve(3);
        links.push_back(std::move(location_link));
        links.push_back(std::move(scale_link));
        links.push_back(std::move(shape_link));
        return LinkController(std::move(links));
    }

   private:
    std::vector<std::unique_ptr<ILinkFunction>> links_;
};

}  // namespace bestfit::numerics::functions
