// ported from: Numerics/Mathematics/Linear Algebra/Support/Vector.cs @ a2c4dbf
//
// Minimal dense vector: storage, length, indexed access, and array conversion -- the
// subset that Matrix and CholeskyDecomposition actually call. Omitted (not needed by
// either): the `Header` UI-metadata property, `Clone`/`Clear`/`CopyFrom`/`ToList`,
// `Sum`/`Norm`/`NormSquared`, the static `Distance`/`DotProduct`/`Project` helpers, all
// arithmetic (`Add`/`Subtract`/`Multiply`/`Divide`) and their operator overloads. None of
// Matrix's or CholeskyDecomposition's own methods call them; add them when a later Phase
// 2 target (e.g. MultivariateNormal) needs one.
#pragma once
#include <utility>
#include <vector>

namespace bestfit::numerics::math::linalg {

class Vector {
   public:
    // Construct a new vector with specified length, zero-initialized.
    explicit Vector(int length) : data_(static_cast<std::size_t>(length), 0.0) {}

    // Construct a new vector with specified length and fill it with a constant value.
    Vector(int length, double fill) : data_(static_cast<std::size_t>(length), fill) {}

    // Construct a new vector based on an initial array.
    explicit Vector(std::vector<double> initial_array) : data_(std::move(initial_array)) {}

    // The length of the vector.
    int length() const { return static_cast<int>(data_.size()); }

    // Get/set the element at the specific index.
    double operator[](int index) const { return data_[static_cast<std::size_t>(index)]; }
    double& operator[](int index) { return data_[static_cast<std::size_t>(index)]; }

    // Returns the vector as an array (copy).
    std::vector<double> to_array() const { return data_; }

   private:
    std::vector<double> data_;
};

}  // namespace bestfit::numerics::math::linalg
