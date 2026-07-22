// ported from: Numerics/Data/Interpolation/Support/Transform.cs @ 2a0357a
//
// Data transformation applied before/after interpolation (Linear/Bilinear). Member names
// mirror the C# enum verbatim; the [Display(Name = ...)] attributes (WPF UI labels) are
// not ported.
#pragma once

namespace corehydro::numerics::data {

enum class Transform {
    None,
    Logarithmic,
    NormalZ
};

}  // namespace corehydro::numerics::data
