// ported from: Numerics/Data/Interpolation/Support/Transform.cs @ a2c4dbf
//
// Data transformation applied before/after interpolation (Linear/Bilinear). Member names
// mirror the C# enum verbatim; the [Display(Name = ...)] attributes (WPF UI labels) are
// not ported.
#pragma once

namespace bestfit::numerics::data {

enum class Transform {
    None,
    Logarithmic,
    NormalZ
};

}  // namespace bestfit::numerics::data
