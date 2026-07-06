// ported from: RMC.BestFit/Models/TimeSeries/TransformType.cs @ fc28c0c
//
// The C# file is named TransformType.cs but the ENUM it declares is `Transform` (in namespace
// RMC.BestFit.Models). The AR/MA property that HOLDS a value of this enum is named
// `TransformType`; the enum TYPE itself is `Transform`. This port mirrors the C# names exactly:
// the type is `bestfit::models::Transform` and the model accessor is `transform_type()`.
// Enumerators are kept in the C# declaration order (None, Logarithmic, BoxCox, YeoJohnson) and
// are NOT renamed.
#pragma once

namespace bestfit::models {

// Enumeration of time series transform types.
enum class Transform {
    None,
    Logarithmic,
    BoxCox,
    YeoJohnson,
};

}  // namespace bestfit::models
