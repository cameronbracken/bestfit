// ported from: Numerics/Data/Interpolation/Support/SortOrder.cs @ 2a0357a
//
// Sort order of the x-values driving the Interpolater search machinery.
#pragma once

namespace corehydro::numerics::data {

enum class SortOrder {
    Ascending,
    Descending,
    None
};

}  // namespace corehydro::numerics::data
