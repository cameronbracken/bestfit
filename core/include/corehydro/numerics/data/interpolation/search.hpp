// ported from: Numerics/Data/Interpolation/Support/Search.cs @ 2a0357a
//
// Only the two overloads this port's callers actually need: `sequential` (SNIS.cs's
// `Search.Sequential(rndOut[i], cdf, idx)` posterior-weight lookup during output
// resampling) and `bisection` (Histogram.cs's `GetBinIndexOf` bin lookup -- see
// histogram.hpp; not called out in the original task brief, which scoped this file to
// SNIS's Sequential overload alone, but Histogram's own faithful port requires it, and
// per this repo's standing rule the C# source governs over plan text). The
// OrderedPairedData/Ordinate overloads of Sequential/Bisection, and all three overloads of
// Hunt, are not ported -- no caller in this port's scope needs them.
//
// This is a DIFFERENT algorithm from Interpolater's own sequential_search/
// bisection_search/hunt_search (ported in Phase 2 for the Linear/Bilinear path):
// Interpolater's variants are member functions with correlated-call search-start memory
// and no boundary-sentinel returns, while Search's are free functions returning
// -1/0/(N-1)/N sentinels for out-of-range/exact-endpoint queries. Phase 2 deliberately
// skipped this file entirely (see interpolater.hpp's header comment); this header ports
// the SNIS/Histogram-required subset only.
//
// v2.1.4 sync (Numerics 33dc1af): FIXED, not mirrored -- bisection()'s loop condition used
// to be `x >= values[xm] && order == Ascending` (a logical AND against the order flag), not
// `(x >= values[xm]) == ascending` the way Interpolater's own bisection_search phrases the
// equivalent test. For order == Descending that condition was always false, so the loop
// only ever shrank `xhi`; `xlo` never advanced past `start`, so bisection() in descending
// order always returned `start` instead of the correct bracketing index. Upstream's fix
// splits the loop into separate ascending/descending branches (`x >= values[xm]` for
// ascending, `x < values[xm]` for descending) rather than adopting the equality-test
// phrasing Interpolater/Hunt already used; this port mirrors that same split. Previously
// dead code for every caller in this port's scope (Histogram::get_bin_index_of and SNIS
// both only ever call with the default Ascending order), now oracle-covered in the
// descending direction too via fixtures/special_functions/search.json's
// `*_descending_*` cases. See docs/upstream-csharp-issues.md (marked RESOLVED).
#pragma once
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/sort_order.hpp"

namespace corehydro::numerics::data::search {

// Sequential (linear) search for the lower-bound index of x within `values`, scanning
// forward from `start`. Returns -1 if x is below the first value (above the first, for
// descending order); returns `values.size()` if x is beyond the far endpoint; returns the
// exact endpoint index (0 or N-1) on an exact match; otherwise the lower-bound index found
// by scanning forward from `start`.
inline int sequential(double x, const std::vector<double>& values, int start = 0,
                       SortOrder order = SortOrder::Ascending) {
    int n = static_cast<int>(values.size());
    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start >= n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (order == SortOrder::Ascending) {
        if (x < values[0]) return -1;
        if (x == values[0]) return 0;
        if (x == values[static_cast<std::size_t>(n - 1)]) return n - 1;
        if (x > values[static_cast<std::size_t>(n - 1)]) return n;
        for (int i = start; i < n; ++i)
            if (x <= values[static_cast<std::size_t>(i)]) return i - 1;
    } else {
        if (x > values[0]) return -1;
        if (x == values[0]) return 0;
        if (x == values[static_cast<std::size_t>(n - 1)]) return n - 1;
        if (x < values[static_cast<std::size_t>(n - 1)]) return n;
        for (int i = start; i < n; ++i)
            if (x >= values[static_cast<std::size_t>(i)]) return i - 1;
    }
    return 0;
}

// Bisection search for the lower-bound index of x within `values`; same boundary-sentinel
// conventions as sequential() above, halving the search bracket instead of scanning
// linearly. See the file header for a transcribed-verbatim quirk in descending order.
inline int bisection(double x, const std::vector<double>& values, int start = 0,
                      SortOrder order = SortOrder::Ascending) {
    int n = static_cast<int>(values.size());
    if (start < 0) throw std::out_of_range("The search starting point must be non-negative.");
    if (start >= n)
        throw std::out_of_range(
            "The search starting point cannot be greater than the length of the X array.");

    if (order == SortOrder::Ascending) {
        if (x < values[0]) return -1;
        if (x == values[0]) return 0;
        if (x == values[static_cast<std::size_t>(n - 1)]) return n - 1;
        if (x > values[static_cast<std::size_t>(n - 1)]) return n;
    } else {
        if (x > values[0]) return -1;
        if (x == values[0]) return 0;
        if (x == values[static_cast<std::size_t>(n - 1)]) return n - 1;
        if (x < values[static_cast<std::size_t>(n - 1)]) return n;
    }

    int xlo = start, xhi = n;
    if (order == SortOrder::Ascending) {
        while (xhi - xlo > 1) {
            int xm = xlo + ((xhi - xlo) >> 1);
            if (x >= values[static_cast<std::size_t>(xm)])
                xlo = xm;
            else
                xhi = xm;
        }
    } else {
        while (xhi - xlo > 1) {
            int xm = xlo + ((xhi - xlo) >> 1);
            if (x < values[static_cast<std::size_t>(xm)])
                xlo = xm;
            else
                xhi = xm;
        }
    }
    return xlo;
}

}  // namespace corehydro::numerics::data::search
