// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataFrame.cs @ fc28c0c
// (the "Plotting Positions" region, lines 1104-1488)
//
// Out-of-line definitions of DataFrame::calculate_plotting_positions() and
// DataFrame::apply_langbein_conversion() -- the Hirsch-Stedinger censored plotting
// positions (Hirsch & Stedinger 1987; Bulletin 17C Appendix 5) and the Langbein
// partial-duration conversion. Split from data_frame.hpp purely for file size; this
// header is included by data_frame.hpp after the class definition and must not be
// included directly.
//
// TIE ORDERING: the oracle values (PlottingPositionTests.cs) are sensitive to how C#
// List<T>.Sort orders EQUAL values (it is deterministic but not stable), because tied
// observations receive distinct plotting positions by rank. std::sort/std::stable_sort
// therefore cannot reproduce the C# results; detail::dotnet_list_sort below is a
// faithful port of .NET's ArraySortHelper<T>.IntrospectiveSort (dotnet/runtime,
// src/libraries/System.Private.CoreLib/src/System/Collections/Generic/ArraySortHelper.cs
// -- the algorithm behind List<T>.Sort(Comparison<T>)), verified against dotnet 10
// output for the tie patterns in the B17C examples.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

// (Included by data_frame.hpp; all types it needs are already declared there.)

namespace corehydro::models {

namespace detail {

// C# System.Double.CompareTo semantics: NaN sorts before everything and equals NaN.
inline int compare_double(double x, double y) {
    if (x < y) return -1;
    if (x > y) return 1;
    if (x == y) return 0;
    if (std::isnan(x)) return std::isnan(y) ? 0 : -1;
    return 1;
}

// --- .NET introsort port (see the file header). `Compare` returns an int with C#
// Comparison<T> semantics: negative / zero / positive. ---

template <typename T, typename Compare>
inline void dn_swap_if_greater(T* keys, Compare& cmp, int i, int j) {
    if (cmp(keys[i], keys[j]) > 0) std::swap(keys[i], keys[j]);
}

template <typename T, typename Compare>
inline void dn_insertion_sort(T* keys, int length, Compare& cmp) {
    for (int i = 0; i < length - 1; i++) {
        T t = keys[i + 1];
        int j = i;
        while (j >= 0 && cmp(t, keys[j]) < 0) {
            keys[j + 1] = keys[j];
            j--;
        }
        keys[j + 1] = t;
    }
}

template <typename T, typename Compare>
inline void dn_down_heap(T* keys, int i, int n, Compare& cmp) {
    T d = keys[i - 1];
    while (i <= n / 2) {
        int child = 2 * i;
        if (child < n && cmp(keys[child - 1], keys[child]) < 0) child++;
        if (!(cmp(d, keys[child - 1]) < 0)) break;
        keys[i - 1] = keys[child - 1];
        i = child;
    }
    keys[i - 1] = d;
}

template <typename T, typename Compare>
inline void dn_heap_sort(T* keys, int length, Compare& cmp) {
    int n = length;
    for (int i = n / 2; i >= 1; i--) dn_down_heap(keys, i, n, cmp);
    for (int i = n; i > 1; i--) {
        std::swap(keys[0], keys[i - 1]);
        dn_down_heap(keys, 1, i - 1, cmp);
    }
}

template <typename T, typename Compare>
inline int dn_pick_pivot_and_partition(T* keys, int length, Compare& cmp) {
    int hi = length - 1;
    // Median-of-three pivot: sort lo/mid/hi, pivot = mid, parked at hi - 1.
    int middle = hi >> 1;
    dn_swap_if_greater(keys, cmp, 0, middle);
    dn_swap_if_greater(keys, cmp, 0, hi);
    dn_swap_if_greater(keys, cmp, middle, hi);
    T pivot = keys[middle];
    std::swap(keys[middle], keys[hi - 1]);
    int left = 0, right = hi - 1;
    while (left < right) {
        while (cmp(keys[++left], pivot) < 0) {
        }
        while (cmp(pivot, keys[--right]) < 0) {
        }
        if (left >= right) break;
        std::swap(keys[left], keys[right]);
    }
    if (left != hi - 1) std::swap(keys[left], keys[hi - 1]);
    return left;
}

template <typename T, typename Compare>
inline void dn_intro_sort(T* keys, int length, int depth_limit, Compare& cmp) {
    const int kIntrosortSizeThreshold = 16;
    int partition_size = length;
    while (partition_size > 1) {
        if (partition_size <= kIntrosortSizeThreshold) {
            if (partition_size == 2) {
                dn_swap_if_greater(keys, cmp, 0, 1);
                return;
            }
            if (partition_size == 3) {
                dn_swap_if_greater(keys, cmp, 0, 1);
                dn_swap_if_greater(keys, cmp, 0, 2);
                dn_swap_if_greater(keys, cmp, 1, 2);
                return;
            }
            dn_insertion_sort(keys, partition_size, cmp);
            return;
        }
        if (depth_limit == 0) {
            dn_heap_sort(keys, partition_size, cmp);
            return;
        }
        depth_limit--;
        int p = dn_pick_pivot_and_partition(keys, partition_size, cmp);
        dn_intro_sort(keys + p + 1, partition_size - (p + 1), depth_limit, cmp);
        partition_size = p;
    }
}

// C# List<T>.Sort(Comparison<T>): introsort with depth limit 2*(floor(log2(n)) + 1).
template <typename T, typename Compare>
inline void dotnet_list_sort(std::vector<T>& items, Compare cmp) {
    if (items.size() <= 1) return;
    unsigned n = static_cast<unsigned>(items.size());
    int log2n = 0;
    while (n >>= 1) log2n++;
    dn_intro_sort(items.data(), static_cast<int>(items.size()), 2 * (log2n + 1), cmp);
}

}  // namespace detail

// Provides plotting positions for censored data using the Hirsch-Stedinger plotting
// position formula (C# CalculatePlottingPositions, line 1140).
//
// References: Hirsch & Stedinger (1987) WRR; Cohn, Lane & Baier (1997) WRR; Cohn, Lane
// & Stedinger (2001) WRR; Bulletin 17C Appendix 5 (USGS 2018); the PeakfqSA FORTRAN
// source.
//
// EVENT CADENCE (M4 review follow-up): in the C#, ProcessThresholdSeries() is the FIRST
// statement of CalculatePlottingPositions() itself (lines 1142-1144, "Process threshold
// series to compute NumberBelow/NumberAbove before TotalRecordLength() is called
// below"); the collection-changed events only decide WHEN CalculatePlottingPositions
// runs, not its internal ordering. The C++ therefore calls process_threshold_series()
// here, unconditionally, before anything else -- exactly the C# effective ordering. The
// SuppressCollectionChanged bracketing and the final RaisePropertyChange are WPF event
// plumbing with no C++ counterpart.
inline void DataFrame::calculate_plotting_positions() {
    // Process threshold series to compute NumberBelow/NumberAbove
    // before TotalRecordLength() is called below.
    process_threshold_series();

    // The plotting position parameter
    double alpha = plotting_parameter_;
    int total_years = total_record_length();

    std::size_t threshold_count = threshold_series_.count();
    // Probability of exceedance
    std::vector<double> pej(threshold_count, 0.0);
    // Condition probability that a value falls between the j-th and (j-1)-th threshold.
    std::vector<double> qej(threshold_count, 0.0);
    // The number of values that exceed threshold j but not higher thresholds (j-1).
    std::vector<int> kj(threshold_count, 0);
    // The total number of years the threshold applies
    std::vector<int> nj(threshold_count, 0);
    std::vector<int> durations(threshold_count, 0);
    // The total number of values that exceed higher thresholds during period Nj
    std::vector<int> kl(threshold_count, 0);
    // The values above / below thresholds (references into the owned series, so the
    // plotting-position writes below land on the real ordinates, as in the C#).
    std::vector<Data*> above_values;
    std::vector<Data*> below_values;

    // Sort thresholds from largest to smallest. The C# sorts a CLONE of the series, so
    // the per-threshold PlottingPosition write below lands on discarded clones -- that
    // quirk is mirrored deliberately (the owned ThresholdSeries keeps PlottingPosition
    // untouched, exactly like the upstream).
    std::vector<ThresholdData> new_threshold_data_list = threshold_series_.to_list();
    detail::dotnet_list_sort(new_threshold_data_list,
                             [](const ThresholdData& x, const ThresholdData& y) {
                                 return -detail::compare_double(x.value(), y.value());
                             });
    for (std::size_t i = 0; i < new_threshold_data_list.size(); i++) {
        ThresholdData& threshold_data = new_threshold_data_list[i];
        // Determine the number of years that the threshold applies.
        if (i == 0) {
            nj[i] = total_years;
            durations[i] = threshold_data.duration();
        } else {
            // Ensure Nj doesn't go negative if durations exceed totalYears
            nj[i] = std::max(0, total_years - durations[i - 1]);
            durations[i] = durations[i - 1] + threshold_data.duration();
        }

        // Check interval data
        for (std::size_t j = 0; j < interval_series_.count(); j++) {
            if (i == 0) {
                if (interval_series_[j].value() >= threshold_data.value()) {
                    kj[i] += 1;
                }
            } else {
                if (interval_series_[j].value() >= threshold_data.value() &&
                    interval_series_[j].value() < new_threshold_data_list[i - 1].value()) {
                    kj[i] += 1;
                }
                // Determine if the value exceeds higher thresholds during period Nj
                bool kl_bool = false;
                for (int k = static_cast<int>(i) - 1; k >= 0; k -= 1) {
                    if (interval_series_[j].value() >=
                        new_threshold_data_list[static_cast<std::size_t>(k)].value()) {
                        kl_bool = true;
                        // Check if the event occurs during a larger threshold's period
                        if (interval_series_[j].index() >=
                                new_threshold_data_list[static_cast<std::size_t>(k)]
                                    .start_index() &&
                            interval_series_[j].index() <=
                                new_threshold_data_list[static_cast<std::size_t>(k)]
                                    .end_index()) {
                            kl_bool = false;
                            break;
                        }
                    }
                }
                if (kl_bool) {
                    kl[i] += 1;
                }
            }
        }

        // Check uncertain data
        for (std::size_t j = 0; j < uncertain_series_.count(); j++) {
            if (i == 0) {
                if (uncertain_series_[j].value() >= threshold_data.value()) {
                    kj[i] += 1;
                }
            } else {
                if (uncertain_series_[j].value() >= threshold_data.value() &&
                    uncertain_series_[j].value() <
                        new_threshold_data_list[i - 1].value()) {
                    kj[i] += 1;
                }
                // Determine if the value exceeds higher thresholds during period Nj
                bool kl_bool = false;
                for (int k = static_cast<int>(i) - 1; k >= 0; k -= 1) {
                    if (uncertain_series_[j].value() >=
                        new_threshold_data_list[static_cast<std::size_t>(k)].value()) {
                        kl_bool = true;
                        // Check if the event occurs during a larger threshold's period
                        if (uncertain_series_[j].index() >=
                                new_threshold_data_list[static_cast<std::size_t>(k)]
                                    .start_index() &&
                            uncertain_series_[j].index() <=
                                new_threshold_data_list[static_cast<std::size_t>(k)]
                                    .end_index()) {
                            kl_bool = false;
                            break;
                        }
                    }
                }
                if (kl_bool) {
                    kl[i] += 1;
                }
            }
        }

        // Check exact data
        for (std::size_t j = 0; j < exact_series_.count(); j++) {
            if (i == 0) {
                if (exact_series_[j].value() >= threshold_data.value()) {
                    kj[i] += 1;
                }
            } else {
                if (exact_series_[j].value() >= threshold_data.value() &&
                    exact_series_[j].value() < new_threshold_data_list[i - 1].value()) {
                    kj[i] += 1;
                }
                // Determine if the value exceeds higher thresholds during period Nj
                bool kl_bool = false;
                for (int k = static_cast<int>(i) - 1; k >= 0; k -= 1) {
                    if (exact_series_[j].value() >=
                        new_threshold_data_list[static_cast<std::size_t>(k)].value()) {
                        kl_bool = true;
                        // Check if the event occurs during a larger threshold's period
                        if (exact_series_[j].index() >=
                                new_threshold_data_list[static_cast<std::size_t>(k)]
                                    .start_index() &&
                            exact_series_[j].index() <=
                                new_threshold_data_list[static_cast<std::size_t>(k)]
                                    .end_index()) {
                            kl_bool = false;
                            break;
                        }
                    }
                }
                if (kl_bool) {
                    kl[i] += 1;
                }
            }
        }

        // Compute threshold exceedance probability
        if (i == 0) {
            kl[i] = 0;
            int denominator = nj[i] - kl[i];
            // Guard against division by zero: if all observations are below threshold,
            // Qej = 0
            qej[i] = denominator > 0 ? kj[i] / static_cast<double>(denominator) : 0.0;
            pej[i] = qej[i];
        } else {
            int denominator = nj[i] - kl[i];
            // Guard against division by zero: if all observations are below threshold,
            // Qej = 0
            qej[i] = denominator > 0 ? kj[i] / static_cast<double>(denominator) : 0.0;
            pej[i] = pej[i - 1] + (1 - pej[i - 1]) * qej[i];
        }

        // Record plotting position (on the local clone -- see the note above)
        threshold_data.set_plotting_position(pej[i]);
    }

    // Record the values that are above and below thresholds
    // Loop through interval data
    for (std::size_t i = 0; i < interval_series_.count(); i++) {
        bool above = false;
        for (std::size_t j = 0; j < new_threshold_data_list.size(); j++) {
            if (interval_series_[i].value() >= new_threshold_data_list[j].value()) {
                above = true;
                break;
            }
        }
        if (above) {
            above_values.push_back(&interval_series_[i]);
        } else {
            below_values.push_back(&interval_series_[i]);
        }
    }
    // Loop through uncertain data
    for (std::size_t i = 0; i < uncertain_series_.count(); i++) {
        bool above = false;
        for (std::size_t j = 0; j < new_threshold_data_list.size(); j++) {
            if (uncertain_series_[i].value() >= new_threshold_data_list[j].value()) {
                above = true;
                break;
            }
        }
        if (above) {
            above_values.push_back(&uncertain_series_[i]);
        } else {
            below_values.push_back(&uncertain_series_[i]);
        }
    }
    // Loop through exact data
    for (std::size_t i = 0; i < exact_series_.count(); i++) {
        bool above = false;
        for (std::size_t j = 0; j < new_threshold_data_list.size(); j++) {
            if (exact_series_[i].value() >= new_threshold_data_list[j].value()) {
                above = true;
                break;
            }
        }
        if (above) {
            above_values.push_back(&exact_series_[i]);
        } else {
            below_values.push_back(&exact_series_[i]);
        }
    }

    // Sort above- and below-values from largest to smallest (List<T>.Sort semantics --
    // see the tie-ordering note in the file header).
    auto descending_by_value = [](const Data* x, const Data* y) {
        return -detail::compare_double(x->value(), y->value());
    };
    detail::dotnet_list_sort(above_values, descending_by_value);
    detail::dotnet_list_sort(below_values, descending_by_value);
    // The above-value probability of exceedance
    std::vector<double> pi(above_values.size(), 0.0);
    // The below-value probability of exceedance
    std::vector<double> pr(below_values.size(), 0.0);
    // The number of values below
    int nb = static_cast<int>(below_values.size());

    // Compute the above-threshold plotting positions
    int t = 0;
    for (std::size_t i = 0; i < kj.size(); i++) {
        for (int j = 1; j <= kj[i]; j++) {
            t += 1;
            if (i == 0) {
                pi[static_cast<std::size_t>(t - 1)] =
                    qej[i] * (j - alpha) / (kj[i] + 1 - 2 * alpha);
            } else {
                pi[static_cast<std::size_t>(t - 1)] =
                    pej[i - 1] +
                    (1 - pej[i - 1]) * qej[i] * (j - alpha) / (kj[i] + 1 - 2 * alpha);
            }
            above_values[static_cast<std::size_t>(t - 1)]->set_plotting_position(
                pi[static_cast<std::size_t>(t - 1)]);
        }
    }

    // Compute the below-threshold plotting positions
    double below_denominator = nb + 1 - 2 * alpha;
    // Guard against division by zero or negative denominator
    if (below_denominator <= 0)
        below_denominator = 1.0;  // Fallback to prevent invalid computation

    for (int i = 1; i <= static_cast<int>(below_values.size()); i++) {
        if (!pej.empty()) {
            pr[static_cast<std::size_t>(i - 1)] =
                pej[pej.size() - 1] +
                (1 - pej[pej.size() - 1]) * (i - alpha) / below_denominator;
        } else {
            pr[static_cast<std::size_t>(i - 1)] = (i - alpha) / below_denominator;
        }
        below_values[static_cast<std::size_t>(i - 1)]->set_plotting_position(
            pr[static_cast<std::size_t>(i - 1)]);
    }
}

// Apply the Langbein conversion to the plotting positions (C# ApplyLangbeinConversion,
// line 1458). `lambda` is the number of events per block. The SuppressCollectionChanged
// bracketing and RaisePropertyChange are WPF event plumbing with no C++ counterpart.
inline void DataFrame::apply_langbein_conversion(double lambda) {
    for (std::size_t i = 0; i < exact_series_.count(); i++) {
        exact_series_[i].set_plotting_position(
            1 - std::exp(-lambda * exact_series_[i].plotting_position()));
    }
    for (std::size_t i = 0; i < uncertain_series_.count(); i++) {
        uncertain_series_[i].set_plotting_position(
            1 - std::exp(-lambda * uncertain_series_[i].plotting_position()));
    }
    for (std::size_t i = 0; i < interval_series_.count(); i++) {
        interval_series_[i].set_plotting_position(
            1 - std::exp(-lambda * interval_series_[i].plotting_position()));
    }
}

}  // namespace corehydro::models
