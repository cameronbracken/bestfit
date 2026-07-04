// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/DataComponent.cs @ fc28c0c
//
// Represents a single data point's contribution to the log-likelihood for influence
// diagnostics. The C# type is a `readonly struct` (no setters, value semantics, chosen
// upstream for cache locality / GC-pressure reasons in parallel loops). The C++ port keeps
// the "immutable-style" spirit -- every field is set only at construction and exposed via
// getters, no setters exist -- but deliberately does NOT mark the members `const`: `const`
// members would delete the compiler-generated copy/move assignment operators, which
// `std::vector<DataComponent>` usage (sorting, re-assignment, erase-remove) needs. Ordinary
// (non-const) private members plus getter-only public API gives the same external
// immutability with normal value-type ergonomics.
#pragma once
#include <cstdio>
#include <optional>
#include <string>

namespace bestfit::models {

enum class DataComponentType { Exact, Uncertain, Interval, LeftCensored, RightCensored };

class DataComponent {
   public:
    // Exact-observation ctor (C# 4-arg ctor): Count=1, Type=Exact.
    DataComponent(int index, double log_likelihood, double value,
                  std::optional<std::string> name = std::nullopt)
        : index_(index),
          log_likelihood_(log_likelihood),
          value_(value),
          type_(DataComponentType::Exact),
          count_(1),
          name_(std::move(name)) {}

    // Full-specification ctor (C# 6-arg ctor).
    DataComponent(int index, double log_likelihood, double value, DataComponentType type,
                  int count = 1, std::optional<std::string> name = std::nullopt)
        : index_(index),
          log_likelihood_(log_likelihood),
          value_(value),
          type_(type),
          count_(count),
          name_(std::move(name)) {}

    int index() const { return index_; }
    double log_likelihood() const { return log_likelihood_; }
    double value() const { return value_; }
    DataComponentType type() const { return type_; }
    int count() const { return count_; }
    const std::optional<std::string>& name() const { return name_; }

    // True for left- or right-censored observations.
    bool is_censored() const {
        return type_ == DataComponentType::LeftCensored || type_ == DataComponentType::RightCensored;
    }

    // True for censored observations that also represent grouped/threshold data (count > 1).
    bool is_threshold() const { return is_censored() && count_ > 1; }

    // Mirrors the C# `ToString()` switch. Prefixes (`~`, `[]`, `<`, `>`) and the `(n=Count)`
    // suffix for censored types are preserved; the exact `:G4`/`:F4` numeric formatting is
    // approximated with `%.4g` / `%.4f` rather than chased bit-for-bit (not oracle-checked).
    std::string to_string() const {
        std::string name_str =
            (name_.has_value() && !name_->empty()) ? *name_ : "[" + std::to_string(index_) + "]";

        char value_buf[64];
        std::snprintf(value_buf, sizeof value_buf, "%.4g", value_);
        char ll_buf[64];
        std::snprintf(ll_buf, sizeof ll_buf, "%.4f", log_likelihood_);
        std::string value_str(value_buf);
        std::string ll_str(ll_buf);

        switch (type_) {
            case DataComponentType::Exact:
                return name_str + ": " + value_str + " = " + ll_str;
            case DataComponentType::Uncertain:
                return name_str + ": ~" + value_str + " = " + ll_str;
            case DataComponentType::Interval:
                return name_str + ": [" + value_str + "] = " + ll_str;
            case DataComponentType::LeftCensored:
                return name_str + ": <" + value_str + " (n=" + std::to_string(count_) + ") = " + ll_str;
            case DataComponentType::RightCensored:
                return name_str + ": >" + value_str + " (n=" + std::to_string(count_) + ") = " + ll_str;
            default:
                return name_str + ": " + value_str + " = " + ll_str;
        }
    }

   private:
    int index_;
    double log_likelihood_;
    double value_;
    DataComponentType type_;
    int count_;
    std::optional<std::string> name_;
};

}  // namespace bestfit::models
