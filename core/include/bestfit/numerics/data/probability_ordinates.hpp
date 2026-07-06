// ported from: Numerics/Data/Paired Data/ProbabilityOrdinate.cs @ a2c4dbf
//
// ProbabilityOrdinates: a flat collection of EXCEEDANCE probabilities at which a frequency
// curve is tabulated. The C# file is named ProbabilityOrdinate.cs but the class inside is
// ProbabilityOrdinates (plural) and subclasses List<double>; there is NO scalar ordinate type.
//
// This is the mutable probability grid every Phase-8 analysis holds and iterates (A2 builds a
// grid from it; A4's IProbabilityOrdinates wraps it; A5/A6/A7 flip exceedance <-> non-exceedance
// while driving distributions). The never-mutate rule is relaxed for it (it mirrors the C#
// mutable stateful collection).
//
// Deliberately NOT ported (the ~245 lines of WPF plumbing): INotifyCollectionChanged /
// INotifyPropertyChanged, the shadowed `new` mutators (Add/AddRange/Insert/InsertRange/Remove/
// RemoveAt/RemoveRange/Clear/indexer-setter that raise events), SuppressCollectionChanged,
// RaiseCollectionChangedReset, OnPropertyChanged/OnCollectionChanged, and all
// NotifyCollectionChangedEventArgs. Every mutation collapses to a plain std::vector<double> op.
#pragma once

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace bestfit::numerics::data {

// Result of Validate(), mirroring the C# tuple (bool IsValid, List<string> ValidationMessages).
// Kept as a local lightweight struct to avoid a numerics -> models layering inversion:
// models/support/validation_result.hpp lives ABOVE numerics/, so a numerics header must not
// include it. C# governs the field shape (IsValid -> is_valid, ValidationMessages -> messages).
struct ProbabilityOrdinatesValidationResult {
    bool is_valid = true;
    std::vector<std::string> messages;
};

class ProbabilityOrdinates {
   public:
    // The default delimiter used when converting to and from strings (C# DefaultDelimiter = "|").
    static constexpr const char* kDefaultDelimiter = "|";

    // Initializes with the 25 standard default exceedance probabilities (C# line 71). AddDefaults
    // appends, so on a fresh (empty) instance this yields exactly the 25 defaults.
    ProbabilityOrdinates() { add_defaults(); }

    // Initializes populated from a caller-supplied sequence (C# AddRange(probabilities), line 86).
    // C# throws ArgumentNullException on null; in C++ an empty range simply yields an empty
    // collection (null-guard is N/A for value semantics).
    explicit ProbabilityOrdinates(std::vector<double> probabilities)
        : values_(std::move(probabilities)) {}

    // Iterator-pair sequence ctor (same append-from-range semantics) for callers holding a range.
    template <typename InputIt>
    ProbabilityOrdinates(InputIt first, InputIt last) : values_(first, last) {}

    // ---- Collection surface (only what the downstream ports need; YAGNI on the event API) ----
    double& operator[](std::size_t index) { return values_[index]; }
    double operator[](std::size_t index) const { return values_[index]; }
    double& at(std::size_t index) { return values_.at(index); }
    double at(std::size_t index) const { return values_.at(index); }

    std::size_t size() const { return values_.size(); }
    std::size_t count() const { return values_.size(); }  // C# List<T>.Count alias
    bool empty() const { return values_.empty(); }

    std::vector<double>::iterator begin() { return values_.begin(); }
    std::vector<double>::iterator end() { return values_.end(); }
    std::vector<double>::const_iterator begin() const { return values_.begin(); }
    std::vector<double>::const_iterator end() const { return values_.end(); }

    void push_back(double p) { values_.push_back(p); }
    void add(double p) { values_.push_back(p); }  // C# Add alias
    void clear() { values_.clear(); }

    // Read-only view of the backing vector (for callers that need the raw grid).
    const std::vector<double>& values() const { return values_; }

    // Returns a string representation using the default delimiter (C# ToString, line 108).
    std::string to_string() const { return to_delimited_string(kDefaultDelimiter); }

    // Converts the ordinates to a single delimited string (C# ToDelimitedString, line 123). A
    // null/empty delimiter falls back to the default "|". Each value is formatted G17 invariant
    // culture: "%.17g" emits enough precision to round-trip a double exactly.
    std::string to_delimited_string(const std::string& delimiter) const {
        const std::string sep = delimiter.empty() ? std::string(kDefaultDelimiter) : delimiter;
        std::string out;
        for (std::size_t i = 0; i < values_.size(); ++i) {
            if (i != 0) out += sep;
            out += format_g17(values_[i]);
        }
        return out;
    }

    // Clears and replaces with the default ordinates (C# ResetToDefaults, line 137).
    void reset_to_defaults() {
        values_.clear();
        add_defaults();
    }

    // Clears, then populates from a delimited string (C# FromDelimitedString, line 161). A
    // null/empty delimiter falls back to "|". Empty/whitespace tokens are skipped; tokens that
    // fail to parse are ignored. Empty/whitespace `value` leaves the collection empty.
    void from_delimited_string(const std::string& value,
                               const std::string& delimiter = kDefaultDelimiter) {
        values_.clear();
        if (is_null_or_whitespace(value)) return;

        const std::string sep = delimiter.empty() ? std::string(kDefaultDelimiter) : delimiter;
        for (const std::string& token : split(value, sep)) {
            if (is_null_or_whitespace(token)) continue;
            double p = 0.0;
            if (try_parse(token, p)) values_.push_back(p);
        }
    }

    // Parses a delimited string into a new instance (C# static Parse, line 209): construct (with
    // defaults), clear the defaults, then FromDelimitedString.
    static ProbabilityOrdinates parse(const std::string& value,
                                      const std::string& delimiter = kDefaultDelimiter) {
        ProbabilityOrdinates ordinates;
        ordinates.clear();  // clears defaults
        ordinates.from_delimited_string(value, delimiter);
        return ordinates;
    }

    // Validates the current state (C# Validate, lines 258-299). Rules, in C# order:
    //   - empty collection -> invalid with the fixed message, early return;
    //   - otherwise for each element: p <= 0 || p >= 1 -> invalid and BREAK;
    //   - and for i > 0: p <= this[i-1] (not strictly increasing) -> invalid and BREAK.
    // Reproduces the break-on-first-failure short-circuit and the C# message text.
    ProbabilityOrdinatesValidationResult validate() const {
        ProbabilityOrdinatesValidationResult result;
        result.is_valid = true;

        // No ordinates at all.
        if (values_.empty()) {
            result.is_valid = false;
            result.messages.push_back("At least one exceedance probability must be specified.");
            return result;
        }

        for (std::size_t i = 0; i < values_.size(); ++i) {
            const double p = values_[i];

            // Range check.
            if (p <= 0.0 || p >= 1.0) {
                result.is_valid = false;
                result.messages.push_back(
                    "Exceedance probability at index " + std::to_string(i) + " has value " +
                    format_g17(p) + ", but it must be strictly between 0 and 1.");
                break;
            }

            // Strictly increasing check.
            if (i > 0 && p <= values_[i - 1]) {
                const double prev = values_[i - 1];
                result.messages.push_back(
                    "Exceedance probabilities must be strictly increasing. Value at index " +
                    std::to_string(i - 1) + " is " + format_g17(prev) + " and value at index " +
                    std::to_string(i) + " is " + format_g17(p) + ".");
                result.is_valid = false;
                break;
            }
        }

        return result;
    }

   private:
    // Appends the 25 standard default exceedance probabilities (C# AddDefaults, lines 227-242).
    // APPENDS -- does not clear (use reset_to_defaults() to clear then append). Values and order
    // are oracle-locked, transcribed exactly from the C# array.
    void add_defaults() {
        static const double kDefaults[] = {
            0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
            0.001,    0.002,    0.005,    0.01,    0.02,    0.05,    0.1,    0.2,    0.3,
            0.5,      0.7,      0.8,      0.9,     0.95,    0.98,    0.99};
        for (double d : kDefaults) values_.push_back(d);
    }

    // G17 invariant-culture format: 17 significant digits round-trips a double exactly.
    static std::string format_g17(double value) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.17g", value);
        return std::string(buf);
    }

    // Mirrors C# string.IsNullOrWhiteSpace: true for empty or all-whitespace input.
    static bool is_null_or_whitespace(const std::string& s) {
        for (char c : s) {
            if (!std::isspace(static_cast<unsigned char>(c))) return false;
        }
        return true;
    }

    // Splits on a (possibly multi-char) separator with StringSplitOptions.None semantics --
    // empty tokens between adjacent/trailing separators are retained (and later skipped by the
    // whitespace guard, matching the C# per-token IsNullOrWhiteSpace check).
    static std::vector<std::string> split(const std::string& value, const std::string& sep) {
        std::vector<std::string> tokens;
        std::size_t start = 0;
        while (true) {
            std::size_t pos = value.find(sep, start);
            if (pos == std::string::npos) {
                tokens.push_back(value.substr(start));
                break;
            }
            tokens.push_back(value.substr(start, pos - start));
            start = pos + sep.size();
        }
        return tokens;
    }

    // Invariant-culture double parse (C# double.TryParse, NumberStyles.Float | AllowThousands).
    // strtod is used with the token trimmed; the parse succeeds only if the whole trimmed token
    // is consumed. (Thousands separators are not reproduced -- an out-of-scope input for a
    // probability grid; such a token is simply treated as unparsable and skipped.) NOTE: strtod
    // is LC_NUMERIC locale-sensitive; the core never calls setlocale(), so it runs under the "C"
    // (invariant) locale -- matching the same convention used in models/json_lite.hpp.
    static bool try_parse(const std::string& token, double& out) {
        // Trim leading/trailing whitespace.
        std::size_t b = 0;
        std::size_t e = token.size();
        while (b < e && std::isspace(static_cast<unsigned char>(token[b]))) ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(token[e - 1]))) --e;
        if (b == e) return false;
        const std::string trimmed = token.substr(b, e - b);

        const char* c = trimmed.c_str();
        char* end = nullptr;
        const double v = std::strtod(c, &end);
        if (end != c + trimmed.size()) return false;  // trailing garbage -> not a valid double
        out = v;
        return true;
    }

    std::vector<double> values_;
};

}  // namespace bestfit::numerics::data
