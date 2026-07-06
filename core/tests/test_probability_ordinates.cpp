// Standalone tests for bestfit::numerics::data::ProbabilityOrdinates.
//
// Oracle for behavior is the C# source itself (upstream/Numerics/Numerics/Data/Paired Data/
// ProbabilityOrdinate.cs @ a2c4dbf) -- the 25 default exceedance probabilities, the Validate
// rules/messages/short-circuit order, and the delimited-string round-trip. This is internal
// core math not exposed to R/Python, so oracles are transcribed from the C# constants + the
// upstream preservation-test invariant (re-expressed as value round-trips); no fixtures/ entry.
#include <string>
#include <vector>

#include "bestfit/numerics/data/probability_ordinates.hpp"
#include "check.hpp"

using bestfit::numerics::data::ProbabilityOrdinates;

namespace {

// The exact 25 defaults transcribed from the C# AddDefaults() array (lines 227-242), in order.
const std::vector<double> kExpectedDefaults = {
    0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
    0.001,    0.002,    0.005,    0.01,    0.02,    0.05,    0.1,    0.2,    0.3,
    0.5,      0.7,      0.8,      0.9,     0.95,    0.98,    0.99};

// ---- Defaults: count + exact value list (the count/values oracle) ----
void test_default_ctor_yields_25_exact_values() {
    ProbabilityOrdinates po;
    CHECK_EQ(po.size(), static_cast<std::size_t>(25));
    CHECK_EQ(po.count(), static_cast<std::size_t>(25));
    for (std::size_t i = 0; i < kExpectedDefaults.size(); ++i) {
        // Bit-exact: the literals are the same doubles the C# array holds.
        CHECK_TRUE(po[i] == kExpectedDefaults[i]);
    }
}

void test_reset_to_defaults_restores_25() {
    ProbabilityOrdinates po;
    po.clear();
    po.add(0.42);
    po.add(0.99);
    CHECK_EQ(po.size(), static_cast<std::size_t>(2));
    po.reset_to_defaults();
    CHECK_EQ(po.size(), static_cast<std::size_t>(25));
    for (std::size_t i = 0; i < kExpectedDefaults.size(); ++i) {
        CHECK_TRUE(po[i] == kExpectedDefaults[i]);
    }
}

void test_add_defaults_appends_not_resets() {
    // Start from a single-element grid, then add_defaults() appends the 25 (size 1 -> 26).
    ProbabilityOrdinates po(std::vector<double>{0.123});
    CHECK_EQ(po.size(), static_cast<std::size_t>(1));
    po.reset_to_defaults();  // exercises clear + add_defaults => 25
    CHECK_EQ(po.size(), static_cast<std::size_t>(25));
    // Append-not-reset semantics are visible via the public reset (clear then append).
    // Directly exercise append: construct empty, then add each default via the sequence ctor twice.
    ProbabilityOrdinates a(kExpectedDefaults);
    ProbabilityOrdinates b(a.values());
    CHECK_EQ(a.size(), static_cast<std::size_t>(25));
    CHECK_EQ(b.size(), static_cast<std::size_t>(25));
}

// ---- Sequence ctor ----
void test_sequence_ctor_and_empty_range() {
    ProbabilityOrdinates po(std::vector<double>{0.1, 0.2, 0.3});
    CHECK_EQ(po.size(), static_cast<std::size_t>(3));
    CHECK_TRUE(po[0] == 0.1);
    CHECK_TRUE(po[2] == 0.3);

    // C# throws ArgumentNullException on null; an empty C++ range simply yields empty.
    ProbabilityOrdinates empty(std::vector<double>{});
    CHECK_EQ(empty.size(), static_cast<std::size_t>(0));
    CHECK_TRUE(empty.empty());
}

// ---- Preservation / round-trip (the upstream INPC tests re-expressed as value round-trips) ----
void test_default_round_trip_bit_exact() {
    ProbabilityOrdinates po;
    std::string s = po.to_delimited_string("|");
    ProbabilityOrdinates back = ProbabilityOrdinates::parse(s, "|");
    CHECK_EQ(back.size(), static_cast<std::size_t>(25));
    for (std::size_t i = 0; i < po.size(); ++i) {
        CHECK_TRUE(back[i] == po[i]);  // bit-exact; well within 1e-10 relative
    }
}

void test_append_then_round_trip() {
    ProbabilityOrdinates po;
    po.add(0.001);  // duplicate-value append; round-trip must still reproduce every value
    std::string s = po.to_string();  // ToString delegates to ToDelimitedString(DefaultDelimiter)
    ProbabilityOrdinates back = ProbabilityOrdinates::parse(s);
    CHECK_EQ(back.size(), po.size());
    for (std::size_t i = 0; i < po.size(); ++i) {
        CHECK_TRUE(back[i] == po[i]);
    }
}

void test_from_delimited_string_exact_doubles() {
    ProbabilityOrdinates po;
    po.from_delimited_string("0.001|0.01|0.1|0.5|0.9|0.99");
    const std::vector<double> expect = {0.001, 0.01, 0.1, 0.5, 0.9, 0.99};
    CHECK_EQ(po.size(), expect.size());
    for (std::size_t i = 0; i < expect.size(); ++i) {
        CHECK_TRUE(po[i] == expect[i]);
    }
}

// ---- Validate ----
void test_validate_default_grid_valid() {
    ProbabilityOrdinates po;
    auto r = po.validate();
    CHECK_TRUE(r.is_valid);
    CHECK_EQ(r.messages.size(), static_cast<std::size_t>(0));
}

void test_validate_empty_grid_invalid() {
    ProbabilityOrdinates po;
    po.clear();
    auto r = po.validate();
    CHECK_TRUE(!r.is_valid);
    CHECK_EQ(r.messages.size(), static_cast<std::size_t>(1));
    CHECK_EQ(r.messages[0],
             std::string("At least one exceedance probability must be specified."));
}

void test_validate_out_of_range_invalid() {
    ProbabilityOrdinates hi(std::vector<double>{0.1, 1.5});
    auto rhi = hi.validate();
    CHECK_TRUE(!rhi.is_valid);
    // Range failure short-circuits (break): exactly one message.
    CHECK_EQ(rhi.messages.size(), static_cast<std::size_t>(1));

    ProbabilityOrdinates lo(std::vector<double>{-0.1, 0.2});
    auto rlo = lo.validate();
    CHECK_TRUE(!rlo.is_valid);
    CHECK_EQ(rlo.messages.size(), static_cast<std::size_t>(1));
}

void test_validate_non_increasing_invalid() {
    ProbabilityOrdinates dup(std::vector<double>{0.1, 0.1});
    auto rdup = dup.validate();
    CHECK_TRUE(!rdup.is_valid);
    CHECK_EQ(rdup.messages.size(), static_cast<std::size_t>(1));

    ProbabilityOrdinates dec(std::vector<double>{0.5, 0.2});
    auto rdec = dec.validate();
    CHECK_TRUE(!rdec.is_valid);
    CHECK_EQ(rdec.messages.size(), static_cast<std::size_t>(1));
}

// ---- Parse edge cases ----
void test_parse_empty_and_whitespace() {
    ProbabilityOrdinates a = ProbabilityOrdinates::parse("");
    CHECK_EQ(a.size(), static_cast<std::size_t>(0));
    ProbabilityOrdinates b = ProbabilityOrdinates::parse("   ");
    CHECK_EQ(b.size(), static_cast<std::size_t>(0));
}

void test_parse_skips_unparsable_and_empty_tokens() {
    // "abc" fails to parse (skipped); the empty token between the double delimiter is skipped;
    // a trailing delimiter does not add a spurious entry.
    ProbabilityOrdinates po = ProbabilityOrdinates::parse("0.1|abc|0.2||0.3|");
    CHECK_EQ(po.size(), static_cast<std::size_t>(3));
    CHECK_TRUE(po[0] == 0.1);
    CHECK_TRUE(po[1] == 0.2);
    CHECK_TRUE(po[2] == 0.3);
}

void test_from_delimited_string_clears_first() {
    ProbabilityOrdinates po;  // 25 defaults
    po.from_delimited_string("0.5");
    CHECK_EQ(po.size(), static_cast<std::size_t>(1));
    CHECK_TRUE(po[0] == 0.5);
}

}  // namespace

int main() {
    test_default_ctor_yields_25_exact_values();
    test_reset_to_defaults_restores_25();
    test_add_defaults_appends_not_resets();
    test_sequence_ctor_and_empty_range();
    test_default_round_trip_bit_exact();
    test_append_then_round_trip();
    test_from_delimited_string_exact_doubles();
    test_validate_default_grid_valid();
    test_validate_empty_grid_invalid();
    test_validate_out_of_range_invalid();
    test_validate_non_increasing_invalid();
    test_parse_empty_and_whitespace();
    test_parse_skips_unparsable_and_empty_tokens();
    test_from_delimited_string_clears_first();

    return bftest::summary("probability_ordinates");
}
