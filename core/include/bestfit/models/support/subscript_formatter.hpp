// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/SubscriptFormatter.cs @ fc28c0c
//
// ToSubscript(int): maps the ASCII digits of the invariant-culture integer string to their
// Unicode subscript counterparts U+2080..U+2089 (UTF-8 escapes); any non-digit character
// (e.g. the '-' of a negative value) passes through unchanged, as in the C#.
//
// History: M7 ported this inline as `general_linear_detail::to_subscript` (localized in
// general_linear_function.hpp because it was the only caller). M10 adds a second caller
// (MixtureModel's "Weight (w<sub>)" parameter names), so the helper is hoisted here to its
// proper Models/Support home; general_linear_function.hpp now forwards to this.
#pragma once
#include <string>

namespace bestfit::models {

inline std::string to_subscript(int value) {
    static const char* const kSubscripts[10] = {
        "\xE2\x82\x80", "\xE2\x82\x81", "\xE2\x82\x82", "\xE2\x82\x83", "\xE2\x82\x84",
        "\xE2\x82\x85", "\xE2\x82\x86", "\xE2\x82\x87", "\xE2\x82\x88", "\xE2\x82\x89"};
    const std::string s = std::to_string(value);
    std::string result;
    result.reserve(s.size() * 3);
    for (const char c : s) {
        if (c >= '0' && c <= '9') {
            result += kSubscripts[c - '0'];
        } else {
            result += c;
        }
    }
    return result;
}

}  // namespace bestfit::models
