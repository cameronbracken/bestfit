// Minimal dependency-free test harness for the bestfit C++ core (Phase 0).
// Will be superseded by the fixture-driven doctest runner; kept tiny on purpose.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace bftest {

inline int& failures() {
    static int n = 0;
    return n;
}
inline int& checks() {
    static int n = 0;
    return n;
}

inline void report_pass() { ++checks(); }

inline void report_fail(const char* file, int line, const std::string& msg) {
    ++checks();
    ++failures();
    std::printf("  FAIL %s:%d  %s\n", file, line, msg.c_str());
}

inline int summary(const char* suite) {
    if (failures() == 0) {
        std::printf("[PASS] %s  (%d checks)\n", suite, checks());
        return 0;
    }
    std::printf("[FAIL] %s  (%d/%d checks failed)\n", suite, failures(), checks());
    return 1;
}

}  // namespace bftest

#define CHECK_EQ(actual, expected)                                                       \
    do {                                                                                 \
        auto _a = (actual);                                                              \
        auto _e = (expected);                                                            \
        if (_a == _e) {                                                                  \
            ::bftest::report_pass();                                                     \
        } else {                                                                         \
            ::bftest::report_fail(__FILE__, __LINE__,                                    \
                                  std::string(#actual) + " != " + #expected);           \
        }                                                                                \
    } while (0)

// Absolute-tolerance floating comparison.
#define CHECK_NEAR(actual, expected, tol)                                                \
    do {                                                                                 \
        double _a = (actual);                                                            \
        double _e = (expected);                                                          \
        double _t = (tol);                                                               \
        if (std::fabs(_a - _e) <= _t) {                                                  \
            ::bftest::report_pass();                                                     \
        } else {                                                                         \
            char _b[256];                                                                \
            std::snprintf(_b, sizeof(_b), "%s: |%.17g - %.17g| = %.3g > %.3g",           \
                          #actual, _a, _e, std::fabs(_a - _e), _t);                      \
            ::bftest::report_fail(__FILE__, __LINE__, _b);                               \
        }                                                                                \
    } while (0)
