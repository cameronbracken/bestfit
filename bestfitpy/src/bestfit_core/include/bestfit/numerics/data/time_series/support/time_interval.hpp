// ported from: Numerics/Data/Time Series/Support/TimeInterval.cs @ a2c4dbf
//
// Enumeration of available time-series time intervals, in C# declaration order. The C#
// [Description] attributes ("1-Min", "5-Min", ...) are WPF/display metadata and are skipped
// (no display consumer in this port's scope). The directory mirrors the C# folder layout
// (Numerics/Data/Time Series/Support/); the namespace stays flat like C#'s Numerics.Data.
#pragma once

namespace bestfit::numerics::data {

enum class TimeInterval {
    OneMinute,
    FiveMinute,
    FifteenMinute,
    ThirtyMinute,
    OneHour,
    SixHour,
    TwelveHour,
    OneDay,
    SevenDay,
    OneMonth,
    OneQuarter,
    OneYear,
    Irregular,
};

}  // namespace bestfit::numerics::data
