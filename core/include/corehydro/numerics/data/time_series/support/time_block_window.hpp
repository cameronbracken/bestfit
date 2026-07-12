// ported from: Numerics/Data/Time Series/Support/TimeBlockWindow.cs @ a2c4dbf
//
// Enumeration of time block window options. Ported for PointProcessModel's TimeBlock
// property (M12); the TimeSeries container that consumes it upstream (CreateBlockSeries)
// is NOT ported (the seasonal DATA path deferral -- see point_process_model.hpp). The
// directory mirrors the C# folder layout (Numerics/Data/Time Series/Support/); the
// namespace stays flat like C#'s Numerics.Data.
#pragma once

namespace corehydro::numerics::data {

enum class TimeBlockWindow {
    // A full calendar year of 365 days from January 1st to December 31st.
    CalendarYear,

    // A full year of 12 months from October 1st to September 30th.
    WaterYear,

    // A custom, user specified year.
    CustomYear,

    // Quarter of a year, a 3 month period.
    Quarter,

    // A month of time.
    Month,
};

}  // namespace corehydro::numerics::data
