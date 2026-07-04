// ported from: (shared return type) RMC-BestFit/src/RMC.BestFit/Models/Support/
//               {IUnivariateModel,QuantilePrior}.cs @ fc28c0c
//
// The C# Validate() surface returns the anonymous tuple
// `(bool IsValid, List<string> ValidationMessages)` on several Models types (IUnivariateModel,
// QuantilePrior, IModel/ModelBase, ...). C++ has no named-tuple equivalent, so this port gives
// the tuple ONE named struct that every ported Validate() returns -- a small support header of
// its own rather than a redefinition per file (DRY; the earlier ModelBase slice deferred
// Validate() entirely, so this is the first place the type is needed).
#pragma once
#include <string>
#include <vector>

namespace bestfit::models {

// Mirrors the C# `(bool IsValid, List<string> ValidationMessages)` tuple. Aggregate: valid by
// default with no messages, matching the C# Validate() bodies that start from
// `bool isValid = true; var messageList = new List<string>();`.
struct ValidationResult {
    bool is_valid = true;
    std::vector<std::string> validation_messages;
};

}  // namespace bestfit::models
