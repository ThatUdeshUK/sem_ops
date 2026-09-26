//===----------------------------------------------------------------------===//
//                         SemOps
//
// sem_ops/prompt.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

//! An output placeholder `{name TYPE}` declared in a prompt
struct PromptOutputColumn {
	string name;
	LogicalType type;
};

//! An input placeholder `{{column}}` or `{{table.column}}` referenced by a prompt
struct PromptInputColumn {
	//! The referenced table (empty if unqualified)
	string table;
	string column;

	string ToString() const {
		return table.empty() ? column : table + "." + column;
	}
};

//! The placeholders of a prompt, in order of first appearance
struct ParsedPrompt {
	vector<PromptInputColumn> inputs;
	vector<PromptOutputColumn> outputs;

	static ParsedPrompt Parse(const string &prompt);
};

//! Maps the type keyword of an output placeholder to its logical type
LogicalType PromptTypeToLogicalType(const string &type);
//! The JSON schema type name of a supported output type
string LogicalTypeToJSONType(const LogicalType &type);

} // namespace duckdb
