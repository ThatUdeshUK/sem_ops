#include "sem_ops/prompt.hpp"

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

#include <regex>

namespace duckdb {

// `{{column}}` or `{{table.column}}`
static constexpr const char *INPUT_REGEX = R"(\{\{\s*([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?\s*\}\})";
// `{name TYPE}`
static constexpr const char *OUTPUT_REGEX =
    R"(\{\s*([A-Za-z_][A-Za-z0-9_]*)\s+(INTEGER|VARCHAR|BOOLEAN|BOOL|DOUBLE)\s*\})";

LogicalType PromptTypeToLogicalType(const string &type) {
	auto upper = StringUtil::Upper(type);
	if (upper == "VARCHAR") {
		return LogicalType::VARCHAR;
	}
	if (upper == "INTEGER") {
		return LogicalType::INTEGER;
	}
	if (upper == "BOOLEAN" || upper == "BOOL") {
		return LogicalType::BOOLEAN;
	}
	if (upper == "DOUBLE") {
		return LogicalType::DOUBLE;
	}
	throw InvalidInputException("Unsupported prompt output type \"%s\"", type);
}

string LogicalTypeToJSONType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::VARCHAR:
		return "string";
	case LogicalTypeId::INTEGER:
		return "integer";
	case LogicalTypeId::DOUBLE:
		return "number";
	case LogicalTypeId::BOOLEAN:
		return "boolean";
	default:
		throw InternalException("Unsupported prompt output type %s", type.ToString());
	}
}

ParsedPrompt ParsedPrompt::Parse(const string &prompt) {
	ParsedPrompt result;

	case_insensitive_set_t seen_inputs;
	const std::regex input_re(INPUT_REGEX);
	for (auto it = std::sregex_iterator(prompt.begin(), prompt.end(), input_re); it != std::sregex_iterator(); ++it) {
		auto &match = *it;
		PromptInputColumn input;
		if (match[2].matched) {
			input.table = match[1].str();
			input.column = match[2].str();
		} else {
			input.column = match[1].str();
		}
		if (seen_inputs.insert(input.ToString()).second) {
			result.inputs.push_back(std::move(input));
		}
	}

	case_insensitive_set_t seen_outputs;
	const std::regex output_re(OUTPUT_REGEX, std::regex_constants::icase);
	for (auto it = std::sregex_iterator(prompt.begin(), prompt.end(), output_re); it != std::sregex_iterator(); ++it) {
		auto &match = *it;
		auto name = match[1].str();
		if (!seen_outputs.insert(name).second) {
			throw InvalidInputException("Prompt declares the output column \"%s\" more than once", name);
		}
		result.outputs.push_back({std::move(name), PromptTypeToLogicalType(match[2].str())});
	}
	return result;
}

} // namespace duckdb
