#include "sem_ops/llm_predictor.hpp"

#include "sem_ops/prompt.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/main/client_context.hpp"

#include <future>
#include <regex>
#include <set>
#include <sstream>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Response parsing
//===--------------------------------------------------------------------===//
string ExtractJSON(const string &text) {
	auto start = text.find_first_of("{[");
	if (start == string::npos) {
		throw InvalidInputException("No JSON start found");
	}
	const char open = text[start];
	const char close = open == '{' ? '}' : ']';
	idx_t depth = 0;
	bool in_string = false;
	bool escaped = false;
	for (idx_t i = start; i < text.size(); i++) {
		auto c = text[i];
		if (in_string) {
			if (escaped) {
				escaped = false;
			} else if (c == '\\') {
				escaped = true;
			} else if (c == '"') {
				in_string = false;
			}
			continue;
		}
		if (c == '"') {
			in_string = true;
		} else if (c == open) {
			depth++;
		} else if (c == close) {
			depth--;
			if (depth == 0) {
				return text.substr(start, i - start + 1);
			}
		}
	}
	throw InvalidInputException("No matching JSON end found");
}

static string ExtractContent(const Json &completion) {
	if (!completion.contains("choices") || !completion["choices"].is_array()) {
		return string();
	}
	for (auto &choice : completion["choices"]) {
		if (choice.contains("message") && choice["message"].contains("content") &&
		    choice["message"]["content"].is_string()) {
			return choice["message"]["content"].get<string>();
		}
	}
	return string();
}

static Value ExtractLongestInteger(const string &input) {
	const std::regex re("\\d+");
	string longest;
	for (auto it = std::sregex_iterator(input.begin(), input.end(), re); it != std::sregex_iterator(); ++it) {
		if (it->str().size() > longest.size()) {
			longest = it->str();
		}
	}
	if (longest.empty()) {
		return Value(LogicalType::INTEGER);
	}
	return Value::INTEGER(std::stoi(longest));
}

static double ExtractDouble(const Json &value) {
	if (value.is_number()) {
		return value.get<double>();
	}
	if (value.is_string()) {
		return std::stod(value.get<string>());
	}
	throw InvalidInputException("Value is not a number or string");
}

Value LlmPredictor::ConvertValue(const Json &value, const LogicalType &type) const {
	if (value.is_null()) {
		return Value(type);
	}
	try {
		switch (type.id()) {
		case LogicalTypeId::VARCHAR:
			return Value(value.is_string() ? value.get<string>() : value.dump());
		case LogicalTypeId::INTEGER:
			if (value.is_string()) {
				return ExtractLongestInteger(value.get<string>());
			}
			if (value.is_number()) {
				return Value::INTEGER(value.get<int32_t>());
			}
			break;
		case LogicalTypeId::DOUBLE:
			return Value::DOUBLE(ExtractDouble(value));
		case LogicalTypeId::BOOLEAN:
			if (value.is_boolean()) {
				return Value::BOOLEAN(value.get<bool>());
			}
			if (value.is_string()) {
				auto str = StringUtil::Lower(value.get<string>());
				if (str == "true" || str == "false") {
					return Value::BOOLEAN(str == "true");
				}
			}
			break;
		default:
			break;
		}
	} catch (std::exception &) {
		// unparseable values become NULL
	}
	return Value(type);
}

void LlmPredictor::PopulateRow(const Json &values, idx_t row, PredictionRows &output) const {
	for (idx_t col = 0; col < info.result_names.size(); col++) {
		auto &name = info.result_names[col];
		if (!values.is_object() || !values.contains(name)) {
			output[row][col] = Value(info.result_types[col]);
			continue;
		}
		output[row][col] = ConvertValue(values[name], info.result_types[col]);
	}
}

//===--------------------------------------------------------------------===//
// Prompt construction
//===--------------------------------------------------------------------===//
LlmPredictor::LlmPredictor(PredictInfo info_p)
    : info(std::move(info_p)), client(info.base_api, info.token, info.settings.timeout_seconds) {
	Json properties = Json::object();
	Json required = Json::array();
	for (idx_t i = 0; i < info.result_names.size(); i++) {
		properties[info.result_names[i]] = {{"type", LogicalTypeToJSONType(info.result_types[i])}};
		required.push_back(info.result_names[i]);
	}
	row_schema = {
	    {"type", "object"}, {"properties", properties}, {"required", required}, {"additionalProperties", false}};
}

//! Renders the inputs [begin, end) of a row as "name = `value`, ..."
static string FormatInputs(const PredictInfo &info, DataChunk &input, idx_t row, idx_t begin, idx_t end) {
	std::stringstream ss;
	for (idx_t i = begin; i < end; i++) {
		auto value = input.GetValue(info.input_mask[i], row);
		ss << info.input_names[i] << " = `" << (value.IsNull() ? "NULL" : value.ToSQLString()) << "`, ";
	}
	return ss.str();
}

string LlmPredictor::EmbedPrompt(DataChunk &input, idx_t row) const {
	return FormatInputs(info, input, row, 0, info.input_mask.size());
}

string LlmPredictor::SystemMessage(bool is_array) const {
	const string suffix = ". Do not include any extra text, explanations, language specifier, produce {<key>: <single "
	                      "value>} for JSON objects. The JSON must be parsable by a standard parser.";
	if (is_array) {
		return "You are a helpful assistant. Always respond **only** with valid single JSON array where each object "
		       "is in format " +
		       row_schema.dump() + suffix;
	}
	return "You are a helpful assistant. Always respond **only** with valid JSON object (i.e. not an array) in "
	       "format " +
	       row_schema.dump() + suffix;
}

Json LlmPredictor::SingleResponseFormat() const {
	return {{"type", "json_schema"},
	        {"json_schema", {{"name", "json_response"}, {"strict", true}, {"schema", row_schema}}}};
}

Json LlmPredictor::ArrayResponseFormat(idx_t n_rows) const {
	Json array_schema = {{"type", "array"}, {"items", row_schema}};
	if (n_rows > 0) {
		array_schema["minItems"] = n_rows;
		array_schema["maxItems"] = n_rows;
	}
	Json schema = {{"type", "object"},
	               {"additionalProperties", false},
	               {"required", {"output_array"}},
	               {"properties", {{"output_array", array_schema}}}};
	return {{"type", "json_schema"},
	        {"json_schema", {{"name", "json_response"}, {"strict", true}, {"schema", schema}}}};
}

//===--------------------------------------------------------------------===//
// API calls
//===--------------------------------------------------------------------===//
void LlmPredictor::AddUsage(const Json &completion) {
	lock_guard<mutex> guard(stats_lock);
	stats.llm_calls++;
	if (!completion.contains("usage") || !completion["usage"].is_object()) {
		return;
	}
	auto &usage = completion["usage"];
	auto read = [&](const char *key) -> idx_t {
		return usage.contains(key) && usage[key].is_number_unsigned() ? usage[key].get<idx_t>() : 0;
	};
	stats.total_tokens += read("total_tokens");
	stats.input_tokens += read("prompt_tokens");
	stats.output_tokens += read("completion_tokens");
}

void LlmPredictor::FlushStats(ClientContext &context) {
	PredictStats flushed;
	{
		lock_guard<mutex> guard(stats_lock);
		flushed = stats;
		stats = PredictStats();
	}
	if (flushed.llm_calls == 0) {
		return;
	}
	DUCKDB_LOG_INFO(context, StringUtil::Format("sem_ops: model=%s calls=%llu input_tokens=%llu output_tokens=%llu "
	                                            "total_tokens=%llu",
	                                            info.model_path, flushed.llm_calls, flushed.input_tokens,
	                                            flushed.output_tokens, flushed.total_tokens));
}

void LlmPredictor::CheckApiError(const Json &completion, const string &call_name, bool throw_client_errors) {
	auto code =
	    completion.contains("code") && completion["code"].is_number_integer() ? completion["code"].get<int>() : -1;
	auto reason = completion.contains("error") && completion["error"].is_string()
	                  ? completion["error"].get<string>()
	                  : completion.value("error", Json()).dump();
	if (code == -1) {
		throw IOException("%s failed! Connection error: %s. Check that the configured API is valid.", call_name,
		                  reason);
	}
	if (code == 401) {
		throw IOException("%s failed with 401: %s. Check that the configured API key/secret is valid.", call_name,
		                  reason);
	}
	if (code == 404) {
		throw IOException("%s failed with 404: %s. Check that the configured API and model are valid.", call_name,
		                  reason);
	}
	// rate limits, timeouts and server errors are transient: the caller falls back or yields NULL
	bool transient = code == 408 || code == 429 || code >= 500;
	if (throw_client_errors && !transient) {
		throw IOException("%s failed with %d: %s", call_name, code, reason);
	}
}

Json LlmPredictor::ChatCompletion(const Json &request) {
	auto completion = client.Post("chat/completions", request);
	AddUsage(completion);
	return completion;
}

LlmPredictor::CallResult LlmPredictor::PredictOne(const string &input) {
	Json request;
	request["model"] = info.model_path;
	request["messages"] = {{{"role", "system"}, {"content", SystemMessage(false)}},
	                       {{"role", "user"}, {"content", info.prompt + ";\n" + input}}};
	request["response_format"] = SingleResponseFormat();

	auto completion = ChatCompletion(request);
	CallResult result;
	if (completion.contains("error")) {
		CheckApiError(completion, "LLM call", true);
		return result;
	}
	result.output = ExtractContent(completion);
	result.success = !result.output.empty();
	return result;
}

LlmPredictor::BatchResult LlmPredictor::PredictBatch(const vector<PredictGroup> &groups, idx_t first_group,
                                                     idx_t group_count) {
	BatchResult result;
	result.first_group = first_group;
	result.group_count = group_count;
	if (!info.settings.use_batch) {
		for (idx_t i = 0; i < group_count; i++) {
			result.outputs.push_back(PredictOne(groups[first_group + i].input));
		}
		return result;
	}

	result.is_array = true;
	std::stringstream prompt;
	prompt << info.prompt << "; \nRespond with a JSON object for each the following " << group_count << " inputs:\n[";
	for (idx_t i = 0; i < group_count; i++) {
		prompt << "{" << groups[first_group + i].input << "},\n";
	}
	prompt << "]";

	Json request;
	request["model"] = info.model_path;
	request["messages"] = {{{"role", "system"}, {"content", SystemMessage(true)}},
	                       {{"role", "user"}, {"content", prompt.str()}}};
	request["response_format"] = ArrayResponseFormat(group_count);

	auto completion = ChatCompletion(request);
	if (completion.contains("error")) {
		// falls back to row-wise calls
		CheckApiError(completion, "Batch call", false);
		return result;
	}
	CallResult output;
	output.output = ExtractContent(completion);
	output.success = !output.output.empty();
	result.outputs.push_back(std::move(output));
	return result;
}

LlmPredictor::CallResult LlmPredictor::PredictAgg(const string &input) {
	Json request;
	request["model"] = info.model_path;
	request["messages"] = {
	    {{"role", "system"},
	     {"content", "You are a helpful assistant. Always respond with a plain text. Do not include any explanations, "
	                 "given inputs or language specifiers."}},
	    {{"role", "user"},
	     {"content", info.prompt + "; Consider all of the following inputs and produce a single output: \n" + input}}};

	auto completion = ChatCompletion(request);
	CallResult result;
	if (completion.contains("error")) {
		CheckApiError(completion, "Aggregate LLM call", true);
		return result;
	}
	result.output = ExtractContent(completion);
	result.success = true;
	return result;
}

//===--------------------------------------------------------------------===//
// Result propagation
//===--------------------------------------------------------------------===//
void LlmPredictor::ApplySingleResult(const string &llm_out, const PredictGroup &group, PredictionRows &output) {
	if (info.settings.use_cache) {
		cache[group.input] = llm_out;
	}
	Json values;
	try {
		values = Json::parse(ExtractJSON(llm_out));
	} catch (std::exception &) {
		// unparseable outputs leave the rows NULL
		return;
	}
	for (auto row : group.rows) {
		PopulateRow(values, row, output);
	}
}

bool LlmPredictor::ApplyArrayResult(const string &llm_out, const vector<PredictGroup> &groups, idx_t first_group,
                                    idx_t group_count, PredictionRows &output, vector<idx_t> &failed_groups) {
	Json values;
	try {
		values = Json::parse(ExtractJSON(llm_out));
	} catch (std::exception &) {
		return false;
	}
	if (values.is_object() && values.contains("output_array")) {
		values = values["output_array"];
	}
	if (!values.is_array()) {
		return false;
	}
	idx_t parsed = MinValue<idx_t>(values.size(), group_count);
	for (idx_t i = 0; i < parsed; i++) {
		auto &group = groups[first_group + i];
		if (info.settings.use_cache) {
			cache[group.input] = values[i].dump();
		}
		for (auto row : group.rows) {
			PopulateRow(values[i], row, output);
		}
	}
	// groups the model did not answer fall back to row-wise calls
	for (idx_t i = parsed; i < group_count; i++) {
		failed_groups.push_back(first_group + i);
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Predictions
//===--------------------------------------------------------------------===//
static void InitializeOutput(const PredictInfo &info, idx_t rows, PredictionRows &output) {
	vector<Value> null_row;
	for (auto &type : info.result_types) {
		null_row.emplace_back(type);
	}
	output.assign(rows, null_row);
}

void LlmPredictor::PredictChunk(DataChunk &input, idx_t rows, PredictionRows &output) {
	InitializeOutput(info, rows, output);

	// group rows with identical prompt inputs, and answer rows seen in previous chunks from the cache
	vector<PredictGroup> groups;
	unordered_map<string, idx_t> group_index;
	for (idx_t row = 0; row < rows; row++) {
		auto prompt_input = EmbedPrompt(input, row);
		if (!info.settings.use_cache) {
			groups.push_back({std::move(prompt_input), {row}});
			continue;
		}
		auto cached = cache.find(prompt_input);
		if (cached != cache.end()) {
			try {
				PopulateRow(Json::parse(ExtractJSON(cached->second)), row, output);
			} catch (std::exception &) {
				// unparseable outputs leave the row NULL
			}
			continue;
		}
		auto entry = group_index.find(prompt_input);
		if (entry == group_index.end()) {
			group_index[prompt_input] = groups.size();
			groups.push_back({std::move(prompt_input), {row}});
		} else {
			groups[entry->second].rows.push_back(row);
		}
	}
	if (groups.empty()) {
		return;
	}

	const idx_t n_groups = groups.size();
	const idx_t n_threads = info.settings.n_threads;
	// spread the groups over all threads if there are fewer than n_threads full batches
	idx_t batch_size = info.settings.batch_size;
	if (n_threads * batch_size > n_groups && n_groups >= n_threads) {
		batch_size = (n_groups + n_threads - 1) / n_threads;
	}
	const idx_t rounds = (n_groups + batch_size - 1) / batch_size;

	vector<idx_t> failed_groups;
	for (idx_t batch = 0; batch < rounds; batch += n_threads) {
		vector<std::future<BatchResult>> futures;
		for (idx_t run = 0; run < n_threads && batch + run < rounds; run++) {
			auto first_group = (batch + run) * batch_size;
			auto group_count = MinValue<idx_t>(batch_size, n_groups - first_group);
			futures.push_back(std::async(std::launch::async, [this, &groups, first_group, group_count]() {
				return PredictBatch(groups, first_group, group_count);
			}));
		}
		for (auto &future : futures) {
			auto result = future.get();
			if (!result.is_array) {
				for (idx_t i = 0; i < result.outputs.size(); i++) {
					if (result.outputs[i].success) {
						ApplySingleResult(result.outputs[i].output, groups[result.first_group + i], output);
					}
				}
				continue;
			}
			bool applied = !result.outputs.empty() && result.outputs[0].success &&
			               ApplyArrayResult(result.outputs[0].output, groups, result.first_group, result.group_count,
			                                output, failed_groups);
			if (!applied) {
				for (idx_t i = 0; i < result.group_count; i++) {
					failed_groups.push_back(result.first_group + i);
				}
			}
		}
	}

	// retry the groups of failed batches one call at a time
	for (idx_t offset = 0; offset < failed_groups.size(); offset += n_threads) {
		vector<std::future<CallResult>> futures;
		vector<idx_t> future_groups;
		for (idx_t run = 0; run < n_threads && offset + run < failed_groups.size(); run++) {
			auto group_idx = failed_groups[offset + run];
			future_groups.push_back(group_idx);
			futures.push_back(std::async(std::launch::async,
			                             [this, &groups, group_idx]() { return PredictOne(groups[group_idx].input); }));
		}
		for (idx_t i = 0; i < futures.size(); i++) {
			auto result = futures[i].get();
			if (result.success) {
				ApplySingleResult(result.output, groups[future_groups[i]], output);
			}
		}
	}
}

void LlmPredictor::PredictJoin(DataChunk &input, idx_t rows, idx_t n_left_cols, PredictionRows &output) {
	D_ASSERT(n_left_cols <= info.input_mask.size());
	InitializeOutput(info, rows, output);

	// deduplicate the left and right sides, and record each row's (left, right) pair
	unordered_map<string, idx_t> left_index;
	unordered_map<string, idx_t> right_index;
	vector<string> left_unique;
	vector<string> right_unique;
	vector<pair<idx_t, idx_t>> row_pairs;
	for (idx_t row = 0; row < rows; row++) {
		auto left = FormatInputs(info, input, row, 0, n_left_cols);
		auto right = FormatInputs(info, input, row, n_left_cols, info.input_mask.size());
		if (left_index.find(left) == left_index.end()) {
			left_index[left] = left_unique.size();
			left_unique.push_back(left);
		}
		if (right_index.find(right) == right_index.end()) {
			right_index[right] = right_unique.size();
			right_unique.push_back(right);
		}
		row_pairs.emplace_back(left_index[left], right_index[right]);
	}

	std::stringstream prompt;
	prompt << info.prompt << "\n\nLeft items:\n";
	for (idx_t i = 0; i < left_unique.size(); i++) {
		prompt << "[" << i << "] " << left_unique[i] << "\n";
	}
	prompt << "\nRight items:\n";
	for (idx_t i = 0; i < right_unique.size(); i++) {
		prompt << "[" << i << "] " << right_unique[i] << "\n";
	}
	prompt << "\nReturn only the (left_id, right_id) index pairs that match according to the prompt above.";

	Json pair_schema = {{"type", "object"},
	                    {"additionalProperties", false},
	                    {"required", {"left_id", "right_id"}},
	                    {"properties", {{"left_id", {{"type", "integer"}}}, {"right_id", {{"type", "integer"}}}}}};
	Json join_schema = {{"type", "object"},
	                    {"additionalProperties", false},
	                    {"required", {"matching_pairs"}},
	                    {"properties", {{"matching_pairs", {{"type", "array"}, {"items", pair_schema}}}}}};

	Json request;
	request["model"] = info.model_path;
	request["messages"] = {
	    {{"role", "system"},
	     {"content", "You are a data matching assistant. Given enumerated left items and right items, return only the "
	                 "index pairs that match according to the user's criteria. Respond with matching_pairs as an array "
	                 "of {\"left_id\": <int>, \"right_id\": <int>} objects. Omit non-matching pairs entirely."}},
	    {{"role", "user"}, {"content", prompt.str()}}};
	request["response_format"] = {
	    {"type", "json_schema"},
	    {"json_schema", {{"name", "join_response"}, {"strict", true}, {"schema", join_schema}}}};

	auto completion = ChatCompletion(request);
	if (completion.contains("error")) {
		// the match result stays unknown (NULL)
		CheckApiError(completion, "Join LLM call", true);
		return;
	}

	std::set<pair<idx_t, idx_t>> matched_pairs;
	try {
		auto response = Json::parse(ExtractJSON(ExtractContent(completion)));
		if (response.contains("matching_pairs") && response["matching_pairs"].is_array()) {
			for (auto &entry : response["matching_pairs"]) {
				auto left_id = entry.at("left_id").get<idx_t>();
				auto right_id = entry.at("right_id").get<idx_t>();
				if (left_id < left_unique.size() && right_id < right_unique.size()) {
					matched_pairs.emplace(left_id, right_id);
				}
			}
		}
	} catch (std::exception &) {
		// an unparseable response leaves the match result unknown (NULL)
		return;
	}

	for (idx_t row = 0; row < rows; row++) {
		bool is_match = matched_pairs.count(row_pairs[row]) > 0;
		for (idx_t col = 0; col < info.result_types.size(); col++) {
			if (info.result_types[col].id() == LogicalTypeId::BOOLEAN) {
				output[row][col] = Value::BOOLEAN(is_match);
			}
		}
	}
}

vector<Value> LlmPredictor::PredictString(const vector<string> &inputs) {
	vector<Value> result;
	const idx_t n_threads = info.settings.n_threads;
	for (idx_t offset = 0; offset < inputs.size(); offset += n_threads) {
		vector<std::future<CallResult>> futures;
		for (idx_t i = offset; i < inputs.size() && i < offset + n_threads; i++) {
			futures.push_back(std::async(std::launch::async, [this, &inputs, i]() { return PredictAgg(inputs[i]); }));
		}
		for (auto &future : futures) {
			auto call = future.get();
			result.push_back(call.success ? Value(call.output) : Value(LogicalType::VARCHAR));
		}
	}
	return result;
}

PredictionRows LlmPredictor::ScanChunk() {
	Json request;
	request["model"] = info.model_path;
	request["messages"] = {{{"role", "system"}, {"content", SystemMessage(true)}},
	                       {{"role", "user"}, {"content", info.prompt}}};
	request["response_format"] = ArrayResponseFormat(0);

	auto completion = ChatCompletion(request);
	if (completion.contains("error")) {
		CheckApiError(completion, "LLM scan call", true);
		throw IOException("LLM scan call failed: %s", completion["error"].dump());
	}
	auto content = ExtractContent(completion);
	Json values;
	try {
		values = Json::parse(ExtractJSON(content));
	} catch (std::exception &ex) {
		throw InvalidInputException("LLM scan call returned invalid JSON (%s): %s", ex.what(), content);
	}
	if (values.is_object() && values.contains("output_array")) {
		values = values["output_array"];
	}
	if (!values.is_array()) {
		throw InvalidInputException("LLM scan call did not return an array: %s", content);
	}
	PredictionRows output;
	InitializeOutput(info, values.size(), output);
	for (idx_t row = 0; row < values.size(); row++) {
		PopulateRow(values[row], row, output);
	}
	return output;
}

//===--------------------------------------------------------------------===//
// Embeddings
//===--------------------------------------------------------------------===//
vector<vector<float>> LlmPredictor::EmbedBatch(const vector<string> &inputs) {
	Json request;
	request["model"] = info.model_path;
	request["input"] = inputs;
	request["dimensions"] = EMBEDDING_DIMENSIONS;

	auto response = client.Post("embeddings", request);
	AddUsage(response);
	if (response.contains("error")) {
		CheckApiError(response, "Embedding call", true);
		throw IOException("Embedding call failed: %s", response["error"].dump());
	}
	if (!response.contains("data") || !response["data"].is_array() || response["data"].size() != inputs.size()) {
		throw InvalidInputException("Embedding call returned an unexpected response");
	}
	vector<vector<float>> result(inputs.size());
	for (auto &entry : response["data"]) {
		auto index = entry.value("index", idx_t(0));
		if (index >= inputs.size() || !entry.contains("embedding") || !entry["embedding"].is_array()) {
			throw InvalidInputException("Embedding call returned an unexpected response");
		}
		result[index] = entry["embedding"].get<vector<float>>();
		if (result[index].size() != EMBEDDING_DIMENSIONS) {
			throw InvalidInputException("Embedding dimension mismatch: expected %llu, got %llu", EMBEDDING_DIMENSIONS,
			                            result[index].size());
		}
	}
	return result;
}

vector<vector<float>> LlmPredictor::Embed(const vector<Value> &inputs) {
	vector<vector<float>> result(inputs.size());
	vector<idx_t> rows;
	vector<string> texts;
	for (idx_t row = 0; row < inputs.size(); row++) {
		if (!inputs[row].IsNull()) {
			rows.push_back(row);
			texts.push_back(inputs[row].ToString());
		}
	}

	const idx_t batch_size = info.settings.batch_size;
	const idx_t n_threads = info.settings.n_threads;
	for (idx_t offset = 0; offset < texts.size(); offset += batch_size * n_threads) {
		vector<std::future<vector<vector<float>>>> futures;
		vector<idx_t> batch_offsets;
		for (idx_t start = offset; start < texts.size() && start < offset + batch_size * n_threads;
		     start += batch_size) {
			auto end = MinValue<idx_t>(start + batch_size, texts.size());
			batch_offsets.push_back(start);
			futures.push_back(std::async(std::launch::async, [this, &texts, start, end]() {
				return EmbedBatch(vector<string>(texts.begin() + static_cast<int64_t>(start),
				                                 texts.begin() + static_cast<int64_t>(end)));
			}));
		}
		for (idx_t i = 0; i < futures.size(); i++) {
			auto embeddings = futures[i].get();
			for (idx_t j = 0; j < embeddings.size(); j++) {
				result[rows[batch_offsets[i] + j]] = std::move(embeddings[j]);
			}
		}
	}
	return result;
}

} // namespace duckdb
