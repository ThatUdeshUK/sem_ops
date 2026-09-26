//===----------------------------------------------------------------------===//
//                         SemOps
//
// sem_ops/llm_predictor.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sem_ops/llm_client.hpp"
#include "sem_ops/settings.hpp"

#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/types/data_chunk.hpp"

namespace duckdb {

class ClientContext;

//! Everything a predictor needs to know about one LLM / embedding call site
struct PredictInfo {
	//! The model identifier sent to the API
	string model_path;
	string prompt;
	string base_api;
	//! The resolved bearer token (empty: fall back to OPENAI_API_KEY)
	string token;
	SemOpsSettings settings;

	//! The columns of the input chunk that feed the prompt, and the placeholder names they fill
	vector<idx_t> input_mask;
	vector<string> input_names;
	//! The output columns declared by the prompt
	vector<string> result_names;
	vector<LogicalType> result_types;
};

struct PredictStats {
	idx_t llm_calls = 0;
	idx_t input_tokens = 0;
	idx_t output_tokens = 0;
	idx_t total_tokens = 0;
};

//! The typed outputs of a prediction, one row per input row
using PredictionRows = vector<vector<Value>>;

//! The dimension of the embeddings produced by EMBED models
static constexpr idx_t EMBEDDING_DIMENSIONS = 384;

//! Calls an OpenAI-compatible API to evaluate prompts over rows (port of iPDb's LlmApiPredictor)
class LlmPredictor {
public:
	explicit LlmPredictor(PredictInfo info);

	//! Predicts the output columns for the first `rows` rows of `input`
	void PredictChunk(DataChunk &input, idx_t rows, PredictionRows &output);
	//! Treats input column 0 as the left and the other input columns as the right side of a join, and decides for all
	//! pairs at once which ones match. Writes the result into the BOOLEAN output column.
	void PredictJoin(DataChunk &input, idx_t rows, idx_t n_left_cols, PredictionRows &output);
	//! Produces one plain-text answer per input (used by aggregates); NULL if the call failed
	vector<Value> PredictString(const vector<string> &inputs);
	//! Generates rows from the prompt alone
	PredictionRows ScanChunk();
	//! Embeds each input; NULL inputs produce empty vectors
	vector<vector<float>> Embed(const vector<Value> &inputs);

	//! Logs the calls and tokens used since the last flush (visible in duckdb_logs when logging is enabled)
	void FlushStats(ClientContext &context);
	const PredictInfo &GetInfo() const {
		return info;
	}

private:
	//! The rows that share one LLM call: the prompt input and the output rows it fills
	struct PredictGroup {
		string input;
		vector<idx_t> rows;
	};
	//! The result of a single API call
	struct CallResult {
		bool success = false;
		string output;
	};
	//! The result of one batch of groups
	struct BatchResult {
		idx_t first_group = 0;
		idx_t group_count = 0;
		bool is_array = false;
		vector<CallResult> outputs;
	};

	string EmbedPrompt(DataChunk &input, idx_t row) const;
	Json SingleResponseFormat() const;
	Json ArrayResponseFormat(idx_t n_rows) const;
	string SystemMessage(bool is_array) const;

	Json ChatCompletion(const Json &request);
	BatchResult PredictBatch(const vector<PredictGroup> &groups, idx_t first_group, idx_t group_count);
	CallResult PredictOne(const string &input);
	CallResult PredictAgg(const string &input);
	vector<vector<float>> EmbedBatch(const vector<string> &inputs);

	void ApplySingleResult(const string &llm_out, const PredictGroup &group, PredictionRows &output);
	bool ApplyArrayResult(const string &llm_out, const vector<PredictGroup> &groups, idx_t first_group,
	                      idx_t group_count, PredictionRows &output, vector<idx_t> &failed_groups);
	void PopulateRow(const Json &values, idx_t row, PredictionRows &output) const;
	Value ConvertValue(const Json &value, const LogicalType &type) const;

	void AddUsage(const Json &completion);
	//! Throws for errors that retrying cannot fix. Client errors (4xx) only throw if `throw_client_errors` is set,
	//! since a batch call can fail with a client error (e.g. a too long prompt) that the per-row fallback avoids.
	static void CheckApiError(const Json &completion, const string &call_name, bool throw_client_errors);

private:
	PredictInfo info;
	LlmClient client;
	//! The JSON schema of one output row
	Json row_schema;

	mutex stats_lock;
	PredictStats stats;
	//! Outputs of previous chunks, by prompt input (exact tuple deduplication)
	unordered_map<string, string> cache;
};

//! Extracts the first JSON object or array embedded in the text
string ExtractJSON(const string &text);

} // namespace duckdb
