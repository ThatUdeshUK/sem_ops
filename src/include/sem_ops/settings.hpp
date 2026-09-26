//===----------------------------------------------------------------------===//
//                         SemOps
//
// sem_ops/settings.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

class ClientContext;
struct DBConfig;

//! The session settings controlling LLM inference, optionally overridden per model through its OPTIONS
struct SemOpsSettings {
	//! Rows per batched LLM call (`ml_batch_size`, option `batch_size`)
	idx_t batch_size = 16;
	//! Whether rows are batched into a single LLM call (`llm_use_batch`, option `use_batch`)
	bool use_batch = true;
	//! Whether rows with identical inputs share one LLM call (`llm_use_cache`, option `use_cache`)
	bool use_cache = true;
	//! Maximum output tokens per LLM call (`llm_max_tokens`, option `llm_max_tokens`)
	idx_t max_tokens = 512;
	//! Number of concurrent LLM calls (`llm_no_threads`, option `n_threads`)
	idx_t n_threads = 16;
	//! Requests per minute budget (option `req_per_min`)
	idx_t req_per_min = 500;
	//! Timeout of a single API request in seconds (`llm_timeout`, option `timeout`)
	idx_t timeout_seconds = 120;

	//! Reads the session settings, then applies the model OPTIONS on top
	static SemOpsSettings Get(ClientContext &context, const case_insensitive_map_t<Value> &options);
	//! Registers the settings with the database
	static void Register(DBConfig &config);
	//! The `model_select_strategy` setting
	static string GetModelSelectStrategy(ClientContext &context);
};

} // namespace duckdb
