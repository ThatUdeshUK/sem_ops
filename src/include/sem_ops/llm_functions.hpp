//===----------------------------------------------------------------------===//
//                         SemOps
//
// sem_ops/llm_functions.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

//! The functions the semantic operator syntax is rewritten into. They can also be called directly.
//!   llm_predict(model, prompt, inputs...)     LLM model PROMPT '...' / LLM '...' (scalar)
//!   llm_reduce(model, prompt, text)            AGG LLM model (PROMPT '...') (applied to a string_agg of the inputs)
//!   llm_predict_table(TABLE, model, prompt)    LLM model (PROMPT '...' ON t) / PREDICT(model, PROMPT '...', t)
//!   llm_scan(model, prompt)                    FROM LLM model PROMPT '...'
//!   llm_embed(TABLE, model, column)            EMBED model(COLUMN c, t)
//!   sem_ops_models()                           lists the registered models
//! An empty model name selects a model according to `model_select_strategy`.
struct LlmFunctions {
	static constexpr const char *PREDICT = "llm_predict";
	static constexpr const char *REDUCE = "llm_reduce";
	static constexpr const char *PREDICT_TABLE = "llm_predict_table";
	static constexpr const char *SCAN = "llm_scan";
	static constexpr const char *EMBED = "llm_embed";
	static constexpr const char *MODELS = "sem_ops_models";

	static void Register(ExtensionLoader &loader);
};

} // namespace duckdb
