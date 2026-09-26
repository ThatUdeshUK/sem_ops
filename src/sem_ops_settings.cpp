#include "sem_ops/settings.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"

namespace duckdb {

static constexpr const char *BATCH_SIZE_SETTING = "ml_batch_size";
static constexpr const char *USE_BATCH_SETTING = "llm_use_batch";
static constexpr const char *USE_CACHE_SETTING = "llm_use_cache";
static constexpr const char *MAX_TOKENS_SETTING = "llm_max_tokens";
static constexpr const char *N_THREADS_SETTING = "llm_no_threads";
static constexpr const char *TIMEOUT_SETTING = "llm_timeout";
static constexpr const char *MODEL_SELECT_STRATEGY_SETTING = "model_select_strategy";

static Value GetSetting(ClientContext &context, const char *name) {
	Value result;
	if (!context.TryGetCurrentSetting(name, result)) {
		throw InternalException("sem_ops setting \"%s\" is not registered", name);
	}
	return result;
}

//! Applies a model option, cast to the type of the setting it overrides
template <class T>
static void ApplyOption(const case_insensitive_map_t<Value> &options, const string &name, const LogicalType &type,
                        T &target) {
	auto entry = options.find(name);
	if (entry == options.end()) {
		return;
	}
	auto value = entry->second.DefaultTryCastAs(type);
	if (!value || value->IsNull()) {
		throw InvalidInputException("Invalid value \"%s\" for model option \"%s\": expected %s",
		                            entry->second.ToString(), name, type.ToString());
	}
	target = value->GetValue<T>();
}

SemOpsSettings SemOpsSettings::Get(ClientContext &context, const case_insensitive_map_t<Value> &options) {
	SemOpsSettings result;
	result.batch_size = UBigIntValue::Get(GetSetting(context, BATCH_SIZE_SETTING));
	result.use_batch = BooleanValue::Get(GetSetting(context, USE_BATCH_SETTING));
	result.use_cache = BooleanValue::Get(GetSetting(context, USE_CACHE_SETTING));
	result.max_tokens = UBigIntValue::Get(GetSetting(context, MAX_TOKENS_SETTING));
	result.n_threads = UBigIntValue::Get(GetSetting(context, N_THREADS_SETTING));
	result.timeout_seconds = UBigIntValue::Get(GetSetting(context, TIMEOUT_SETTING));

	ApplyOption(options, "batch_size", LogicalType::UBIGINT, result.batch_size);
	ApplyOption(options, "use_batch", LogicalType::BOOLEAN, result.use_batch);
	ApplyOption(options, "use_cache", LogicalType::BOOLEAN, result.use_cache);
	ApplyOption(options, "llm_max_tokens", LogicalType::UBIGINT, result.max_tokens);
	ApplyOption(options, "n_threads", LogicalType::UBIGINT, result.n_threads);
	ApplyOption(options, "req_per_min", LogicalType::UBIGINT, result.req_per_min);
	ApplyOption(options, "timeout", LogicalType::UBIGINT, result.timeout_seconds);

	result.batch_size = MaxValue<idx_t>(result.batch_size, 1);
	result.n_threads = MaxValue<idx_t>(result.n_threads, 1);
	return result;
}

string SemOpsSettings::GetModelSelectStrategy(ClientContext &context) {
	return StringValue::Get(GetSetting(context, MODEL_SELECT_STRATEGY_SETTING));
}

static void SetModelSelectStrategy(ClientContext &context, SetScope scope, Value &parameter) {
	auto strategy = StringUtil::Lower(parameter.ToString());
	if (strategy != "first" && strategy != "random") {
		throw InvalidInputException("Unsupported model_select_strategy \"%s\": expected 'first' or 'random'",
		                            parameter.ToString());
	}
	parameter = Value(strategy);
}

void SemOpsSettings::Register(DBConfig &config) {
	config.AddExtensionOption(BATCH_SIZE_SETTING, "Rows per batched LLM call", LogicalType::UBIGINT,
	                          Value::UBIGINT(16));
	config.AddExtensionOption(USE_BATCH_SETTING, "Batch multiple rows into a single LLM call", LogicalType::BOOLEAN,
	                          Value::BOOLEAN(true));
	config.AddExtensionOption(USE_CACHE_SETTING,
	                          "Send rows with identical LLM inputs once and share the result within a query",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true));
	config.AddExtensionOption(MAX_TOKENS_SETTING, "Maximum output tokens per LLM call", LogicalType::UBIGINT,
	                          Value::UBIGINT(512));
	config.AddExtensionOption(N_THREADS_SETTING, "Number of concurrent LLM calls", LogicalType::UBIGINT,
	                          Value::UBIGINT(16));
	config.AddExtensionOption(TIMEOUT_SETTING, "Timeout of a single LLM API request in seconds", LogicalType::UBIGINT,
	                          Value::UBIGINT(120));
	config.AddExtensionOption(MODEL_SELECT_STRATEGY_SETTING,
	                          "How an LLM call without a model name picks a model: 'first' or 'random'",
	                          LogicalType::VARCHAR, Value("first"), SetModelSelectStrategy);
}

} // namespace duckdb
