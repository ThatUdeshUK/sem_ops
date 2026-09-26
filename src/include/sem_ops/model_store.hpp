//===----------------------------------------------------------------------===//
//                         SemOps
//
// sem_ops/model_store.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/parser/sql_statement.hpp"

namespace duckdb {

class ClientContext;

enum class SemModelType : uint8_t { LLM, EMBED };

string SemModelTypeToString(SemModelType type);

//! A model registered through CREATE LLM MODEL / CREATE EMBED MODEL
struct SemModel {
	string name;
	SemModelType type = SemModelType::LLM;
	//! The model identifier sent to the API (e.g. 'gpt-4o-mini')
	string path;
	//! The OpenAI-compatible base URL (empty: fall back to OPENAI_API_BASE)
	string base_api;
	//! The name of the secret holding the bearer token (empty: fall back to OPENAI_API_KEY)
	string secret;
	case_insensitive_map_t<Value> options;
	bool temporary = false;
};

//! The parsed form of a CREATE MODEL statement
struct CreateModelInput {
	SemModel model;
	bool or_replace = false;
	bool if_not_exists = false;
};

//! Models are rows of a hidden table so they are transactional and persist with the database:
//! `__sem_ops_models` in the default schema for persistent models, `temp.__sem_ops_temp_models` for TEMP models.
class ModelStore {
public:
	static constexpr const char *TABLE_NAME = "__sem_ops_models";
	static constexpr const char *TEMP_TABLE_NAME = "__sem_ops_temp_models";

public:
	//! The statements that register a model
	static unique_ptr<SQLStatement> CreateModelStatement(const CreateModelInput &input);
	//! The statements that remove a model
	static unique_ptr<SQLStatement> DropModelStatement(const string &name, bool if_exists);

	//! All registered models; temporary models come first
	static vector<SemModel> ListModels(ClientContext &context);
	//! The model to use for a call: the named model, or one picked by `model_select_strategy` if the name is empty
	static SemModel GetModel(ClientContext &context, const string &name, SemModelType type);
	//! The bearer token of a secret
	static string GetSecretToken(ClientContext &context, const string &secret_name);
};

} // namespace duckdb
