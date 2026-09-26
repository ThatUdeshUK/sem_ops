#include "sem_ops/model_store.hpp"

#include "sem_ops/settings.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/random_engine.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/multi_statement.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

namespace duckdb {

string SemModelTypeToString(SemModelType type) {
	switch (type) {
	case SemModelType::LLM:
		return "LLM";
	case SemModelType::EMBED:
		return "EMBED";
	default:
		throw InternalException("Unknown SemModelType");
	}
}

static SemModelType SemModelTypeFromString(const string &type) {
	if (StringUtil::CIEquals(type, "LLM")) {
		return SemModelType::LLM;
	}
	if (StringUtil::CIEquals(type, "EMBED")) {
		return SemModelType::EMBED;
	}
	throw InvalidInputException("Unknown model type \"%s\" in the model table", type);
}

//===--------------------------------------------------------------------===//
// CREATE / DROP MODEL
//===--------------------------------------------------------------------===//
static string SQLString(const string &value) {
	return Value(value).ToSQLString();
}

static string SQLStringOrNull(const string &value) {
	return value.empty() ? "NULL" : SQLString(value);
}

static string QualifiedTableName(bool temporary) {
	return temporary ? string("temp.main.") + ModelStore::TEMP_TABLE_NAME : string(ModelStore::TABLE_NAME);
}

static string CreateTableSQL(bool temporary) {
	return StringUtil::Format("CREATE %sTABLE IF NOT EXISTS %s (name VARCHAR PRIMARY KEY, model_type VARCHAR NOT NULL, "
	                          "model_path VARCHAR NOT NULL, base_api VARCHAR, secret VARCHAR, "
	                          "options MAP(VARCHAR, VARCHAR));",
	                          temporary ? "TEMP " : "", QualifiedTableName(temporary));
}

static string ModelExistsSQL(bool temporary, const string &name) {
	return StringUtil::Format("EXISTS (SELECT 1 FROM %s WHERE lower(name) = lower(%s))", QualifiedTableName(temporary),
	                          SQLString(name));
}

static string ErrorSQL(const string &message) {
	return "error(" + SQLString(message) + ")";
}

static string OptionsSQL(const case_insensitive_map_t<Value> &options) {
	vector<string> entries;
	for (auto &entry : options) {
		entries.push_back(SQLString(entry.first) + ": " + SQLString(entry.second.ToString()));
	}
	return "MAP {" + StringUtil::Join(entries, ", ") + "}";
}

static unique_ptr<SQLStatement> ParseMultiStatement(const string &sql) {
	Parser parser;
	parser.ParseQuery(sql);
	auto result = make_uniq<MultiStatement>();
	for (auto &statement : parser.statements) {
		result->statements.push_back(std::move(statement));
	}
	return std::move(result);
}

unique_ptr<SQLStatement> ModelStore::CreateModelStatement(const CreateModelInput &input) {
	auto &model = input.model;
	auto table = QualifiedTableName(model.temporary);

	vector<string> guards;
	if (input.if_not_exists) {
		guards.push_back("NOT " + ModelExistsSQL(model.temporary, model.name));
	} else if (!input.or_replace) {
		guards.push_back(StringUtil::Format("CASE WHEN %s THEN %s ELSE true END",
		                                    ModelExistsSQL(model.temporary, model.name),
		                                    ErrorSQL("Model with name \"" + model.name + "\" already exists")));
	}
	if (!model.secret.empty()) {
		guards.push_back(StringUtil::Format(
		    "CASE WHEN EXISTS (SELECT 1 FROM duckdb_secrets() WHERE lower(name) = lower(%s)) THEN true ELSE %s END",
		    SQLString(model.secret),
		    ErrorSQL("Secret \"" + model.secret + "\" for the API of model \"" + model.name + "\" does not exist")));
	}
	if (guards.empty()) {
		guards.push_back("true");
	}

	string sql = CreateTableSQL(model.temporary);
	if (input.or_replace) {
		sql += StringUtil::Format("DELETE FROM %s WHERE lower(name) = lower(%s);", table, SQLString(model.name));
	}
	sql += StringUtil::Format("INSERT INTO %s SELECT %s, %s, %s, %s, %s, %s WHERE %s;", table, SQLString(model.name),
	                          SQLString(SemModelTypeToString(model.type)), SQLString(model.path),
	                          SQLStringOrNull(model.base_api), SQLStringOrNull(model.secret), OptionsSQL(model.options),
	                          StringUtil::Join(guards, " AND "));
	return ParseMultiStatement(sql);
}

unique_ptr<SQLStatement> ModelStore::DropModelStatement(const string &name, bool if_exists) {
	auto persistent_table = QualifiedTableName(false);
	auto temp_table = QualifiedTableName(true);

	string sql = CreateTableSQL(false) + CreateTableSQL(true);
	if (!if_exists) {
		// raise the error from a statement that is always evaluated exactly once, without inserting anything
		sql += StringUtil::Format(
		    "INSERT INTO %s SELECT NULL, NULL, NULL, NULL, NULL, NULL WHERE CASE WHEN %s OR %s THEN false ELSE %s END;",
		    temp_table, ModelExistsSQL(true, name), ModelExistsSQL(false, name),
		    ErrorSQL("Model with name \"" + name + "\" does not exist"));
	}
	// a temporary model shadows a persistent one of the same name, so only drop the persistent one if there is none
	sql += StringUtil::Format("DELETE FROM %s WHERE lower(name) = lower(%s) AND NOT %s;", persistent_table,
	                          SQLString(name), ModelExistsSQL(true, name));
	sql += StringUtil::Format("DELETE FROM %s WHERE lower(name) = lower(%s);", temp_table, SQLString(name));
	return ParseMultiStatement(sql);
}

//===--------------------------------------------------------------------===//
// Lookup
//===--------------------------------------------------------------------===//
static case_insensitive_map_t<Value> ReadOptions(const Value &options) {
	case_insensitive_map_t<Value> result;
	if (options.IsNull()) {
		return result;
	}
	for (auto &entry : MapValue::GetChildren(options)) {
		auto &key_value = StructValue::GetChildren(entry);
		result[key_value[0].ToString()] = key_value[1];
	}
	return result;
}

static void ScanModelTable(ClientContext &context, optional_ptr<TableCatalogEntry> table, bool temporary,
                           vector<SemModel> &result) {
	if (!table) {
		return;
	}
	if (!table->IsDuckTable()) {
		throw InvalidInputException("The model table \"%s\" is not a DuckDB table", table->name.GetIdentifierName());
	}
	auto &storage = table->Cast<DuckTableEntry>().GetStorage();
	auto &transaction = DuckTransaction::Get(context, table->ParentCatalog());

	vector<StorageIndex> column_ids;
	vector<LogicalType> types;
	auto table_types = storage.GetTypes();
	for (idx_t i = 0; i < table_types.size(); i++) {
		column_ids.emplace_back(i);
		types.push_back(table_types[i]);
	}
	if (types.size() != 6) {
		throw InvalidInputException("The model table \"%s\" has an unexpected layout", table->name.GetIdentifierName());
	}

	TableScanState state;
	storage.InitializeScan(context, transaction, state, column_ids);
	DataChunk chunk;
	chunk.Initialize(Allocator::Get(context), types);
	while (true) {
		chunk.Reset();
		storage.Scan(transaction, chunk, state);
		if (chunk.size() == 0) {
			break;
		}
		for (idx_t row = 0; row < chunk.size(); row++) {
			SemModel model;
			model.name = chunk.GetValue(0, row).ToString();
			model.type = SemModelTypeFromString(chunk.GetValue(1, row).ToString());
			model.path = chunk.GetValue(2, row).ToString();
			auto base_api = chunk.GetValue(3, row);
			model.base_api = base_api.IsNull() ? string() : base_api.ToString();
			auto secret = chunk.GetValue(4, row);
			model.secret = secret.IsNull() ? string() : secret.ToString();
			model.options = ReadOptions(chunk.GetValue(5, row));
			model.temporary = temporary;
			result.push_back(std::move(model));
		}
	}
}

vector<SemModel> ModelStore::ListModels(ClientContext &context) {
	vector<SemModel> result;
	auto temp_table = Catalog::GetEntry<TableCatalogEntry>(
	    context, QualifiedName(Identifier::TempCatalog(), Identifier::DefaultSchema(), TEMP_TABLE_NAME),
	    OnEntryNotFound::RETURN_NULL);
	ScanModelTable(context, temp_table, true, result);
	auto table = Catalog::GetEntry<TableCatalogEntry>(context, QualifiedName(TABLE_NAME), OnEntryNotFound::RETURN_NULL);
	ScanModelTable(context, table, false, result);
	return result;
}

SemModel ModelStore::GetModel(ClientContext &context, const string &name, SemModelType type) {
	vector<SemModel> candidates;
	for (auto &model : ListModels(context)) {
		if (!name.empty() && StringUtil::CIEquals(model.name, name)) {
			if (model.type != type) {
				throw BinderException("Model \"%s\" is an %s model, but an %s model is required here", model.name,
				                      SemModelTypeToString(model.type), SemModelTypeToString(type));
			}
			return model;
		}
		if (model.type == type) {
			candidates.push_back(std::move(model));
		}
	}
	if (!name.empty()) {
		throw BinderException("Model with name \"%s\" does not exist", name);
	}
	if (candidates.empty()) {
		throw BinderException("No %s models available for model selection", SemModelTypeToString(type));
	}
	if (SemOpsSettings::GetModelSelectStrategy(context) == "random") {
		RandomEngine random;
		return candidates[random.NextRandomInteger(0, NumericCast<uint32_t>(candidates.size()))];
	}
	return candidates[0];
}

string ModelStore::GetSecretToken(ClientContext &context, const string &secret_name) {
	auto &secret_manager = SecretManager::Get(context);
	auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
	auto secret_entry = secret_manager.GetSecretByName(transaction, secret_name);
	if (!secret_entry) {
		throw InvalidInputException("Secret \"%s\" for the API is not found", secret_name);
	}
	auto &kv_secret = secret_entry->secret->Cast<KeyValueSecret>();
	auto token = kv_secret.TryGetValue("bearer_token");
	if (token.IsNull()) {
		throw InvalidInputException("Secret \"%s\" has no bearer_token", secret_name);
	}
	return token.ToString();
}

} // namespace duckdb
