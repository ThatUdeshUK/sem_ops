#include "sem_ops/llm_functions.hpp"

#include "sem_ops/llm_predictor.hpp"
#include "sem_ops/model_store.hpp"
#include "sem_ops/prompt.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/vector_writer.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------------===//
static PredictInfo MakePredictInfo(ClientContext &context, const string &model_name, SemModelType type,
                                   const string &prompt) {
	auto model = ModelStore::GetModel(context, model_name, type);
	PredictInfo info;
	info.model_path = model.path;
	info.prompt = prompt;
	info.base_api = model.base_api;
	if (!model.secret.empty()) {
		info.token = ModelStore::GetSecretToken(context, model.secret);
	}
	info.settings = SemOpsSettings::Get(context, model.options);
	return info;
}

static string GetModelName(const Value &value) {
	return value.IsNull() ? string() : StringValue::Get(value);
}

static string GetPrompt(const Value &value, const char *function_name) {
	if (value.IsNull()) {
		throw BinderException("%s: the prompt cannot be NULL", function_name);
	}
	return StringValue::Get(value);
}

static bool PredictInfoEquals(const PredictInfo &a, const PredictInfo &b) {
	return a.model_path == b.model_path && a.prompt == b.prompt && a.base_api == b.base_api &&
	       a.input_mask == b.input_mask && a.input_names == b.input_names && a.result_names == b.result_names &&
	       a.result_types == b.result_types;
}

//! Writes column `col` of rows [offset, offset + count) into a flat vector
static void WriteColumn(Vector &result, const PredictionRows &rows, idx_t col, idx_t offset, idx_t count) {
	auto &type = result.GetType();
	switch (type.id()) {
	case LogicalTypeId::VARCHAR: {
		auto writer = FlatVector::Writer<string_t>(result, count, 0);
		for (idx_t i = 0; i < count; i++) {
			auto &value = rows[offset + i][col];
			if (value.IsNull()) {
				writer.WriteNull();
			} else {
				writer.WriteValue(string_t(StringValue::Get(value)));
			}
		}
		break;
	}
	case LogicalTypeId::INTEGER: {
		auto writer = FlatVector::Writer<int32_t>(result, count, 0);
		for (idx_t i = 0; i < count; i++) {
			auto &value = rows[offset + i][col];
			if (value.IsNull()) {
				writer.WriteNull();
			} else {
				writer.WriteValue(IntegerValue::Get(value));
			}
		}
		break;
	}
	case LogicalTypeId::DOUBLE: {
		auto writer = FlatVector::Writer<double>(result, count, 0);
		for (idx_t i = 0; i < count; i++) {
			auto &value = rows[offset + i][col];
			if (value.IsNull()) {
				writer.WriteNull();
			} else {
				writer.WriteValue(DoubleValue::Get(value));
			}
		}
		break;
	}
	case LogicalTypeId::BOOLEAN: {
		auto writer = FlatVector::Writer<bool>(result, count, 0);
		for (idx_t i = 0; i < count; i++) {
			auto &value = rows[offset + i][col];
			if (value.IsNull()) {
				writer.WriteNull();
			} else {
				writer.WriteValue(BooleanValue::Get(value));
			}
		}
		break;
	}
	default:
		throw InternalException("Unsupported LLM output type %s", type.ToString());
	}
}

//! Maps the input placeholders of a prompt to columns of the input table. Without placeholders all columns are used.
static void BindTableInputs(const ParsedPrompt &prompt, const vector<Identifier> &names, PredictInfo &info,
                            const char *function_name) {
	if (prompt.inputs.empty()) {
		for (idx_t i = 0; i < names.size(); i++) {
			info.input_mask.push_back(i);
			info.input_names.push_back(names[i].GetIdentifierName());
		}
		return;
	}
	for (auto &input : prompt.inputs) {
		if (!input.table.empty()) {
			throw BinderException("%s: the prompt cannot reference qualified columns ({{%s}})", function_name,
			                      input.ToString());
		}
		optional_idx index;
		for (idx_t i = 0; i < names.size(); i++) {
			if (StringUtil::CIEquals(names[i].GetIdentifierName(), input.column)) {
				index = i;
				break;
			}
		}
		if (!index.IsValid()) {
			throw BinderException("%s: the input relation has no column \"%s\" referenced by the prompt", function_name,
			                      input.column);
		}
		info.input_mask.push_back(index.GetIndex());
		info.input_names.push_back(input.column);
	}
}

//===--------------------------------------------------------------------===//
// llm_predict(model, prompt, inputs...)
//===--------------------------------------------------------------------===//
struct LlmPredictBindData : public FunctionData {
	explicit LlmPredictBindData(PredictInfo info_p, bool is_join_p) : info(std::move(info_p)), is_join(is_join_p) {
	}

	PredictInfo info;
	//! Multiple inputs with a BOOLEAN output are evaluated as a semantic join over all input pairs of a chunk
	bool is_join;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<LlmPredictBindData>(info, is_join);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<LlmPredictBindData>();
		return is_join == other.is_join && PredictInfoEquals(info, other.info);
	}
};

struct LlmPredictLocalState : public FunctionLocalState {
	explicit LlmPredictLocalState(const LlmPredictBindData &bind_data_p)
	    : bind_data(bind_data_p), predictor(make_uniq<LlmPredictor>(bind_data_p.info)) {
	}

	const LlmPredictBindData &bind_data;
	unique_ptr<LlmPredictor> predictor;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		return make_uniq<LlmPredictLocalState>(bind_data->Cast<LlmPredictBindData>());
	}
};

static unique_ptr<FunctionData> LlmPredictBind(BindScalarFunctionInput &input) {
	auto &context = input.GetClientContext();
	auto &arguments = input.GetArguments();
	auto model_name = GetModelName(input.GetConstant(0));
	auto prompt = GetPrompt(input.GetConstant(1), LlmFunctions::PREDICT);

	auto parsed = ParsedPrompt::Parse(prompt);
	if (parsed.outputs.size() != 1) {
		throw BinderException("%s: the prompt must declare exactly one output column {name TYPE}, found %llu",
		                      LlmFunctions::PREDICT, parsed.outputs.size());
	}
	auto input_count = arguments.size() - 2;
	if (!parsed.inputs.empty() && parsed.inputs.size() != input_count) {
		throw BinderException("%s: the prompt references %llu input columns, but %llu inputs were passed",
		                      LlmFunctions::PREDICT, parsed.inputs.size(), input_count);
	}

	auto info = MakePredictInfo(context, model_name, SemModelType::LLM, prompt);
	for (idx_t i = 0; i < input_count; i++) {
		info.input_mask.push_back(i + 2);
		info.input_names.push_back(parsed.inputs.empty() ? "input" + to_string(i) : parsed.inputs[i].ToString());
	}
	info.result_names.push_back(parsed.outputs[0].name);
	info.result_types.push_back(parsed.outputs[0].type);

	bool is_join = input_count > 1 && info.result_types[0].id() == LogicalTypeId::BOOLEAN;
	input.GetBoundFunction().SetReturnType(info.result_types[0]);
	return make_uniq<LlmPredictBindData>(std::move(info), is_join);
}

static void LlmPredictFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<LlmPredictLocalState>();
	auto count = args.size();
	PredictionRows rows;
	if (lstate.bind_data.is_join) {
		lstate.predictor->PredictJoin(args, count, 1, rows);
	} else {
		lstate.predictor->PredictChunk(args, count, rows);
	}
	lstate.predictor->FlushStats(state.GetContext());
	WriteColumn(result, rows, 0, 0, count);
}

static ScalarFunction GetLlmPredictFunction() {
	ScalarFunction fun(LlmFunctions::PREDICT, {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::ANY,
	                   LlmPredictFunction, LlmPredictBind, nullptr, LlmPredictLocalState::Init, LogicalType::ANY,
	                   FunctionStability::VOLATILE, FunctionNullHandling::SPECIAL_HANDLING);
	fun.GetSignature().GetParameter(0).SetName("model");
	fun.GetSignature().GetParameter(1).SetName("prompt");
	fun.SetFallible();
	return fun;
}

//===--------------------------------------------------------------------===//
// llm_reduce(model, prompt, text)
//===--------------------------------------------------------------------===//
struct LlmReduceBindData : public FunctionData {
	explicit LlmReduceBindData(PredictInfo info_p) : info(std::move(info_p)) {
	}

	PredictInfo info;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<LlmReduceBindData>(info);
	}
	bool Equals(const FunctionData &other_p) const override {
		return PredictInfoEquals(info, other_p.Cast<LlmReduceBindData>().info);
	}
};

struct LlmReduceLocalState : public FunctionLocalState {
	explicit LlmReduceLocalState(const PredictInfo &info) : predictor(make_uniq<LlmPredictor>(info)) {
	}

	unique_ptr<LlmPredictor> predictor;

	static unique_ptr<FunctionLocalState> Init(ExpressionState &state, const BoundFunctionExpression &expr,
	                                           FunctionData *bind_data) {
		return make_uniq<LlmReduceLocalState>(bind_data->Cast<LlmReduceBindData>().info);
	}
};

static unique_ptr<FunctionData> LlmReduceBind(BindScalarFunctionInput &input) {
	auto &context = input.GetClientContext();
	auto model_name = GetModelName(input.GetConstant(0));
	auto prompt = GetPrompt(input.GetConstant(1), LlmFunctions::REDUCE);
	return make_uniq<LlmReduceBindData>(MakePredictInfo(context, model_name, SemModelType::LLM, prompt));
}

static void LlmReduceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &lstate = ExecuteFunctionState::GetFunctionState(state)->Cast<LlmReduceLocalState>();
	auto count = args.size();
	vector<idx_t> rows;
	vector<string> inputs;
	for (idx_t row = 0; row < count; row++) {
		auto value = args.GetValue(2, row);
		if (!value.IsNull()) {
			rows.push_back(row);
			inputs.push_back(StringValue::Get(value));
		}
	}
	auto outputs = lstate.predictor->PredictString(inputs);
	lstate.predictor->FlushStats(state.GetContext());

	PredictionRows predictions(count, vector<Value> {Value(LogicalType::VARCHAR)});
	for (idx_t i = 0; i < rows.size(); i++) {
		predictions[rows[i]][0] = outputs[i];
	}
	WriteColumn(result, predictions, 0, 0, count);
}

static ScalarFunction GetLlmReduceFunction() {
	ScalarFunction fun(LlmFunctions::REDUCE, {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                   LogicalType::VARCHAR, LlmReduceFunction, LlmReduceBind, nullptr, LlmReduceLocalState::Init,
	                   LogicalType(LogicalTypeId::INVALID), FunctionStability::VOLATILE,
	                   FunctionNullHandling::SPECIAL_HANDLING);
	fun.GetSignature().GetParameter(0).SetName("model");
	fun.GetSignature().GetParameter(1).SetName("prompt");
	fun.GetSignature().GetParameter(2).SetName("text");
	fun.SetFallible();
	return fun;
}

//===--------------------------------------------------------------------===//
// llm_predict_table(TABLE, model, prompt) / llm_embed(TABLE, model, column)
//===--------------------------------------------------------------------===//
struct LlmTableBindData : public TableFunctionData {
	explicit LlmTableBindData(PredictInfo info_p) : info(std::move(info_p)) {
	}

	PredictInfo info;
};

struct LlmTableLocalState : public LocalTableFunctionState {
	explicit LlmTableLocalState(const PredictInfo &info) : predictor(make_uniq<LlmPredictor>(info)) {
	}

	unique_ptr<LlmPredictor> predictor;
};

static unique_ptr<LocalTableFunctionState> LlmTableInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                             GlobalTableFunctionState *global_state) {
	return make_uniq<LlmTableLocalState>(input.bind_data->Cast<LlmTableBindData>().info);
}

static unique_ptr<FunctionData> LlmPredictTableBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<Identifier> &names) {
	auto model_name = GetModelName(input.inputs[1]);
	auto prompt = GetPrompt(input.inputs[2], LlmFunctions::PREDICT_TABLE);
	auto parsed = ParsedPrompt::Parse(prompt);
	if (parsed.outputs.empty()) {
		throw BinderException("%s: the prompt must declare at least one output column {name TYPE}",
		                      LlmFunctions::PREDICT_TABLE);
	}

	auto info = MakePredictInfo(context, model_name, SemModelType::LLM, prompt);
	BindTableInputs(parsed, input.input_table_names, info, LlmFunctions::PREDICT_TABLE);
	return_types = input.input_table_types;
	names = input.input_table_names;
	for (auto &output : parsed.outputs) {
		info.result_names.push_back(output.name);
		info.result_types.push_back(output.type);
		return_types.push_back(output.type);
		names.emplace_back(output.name);
	}
	return make_uniq<LlmTableBindData>(std::move(info));
}

static OperatorResultType LlmPredictTableFunction(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
                                                  DataChunk &output) {
	auto &state = data.local_state->Cast<LlmTableLocalState>();
	auto count = input.size();
	PredictionRows rows;
	state.predictor->PredictChunk(input, count, rows);
	state.predictor->FlushStats(context.client);

	auto input_columns = input.ColumnCount();
	for (idx_t col = 0; col < input_columns; col++) {
		output.data[col].Reference(input.data[col]);
	}
	for (idx_t col = input_columns; col < output.ColumnCount(); col++) {
		WriteColumn(output.data[col], rows, col - input_columns, 0, count);
	}
	output.CheckCardinality(count);
	return OperatorResultType::NEED_MORE_INPUT;
}

static unique_ptr<FunctionData> LlmEmbedBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<Identifier> &names) {
	auto model_name = GetModelName(input.inputs[1]);
	if (input.inputs[2].IsNull()) {
		throw BinderException("%s: the column cannot be NULL", LlmFunctions::EMBED);
	}
	auto column = StringValue::Get(input.inputs[2]);

	auto info = MakePredictInfo(context, model_name, SemModelType::EMBED, string());
	ParsedPrompt column_input;
	column_input.inputs.push_back({string(), column});
	BindTableInputs(column_input, input.input_table_names, info, LlmFunctions::EMBED);

	return_types = input.input_table_types;
	names = input.input_table_names;
	return_types.push_back(LogicalType::ARRAY(LogicalType::FLOAT, EMBEDDING_DIMENSIONS));
	names.emplace_back("vec");
	return make_uniq<LlmTableBindData>(std::move(info));
}

static OperatorResultType LlmEmbedFunction(ExecutionContext &context, TableFunctionInput &data, DataChunk &input,
                                           DataChunk &output) {
	auto &state = data.local_state->Cast<LlmTableLocalState>();
	auto &info = state.predictor->GetInfo();
	auto count = input.size();

	vector<Value> texts;
	for (idx_t row = 0; row < count; row++) {
		texts.push_back(input.GetValue(info.input_mask[0], row));
	}
	auto embeddings = state.predictor->Embed(texts);
	state.predictor->FlushStats(context.client);

	auto input_columns = input.ColumnCount();
	for (idx_t col = 0; col < input_columns; col++) {
		output.data[col].Reference(input.data[col]);
	}
	auto &vec = output.data[input_columns];
	FlatVector::SetSize(vec, count);
	auto child_data = FlatVector::GetDataMutable<float>(ArrayVector::GetChildMutable(vec));
	for (idx_t row = 0; row < count; row++) {
		if (embeddings[row].empty()) {
			FlatVector::SetNull(vec, row, true);
			continue;
		}
		memcpy(child_data + row * EMBEDDING_DIMENSIONS, embeddings[row].data(), EMBEDDING_DIMENSIONS * sizeof(float));
	}
	output.CheckCardinality(count);
	return OperatorResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// llm_scan(model, prompt)
//===--------------------------------------------------------------------===//
struct LlmScanGlobalState : public GlobalTableFunctionState {
	bool executed = false;
	PredictionRows rows;
	idx_t offset = 0;
};

static unique_ptr<FunctionData> LlmScanBind(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<Identifier> &names) {
	auto model_name = GetModelName(input.inputs[0]);
	auto prompt = GetPrompt(input.inputs[1], LlmFunctions::SCAN);
	auto parsed = ParsedPrompt::Parse(prompt);
	if (parsed.outputs.empty()) {
		throw BinderException("%s: the prompt must declare at least one output column {name TYPE}", LlmFunctions::SCAN);
	}
	if (!parsed.inputs.empty()) {
		throw BinderException("%s: the prompt cannot reference input columns ({{%s}}) without an input relation, use "
		                      "LLM model (PROMPT '...' ON table) instead",
		                      LlmFunctions::SCAN, parsed.inputs[0].ToString());
	}

	auto info = MakePredictInfo(context, model_name, SemModelType::LLM, prompt);
	for (auto &output : parsed.outputs) {
		info.result_names.push_back(output.name);
		info.result_types.push_back(output.type);
		return_types.push_back(output.type);
		names.emplace_back(output.name);
	}
	return make_uniq<LlmTableBindData>(std::move(info));
}

static unique_ptr<GlobalTableFunctionState> LlmScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<LlmScanGlobalState>();
}

static void LlmScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<LlmScanGlobalState>();
	if (!state.executed) {
		LlmPredictor predictor(data.bind_data->Cast<LlmTableBindData>().info);
		state.rows = predictor.ScanChunk();
		predictor.FlushStats(context);
		state.executed = true;
	}
	auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.rows.size() - state.offset);
	for (idx_t col = 0; col < output.ColumnCount(); col++) {
		WriteColumn(output.data[col], state.rows, col, state.offset, count);
	}
	state.offset += count;
	output.CheckCardinality(count);
}

//===--------------------------------------------------------------------===//
// sem_ops_models()
//===--------------------------------------------------------------------===//
struct ModelsGlobalState : public GlobalTableFunctionState {
	vector<SemModel> models;
	idx_t offset = 0;
};

static unique_ptr<FunctionData> ModelsBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<Identifier> &names) {
	names.emplace_back("name");
	return_types.push_back(LogicalType::VARCHAR);
	names.emplace_back("model_type");
	return_types.push_back(LogicalType::VARCHAR);
	names.emplace_back("model_path");
	return_types.push_back(LogicalType::VARCHAR);
	names.emplace_back("base_api");
	return_types.push_back(LogicalType::VARCHAR);
	names.emplace_back("secret");
	return_types.push_back(LogicalType::VARCHAR);
	names.emplace_back("options");
	return_types.push_back(LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR));
	names.emplace_back("temporary");
	return_types.push_back(LogicalType::BOOLEAN);
	return make_uniq<TableFunctionData>();
}

static unique_ptr<GlobalTableFunctionState> ModelsInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<ModelsGlobalState>();
	result->models = ModelStore::ListModels(context);
	return std::move(result);
}

static Value OptionalString(const string &value) {
	return value.empty() ? Value(LogicalType::VARCHAR) : Value(value);
}

static void ModelsFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<ModelsGlobalState>();
	idx_t count = 0;
	for (; state.offset < state.models.size() && count < STANDARD_VECTOR_SIZE; state.offset++, count++) {
		auto &model = state.models[state.offset];
		vector<Value> keys;
		vector<Value> values;
		for (auto &option : model.options) {
			keys.emplace_back(option.first);
			values.emplace_back(option.second.ToString());
		}
		output.data[0].Append(Value(model.name));
		output.data[1].Append(Value(SemModelTypeToString(model.type)));
		output.data[2].Append(Value(model.path));
		output.data[3].Append(OptionalString(model.base_api));
		output.data[4].Append(OptionalString(model.secret));
		output.data[5].Append(
		    Value::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR, std::move(keys), std::move(values)));
		output.data[6].Append(Value::BOOLEAN(model.temporary));
	}
	output.CheckCardinality(count);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
static InsertionOrderPreservingMap<string> LlmTableToString(TableFunctionToStringInput &input) {
	InsertionOrderPreservingMap<string> result;
	auto &info = input.bind_data->Cast<LlmTableBindData>().info;
	result["Model"] = info.model_path;
	if (!info.prompt.empty()) {
		result["Prompt"] = info.prompt;
	}
	return result;
}

void LlmFunctions::Register(ExtensionLoader &loader) {
	loader.RegisterFunction(GetLlmPredictFunction());
	loader.RegisterFunction(GetLlmReduceFunction());

	TableFunction predict_table(PREDICT_TABLE, {LogicalType::TABLE, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                            nullptr, LlmPredictTableBind, nullptr, LlmTableInitLocal);
	predict_table.in_out_function = LlmPredictTableFunction;
	predict_table.to_string = LlmTableToString;
	loader.RegisterFunction(predict_table);

	TableFunction embed(EMBED, {LogicalType::TABLE, LogicalType::VARCHAR, LogicalType::VARCHAR}, nullptr, LlmEmbedBind,
	                    nullptr, LlmTableInitLocal);
	embed.in_out_function = LlmEmbedFunction;
	embed.to_string = LlmTableToString;
	loader.RegisterFunction(embed);

	TableFunction scan(SCAN, {LogicalType::VARCHAR, LogicalType::VARCHAR}, LlmScanFunction, LlmScanBind,
	                   LlmScanInitGlobal);
	scan.to_string = LlmTableToString;
	loader.RegisterFunction(scan);

	TableFunction models(MODELS, {}, ModelsFunction, ModelsBind, ModelsInitGlobal);
	loader.RegisterFunction(models);
}

} // namespace duckdb
