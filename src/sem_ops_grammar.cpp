#include "sem_ops/grammar.hpp"

#include "sem_ops/llm_functions.hpp"
#include "sem_ops/model_store.hpp"
#include "sem_ops/prompt.hpp"

#include "duckdb/common/sql_identifier.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/connection_manager.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/peg/compiled_grammar.hpp"
#include "duckdb/parser/peg/transformer/peg_transformer.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Parse result helpers
//===--------------------------------------------------------------------===//
static ListParseResult &AsList(ParseResult &parse_result) {
	return parse_result.Cast<ListParseResult>();
}

static optional_ptr<ParseResult> GetOptional(ListParseResult &list, idx_t index) {
	auto &optional = list.Child<OptionalParseResult>(index);
	return optional.HasResult() ? &optional.GetResult() : nullptr;
}

static string GetString(ParseResult &parse_result) {
	if (parse_result.type != ParseResultType::STRING) {
		auto &list = AsList(parse_result);
		return GetString(list.GetChild(0));
	}
	auto expression = parse_result.Cast<StringLiteralParseResult>().ToExpression();
	if (expression->GetExpressionClass() != ExpressionClass::CONSTANT) {
		throw ParserException("Expected a plain string literal");
	}
	return expression->Cast<ConstantExpression>().GetLiteral().ToValue().ToString();
}

static string GetIdentifier(PEGTransformer &transformer, ParseResult &parse_result) {
	return transformer.Transform<Identifier>(parse_result).GetIdentifierName();
}

static string GetOptionalIdentifier(PEGTransformer &transformer, ListParseResult &list, idx_t index) {
	auto result = GetOptional(list, index);
	return result ? GetIdentifier(transformer, *result) : string();
}

static unique_ptr<TransformProcess> MakeProcess(PEGTransformer &transformer, ParseResult &parse_result,
                                                transform_finalize_function_t finalize) {
	return make_uniq<FinalizeTransformProcess>(transformer, parse_result, std::move(finalize));
}

template <class T>
static unique_ptr<TransformResultValue> MakeResult(T value) {
	return make_uniq<TypedTransformResult<T>>(std::move(value));
}

//===--------------------------------------------------------------------===//
// CREATE / DROP MODEL
//===--------------------------------------------------------------------===//
static case_insensitive_map_t<Value> TransformModelOptions(PEGTransformer &transformer, ParseResult &parse_result) {
	case_insensitive_map_t<Value> options;
	auto expression = transformer.Transform<unique_ptr<ParsedExpression>>(AsList(parse_result).GetChild(1));
	auto &function = expression->Cast<FunctionExpression>();
	for (auto &argument : function.GetArguments()) {
		auto &value = argument.GetExpression();
		if (value.GetExpressionClass() != ExpressionClass::CONSTANT) {
			throw ParserException("Model OPTIONS must be constants, got \"%s\"", value.ToString());
		}
		options[argument.GetName().GetIdentifierName()] = value.Cast<ConstantExpression>().GetLiteral().ToValue();
	}
	return options;
}

static unique_ptr<TransformResultValue> TransformCreateModel(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list = AsList(parse_result);
	CreateModelInput input;
	auto &model = input.model;

	input.or_replace = GetOptional(list, 1) != nullptr;
	auto temporary = GetOptional(list, 2);
	if (temporary) {
		auto &persistence = AsList(*temporary).Child<ChoiceParseResult>(0).GetResult();
		model.temporary = persistence.name != "Persistent";
	}
	auto &kind = AsList(list.GetChild(3)).Child<ChoiceParseResult>(0).GetResult();
	model.type = kind.name == "SemLlmKind" ? SemModelType::LLM : SemModelType::EMBED;
	input.if_not_exists = GetOptional(list, 5) != nullptr;
	model.name = GetIdentifier(transformer, list.GetChild(6));
	model.path = GetString(list.GetChild(8));

	bool on_prompt = GetOptional(list, 9) != nullptr;
	auto api = GetOptional(list, 10);
	if (api) {
		model.base_api = GetString(AsList(*api).GetChild(1));
	}
	auto secret = GetOptional(list, 11);
	if (secret) {
		model.secret = GetIdentifier(transformer, AsList(*secret).GetChild(1));
	}
	auto options = GetOptional(list, 12);
	if (options) {
		model.options = TransformModelOptions(transformer, *options);
	}

	if (input.or_replace && input.if_not_exists) {
		throw ParserException("Cannot specify both OR REPLACE and IF NOT EXISTS for CREATE MODEL");
	}
	if (model.type == SemModelType::LLM && !on_prompt) {
		throw ParserException("CREATE LLM MODEL requires ON PROMPT");
	}
	if (model.type == SemModelType::EMBED && on_prompt) {
		throw ParserException("CREATE EMBED MODEL does not accept ON PROMPT");
	}
	if (StringUtil::Contains(StringUtil::Lower(model.path), ".gguf")) {
		throw ParserException("Local .gguf models are not supported, serve the model through an OpenAI-compatible API "
		                      "(e.g. the llama.cpp server) and set its URL with API '...'");
	}
	if (!model.base_api.empty() && !StringUtil::StartsWith(model.base_api, "http://") &&
	    !StringUtil::StartsWith(model.base_api, "https://")) {
		throw ParserException("The model API \"%s\" must be an http:// or https:// URL", model.base_api);
	}
	return MakeResult(ModelStore::CreateModelStatement(input));
}

static unique_ptr<TransformResultValue> TransformDropModel(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list = AsList(parse_result);
	auto if_exists = GetOptional(list, 2) != nullptr;
	auto name = GetIdentifier(transformer, list.GetChild(3));
	return MakeResult(ModelStore::DropModelStatement(name, if_exists));
}

//===--------------------------------------------------------------------===//
// Expressions
//===--------------------------------------------------------------------===//
static unique_ptr<ParsedExpression> MakeColumnRef(const PromptInputColumn &input) {
	if (input.table.empty()) {
		return make_uniq<ColumnRefExpression>(Identifier(input.column));
	}
	return make_uniq<ColumnRefExpression>(Identifier(input.column), Identifier(input.table));
}

//! LLM [model] PROMPT '...' -> llm_predict('model', '...', <input columns>)
static unique_ptr<ParsedExpression> MakePredictExpression(PEGTransformer &transformer, ParseResult &parse_result,
                                                          const string &model_name, const string &prompt) {
	auto parsed = ParsedPrompt::Parse(prompt);
	if (parsed.outputs.size() > 1) {
		throw ParserException("Scalar LLM expressions support only one output column, but the prompt \"%s\" declares "
		                      "%llu",
		                      prompt, parsed.outputs.size());
	}
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(ConstantExpression::String(model_name));
	children.push_back(ConstantExpression::String(prompt));
	for (auto &input : parsed.inputs) {
		children.push_back(MakeColumnRef(input));
	}
	auto result = make_uniq<FunctionExpression>(Identifier(LlmFunctions::PREDICT), std::move(children));
	if (!parsed.outputs.empty()) {
		result->SetAlias(Identifier(parsed.outputs[0].name));
	}
	transformer.SetQueryLocation(*result, parse_result.GetLocation());
	return std::move(result);
}

static unique_ptr<TransformResultValue> TransformLlmModelExpression(PEGTransformer &transformer,
                                                                    ParseResult &parse_result) {
	auto &list = AsList(parse_result);
	auto model_name = GetIdentifier(transformer, list.GetChild(1));
	auto prompt = GetString(list.GetChild(3));
	return MakeResult(MakePredictExpression(transformer, parse_result, model_name, prompt));
}

static unique_ptr<TransformResultValue> TransformLlmPromptExpression(PEGTransformer &transformer,
                                                                     ParseResult &parse_result) {
	auto prompt = GetString(AsList(parse_result).GetChild(1));
	return MakeResult(MakePredictExpression(transformer, parse_result, string(), prompt));
}

//! AGG LLM [model] (PROMPT '...') -> llm_reduce('model', '...', string_agg(<one line per row>, ''))
static unique_ptr<TransformResultValue> TransformAggLlmExpression(PEGTransformer &transformer,
                                                                  ParseResult &parse_result) {
	auto &list = AsList(parse_result);
	auto model_name = GetOptionalIdentifier(transformer, list, 2);
	auto prompt = GetString(list.GetChild(5));
	auto parsed = ParsedPrompt::Parse(prompt);
	if (parsed.inputs.empty()) {
		throw ParserException("AGG LLM requires the prompt to reference at least one input column {{column}}");
	}

	// each row contributes "\n{a=`<a>`, b=`<b>`}"; rows with a NULL input are skipped like any aggregate input
	vector<string> fields;
	for (auto &input : parsed.inputs) {
		auto column = input.table.empty()
		                  ? SQLIdentifier::ToString(input.column)
		                  : SQLIdentifier::ToString(input.table) + "." + SQLIdentifier::ToString(input.column);
		fields.push_back(Value(input.ToString() + "=`").ToSQLString() + " || CAST(" + column + " AS VARCHAR) || '`'");
	}
	auto row_text = "chr(10) || '{' || " + StringUtil::Join(fields, " || ', ' || ") + " || '}'";
	auto aggregate = Parser::ParseExpressionList("string_agg(" + row_text + ", '')");
	D_ASSERT(aggregate.size() == 1);

	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(ConstantExpression::String(model_name));
	children.push_back(ConstantExpression::String(prompt));
	children.push_back(std::move(aggregate[0]));
	auto result = make_uniq<FunctionExpression>(Identifier(LlmFunctions::REDUCE), std::move(children));
	if (!parsed.outputs.empty()) {
		result->SetAlias(Identifier(parsed.outputs[0].name));
	}
	transformer.SetQueryLocation(*result, parse_result.GetLocation());
	return MakeResult(unique_ptr<ParsedExpression>(std::move(result)));
}

//===--------------------------------------------------------------------===//
// Table references
//===--------------------------------------------------------------------===//
//! Wraps a table reference in the subquery argument of a table in-out function: (SELECT * FROM <ref>)
static unique_ptr<ParsedExpression> MakeTableArgument(unique_ptr<TableRef> source) {
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	select_node->from_table = std::move(source);
	auto select = make_uniq<SelectStatement>();
	select->node = std::move(select_node);
	auto subquery = make_uniq<SubqueryExpression>();
	subquery->SubqueryMutable() = std::move(select);
	subquery->GetSubqueryTypeMutable() = SubqueryType::SCALAR;
	return std::move(subquery);
}

static unique_ptr<TableRef> MakeTableFunction(PEGTransformer &transformer, ParseResult &parse_result,
                                              const char *function_name, vector<unique_ptr<ParsedExpression>> arguments,
                                              optional_ptr<ParseResult> alias) {
	auto result = make_uniq<TableFunctionRef>();
	result->function = make_uniq<FunctionExpression>(Identifier(function_name), std::move(arguments));
	if (alias) {
		auto table_alias = transformer.Transform<TableAlias>(*alias);
		result->alias = std::move(table_alias.name);
		result->column_name_alias = std::move(table_alias.column_name_alias);
	}
	transformer.SetQueryLocation(*result, parse_result.GetLocation());
	return std::move(result);
}

static unique_ptr<TableRef> MakePredictTable(PEGTransformer &transformer, ParseResult &parse_result,
                                             const string &model_name, const string &prompt, ParseResult &source,
                                             optional_ptr<ParseResult> alias) {
	vector<unique_ptr<ParsedExpression>> arguments;
	arguments.push_back(MakeTableArgument(transformer.Transform<unique_ptr<TableRef>>(source)));
	arguments.push_back(ConstantExpression::String(model_name));
	arguments.push_back(ConstantExpression::String(prompt));
	return MakeTableFunction(transformer, parse_result, LlmFunctions::PREDICT_TABLE, std::move(arguments), alias);
}

static unique_ptr<TableRef> MakeScanTable(PEGTransformer &transformer, ParseResult &parse_result,
                                          const string &model_name, const string &prompt,
                                          optional_ptr<ParseResult> alias) {
	vector<unique_ptr<ParsedExpression>> arguments;
	arguments.push_back(ConstantExpression::String(model_name));
	arguments.push_back(ConstantExpression::String(prompt));
	return MakeTableFunction(transformer, parse_result, LlmFunctions::SCAN, std::move(arguments), alias);
}

//! LLM [model] (PROMPT '...' ON table_ref) [alias]
static unique_ptr<TransformResultValue> TransformLlmTableRef(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list = AsList(parse_result);
	auto model_name = GetOptionalIdentifier(transformer, list, 1);
	auto prompt = GetString(list.GetChild(4));
	return MakeResult(
	    MakePredictTable(transformer, parse_result, model_name, prompt, list.GetChild(6), GetOptional(list, 8)));
}

//! LLM model PROMPT '...' [alias]
static unique_ptr<TransformResultValue> TransformLlmModelScanRef(PEGTransformer &transformer,
                                                                 ParseResult &parse_result) {
	auto &list = AsList(parse_result);
	auto model_name = GetIdentifier(transformer, list.GetChild(1));
	auto prompt = GetString(list.GetChild(3));
	return MakeResult(MakeScanTable(transformer, parse_result, model_name, prompt, GetOptional(list, 4)));
}

//! LLM '...' [alias]
static unique_ptr<TransformResultValue> TransformLlmPromptScanRef(PEGTransformer &transformer,
                                                                  ParseResult &parse_result) {
	auto &list = AsList(parse_result);
	auto prompt = GetString(list.GetChild(1));
	return MakeResult(MakeScanTable(transformer, parse_result, string(), prompt, GetOptional(list, 2)));
}

//! PREDICT(model, PROMPT '...', table_ref) [alias]
static unique_ptr<TransformResultValue> TransformPredictTableRef(PEGTransformer &transformer,
                                                                 ParseResult &parse_result) {
	auto &list = AsList(parse_result);
	auto model_name = GetIdentifier(transformer, list.GetChild(2));
	auto prompt = GetString(list.GetChild(5));
	return MakeResult(
	    MakePredictTable(transformer, parse_result, model_name, prompt, list.GetChild(7), GetOptional(list, 9)));
}

//! EMBED model(COLUMN column, table_ref) [alias]
static unique_ptr<TransformResultValue> TransformEmbedTableRef(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list = AsList(parse_result);
	vector<unique_ptr<ParsedExpression>> arguments;
	arguments.push_back(MakeTableArgument(transformer.Transform<unique_ptr<TableRef>>(list.GetChild(6))));
	arguments.push_back(ConstantExpression::String(GetIdentifier(transformer, list.GetChild(1))));
	arguments.push_back(ConstantExpression::String(GetIdentifier(transformer, list.GetChild(4))));
	return MakeResult(
	    MakeTableFunction(transformer, parse_result, LlmFunctions::EMBED, std::move(arguments), GetOptional(list, 8)));
}

//===--------------------------------------------------------------------===//
// Grammar
//===--------------------------------------------------------------------===//
template <class T>
static grammar_transform_process_function_t Process(T finalize) {
	return [finalize](PEGTransformer &transformer, ParseResult &parse_result) {
		return MakeProcess(transformer, parse_result, finalize);
	};
}

SemOpsGrammarExtension::SemOpsGrammarExtension()
    : GrammarExtension(NAME, "Semantic operators from iPDb: CREATE LLM/EMBED MODEL, LLM ... PROMPT, AGG LLM, "
                             "PREDICT and EMBED") {
}

vector<GrammarChange> SemOpsGrammarExtension::GetChanges() const {
	vector<GrammarChange> changes;

	// statements
	changes.push_back(GrammarChange::AddRule(
	    "SemCreateModelStatement <- 'CREATE' OrReplace? Temporary? SemModelKind 'MODEL' IfNotExists? ColId 'PATH' "
	    "StringLiteral SemOnPrompt? SemModelApi? SemModelSecret? SemModelOptions?",
	    Process(TransformCreateModel)));
	changes.push_back(GrammarChange::AddRule("SemModelKind <- SemLlmKind / SemEmbedKind"));
	changes.push_back(GrammarChange::AddRule("SemLlmKind <- 'LLM'"));
	changes.push_back(GrammarChange::AddRule("SemEmbedKind <- 'EMBED'"));
	changes.push_back(GrammarChange::AddRule("SemOnPrompt <- 'ON' 'PROMPT'"));
	changes.push_back(GrammarChange::AddRule("SemModelApi <- 'API' StringLiteral"));
	changes.push_back(GrammarChange::AddRule("SemModelSecret <- 'SECRET' ColId"));
	changes.push_back(GrammarChange::AddRule("SemModelOptions <- 'OPTIONS' StructExpression"));
	changes.push_back(
	    GrammarChange::AddRule("SemDropModelStatement <- 'DROP' 'MODEL' IfExists? ColId", Process(TransformDropModel)));
	changes.push_back(GrammarChange::PrependChoice("Statement", "SemCreateModelStatement"));
	changes.push_back(GrammarChange::PrependChoice("Statement", "SemDropModelStatement"));

	// expressions
	changes.push_back(GrammarChange::AddRule("SemAggLlmExpression <- 'AGG' 'LLM' ColId? '(' 'PROMPT' StringLiteral ')'",
	                                         Process(TransformAggLlmExpression)));
	changes.push_back(GrammarChange::AddRule("SemLlmModelExpression <- 'LLM' ColId 'PROMPT' StringLiteral",
	                                         Process(TransformLlmModelExpression)));
	changes.push_back(
	    GrammarChange::AddRule("SemLlmPromptExpression <- 'LLM' StringLiteral", Process(TransformLlmPromptExpression)));
	changes.push_back(GrammarChange::PrependChoice("SingleExpression", "SemLlmPromptExpression"));
	changes.push_back(GrammarChange::PrependChoice("SingleExpression", "SemLlmModelExpression"));
	changes.push_back(GrammarChange::PrependChoice("SingleExpression", "SemAggLlmExpression"));

	// table references
	changes.push_back(GrammarChange::AddRule(
	    "SemLlmTableRef <- 'LLM' ColId? '(' 'PROMPT' StringLiteral 'ON' TableRef ')' TableAlias?",
	    Process(TransformLlmTableRef)));
	changes.push_back(GrammarChange::AddRule("SemLlmModelScanRef <- 'LLM' ColId 'PROMPT' StringLiteral TableAlias?",
	                                         Process(TransformLlmModelScanRef)));
	changes.push_back(GrammarChange::AddRule("SemLlmPromptScanRef <- 'LLM' StringLiteral TableAlias?",
	                                         Process(TransformLlmPromptScanRef)));
	changes.push_back(GrammarChange::AddRule(
	    "SemPredictTableRef <- 'PREDICT' '(' ColId ',' 'PROMPT' StringLiteral ',' TableRef ')' TableAlias?",
	    Process(TransformPredictTableRef)));
	changes.push_back(
	    GrammarChange::AddRule("SemEmbedTableRef <- 'EMBED' ColId '(' 'COLUMN' ColId ',' TableRef ')' TableAlias?",
	                           Process(TransformEmbedTableRef)));
	changes.push_back(GrammarChange::PrependChoice("InnerTableRef", "SemEmbedTableRef"));
	changes.push_back(GrammarChange::PrependChoice("InnerTableRef", "SemPredictTableRef"));
	changes.push_back(GrammarChange::PrependChoice("InnerTableRef", "SemLlmPromptScanRef"));
	changes.push_back(GrammarChange::PrependChoice("InnerTableRef", "SemLlmModelScanRef"));
	changes.push_back(GrammarChange::PrependChoice("InnerTableRef", "SemLlmTableRef"));
	return changes;
}

//===--------------------------------------------------------------------===//
// Activation
//===--------------------------------------------------------------------===//
//! Activates the grammar once, at the first query boundary of a connection that was open before the extension loaded
class PendingGrammarActivation : public ClientContextState {
public:
	explicit PendingGrammarActivation(shared_ptr<SemOpsGrammarActivation> activation_p)
	    : activation(std::move(activation_p)) {
	}

	void QueryBegin(ClientContext &context) override {
		ActivateOnce(context);
	}
	void QueryEnd(ClientContext &context) override {
		ActivateOnce(context);
	}

private:
	void ActivateOnce(ClientContext &context) {
		if (activation) {
			activation->Activate(context);
			activation.reset();
		}
	}

	shared_ptr<SemOpsGrammarActivation> activation;
};

void SemOpsGrammarActivation::ActivateOpenConnections(DatabaseInstance &db,
                                                      const shared_ptr<SemOpsGrammarActivation> &activation) {
	for (auto &context : ConnectionManager::Get(db).GetConnectionList()) {
		context->registered_state->GetOrCreate<PendingGrammarActivation>("sem_ops_grammar_activation", activation);
	}
}

void SemOpsGrammarActivation::OnConnectionOpened(ClientContext &context) {
	Activate(context);
}

void SemOpsGrammarActivation::Activate(ClientContext &context) {
	auto &config = ClientConfig::GetConfig(context);
	for (auto &name : config.active_grammar_extensions) {
		if (StringUtil::CIEquals(name, SemOpsGrammarExtension::NAME)) {
			return;
		}
	}
	if (!config.active_grammar_extensions.empty()) {
		auto extensions = config.active_grammar_extensions;
		extensions.push_back(SemOpsGrammarExtension::NAME);
		config.cached_grammar = CompiledGrammar::Create(context, extensions);
		config.active_grammar_extensions = std::move(extensions);
		return;
	}
	lock_guard<mutex> guard(lock);
	if (!grammar) {
		grammar = CompiledGrammar::Create(context, {SemOpsGrammarExtension::NAME});
	}
	config.cached_grammar = grammar;
	config.active_grammar_extensions = {SemOpsGrammarExtension::NAME};
}

} // namespace duckdb
