//===----------------------------------------------------------------------===//
//                         SemOps
//
// sem_ops/grammar.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/parser/grammar_extension.hpp"
#include "duckdb/planner/extension_callback.hpp"

namespace duckdb {

struct CompiledGrammar;

//! Adds the iPDb semantic operator syntax to the PEG parser:
//!   CREATE [OR REPLACE] [TEMP] {LLM | EMBED} MODEL [IF NOT EXISTS] name PATH '...' [ON PROMPT] [API '...']
//!       [SECRET secret] [OPTIONS {...}]
//!   DROP MODEL [IF EXISTS] name
//!   LLM model PROMPT '...' / LLM '...'                      (scalar expression)
//!   AGG LLM [model] (PROMPT '...')                           (aggregate expression)
//!   LLM [model] (PROMPT '...' ON table_ref)                  (table)
//!   LLM model PROMPT '...' / LLM '...'                       (table, generates rows)
//!   PREDICT(model, PROMPT '...', table_ref)                  (table)
//!   EMBED model(COLUMN column, table_ref)                    (table)
class SemOpsGrammarExtension : public GrammarExtension {
public:
	static constexpr const char *NAME = "sem_ops";

	SemOpsGrammarExtension();

	vector<GrammarChange> GetChanges() const override;
};

//! Activates the grammar extension on every new connection
class SemOpsGrammarActivation : public ExtensionCallback {
public:
	void OnConnectionOpened(ClientContext &context) override;

	//! Adds the grammar extension to the active grammar extensions of the connection
	void Activate(ClientContext &context);
	//! Activates the grammar on connections that were already open when the extension was loaded. A connection's
	//! settings may only be changed by its own thread, so this happens at the next query boundary of each connection.
	static void ActivateOpenConnections(DatabaseInstance &db, const shared_ptr<SemOpsGrammarActivation> &activation);

private:
	mutex lock;
	//! The grammar compiled with only this extension, shared by the connections that use nothing else
	shared_ptr<CompiledGrammar> grammar;
};

} // namespace duckdb
