#define DUCKDB_EXTENSION_MAIN

#include "sem_ops_extension.hpp"

#include "sem_ops/grammar.hpp"
#include "sem_ops/llm_functions.hpp"
#include "sem_ops/settings.hpp"

#include "duckdb/main/config.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	auto &config = DBConfig::GetConfig(db);
	SemOpsSettings::Register(config);
	LlmFunctions::Register(loader);
	GrammarExtension::Register(db, make_shared_ptr<SemOpsGrammarExtension>());
	auto activation = make_shared_ptr<SemOpsGrammarActivation>();
	ExtensionCallback::Register(config, activation);
	SemOpsGrammarActivation::ActivateOpenConnections(db, activation);
}

void SemOpsExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string SemOpsExtension::Name() {
	return "sem_ops";
}

std::string SemOpsExtension::Version() const {
#ifdef EXT_VERSION_SEM_OPS
	return EXT_VERSION_SEM_OPS;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(sem_ops, loader) {
	duckdb::LoadInternal(loader);
}
}
