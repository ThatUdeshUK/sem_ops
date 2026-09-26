//===----------------------------------------------------------------------===//
//                         SemOps
//
// sem_ops/llm_client.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"

#include <nlohmann/json.hpp>

namespace duckdb {

using Json = nlohmann::json;

//! A minimal client for OpenAI-compatible APIs (chat/completions, embeddings). Safe to use from multiple threads.
class LlmClient {
public:
	static constexpr const char *DEFAULT_BASE_URL = "https://api.openai.com/v1/";

	//! An empty base_url falls back to OPENAI_API_BASE, an empty token to OPENAI_API_KEY
	LlmClient(const string &base_url, const string &token, idx_t timeout_seconds);

	//! POSTs the body to <base_url><endpoint>. Transport and HTTP errors are returned as {"error": ..., "code": ...},
	//! with code -1 for transport errors.
	Json Post(const string &endpoint, const Json &body) const;

	const string &GetBaseURL() const {
		return base_url;
	}

private:
	string base_url;
	string proto_host_port;
	string path_prefix;
	string token;
	idx_t timeout_seconds;
};

} // namespace duckdb
