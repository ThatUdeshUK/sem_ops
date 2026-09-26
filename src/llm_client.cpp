#include "sem_ops/llm_client.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT
// never wait for a `100 Continue` before sending a request body: prompts are large, and servers that ignore
// `Expect: 100-continue` would delay every call by a second
#define CPPHTTPLIB_EXPECT_100_THRESHOLD 0
#include "httplib.hpp"

#include <cstdlib>

namespace duckdb {

static string GetEnvironmentVariable(const char *name) {
	auto value = std::getenv(name);
	return value ? string(value) : string();
}

LlmClient::LlmClient(const string &base_url_p, const string &token_p, idx_t timeout_seconds_p)
    : base_url(base_url_p), token(token_p), timeout_seconds(timeout_seconds_p) {
	if (base_url.empty()) {
		base_url = GetEnvironmentVariable("OPENAI_API_BASE");
	}
	if (base_url.empty()) {
		base_url = DEFAULT_BASE_URL;
	}
	if (!StringUtil::EndsWith(base_url, "/")) {
		base_url += "/";
	}
	if (token.empty()) {
		token = GetEnvironmentVariable("OPENAI_API_KEY");
	}

	auto scheme_end = base_url.find("://");
	if (scheme_end == string::npos) {
		throw InvalidInputException("Invalid API URL \"%s\": expected http:// or https://", base_url);
	}
	auto path_start = base_url.find('/', scheme_end + 3);
	proto_host_port = base_url.substr(0, path_start);
	path_prefix = base_url.substr(path_start);
}

static Json ErrorResponse(const string &error, int32_t code) {
	Json result;
	result["error"] = error;
	result["code"] = code;
	return result;
}

Json LlmClient::Post(const string &endpoint, const Json &body) const {
	duckdb_httplib_openssl::Client client(proto_host_port);
	auto timeout = static_cast<time_t>(timeout_seconds);
	client.set_connection_timeout(timeout, 0);
	client.set_read_timeout(timeout, 0);
	client.set_write_timeout(timeout, 0);
	client.set_follow_location(false);
	client.set_keep_alive(false);
	if (!token.empty()) {
		client.set_bearer_token_auth(token);
	}

	auto response =
	    client.Post(path_prefix + endpoint, duckdb_httplib_openssl::Headers(), body.dump(), "application/json");
	if (!response) {
		return ErrorResponse(duckdb_httplib_openssl::to_string(response.error()), -1);
	}
	if (response->status < 200 || response->status >= 300) {
		string reason = response->reason;
		auto error_body = Json::parse(response->body, nullptr, false);
		if (!error_body.is_discarded() && error_body.contains("error")) {
			auto &error = error_body["error"];
			reason = error.is_object() && error.contains("message") ? error["message"].dump() : error.dump();
		}
		return ErrorResponse(reason, response->status);
	}
	auto result = Json::parse(response->body, nullptr, false);
	if (result.is_discarded()) {
		return ErrorResponse("Response is not valid JSON: " + response->body.substr(0, 200), response->status);
	}
	return result;
}

} // namespace duckdb
