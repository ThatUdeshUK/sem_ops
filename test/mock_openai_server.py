#!/usr/bin/env python3
"""A deterministic OpenAI-compatible server for testing the sem_ops extension without a real LLM.

Usage: python3 test/mock_openai_server.py [port]   (default port 8765)

Every answer is derived from the first backtick-quoted input value of a row (`'text'` -> text):
  string  property -> the value upper-cased
  integer property -> the length of the value
  number  property -> the length of the value / 2
  boolean property -> whether the value contains "good"
The property `batch_info` instead holds the number of rows sent in the request, which exposes the batching.

Special model names:
  mock-bad-batch   answers batched requests with invalid JSON, so the extension falls back to row-wise calls
  mock-missing     responds with 404 to every request
  mock-auth        responds with 401 unless the request carries `Authorization: Bearer sk-test`
"""

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EMBEDDING_DIMENSIONS = 384


def row_value(row_text):
    match = re.search(r"`([^`]*)`", row_text)
    if not match:
        return ""
    value = match.group(1)
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        value = value[1:-1].replace("''", "'")
    return value


def answer_row(schema, value, batch_rows):
    result = {}
    for name, prop in schema.get("properties", {}).items():
        kind = prop.get("type")
        if name == "batch_info":
            result[name] = batch_rows
        elif kind == "string":
            result[name] = value.upper()
        elif kind == "integer":
            result[name] = len(value)
        elif kind == "number":
            result[name] = len(value) / 2
        elif kind == "boolean":
            result[name] = "good" in value.lower()
        else:
            result[name] = None
    return result


def answer_join(user):
    left, right = user.split("Right items:")
    left_items = re.findall(r"\[(\d+)\] (.*)", left)
    right_items = re.findall(r"\[(\d+)\] (.*)", right)
    pairs = []
    for left_id, left_text in left_items:
        for right_id, right_text in right_items:
            if row_value(left_text).lower() == row_value(right_text).lower():
                pairs.append({"left_id": int(left_id), "right_id": int(right_id)})
    return {"matching_pairs": pairs}


def answer_chat(request):
    model = request.get("model", "")
    user = request["messages"][-1]["content"]
    response_format = request.get("response_format")
    if response_format is None:
        rows = [line for line in user.splitlines() if line.startswith("{")]
        return f"{len(rows)} rows: " + ", ".join(sorted(row_value(row) for row in rows))

    schema = response_format["json_schema"]["schema"]
    properties = schema.get("properties", {})
    if "matching_pairs" in properties:
        return json.dumps(answer_join(user))
    if "output_array" in properties:
        if model == "mock-bad-batch":
            return "this is not JSON"
        row_schema = properties["output_array"]["items"]
        rows = re.findall(r"^\[?\{(.*)\},$", user, re.MULTILINE)
        if not rows:
            # table generation: no inputs, produce a fixed set of rows
            return json.dumps({"output_array": [answer_row(row_schema, name, 3) for name in ("alpha", "beta", "good")]})
        return json.dumps({"output_array": [answer_row(row_schema, row_value(row), len(rows)) for row in rows]})
    return json.dumps(answer_row(schema, row_value(user.split(";\n", 1)[-1]), 1))


def embed(text):
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[0] = float(len(text))
    for i, char in enumerate(text[: EMBEDDING_DIMENSIONS - 1]):
        vector[i + 1] = ord(char) / 256
    return vector


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if request.get("model") == "mock-missing":
            self.reply(404, {"error": {"message": "The model `mock-missing` does not exist"}})
            return
        if request.get("model") == "mock-auth" and self.headers.get("Authorization") != "Bearer sk-test":
            self.reply(401, {"error": {"message": "Incorrect API key provided"}})
            return
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if self.path.endswith("/chat/completions"):
            content = answer_chat(request)
            self.reply(
                200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}], "usage": usage}
            )
        elif self.path.endswith("/embeddings"):
            data = [{"index": i, "embedding": embed(text)} for i, text in enumerate(request["input"])]
            self.reply(200, {"data": data, "usage": usage})
        else:
            self.reply(404, {"error": {"message": f"Unknown endpoint {self.path}"}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
