# SemOps: A Semantic Operator Extension for DuckDB

SemOps brings the relational semantic operators of [iPDb](https://github.com/purduedb/ipdb) to DuckDB as a loadable extension. You can use them in a regular DuckDB install; you don't need a custom build.

## Background

iPDb runs AI inference inside the database through native relational operators. It adds SQL syntax for registering models (`CREATE MODEL`, `CREATE LLM MODEL`) and operators for running them over tables (`PREDICT`, `LLM`), for example:

```sql
CREATE LLM MODEL o4_mini PATH 'o4-mini' ON PROMPT API 'https://api.openai.com/v1/' SECRET openai_key;

SELECT * FROM LLM o4_mini (PROMPT 'extract the {location VARCHAR} and {salary DOUBLE} for job {{description}}' ON jobs);
```

iPDb was built as a **fork** of DuckDB. At the time, extensions could only extend DuckDB in limited ways. They couldn't provide:

- Table functions with the flexibility the operators need.
- Fine-grained control over physical operators and the optimizer.
- New SQL syntax added to the parser.

DuckDB v2.0 extensions are much more capable, so the semantic operators can now live in an extension instead of a fork. SemOps is that port.

## Status

Work in progress, targeting DuckDB v2.0 (`v2.0-cyanoptera`). The LLM and embedding operators of iPDb, which call an OpenAI-compatible API, are ported. The tabular, LM and GNN models (ONNX/torch predictors) and local `.gguf` models (llama.cpp) are not.

## Usage

The syntax is enabled on every connection opened after the extension is loaded. A connection that loads it with `LOAD` gets it from its next statement; other connections that were already open get it after their next statement, or right away with `SET active_grammar_extensions = ['sem_ops']`.

### Models

```sql
CREATE SECRET openai_key (TYPE http, BEARER_TOKEN 'sk-...');

CREATE [OR REPLACE] [TEMP] LLM MODEL [IF NOT EXISTS] o4_mini PATH 'o4-mini' ON PROMPT
    [API 'https://api.openai.com/v1/'] [SECRET openai_key] [OPTIONS {batch_size: 32, n_threads: 8}];
CREATE [OR REPLACE] [TEMP] EMBED MODEL [IF NOT EXISTS] embedder PATH 'text-embedding-3-small'
    [API '...'] [SECRET ...] [OPTIONS {...}];
DROP MODEL [IF EXISTS] o4_mini;

SELECT * FROM sem_ops_models();
```

`PATH` is the model name sent to the API. Without `API` the base URL is `OPENAI_API_BASE`, or `https://api.openai.com/v1/`. Without `SECRET` the token is `OPENAI_API_KEY`. Models are rows of a hidden table (`__sem_ops_models` in the current schema, `temp.__sem_ops_temp_models` for `TEMP` models), so they persist with the database and model DDL is transactional.

### Operators

Prompts reference input columns as `{{column}}` (or `{{table.column}}`) and declare typed output columns as `{name TYPE}`, where `TYPE` is `VARCHAR`, `INTEGER`, `DOUBLE` or `BOOLEAN`. The model name is optional everywhere except `EMBED`; without it, a model is picked by `model_select_strategy` (`first` or `random`).

```sql
-- scalar: one output column, usable in SELECT, WHERE, JOIN ... ON and GROUP BY
SELECT id, LLM o4_mini PROMPT 'what is the {sentiment VARCHAR} of {{review}}' FROM reviews;
SELECT * FROM reviews WHERE LLM 'is {{review}} {positive BOOLEAN}';

-- semantic join: all pairs of a chunk are decided by one call
SELECT p.name, o.name FROM product p JOIN product o
    ON LLM o4_mini PROMPT 'is CPU {{o.name}} {compatible BOOLEAN} with motherboard {{p.name}}';

-- table: appends the output columns to the input relation
SELECT * FROM LLM o4_mini (PROMPT 'extract the {genre VARCHAR} and {year INTEGER} from {{plot}}' ON movies);
SELECT * FROM PREDICT(o4_mini, PROMPT 'extract the {genre VARCHAR} from {{plot}}', movies);

-- table generation without an input relation
SELECT * FROM LLM o4_mini PROMPT 'list all the {state VARCHAR} and their {state_tax DOUBLE} in the US';

-- aggregate: one call per group
SELECT category, AGG LLM o4_mini (PROMPT 'summarize the {summary VARCHAR} of {{review}}') FROM reviews GROUP BY category;

-- embeddings: appends vec FLOAT[384]
SELECT * FROM EMBED embedder(COLUMN review, reviews);
```

The syntax is rewritten into plain functions, which can also be called directly: `llm_predict(model, prompt, inputs...)`, `llm_predict_table(TABLE, model, prompt)`, `llm_scan(model, prompt)`, `llm_reduce(model, prompt, text)` and `llm_embed(TABLE, model, column)`.

### Settings

| Setting | Default | Model option | Meaning |
|---|---|---|---|
| `llm_use_batch` | `true` | `use_batch` | Send a batch of rows in one call, falling back to one call per row for rows the batch could not answer |
| `ml_batch_size` | `16` | `batch_size` | Rows per batched call |
| `llm_use_cache` | `true` | `use_cache` | Send rows with identical inputs once per query and share the result |
| `llm_no_threads` | `16` | `n_threads` | Concurrent calls |
| `llm_timeout` | `120` | `timeout` | Timeout of one API request in seconds |
| `llm_max_tokens` | `512` | `llm_max_tokens` | Accepted for iPDb compatibility, not sent to the API (as in iPDb) |
| `model_select_strategy` | `first` | | How a model is picked when none is named |

The number of API calls and tokens used are logged at the `INFO` level (`CALL enable_logging(level = 'info')`).

## Caveats

### No model catalog

iPDb stores models as entries in the new DuckDB catalog (`ModelCatalogEntry`). An extension cannot add catalog entry types, so SemOps stores models as rows of a hidden table instead.

- **Models are ordinary tables.** `__sem_ops_models` and `temp.__sem_ops_temp_models` appear in `SHOW ALL TABLES`, `duckdb_tables()` and `information_schema`, are exported with the database, and can be modified or dropped directly with plain SQL. Models do not appear in `duckdb_*` catalog functions; list them with `sem_ops_models()`.
- **Model DDL is rewritten into SQL.** `CREATE MODEL` and `DROP MODEL` run as a sequence of `CREATE TABLE IF NOT EXISTS` / `DELETE` / `INSERT` statements. As a result:
  - Persistent models cannot be created in a read-only database.
  - `DROP MODEL` fails in a read-only database even for a `TEMP` model, because it ensures both tables exist.
- Models can only be stored in DuckDB-format databases, not in attached databases of other formats (e.g. Postgres, SQLite).

### Optimizations missing compared to iPDb

The semantic operators are volatile functions to DuckDB. The optimizer treats them as opaque and knows nothing about their cost. Specifically:

- **Semantic join decomposition** is not ported.
- **LIMIT pushdown into LLM filters**  is not ported. 
- **Speculative Tuple Inference** is not ported.
- **Cost-aware model selection** is not ported;
- **No LLM metrics in `EXPLAIN ANALYZE`.** Call and token counts only go to the log.

### Other caveats

- **Only remote models.** Local `.gguf` models (llama.cpp) are not supported. Serve local models through an OpenAI-compatible server instead.

## Building

Clone with submodules (`duckdb` and `extension-ci-tools`):

```sh
git clone --recurse-submodules <repo-url>
# or, in an existing clone
git submodule update --init --recursive
```

Dependencies are managed with [vcpkg](https://vcpkg.io/en/getting-started):

```sh
git clone https://github.com/Microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh
export VCPKG_TOOLCHAIN_PATH=`pwd`/vcpkg/scripts/buildsystems/vcpkg.cmake
```

Build (installing [ninja](https://ninja-build.org/) and [ccache](https://ccache.dev/) makes rebuilds much faster):

```sh
GEN=ninja make          # release
GEN=ninja make debug    # debug
```

This produces:

- `build/release/duckdb`: the DuckDB shell with the extension preloaded
- `build/release/test/unittest`: the DuckDB test runner with the extension linked in
- `build/release/extension/sem_ops/sem_ops.duckdb_extension`: the loadable extension binary

## Testing

Tests are [SQLLogicTests](https://duckdb.org/dev/sqllogictest/intro.html) in `test/sql`:

```sh
make test
./build/release/test/unittest test/sql/sem_ops.test   # a single file
```

`test/sql/sem_ops_llm.test` exercises the operators against a deterministic mock of the OpenAI API and is skipped unless `SEM_OPS_MOCK_API` is set:

```sh
python3 test/mock_openai_server.py 8765 &
SEM_OPS_MOCK_API=http://127.0.0.1:8765/v1/ ./build/release/test/unittest test/sql/sem_ops_llm.test
```

## Loading a built binary

Binaries not distributed through the DuckDB community repository are unsigned, so start DuckDB with unsigned extensions allowed:

```sh
duckdb -unsigned
```

```sql
LOAD '/path/to/sem_ops.duckdb_extension';
```

## References

- iPDb: https://github.com/purduedb/ipdb
- U. Kumarasinghe, et al. *iPDB: Optimizing Semantic SQL Queries.* arXiv:2601.16432, 2026.
- DuckDB extension template documentation: [docs/README.md](docs/README.md)
