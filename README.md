# SemOps: A Semantic Operator Extension for DuckDB

SemOps brings the relational semantic operators of [iPDb](https://github.com/purduedb/ipdb) to DuckDB as a loadable extension. You can use them in a regular DuckDB install; you don't need a custom build.

## Background

iPDb runs AI inference inside the database through native relational operators. It adds SQL syntax for registering models (`CREATE MODEL`, `CREATE LLM MODEL`) and operators for running them over tables (`PREDICT`, `LLM`), for example:

```sql
CREATE LLM MODEL o4_mini PATH 'o4-mini' ON PROMPT API 'https://api.openai.com' SECRET openai_key;

SELECT * FROM LLM o4_mini (PROMPT 'extract the {s:location} and {d:salary} for job {description}', job);
```

iPDb was built as a **fork** of DuckDB. At the time, extensions could only extend DuckDB in limited ways. They couldn't provide:

- Table functions with the flexibility the operators need.
- Fine-grained control over physical operators and the optimizer.
- New SQL syntax added to the parser.

DuckDB v2.0 extensions are much more capable, so the semantic operators can now live in an extension instead of a fork. SemOps is that port.

## Status

Work in progress. The repository is based on the [DuckDB extension template](https://github.com/duckdb/extension-template) and targets DuckDB v2.0 (`v2.0-cyanoptera`). The extension is still built under the template name `waddle` and only contains sample functions; the iPDb operators have not been ported yet.

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
- `build/release/extension/waddle/waddle.duckdb_extension`: the loadable extension binary

## Testing

Tests are [SQLLogicTests](https://duckdb.org/dev/sqllogictest/intro.html) in `test/sql`:

```sh
make test
./build/release/test/unittest test/sql/waddle.test   # a single file
```

## Loading a built binary

Binaries not distributed through the DuckDB community repository are unsigned, so start DuckDB with unsigned extensions allowed:

```sh
duckdb -unsigned
```

```sql
LOAD '/path/to/waddle.duckdb_extension';
```

## References

- iPDb: https://github.com/purduedb/ipdb
- U. Kumarasinghe, et al. *iPDB: Optimizing Semantic SQL Queries.* arXiv:2601.16432, 2026.
- DuckDB extension template documentation: [docs/README.md](docs/README.md)
