cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME sem_ops)

# Dependencies come from ./vcpkg.json. OpenSSL backs the HTTPS client (DuckDB's
# bundled httplib), nlohmann-json builds and parses the OpenAI-compatible
# request/response bodies.
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD
    "17"
    CACHE STRING "C++ standard to enforce")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src/include)

set(EXTENSION_SOURCES
    src/sem_ops_extension.cpp
    src/sem_ops_grammar.cpp
    src/sem_ops_settings.cpp
    src/model_store.cpp
    src/prompt.cpp
    src/llm_client.cpp
    src/llm_predictor.cpp
    src/llm_functions.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link the dependencies into both the static library and the loadable extension
target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto
                      nlohmann_json::nlohmann_json)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto
                      nlohmann_json::nlohmann_json)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
