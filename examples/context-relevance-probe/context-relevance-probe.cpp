#define CONTEXT_STABLE_KEY_PROBE_LIBRARY
#include "../context-stable-key-probe/context-stable-key-probe.cpp"

#include <cstdlib>
#include <sstream>

namespace {

struct persisted_key {
    std::string source;
    std::vector<float> values;
};

std::string json_string_field(const std::string & line, const std::string & key) {
    const std::string marker = "\"" + key + "\":\"";
    const size_t start = line.find(marker);
    if (start == std::string::npos) return {};
    const size_t value_start = start + marker.size();
    const size_t value_end = line.find('"', value_start);
    if (value_end == std::string::npos) throw std::runtime_error("malformed JSON metadata field: " + key);
    return line.substr(value_start, value_end - value_start);
}

int64_t json_integer_field(const std::string & line, const std::string & key) {
    const std::string marker = "\"" + key + "\":";
    const size_t start = line.find(marker);
    if (start == std::string::npos) throw std::runtime_error("missing JSON metadata field: " + key);
    size_t value_start = start + marker.size();
    while (value_start < line.size() && std::isspace((unsigned char) line[value_start])) ++value_start;
    size_t value_end = value_start;
    while (value_end < line.size() && (std::isdigit((unsigned char) line[value_end]) || line[value_end] == '-')) ++value_end;
    return std::stoll(line.substr(value_start, value_end - value_start));
}

std::map<std::string, std::vector<float>> load_persisted_k1(const std::filesystem::path & root) {
    std::ifstream metadata(root / "vector_metadata.jsonl");
    std::ifstream binary(root / "vectors.f32", std::ios::binary);
    if (!metadata || !binary) {
        throw std::runtime_error("Step-17 persisted vector artifacts are missing under " + root.string());
    }

    std::map<std::string, std::vector<float>> out;
    std::string line;
    while (std::getline(metadata, line)) {
        if (json_string_field(line, "role") != "stored_source" ||
            json_string_field(line, "vector_type") != "K1_source_token_hidden") {
            continue;
        }
        const std::string source = json_string_field(line, "source");
        const int64_t dimension = json_integer_field(line, "dimension");
        if (dimension != KEY_WIDTH || source.empty()) throw std::runtime_error("invalid persisted K1 metadata");
        std::vector<float> values(KEY_WIDTH);
        binary.seekg(json_integer_field(line, "byte_offset"));
        binary.read(reinterpret_cast<char *>(values.data()), (std::streamsize) values.size()*sizeof(float));
        if (!binary) throw std::runtime_error("failed to read persisted K1 vector for " + source);
        if (!out.emplace(source, std::move(values)).second) throw std::runtime_error("duplicate persisted K1 key: " + source);
    }
    if (out.size() != SYMBOLS.size()) throw std::runtime_error("expected eight persisted Step-17 K1 source keys");
    return out;
}

void write_double_array(std::ostream & out, const std::vector<double> & values) {
    out << '[' << std::setprecision(17);
    for (size_t index = 0; index < values.size(); ++index) out << (index ? "," : "") << values[index];
    out << ']';
}

std::string hex_bytes(const std::string & value) {
    static constexpr char HEX[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size()*2);
    for (unsigned char byte : value) {
        out.push_back(HEX[byte >> 4]);
        out.push_back(HEX[byte & 0xf]);
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const options opts = parse_args(argc, argv);
        const std::filesystem::path root(opts.output_dir);
        std::filesystem::create_directories(root);
        const char * stored_env = std::getenv("STEP17_ARTIFACT_DIR");
        const std::filesystem::path stored_root = stored_env
            ? std::filesystem::path(stored_env)
            : std::filesystem::path("/workspace/qwen35_context_stable_key_probe_raw");

        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.gpu_layers;
        llama_model * public_model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (!public_model) throw std::runtime_error("model load failed");
        auto & model = *public_model;
        if (model.hparams.n_embd != KEY_WIDTH || model.hparams.n_layer() != 32) {
            throw std::runtime_error("probe requires Qwen3.5-4B dimensions 32x2560");
        }

        key_trace_collector collector;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.ctx_size;
        context_params.n_batch = opts.ctx_size;
        context_params.n_ubatch = opts.ctx_size;
        context_params.n_seq_max = 1;
        context_params.cb_eval = key_eval_callback;
        context_params.cb_eval_user_data = &collector;
        llama_context * ctx = llama_init_from_model(public_model, context_params);
        if (!ctx) throw std::runtime_error("context creation failed");
        const llama_vocab * vocab = llama_model_get_vocab(public_model);

        const auto persisted_k1 = load_persisted_k1(stored_root);
        std::map<std::string, std::vector<float>> static_embeddings;
        for (const auto & symbol : SYMBOLS) {
            static_embeddings[symbol] = normalized(load_embedding_row(model, canonical_symbol_token(vocab, symbol)));
        }

        const std::array<std::tuple<std::string, std::string, std::string>, 16> queries = {{
            {"relevant_A_arrow", "relevant", "A ->"},
            {"relevant_A_input", "relevant", "Input: A"},
            {"relevant_A_symbol", "relevant", "Symbol: A"},
            {"relevant_A_current", "relevant", "Current symbol: A"},
            {"relevant_B_arrow", "relevant", "B ->"},
            {"relevant_B_input", "relevant", "Input: B"},
            {"relevant_B_symbol", "relevant", "Symbol: B"},
            {"relevant_B_current", "relevant", "Current symbol: B"},
            {"irrelevant_A_letter_before", "irrelevant", "The letter A comes before B."},
            {"irrelevant_A_grade", "irrelevant", "Grade A is considered excellent."},
            {"irrelevant_A_word", "irrelevant", "The word A appears in this sentence."},
            {"irrelevant_A_first", "irrelevant", "A is the first letter of the alphabet."},
            {"irrelevant_B_vitamin", "irrelevant", "Vitamin B is found in many foods."},
            {"irrelevant_B_letter_after", "irrelevant", "The letter B comes after A."},
            {"irrelevant_B_plan", "irrelevant", "Plan B is the backup option."},
            {"irrelevant_B_word", "irrelevant", "The word B appears in this sentence."},
        }};

        std::vector<key_observation> observations;
        int32_t id = 0;
        for (const auto & [variant, role, prompt] : queries) {
            const std::string source = variant.find("_A_") != std::string::npos ? "A" : "B";
            observations.push_back(evaluate_key(ctx, vocab, collector, prompt, source, role, variant, id++));
        }
        for (auto & observation : observations) observation.k2 = static_embeddings.at(observation.source);

        std::ofstream vector_bin(root / "vectors.f32", std::ios::binary | std::ios::trunc);
        std::ofstream vector_meta(root / "vector_metadata.jsonl", std::ios::trunc);
        if (!vector_bin || !vector_meta) throw std::runtime_error("cannot open vector artifacts");
        const auto write_vector = [&](int32_t observation_id, const std::string & role,
                                      const std::string & variant, const std::string & source,
                                      const std::string & type, const std::vector<float> & values) {
            const std::streamoff offset = vector_bin.tellp();
            vector_bin.write(reinterpret_cast<const char *>(values.data()), (std::streamsize) values.size()*sizeof(float));
            vector_meta << "{\"observation_id\":" << observation_id
                        << ",\"role\":\"" << json_escape(role)
                        << "\",\"variant\":\"" << json_escape(variant)
                        << "\",\"source\":\"" << json_escape(source)
                        << "\",\"vector_type\":\"" << type
                        << "\",\"byte_offset\":" << offset
                        << ",\"dimension\":" << values.size() << "}\n";
        };
        for (const auto & observation : observations) {
            write_vector(observation.id, observation.role, observation.variant, observation.source,
                         "source_hidden_raw", observation.source_hidden);
            write_vector(observation.id, observation.role, observation.variant, observation.source,
                         "K1_source_token_hidden", observation.k1);
            write_vector(observation.id, observation.role, observation.variant, observation.source,
                         "K2_static_source_embedding", observation.k2);
        }
        for (const auto & source : SYMBOLS) {
            write_vector(-1, "stored_key", "step17_original", source,
                         "K1_source_token_hidden", persisted_k1.at(source));
            write_vector(-1, "stored_key", "step17_original", source,
                         "K2_static_source_embedding", static_embeddings.at(source));
        }

        std::ofstream observation_out(root / "observation_metadata.jsonl", std::ios::trunc);
        std::ofstream matrix_out(root / "similarity_matrices.jsonl", std::ios::trunc);
        if (!observation_out || !matrix_out) throw std::runtime_error("cannot open metadata artifacts");
        for (const auto & observation : observations) {
            observation_out << "{\"observation_id\":" << observation.id
                           << ",\"role\":\"" << json_escape(observation.role)
                           << "\",\"variant\":\"" << json_escape(observation.variant)
                           << "\",\"expected_source\":\"" << observation.source
                           << "\",\"prompt\":\"" << json_escape(observation.prompt)
                           << "\",\"prompt_bytes_hex\":\"" << hex_bytes(observation.prompt) << "\""
                           << ",\"prompt_token_ids\":";
            write_tokens(observation_out, observation.prompt_tokens);
            observation_out << ",\"source_position\":" << observation.location.position
                           << ",\"actual_source_token_id\":" << observation.location.actual_token
                           << ",\"canonical_source_token_id\":" << observation.location.canonical_token
                           << ",\"actual_source_piece\":\"" << json_escape(observation.location.actual_piece)
                           << "\",\"source_match_method\":\"" << observation.location.match_method
                           << "\"}\n";
        }

        const std::array<std::string, 2> key_types = {
            "K1_source_token_hidden", "K2_static_source_embedding",
        };
        for (const auto & key_type : key_types) {
            for (const auto & observation : observations) {
                const std::vector<float> & query = key_type == "K1_source_token_hidden"
                    ? observation.k1 : observation.k2;
                std::vector<double> similarities;
                for (const auto & source : SYMBOLS) {
                    const auto & stored = key_type == "K1_source_token_hidden"
                        ? persisted_k1.at(source) : static_embeddings.at(source);
                    similarities.push_back(cosine(query, stored));
                }
                std::vector<size_t> order(SYMBOLS.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
                    return similarities[left] > similarities[right];
                });
                const size_t first = order[0];
                const size_t second = order[1];
                const double margin = similarities[first] - similarities[second];
                const double ratio = std::abs(similarities[second]) > 1e-15
                    ? similarities[first] / similarities[second] : 1e300;
                matrix_out << std::setprecision(17)
                           << "{\"key_type\":\"" << key_type
                           << "\",\"observation_id\":" << observation.id
                           << ",\"role\":\"" << observation.role
                           << "\",\"variant\":\"" << json_escape(observation.variant)
                           << "\",\"expected_source\":\"" << observation.source
                           << "\",\"similarities\":";
                write_double_array(matrix_out, similarities);
                matrix_out << ",\"top1_source\":\"" << SYMBOLS[first]
                           << "\",\"top1_cosine\":" << similarities[first]
                           << ",\"top2_source\":\"" << SYMBOLS[second]
                           << "\",\"top2_cosine\":" << similarities[second]
                           << ",\"margin\":" << margin
                           << ",\"ratio\":" << ratio
                           << ",\"correct\":" << json_bool(SYMBOLS[first] == observation.source) << "}\n";
            }
        }

        double stored_norm_error = 0;
        for (const auto & source : SYMBOLS) {
            double norm = 0;
            for (float value : persisted_k1.at(source)) norm += (double) value*value;
            stored_norm_error = std::max(stored_norm_error, std::abs(std::sqrt(norm) - 1.0));
        }
        std::ofstream validation(root / "validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"stored_k1_count\":8,\"query_count\":16,\"relevant_count\":8,\"irrelevant_count\":8"
                   << ",\"stored_k1_norm_max_error\":" << stored_norm_error
                   << ",\"stored_step17_artifacts_reused\":true"
                   << ",\"qwen_weights_modified\":false,\"learning_replayed\":false"
                   << ",\"sidecar_injection\":false,\"gradients\":false,\"optimizer\":false}\n";
        std::ofstream runtime(root / "runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ",\"model_decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size
                << ",\"clean_kv_per_forward\":true,\"stored_step17_artifact_dir\":\""
                << json_escape(stored_root.string()) << "\"}\n";
        std::ofstream memory(root / "memory_metrics.json", std::ios::trunc);
        memory << "{\"query_raw_vectors\":16,\"query_normalized_vectors\":32,\"stored_normalized_vectors\":16"
               << ",\"dimension\":2560,\"dense_plastic_matrix\":false}\n";
        std::ofstream config(root / "probe_metadata.json", std::ios::trunc);
        config << "{\"experiment\":\"qwen35_context_relevance_probe\",\"stored_key_artifact\":\""
               << json_escape(stored_root.string())
               << "\",\"stored_key_provenance\":\"Step-17 persisted K1 source-token hidden vectors\","
               << "\"relevant_contexts\":[\"A ->\",\"Input: A\",\"Symbol: A\",\"Current symbol: A\",\"B ->\",\"Input: B\",\"Symbol: B\",\"Current symbol: B\"],"
               << "\"irrelevant_contexts\":[\"The letter A comes before B.\",\"Grade A is considered excellent.\",\"The word A appears in this sentence.\",\"A is the first letter of the alphabet.\",\"Vitamin B is found in many foods.\",\"The letter B comes after A.\",\"Plan B is the backup option.\",\"The word B appears in this sentence.\"],"
               << "\"key_definitions\":{\"K1_source_token_hidden\":\"normalized final-layer hidden state at actual queried source-token position\",\"K2_static_source_embedding\":\"normalized frozen input embedding of canonical source token\"},"
               << "\"no_learning_or_injection\":true}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
