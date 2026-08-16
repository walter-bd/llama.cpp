#define CONTEXT_STABLE_KEY_PROBE_LIBRARY
#include "../context-stable-key-probe/context-stable-key-probe.cpp"

#include <array>
#include <cctype>
#include <fstream>
#include <numeric>
#include <sstream>

namespace {

const std::array<std::string, 8> PROTECTIVE_PROMPTS = {
    "The capital of France is",
    "Two plus two equals",
    "A common week has this many days:",
    "Water freezes at zero degrees",
    "The season after spring is",
    "A triangle has this many sides:",
    "The opposite of narrow is",
    "Monday, Tuesday,",
};

struct final_observation {
    int32_t id = -1;
    std::string role;
    std::string variant;
    std::string source;
    std::string prompt;
    std::vector<llama_token> prompt_tokens;
    std::vector<float> final_hidden;
    int32_t greedy = -1;
};

struct query_record {
    key_observation observed;
    std::vector<float> query_only_final;
    std::vector<float> c0;
    std::vector<float> c1_delta_raw;
    std::vector<float> c1;
    std::vector<float> k1;
};

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

std::string json_string_field(const std::string & line, const std::string & key) {
    const std::string marker = "\"" + key + "\":\"";
    const size_t start = line.find(marker);
    if (start == std::string::npos) return {};
    const size_t value_start = start + marker.size();
    const size_t value_end = line.find('\"', value_start);
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

void write_double_array(std::ostream & out, const std::vector<double> & values) {
    out << '[' << std::setprecision(17);
    for (size_t index = 0; index < values.size(); ++index) {
        out << (index ? "," : "") << values[index];
    }
    out << ']';
}

std::vector<float> subtract(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("context subtraction dimension mismatch");
    std::vector<float> result(left.size());
    for (size_t index = 0; index < left.size(); ++index) result[index] = left[index] - right[index];
    return result;
}

final_observation evaluate_final_only(
        llama_context * ctx,
        const llama_vocab * vocab,
        key_trace_collector & collector,
        const std::string & prompt,
        const std::string & role,
        const std::string & variant,
        const std::string & source,
        int32_t id) {
    const auto tokens = tokenize(vocab, prompt, true);
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    llama_memory_clear(llama_get_memory(ctx), true);
    collector.clear();
    collector.enabled = true;
    for (size_t position = 0; position < tokens.size(); ++position) {
        collector.capture_final = position + 1 == tokens.size();
        llama_token token = tokens[position];
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            collector.enabled = false;
            collector.clear();
            throw std::runtime_error("prompt decode failed");
        }
        ++decode_calls;
        llama_synchronize(ctx);
    }
    collector.capture_final = false;
    collector.enabled = false;
    if (collector.final_hidden.size() != KEY_WIDTH) {
        throw std::runtime_error("missing final hidden state");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) throw std::runtime_error("missing logits");
    const auto greedy = (int32_t) std::distance(
        logits, std::max_element(logits, logits + n_vocab));
    return {id, role, variant, source, prompt, tokens,
            std::move(collector.final_hidden), greedy};
}

std::map<std::string, std::vector<float>> load_persisted_k1_local(const std::filesystem::path & root) {
    // Reuse Step-17's exact normalized source-token vectors. This is only an
    // identity-channel check; context scores never use these vectors.
    std::ifstream metadata(root / "vector_metadata.jsonl");
    std::ifstream binary(root / "vectors.f32", std::ios::binary);
    if (!metadata || !binary) throw std::runtime_error("missing Step-17 K1 artifacts: " + root.string());
    std::map<std::string, std::vector<float>> result;
    std::string line;
    while (std::getline(metadata, line)) {
        if (json_string_field(line, "role") != "stored_source" ||
            json_string_field(line, "vector_type") != "K1_source_token_hidden") continue;
        const std::string source = json_string_field(line, "source");
        const int64_t dimension = json_integer_field(line, "dimension");
        if (dimension != KEY_WIDTH || source.empty()) throw std::runtime_error("invalid Step-17 K1 metadata");
        std::vector<float> values(KEY_WIDTH);
        binary.seekg(json_integer_field(line, "byte_offset"));
        binary.read(reinterpret_cast<char *>(values.data()), (std::streamsize) values.size()*sizeof(float));
        if (!binary) throw std::runtime_error("failed to read Step-17 K1 vector");
        if (!result.emplace(source, std::move(values)).second) throw std::runtime_error("duplicate Step-17 K1 source");
    }
    if (result.size() != SYMBOLS.size()) throw std::runtime_error("expected eight Step-17 K1 source vectors");
    return result;
}

void write_vector(
        std::ofstream & binary,
        std::ofstream & metadata,
        int32_t id,
        const std::string & group,
        const std::string & role,
        const std::string & variant,
        const std::string & source,
        const std::string & prompt,
        const std::vector<llama_token> & prompt_tokens,
        const std::string & type,
        const std::vector<float> & values) {
    if (values.size() != KEY_WIDTH) throw std::runtime_error("invalid vector width");
    const std::streamoff offset = binary.tellp();
    binary.write(reinterpret_cast<const char *>(values.data()), (std::streamsize) values.size()*sizeof(float));
    metadata << std::setprecision(17)
             << "{\"id\":" << id
             << ",\"group\":\"" << json_escape(group)
             << "\",\"role\":\"" << json_escape(role)
             << "\",\"variant\":\"" << json_escape(variant)
             << "\",\"source\":\"" << json_escape(source)
             << "\",\"type\":\"" << json_escape(type)
             << "\",\"prompt\":\"" << json_escape(prompt)
             << "\",\"prompt_bytes_hex\":\"" << hex_bytes(prompt)
             << "\",\"prompt_token_ids\":";
    write_tokens(metadata, prompt_tokens);
    metadata << ",\"offset_bytes\":" << offset
             << ",\"dimension\":" << values.size()
             << ",\"norm\":" << l2(values) << "}\n";
}

std::vector<double> similarities(
        const std::vector<float> & query,
        const std::map<std::string, std::vector<float>> & prototypes) {
    std::vector<double> result;
    for (const auto & source : SYMBOLS) result.push_back(cosine(query, prototypes.at(source)));
    return result;
}

void write_context_row(
        std::ofstream & out,
        const std::string & row_kind,
        const query_record & query,
        const std::string & context_type,
        const std::vector<double> & values,
        const std::vector<float> & query_values,
        const std::map<std::string, std::vector<float>> & prototypes,
        const std::string & excluded_source,
        bool protective) {
    std::vector<double> candidates;
    std::vector<float> centroid_values(KEY_WIDTH, 0.0f);
    for (size_t index = 0; index < values.size(); ++index) {
        if (!protective && SYMBOLS[index] == excluded_source) continue;
        candidates.push_back(values[index]);
        const auto & prototype = prototypes.at(SYMBOLS[index]);
        for (size_t dimension = 0; dimension < prototype.size(); ++dimension) {
            centroid_values[dimension] += prototype[dimension];
        }
    }
    if (candidates.empty()) throw std::runtime_error("empty context candidate set");
    std::sort(candidates.begin(), candidates.end(), std::greater<double>());
    const size_t top_count = std::min<size_t>(3, candidates.size());
    const double top3 = std::accumulate(candidates.begin(), candidates.begin() + top_count, 0.0) / top_count;
    const double max_value = candidates.front();
    const double centroid_similarity = cosine(query_values, normalized(centroid_values));
    out << std::setprecision(17)
        << "{\"row_kind\":\"" << row_kind
        << "\",\"context_type\":\"" << context_type
        << "\",\"role\":\"" << query.observed.role
        << "\",\"variant\":\"" << json_escape(query.observed.variant)
        << "\",\"source\":\"" << query.observed.source
        << "\",\"excluded_source\":\"" << (protective ? "" : query.observed.source)
        << "\",\"similarities\":";
    write_double_array(out, values);
    out << ",\"s_max\":" << max_value
        << ",\"s_centroid\":" << centroid_similarity
        << ",\"s_top3\":" << top3
        << ",\"protective\":" << json_bool(protective) << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const options opts = parse_args(argc, argv);
        const std::filesystem::path root(opts.output_dir);
        std::filesystem::create_directories(root);
        const char * step17_env = std::getenv("STEP17_ARTIFACT_DIR");
        const std::filesystem::path step17_root = step17_env
            ? std::filesystem::path(step17_env)
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
        const auto persisted_k1 = load_persisted_k1_local(step17_root);

        const std::string stored_template = "Observed transition memory.\nSource symbol: {source}\nDestination symbol:";
        const auto stored_prompt = [&](const std::string & source) {
            return "Observed transition memory.\nSource symbol: " + source + "\nDestination symbol:";
        };
        const auto canonical_prompt = [](const std::string & source) { return source; };

        std::map<std::string, key_observation> stored;
        std::map<std::string, key_observation> canonical;
        for (const auto & source : SYMBOLS) {
            const auto prompt = stored_prompt(source);
            stored.emplace(source, evaluate_key(ctx, vocab, collector, prompt, source,
                                                "stored_source", "original", (int32_t) stored.size()));
            canonical.emplace(source, evaluate_key(ctx, vocab, collector, canonical_prompt(source), source,
                                                   "canonical_source", "identity_only", (int32_t) canonical.size()));
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

        std::vector<query_record> query_records;
        int32_t query_id = 0;
        for (const auto & [variant, role, prompt] : queries) {
            const std::string source = variant.find("_A_") != std::string::npos ? "A" : "B";
            auto observed = evaluate_key(ctx, vocab, collector, prompt, source, role, variant, query_id);
            const auto query_only = canonical.at(source).final_hidden;
            const auto delta = subtract(observed.final_hidden, query_only);
            const auto c0 = normalized(observed.final_hidden);
            const auto c1 = normalized(delta);
            const auto k1 = normalized(observed.source_hidden);
            query_records.push_back({std::move(observed), query_only, c0, delta, c1, k1});
            ++query_id;
        }

        std::map<size_t, final_observation> protective;
        for (size_t index = 0; index < PROTECTIVE_PROMPTS.size(); ++index) {
            protective.emplace(index, evaluate_final_only(ctx, vocab, collector, PROTECTIVE_PROMPTS[index],
                                                          "protective", std::to_string(index), "", (int32_t) index));
        }

        std::map<std::string, std::vector<float>> c0_prototypes;
        std::map<std::string, std::vector<float>> c1_prototypes;
        for (const auto & source : SYMBOLS) {
            c0_prototypes[source] = normalized(stored.at(source).final_hidden);
            c1_prototypes[source] = normalized(subtract(stored.at(source).final_hidden,
                                                         canonical.at(source).final_hidden));
        }

        std::ofstream vector_bin(root / "vectors.f32", std::ios::binary | std::ios::trunc);
        std::ofstream vector_meta(root / "vector_metadata.jsonl", std::ios::trunc);
        std::ofstream observation_out(root / "observation_metadata.jsonl", std::ios::trunc);
        std::ofstream prototype_out(root / "prototype_metadata.jsonl", std::ios::trunc);
        std::ofstream matrix_out(root / "similarity_matrices.jsonl", std::ios::trunc);
        if (!vector_bin || !vector_meta || !observation_out || !prototype_out || !matrix_out) {
            throw std::runtime_error("cannot open Step-19 artifacts");
        }

        for (const auto & source : SYMBOLS) {
            const auto & item = stored.at(source);
            prototype_out << "{\"group\":\"stored_source\",\"source\":\"" << source
                          << "\",\"prompt\":\"" << json_escape(item.prompt)
                          << "\",\"prompt_bytes_hex\":\"" << hex_bytes(item.prompt)
                          << "\",\"prompt_token_ids\":";
            write_tokens(prototype_out, item.prompt_tokens);
            prototype_out << "}\n";
            write_vector(vector_bin, vector_meta, item.id, "stored_source", item.role, item.variant,
                         source, item.prompt, item.prompt_tokens, "final_hidden_raw", item.final_hidden);
            write_vector(vector_bin, vector_meta, item.id, "stored_source", item.role, item.variant,
                         source, item.prompt, item.prompt_tokens, "context_c0", c0_prototypes.at(source));
            write_vector(vector_bin, vector_meta, item.id, "stored_source", item.role, item.variant,
                         source, item.prompt, item.prompt_tokens, "context_c1_delta_raw",
                         subtract(item.final_hidden, canonical.at(source).final_hidden));
            write_vector(vector_bin, vector_meta, item.id, "stored_source", item.role, item.variant,
                         source, item.prompt, item.prompt_tokens, "context_c1", c1_prototypes.at(source));
            write_vector(vector_bin, vector_meta, item.id, "stored_source", item.role, item.variant,
                         source, item.prompt, item.prompt_tokens, "K1_source_token_hidden", persisted_k1.at(source));

            const auto & identity = canonical.at(source);
            prototype_out << "{\"group\":\"canonical_source\",\"source\":\"" << source
                          << "\",\"prompt\":\"" << json_escape(identity.prompt)
                          << "\",\"prompt_bytes_hex\":\"" << hex_bytes(identity.prompt)
                          << "\",\"prompt_token_ids\":";
            write_tokens(prototype_out, identity.prompt_tokens);
            prototype_out << "}\n";
            write_vector(vector_bin, vector_meta, identity.id, "canonical_source", identity.role, identity.variant,
                         source, identity.prompt, identity.prompt_tokens, "final_hidden_raw", identity.final_hidden);
        }

        for (const auto & query : query_records) {
            const auto & item = query.observed;
            observation_out << "{\"id\":" << item.id
                           << ",\"role\":\"" << item.role
                           << "\",\"variant\":\"" << json_escape(item.variant)
                           << "\",\"source\":\"" << item.source
                           << "\",\"prompt\":\"" << json_escape(item.prompt)
                           << "\",\"prompt_bytes_hex\":\"" << hex_bytes(item.prompt)
                           << "\",\"prompt_token_ids\":";
            write_tokens(observation_out, item.prompt_tokens);
            observation_out << ",\"source_position\":" << item.location.position
                           << ",\"actual_source_token_id\":" << item.location.actual_token
                           << ",\"canonical_source_token_id\":" << item.location.canonical_token
                           << ",\"actual_source_piece\":\"" << json_escape(item.location.actual_piece)
                           << "\",\"source_match_method\":\"" << item.location.match_method << "\"}\n";
            write_vector(vector_bin, vector_meta, item.id, "query", item.role, item.variant, item.source,
                         item.prompt, item.prompt_tokens, "final_hidden_raw", item.final_hidden);
            write_vector(vector_bin, vector_meta, item.id, "query", item.role, item.variant, item.source,
                         item.prompt, item.prompt_tokens, "query_source_only_final_hidden_raw", query.query_only_final);
            write_vector(vector_bin, vector_meta, item.id, "query", item.role, item.variant, item.source,
                         item.prompt, item.prompt_tokens, "context_c0", query.c0);
            write_vector(vector_bin, vector_meta, item.id, "query", item.role, item.variant, item.source,
                         item.prompt, item.prompt_tokens, "context_c1_delta_raw", query.c1_delta_raw);
            write_vector(vector_bin, vector_meta, item.id, "query", item.role, item.variant, item.source,
                         item.prompt, item.prompt_tokens, "context_c1", query.c1);
            write_vector(vector_bin, vector_meta, item.id, "query", item.role, item.variant, item.source,
                         item.prompt, item.prompt_tokens, "K1_source_token_hidden", query.k1);

            const auto k1_values = similarities(query.k1, persisted_k1);
            const auto k1_order = [&]() {
                std::vector<size_t> order(SYMBOLS.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](size_t left, size_t right) { return k1_values[left] > k1_values[right]; });
                return order;
            }();
            matrix_out << std::setprecision(17)
                       << "{\"row_kind\":\"k1_identity\",\"role\":\"" << item.role
                       << "\",\"variant\":\"" << json_escape(item.variant)
                       << "\",\"source\":\"" << item.source << "\",\"similarities\":";
            write_double_array(matrix_out, k1_values);
            matrix_out << ",\"top1_source\":\"" << SYMBOLS[k1_order[0]]
                       << "\",\"top1_cosine\":" << k1_values[k1_order[0]]
                       << ",\"correct\":" << json_bool(SYMBOLS[k1_order[0]] == item.source) << "}\n";

            write_context_row(matrix_out, "query", query, "C0_full_final_state",
                              similarities(query.c0, c0_prototypes), query.c0, c0_prototypes,
                              item.source, false);
            write_context_row(matrix_out, "query", query, "C1_query_subtracted",
                              similarities(query.c1, c1_prototypes), query.c1, c1_prototypes,
                              item.source, false);
        }

        for (const auto & [index, item] : protective) {
            query_record pseudo;
            pseudo.observed.role = "protective";
            pseudo.observed.variant = std::to_string(index);
            pseudo.observed.source = "";
            const auto c0 = normalized(item.final_hidden);
            write_vector(vector_bin, vector_meta, item.id, "protective", item.role, item.variant, "",
                         item.prompt, item.prompt_tokens, "final_hidden_raw", item.final_hidden);
            write_vector(vector_bin, vector_meta, item.id, "protective", item.role, item.variant, "",
                         item.prompt, item.prompt_tokens, "context_c0", c0);
            write_context_row(matrix_out, "protective", pseudo, "C0_full_final_state",
                              similarities(c0, c0_prototypes), c0, c0_prototypes, "", true);
        }

        double reset_max_diff = 0;
        const auto repeat = evaluate_key(ctx, vocab, collector, stored_prompt("A"), "A",
                                         "stored_source", "reset_check", 10000);
        for (size_t index = 0; index < stored.at("A").final_hidden.size(); ++index) {
            reset_max_diff = std::max(reset_max_diff,
                std::abs((double) stored.at("A").final_hidden[index] - repeat.final_hidden[index]));
        }

        std::ofstream validation(root / "validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"stored_source_count\":8,\"canonical_source_count\":8"
                   << ",\"query_count\":16,\"protective_count\":8"
                   << ",\"matching_source_excluded\":true"
                   << ",\"reset_max_hidden_difference\":" << reset_max_diff
                   << ",\"qwen_weights_modified\":false,\"learning_replayed\":false"
                   << ",\"sidecar_injection\":false,\"gradients\":false,\"optimizer\":false}\n";
        std::ofstream runtime(root / "runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ",\"model_decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size
                << ",\"add_special\":true,\"parse_special\":true,\"clean_kv_per_forward\":true"
                << ",\"step17_artifact_dir\":\"" << json_escape(step17_root.string()) << "\"}\n";
        std::ofstream memory(root / "memory_metrics.json", std::ios::trunc);
        memory << "{\"dimension\":2560,\"stored_source_vectors\":40"
               << ",\"canonical_source_vectors\":8,\"query_vectors\":96"
               << ",\"protective_vectors\":16,\"total_vector_records\":160"
               << ",\"dense_plastic_matrix\":false,\"query_only_vectors_persisted\":true"
               << ",\"delta_vectors_persisted\":true}\n";
        std::ofstream config(root / "probe_metadata.json", std::ios::trunc);
        config << "{\"experiment\":\"qwen35_query_subtracted_context_probe\","
               << "\"stored_prompt_template\":\"" << json_escape(stored_template) << "\","
               << "\"canonical_source_prompt\":\"the source symbol string alone\","
               << "\"query_count\":16,\"protective_count\":8,\"stored_sources\":[\"A\",\"B\",\"C\",\"D\",\"E\",\"F\",\"G\",\"H\"],"
               << "\"context_definitions\":{\"C0\":\"normalize(final_hidden(query))\",\"C1\":\"normalize(final_hidden(query)-final_hidden(canonical_source))\"},"
               << "\"matching_source_excluded\":true,\"no_learning_or_injection\":true}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
