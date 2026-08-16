#define main context_stable_key_probe_included_main
#include "../injection-depth-threshold/injection-depth-threshold.cpp"
#undef main

#include <cctype>
#include <cstring>

namespace {

constexpr int32_t KEY_WIDTH = 2560;
const std::array<std::string, 8> SYMBOLS = {"A", "B", "C", "D", "E", "F", "G", "H"};

struct key_trace_collector {
    bool enabled = false;
    bool capture_source = false;
    bool capture_final = false;
    std::vector<float> source_hidden;
    std::vector<float> final_hidden;

    void clear() {
        capture_source = false;
        capture_final = false;
        source_hidden.clear();
        final_hidden.clear();
    }
};

std::vector<float> copy_vector_at(ggml_tensor * tensor, int64_t position) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error("unsupported result_norm tensor type");
    }
    const int64_t count = tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
    if (position < 0 || position >= count) {
        throw std::runtime_error("requested token position is absent from result_norm tensor");
    }
    std::vector<uint8_t> copied;
    const uint8_t * data = nullptr;
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        data = static_cast<const uint8_t *>(tensor->data);
    } else {
        copied.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, copied.data(), 0, copied.size());
        data = copied.data();
    }
    const int64_t i1 = position % tensor->ne[1];
    const int64_t rem = position / tensor->ne[1];
    const int64_t i2 = rem % tensor->ne[2];
    const int64_t i3 = rem / tensor->ne[2];
    std::vector<float> out(tensor->ne[0]);
    for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
        const size_t offset = i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0];
        out[i0] = tensor_value(data, tensor->type, offset);
    }
    return out;
}

bool key_eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & collector = *static_cast<key_trace_collector *>(user_data);
    if (!collector.enabled || std::strcmp(tensor->name, "result_norm") != 0) return false;
    if (ask) return true;
    // Incremental decoding exposes the current token as row zero. Capture only
    // the rows requested by the caller for this decode step.
    if (collector.capture_source) collector.source_hidden = copy_vector_at(tensor, 0);
    if (collector.capture_final) collector.final_hidden = copy_vector_at(tensor, 0);
    return true;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(128);
    int32_t length = llama_token_to_piece(vocab, token, buffer.data(), (int32_t) buffer.size(), 0, true);
    if (length < 0) {
        buffer.resize((size_t) -length + 1);
        length = llama_token_to_piece(vocab, token, buffer.data(), (int32_t) buffer.size(), 0, true);
    }
    if (length < 0) throw std::runtime_error("failed to decode token piece");
    return std::string(buffer.data(), (size_t) length);
}

std::string trim_ascii(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    while (!value.empty() && is_space((unsigned char) value.front())) value.erase(value.begin());
    while (!value.empty() && is_space((unsigned char) value.back())) value.pop_back();
    return value;
}

llama_token canonical_symbol_token(const llama_vocab * vocab, const std::string & symbol) {
    const auto tokens = tokenize(vocab, symbol, false);
    if (tokens.size() != 1) throw std::runtime_error("symbol is not exactly one token: " + symbol);
    return tokens[0];
}

struct source_location {
    int64_t position = -1;
    llama_token actual_token = -1;
    std::string actual_piece;
    llama_token canonical_token = -1;
    std::string match_method;
};

source_location locate_source(
        const llama_vocab * vocab,
        const std::vector<llama_token> & prompt_tokens,
        const std::string & symbol) {
    const llama_token canonical = canonical_symbol_token(vocab, symbol);
    for (size_t index = 0; index < prompt_tokens.size(); ++index) {
        if (prompt_tokens[index] == canonical) {
            return {(int64_t) index, prompt_tokens[index], token_piece(vocab, prompt_tokens[index]), canonical, "canonical_token_id"};
        }
    }
    for (size_t index = 0; index < prompt_tokens.size(); ++index) {
        const std::string piece = token_piece(vocab, prompt_tokens[index]);
        if (trim_ascii(piece) == symbol) {
            return {(int64_t) index, prompt_tokens[index], piece, canonical, "trimmed_piece"};
        }
    }
    throw std::runtime_error("could not locate source token in prompt: " + symbol);
}

struct key_observation {
    int32_t id = -1;
    std::string role;
    std::string variant;
    std::string prompt;
    std::string source;
    std::vector<llama_token> prompt_tokens;
    source_location location;
    std::vector<float> final_hidden;
    std::vector<float> source_hidden;
    std::vector<float> k0;
    std::vector<float> k1;
    std::vector<float> k2;
    int32_t greedy = -1;
};

std::vector<float> load_embedding_row(const llama_model & model, llama_token token) {
    const ggml_tensor * tensor = model.tok_embd;
    if (!tensor || tensor->ne[0] != KEY_WIDTH || token < 0 || token >= tensor->ne[1]) {
        throw std::runtime_error("invalid token embedding tensor or token id");
    }
    const auto * traits = ggml_get_type_traits(tensor->type);
    if (!traits || !traits->to_float) throw std::runtime_error("token embedding cannot be dequantized");
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    std::vector<float> out(KEY_WIDTH);
    traits->to_float(raw.data() + token*tensor->nb[1], out.data(), KEY_WIDTH);
    return out;
}

key_observation evaluate_key(
        llama_context * ctx,
        const llama_vocab * vocab,
        key_trace_collector & collector,
        const std::string & prompt,
        const std::string & source,
        const std::string & role,
        const std::string & variant,
        int32_t id) {
    auto tokens = tokenize(vocab, prompt, true);
    const auto location = locate_source(vocab, tokens, source);
    llama_memory_clear(llama_get_memory(ctx), true);
    collector.clear();
    collector.enabled = true;
    if (tokens.empty()) {
        collector.enabled = false;
        throw std::runtime_error("empty prompt");
    }
    // Decode one token at a time so result_norm contains the causal state for
    // the source token itself. A single full-prompt decode only exposes the
    // final/logit-selected row through this callback in the current runtime.
    for (size_t position = 0; position < tokens.size(); ++position) {
        collector.capture_source = (int64_t) position == location.position;
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
    collector.capture_source = false;
    collector.capture_final = false;
    collector.enabled = false;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) throw std::runtime_error("missing logits");
    std::vector<float> values(logits, logits + n_vocab);
    if (collector.source_hidden.size() != KEY_WIDTH || collector.final_hidden.size() != KEY_WIDTH) {
        throw std::runtime_error("missing source/final hidden state");
    }
    key_observation out;
    out.id = id;
    out.role = role;
    out.variant = variant;
    out.prompt = prompt;
    out.source = source;
    out.prompt_tokens = tokens;
    out.location = location;
    out.final_hidden = std::move(collector.final_hidden);
    out.source_hidden = std::move(collector.source_hidden);
    out.k0 = normalized(out.final_hidden);
    out.k1 = normalized(out.source_hidden);
    out.greedy = (int32_t) std::distance(values.begin(), std::max_element(values.begin(), values.end()));
    return out;
}

double cosine(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("key dimension mismatch");
    double dot = 0;
    double left_norm = 0;
    double right_norm = 0;
    for (size_t index = 0; index < left.size(); ++index) {
        dot += (double) left[index]*right[index];
        left_norm += (double) left[index]*left[index];
        right_norm += (double) right[index]*right[index];
    }
    if (left_norm <= 0 || right_norm <= 0) return 0;
    return dot/std::sqrt(left_norm*right_norm);
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

void write_tokens(std::ostream & out, const std::vector<llama_token> & tokens) {
    out << '[';
    for (size_t index = 0; index < tokens.size(); ++index) out << (index ? "," : "") << tokens[index];
    out << ']';
}

void write_float_array(std::ostream & out, const std::vector<float> & values) {
    out << '[' << std::setprecision(9);
    for (size_t index = 0; index < values.size(); ++index) out << (index ? "," : "") << values[index];
    out << ']';
}

const std::vector<float> & key_vector(const key_observation & observation, const std::string & type) {
    if (type == "K0_final_state") return observation.k0;
    if (type == "K1_source_token_hidden") return observation.k1;
    throw std::runtime_error("K2 is stored separately");
}

} // namespace

#ifndef CONTEXT_STABLE_KEY_PROBE_LIBRARY
int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const options opts = parse_args(argc, argv);
        const std::filesystem::path root(opts.output_dir);
        std::filesystem::create_directories(root);
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

        std::map<std::string, std::vector<float>> static_embeddings;
        std::map<std::string, llama_token> canonical_tokens;
        for (const auto & symbol : SYMBOLS) {
            canonical_tokens[symbol] = canonical_symbol_token(vocab, symbol);
            static_embeddings[symbol] = normalized(load_embedding_row(model, canonical_tokens.at(symbol)));
        }

        std::vector<key_observation> observations;
        const auto source_prompt = [](const std::string & symbol) {
            return "Observed transition memory.\nSource symbol: " + symbol + "\nDestination symbol:";
        };
        int32_t id = 0;
        for (const auto & source : SYMBOLS) {
            observations.push_back(evaluate_key(ctx, vocab, collector, source_prompt(source), source,
                                                "stored_source", "original", id++));
        }
        const std::array<std::tuple<std::string, std::string, std::string>, 8> variants = {{
            {"A_arrow", "A", "A ->"}, {"A_input", "A", "Input: A"},
            {"A_symbol", "A", "Symbol: A"}, {"A_current", "A", "Current symbol: A"},
            {"B_arrow", "B", "B ->"}, {"B_input", "B", "Input: B"},
            {"B_symbol", "B", "Symbol: B"}, {"B_current", "B", "Current symbol: B"},
        }};
        for (const auto & [variant, source, prompt] : variants) {
            observations.push_back(evaluate_key(ctx, vocab, collector, prompt, source,
                                                "context_variant", variant, id++));
        }
        for (auto & observation : observations) observation.k2 = static_embeddings.at(observation.source);

        std::ofstream vector_bin(root/"vectors.f32", std::ios::binary | std::ios::trunc);
        std::ofstream vector_meta(root/"vector_metadata.jsonl", std::ios::trunc);
        if (!vector_bin || !vector_meta) throw std::runtime_error("cannot open vector artifacts");
        const auto write_vector = [&](const key_observation & observation, const std::string & type,
                                      const std::vector<float> & values) {
            const std::streamoff offset = vector_bin.tellp();
            vector_bin.write(reinterpret_cast<const char *>(values.data()), (std::streamsize) values.size()*sizeof(float));
            vector_meta << "{\"observation_id\":" << observation.id
                        << ",\"role\":\"" << json_escape(observation.role)
                        << "\",\"variant\":\"" << json_escape(observation.variant)
                        << "\",\"source\":\"" << observation.source
                        << "\",\"vector_type\":\"" << type
                        << "\",\"byte_offset\":" << offset
                        << ",\"dimension\":" << values.size() << "}\n";
        };
        for (const auto & observation : observations) {
            write_vector(observation, "final_hidden_raw", observation.final_hidden);
            write_vector(observation, "source_hidden_raw", observation.source_hidden);
            write_vector(observation, "K0_final_state", observation.k0);
            write_vector(observation, "K1_source_token_hidden", observation.k1);
            write_vector(observation, "K2_static_source_embedding", observation.k2);
        }

        std::ofstream observation_out(root/"observation_metadata.jsonl", std::ios::trunc);
        std::ofstream matrix_out(root/"similarity_matrices.jsonl", std::ios::trunc);
        if (!observation_out || !matrix_out) throw std::runtime_error("cannot open metadata artifacts");
        for (const auto & observation : observations) {
            observation_out << "{\"observation_id\":" << observation.id
                           << ",\"role\":\"" << json_escape(observation.role)
                           << "\",\"variant\":\"" << json_escape(observation.variant)
                           << "\",\"source\":\"" << observation.source
                           << "\",\"prompt\":\"" << json_escape(observation.prompt)
                           << "\",\"prompt_token_ids\":";
            write_tokens(observation_out, observation.prompt_tokens);
            observation_out << ",\"source_position\":" << observation.location.position
                           << ",\"actual_source_token_id\":" << observation.location.actual_token
                           << ",\"canonical_source_token_id\":" << observation.location.canonical_token
                           << ",\"actual_source_piece\":\"" << json_escape(observation.location.actual_piece)
                           << "\",\"source_match_method\":\"" << observation.location.match_method
                           << "\",\"greedy_token\":" << observation.greedy << "}\n";
        }

        const std::array<std::string, 3> key_types = {
            "K0_final_state", "K1_source_token_hidden", "K2_static_source_embedding",
        };
        for (const auto & key_type : key_types) {
            std::map<std::string, std::vector<float>> stored;
            for (size_t index = 0; index < SYMBOLS.size(); ++index) {
                if (key_type == "K2_static_source_embedding") stored[SYMBOLS[index]] = static_embeddings.at(SYMBOLS[index]);
                else stored[SYMBOLS[index]] = key_vector(observations[index], key_type);
            }
            for (const auto & observation : observations) {
                const std::vector<float> & query = key_type == "K2_static_source_embedding"
                    ? static_embeddings.at(observation.source) : key_vector(observation, key_type);
                std::vector<double> similarities;
                for (const auto & source : SYMBOLS) similarities.push_back(cosine(query, stored.at(source)));
                std::vector<size_t> order(SYMBOLS.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
                    return similarities[left] > similarities[right];
                });
                const size_t first = order[0];
                const size_t second = order[1];
                matrix_out << std::setprecision(17)
                           << "{\"key_type\":\"" << key_type
                           << "\",\"observation_id\":" << observation.id
                           << ",\"role\":\"" << observation.role
                           << "\",\"variant\":\"" << json_escape(observation.variant)
                           << "\",\"expected_source\":\"" << observation.source
                           << "\",\"similarities\":";
                write_float_array(matrix_out, std::vector<float>(similarities.begin(), similarities.end()));
                matrix_out << ",\"top1_source\":\"" << SYMBOLS[first]
                           << "\",\"top1_cosine\":" << similarities[first]
                           << ",\"top2_source\":\"" << SYMBOLS[second]
                           << "\",\"top2_cosine\":" << similarities[second]
                           << ",\"margin\":" << similarities[first] - similarities[second]
                           << ",\"correct\":" << json_bool(SYMBOLS[first] == observation.source) << "}\n";
            }
        }

        double zero_diff = 0;
        const auto & reference = observations.front();
        const auto repeat = evaluate_key(ctx, vocab, collector, reference.prompt, reference.source,
                                         reference.role, reference.variant, 9999);
        // A second clean forward is the reset/zero-state check; keys are not injected in this probe.
        if (reference.final_hidden.size() != repeat.final_hidden.size()) throw std::runtime_error("reset vector size mismatch");
        for (size_t index = 0; index < reference.final_hidden.size(); ++index) {
            zero_diff = std::max(zero_diff, std::abs((double) reference.final_hidden[index] - repeat.final_hidden[index]));
        }

        std::ofstream validation(root/"validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"zero_state_max_hidden_difference\":" << zero_diff
                   << ",\"post_probe_reset_max_hidden_difference\":" << zero_diff
                   << ",\"qwen_weights_modified\":false,\"learning_replayed\":false"
                   << ",\"sidecar_injection\":false,\"gradients\":false,\"optimizer\":false"
                   << ",\"query_count\":16,\"stored_source_count\":8,\"key_types\":3}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ",\"model_decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size
                << ",\"clean_kv_per_forward\":true}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"normalized_vectors\":48,\"raw_vectors\":32,\"dimension\":2560,\"vectors_f32_bytes\":"
               << (48 + 32)*KEY_WIDTH*sizeof(float)
               << ",\"dense_plastic_matrix\":false}\n";
        std::ofstream config(root/"probe_metadata.json", std::ios::trunc);
        config << "{\"stored_prompt_template\":\"Observed transition memory.\\nSource symbol: {source}\\nDestination symbol:\","
               << "\"query_variants\":8,\"stored_sources\":[\"A\",\"B\",\"C\",\"D\",\"E\",\"F\",\"G\",\"H\"],"
               << "\"key_definitions\":{\"K0\":\"normalized final-layer result_norm at final prompt position\","
               << "\"K1\":\"normalized final-layer result_norm at actual source-token position\","
               << "\"K2\":\"normalized frozen token embedding row for canonical source token\"},"
               << "\"no_learning_or_injection\":true}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
#endif
