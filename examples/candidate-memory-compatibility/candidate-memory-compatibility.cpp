#define CENTERED_OUTCOME_RELEVANCE_LIBRARY
#include "../centered-outcome-relevance/step23_library.cpp"

#include <iostream>

namespace {

struct compatibility_spec {
    std::string id;
    std::string class_label;
    std::string expected_source;
    std::string prompt;
};

struct key_match_summary {
    std::string best_source;
    std::string second_source;
    double best = 0.0;
    double second = 0.0;
};

struct logit_summary {
    std::vector<float> logits;
    llama_token greedy = LLAMA_TOKEN_NULL;
    float greedy_logit = 0.0f;
};

std::vector<float> copy_logits(const llama_vocab * vocab, llama_context * ctx) {
    const int32_t count = llama_vocab_n_tokens(vocab);
    const float * values = llama_get_logits_ith(ctx, -1);
    if (!values || count <= 0) throw std::runtime_error("final logits are unavailable");
    return std::vector<float>(values, values + count);
}

logit_summary summarize_logits(const std::vector<float> & logits) {
    if (logits.empty()) throw std::runtime_error("empty logits");
    logit_summary out;
    out.logits = logits;
    out.greedy = 0;
    for (llama_token token = 1; token < (llama_token) logits.size(); ++token) {
        if (logits[token] > logits[out.greedy]) out.greedy = token;
    }
    out.greedy_logit = logits[out.greedy];
    return out;
}

double token_probability(const std::vector<float> & logits, llama_token token) {
    if (token < 0 || token >= (llama_token) logits.size()) return 0.0;
    const float maximum = *std::max_element(logits.begin(), logits.end());
    long double denominator = 0.0;
    for (const float value : logits) denominator += std::exp((long double) value - maximum);
    return (double) (std::exp((long double) logits[token] - maximum) / denominator);
}

double vector_norm(const std::vector<float> & values) {
    long double squared = 0.0;
    for (const float value : values) squared += (long double) value * value;
    return std::sqrt((double) squared);
}

int token_rank(const std::vector<float> & logits, llama_token token) {
    if (token < 0 || token >= (llama_token) logits.size()) return -1;
    int rank = 1;
    for (llama_token candidate = 0; candidate < (llama_token) logits.size(); ++candidate) {
        if (logits[candidate] > logits[token]) ++rank;
    }
    return rank;
}

llama_token single_token(const llama_vocab * vocab, const std::string & text) {
    const auto tokens = tokenize(vocab, text, false);
    if (tokens.size() != 1) {
        throw std::runtime_error("expected one token for destination: " + text);
    }
    return tokens.front();
}

std::string byte_hex(const std::string & text) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(text.size()*2);
    for (const unsigned char byte : text) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

std::vector<compatibility_spec> heldout_specs() {
    static const std::array<const char *, 8> applicable_templates = {
        "Resolve the memorized pair whose source is X; destination:",
        "Complete the stored source-destination pair beginning with X:",
        "Use associative memory to continue from source X:",
        "The remembered counterpart assigned to source X is:",
        "For the stored transition starting at X, output its destination:",
        "Complete the memorized association X =>",
        "Recall the paired destination for source X:",
        "Continue X according to the previously acquired association:"
    };
    static const std::array<const char *, 8> nonapplicable_templates = {
        "Interpret X only as a literal character; description:",
        "Repeat the displayed source character X:",
        "Discuss the symbol X without completing any stored pair:",
        "Do not retrieve an association; the literal source shown is X:",
        "Treat X as ordinary text rather than a remembered mapping:",
        "The character currently displayed is X; repeat it:",
        "Ignore associative memory and describe the symbol X:",
        "Do not continue any learned transition from X; source:"
    };

    std::vector<compatibility_spec> specs;
    for (const std::string & source : {std::string("A"), std::string("B")}) {
        for (size_t index = 0; index < applicable_templates.size(); ++index) {
            std::string prompt = applicable_templates[index];
            const size_t position = prompt.find('X');
            if (position == std::string::npos) throw std::runtime_error("template lacks X");
            prompt.replace(position, 1, source);
            specs.push_back({"applicable_" + source + "_" + std::to_string(index + 1),
                             "applicable", source, std::move(prompt)});
        }
    }
    for (const std::string & source : {std::string("A"), std::string("B")}) {
        for (size_t index = 0; index < nonapplicable_templates.size(); ++index) {
            std::string prompt = nonapplicable_templates[index];
            size_t position = prompt.find('X');
            if (position == std::string::npos) throw std::runtime_error("template lacks X");
            prompt.replace(position, 1, source);
            position = prompt.find('X');
            if (position != std::string::npos) prompt.replace(position, 1, source);
            specs.push_back({"non_applicable_" + source + "_" + std::to_string(index + 1),
                             "non_applicable", source, std::move(prompt)});
        }
    }
    return specs;
}

key_match_summary rank_keys(const std::vector<float> & query,
                            const std::map<std::string, std::vector<float>> & keys) {
    key_match_summary out;
    out.best = -std::numeric_limits<double>::infinity();
    out.second = -std::numeric_limits<double>::infinity();
    for (const auto & source : SYMBOLS) {
        const double value = cosine(query, keys.at(source));
        if (value > out.best) {
            out.second = out.best;
            out.second_source = out.best_source;
            out.best = value;
            out.best_source = source;
        } else if (value > out.second) {
            out.second = value;
            out.second_source = source;
        }
    }
    return out;
}

std::string mapped_destination(const std::string & source, bool shuffled) {
    if (source == "A") return shuffled ? "E" : "D";
    if (source == "B") return shuffled ? "D" : "E";
    throw std::runtime_error("compatibility requires source A or B: " + source);
}

void write_token_json(std::ostream & out, const llama_vocab * vocab, llama_token token) {
    out << "{\"id\":" << token
        << ",\"piece\":\"" << json_escape(token_piece(vocab, token)) << "\"}";
}

void write_spec_json(std::ostream & out, const compatibility_spec & spec) {
    out << "{\"id\":\"" << json_escape(spec.id)
        << "\",\"class\":\"" << spec.class_label
        << "\",\"expected_source\":\"" << spec.expected_source
        << "\",\"prompt\":\"" << json_escape(spec.prompt)
        << "\",\"prompt_bytes_hex\":\"" << byte_hex(spec.prompt) << "\"}";
}

void write_floats(std::ostream & out, const std::vector<float> & values) {
    out << '[' << std::setprecision(9);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_string_array(std::ostream & out, const std::vector<std::string> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << '"' << json_escape(values[i]) << '"';
    }
    out << ']';
}

} // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const options opts = parse_args(argc, argv);
        const std::filesystem::path output_root(opts.output_dir);
        std::filesystem::create_directories(output_root);

        const char * stored_env = std::getenv("STEP17_ARTIFACT_DIR");
        const std::filesystem::path stored_root = stored_env
            ? std::filesystem::path(stored_env)
            : std::filesystem::path("/workspace/qwen35_context_stable_key_probe_raw");

        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.gpu_layers;
        llama_model * model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (!model) throw std::runtime_error("model load failed");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        auto & model_ref = *model;
        if (model_ref.hparams.n_embd != KEY_WIDTH || model_ref.hparams.n_layer() != 32) {
            throw std::runtime_error("expected Qwen3.5-4B dimensions 32x2560");
        }

        const llama_token token_d = single_token(vocab, "D");
        const llama_token token_e = single_token(vocab, "E");
        const std::map<std::string, std::vector<float>> stored_keys = step23_load_persisted_k1(stored_root);

        key_trace_collector collector;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.ctx_size;
        context_params.n_batch = opts.ctx_size;
        context_params.n_ubatch = opts.ctx_size;
        context_params.n_seq_max = 1;
        context_params.cb_eval = key_eval_callback;
        context_params.cb_eval_user_data = &collector;
        llama_context * ctx = llama_init_from_model(model, context_params);
        if (!ctx) throw std::runtime_error("context creation failed");

        std::ofstream prompts(output_root / "heldout_prompts.json", std::ios::trunc);
        std::ofstream k1_rows(output_root / "k1_rows.jsonl", std::ios::trunc);
        std::ofstream scores(output_root / "compatibility_scores.jsonl", std::ios::trunc);
        if (!prompts || !k1_rows || !scores) throw std::runtime_error("cannot open Step-25 output files");

        const auto specs = heldout_specs();
        prompts << "[\n";
        for (size_t index = 0; index < specs.size(); ++index) {
            if (index) prompts << ",\n";
            write_spec_json(prompts, specs[index]);
        }
        prompts << "\n]\n";

        double zero_reset_diff = 0.0;
        std::vector<float> reference_hidden;
        size_t decode_calls = 0;
        for (size_t index = 0; index < specs.size(); ++index) {
            const auto & spec = specs[index];
            const key_observation observation = evaluate_key(ctx, vocab, collector, spec.prompt,
                                                             spec.expected_source, spec.class_label,
                                                             spec.id, (int32_t) index);
            ++decode_calls;
            const auto key_matches = rank_keys(observation.k1, stored_keys);
            const auto logits = summarize_logits(copy_logits(vocab, ctx));
            const bool valid_source = key_matches.best_source == "A" || key_matches.best_source == "B";
            const std::string selected_source = valid_source ? key_matches.best_source : spec.expected_source;
            const std::string associated = mapped_destination(selected_source, false);
            const std::string shuffled = mapped_destination(selected_source, true);
            const llama_token associated_token = associated == "D" ? token_d : token_e;
            const llama_token shuffled_token = shuffled == "D" ? token_d : token_e;
            const double associated_probability = token_probability(logits.logits, associated_token);
            const double shuffled_probability = token_probability(logits.logits, shuffled_token);
            const double p_mem = token_probability(logits.logits, token_d) + token_probability(logits.logits, token_e);
            const double s_assoc = (double) logits.logits[associated_token] - logits.logits[shuffled_token];
            const double m_greedy = (double) logits.logits[associated_token] - logits.greedy_logit;
            if (index == 0) reference_hidden = observation.final_hidden;
            else if (index == specs.size() - 1) {
                const key_observation repeat = evaluate_key(ctx, vocab, collector, specs.front().prompt,
                                                            specs.front().expected_source, specs.front().class_label,
                                                            specs.front().id, 10000);
                ++decode_calls;
                for (size_t i = 0; i < reference_hidden.size(); ++i) {
                    zero_reset_diff = std::max(zero_reset_diff,
                        std::abs((double) reference_hidden[i] - repeat.final_hidden[i]));
                }
            }

            k1_rows << std::setprecision(17)
                    << "{\"id\":\"" << json_escape(spec.id)
                    << "\",\"class\":\"" << spec.class_label
                    << "\",\"expected_source\":\"" << spec.expected_source
                    << "\",\"k1_winner\":\"" << key_matches.best_source
                    << "\",\"k1_winner_cosine\":" << key_matches.best
                    << ",\"k1_second_source\":\"" << key_matches.second_source
                    << "\",\"k1_second_cosine\":" << key_matches.second
                    << ",\"k1_margin\":" << key_matches.best - key_matches.second
                    << ",\"k1_correct\":" << (key_matches.best_source == spec.expected_source ? "true" : "false")
                    << ",\"prompt_token_ids\":";
            write_tokens(k1_rows, observation.prompt_tokens);
            k1_rows << "}\n";

            scores << std::setprecision(17)
                   << "{\"id\":\"" << json_escape(spec.id)
                   << "\",\"class\":\"" << spec.class_label
                   << "\",\"prompt\":\"" << json_escape(spec.prompt)
                   << "\",\"prompt_bytes_hex\":\"" << byte_hex(spec.prompt)
                   << "\",\"expected_source\":\"" << spec.expected_source
                   << "\",\"k1_winner\":\"" << key_matches.best_source
                   << "\",\"k1_winner_cosine\":" << key_matches.best
                   << ",\"k1_second_source\":\"" << key_matches.second_source
                   << "\",\"k1_second_cosine\":" << key_matches.second
                   << ",\"k1_margin\":" << key_matches.best - key_matches.second
                   << ",\"k1_correct\":" << (key_matches.best_source == spec.expected_source ? "true" : "false")
                   << ",\"valid_primary\":" << (valid_source ? "true" : "false")
                   << ",\"associated_source\":\"" << selected_source
                   << "\",\"associated_destination\":\"" << associated
                   << "\",\"shuffled_destination\":\"" << shuffled
                   << "\",\"associated_token_id\":" << associated_token
                   << ",\"shuffled_token_id\":" << shuffled_token
                   << ",\"associated_logit\":" << logits.logits[associated_token]
                   << ",\"shuffled_logit\":" << logits.logits[shuffled_token]
                   << ",\"S_assoc\":" << s_assoc
                   << ",\"associated_probability\":" << associated_probability
                   << ",\"shuffled_probability\":" << shuffled_probability
                   << ",\"associated_rank\":" << token_rank(logits.logits, associated_token)
                   << ",\"shuffled_rank\":" << token_rank(logits.logits, shuffled_token)
                   << ",\"greedy_token\":" << logits.greedy
                   << ",\"greedy_logit\":" << logits.greedy_logit
                   << ",\"M_greedy\":" << m_greedy
                   << ",\"P_mem\":" << p_mem
                   << ",\"prompt_token_ids\":";
            write_tokens(scores, observation.prompt_tokens);
            scores << "}\n";
        }

        std::ofstream diagnostics(output_root / "source_diagnostics.json", std::ios::trunc);
        if (!diagnostics) throw std::runtime_error("cannot open source diagnostics");
        diagnostics << std::setprecision(17) << "{\"destination_tokens\":{";
        diagnostics << "\"D\":"; write_token_json(diagnostics, vocab, token_d);
        diagnostics << ",\"E\":"; write_token_json(diagnostics, vocab, token_e);
        diagnostics << "},\"neutral\":[";
        for (size_t index = 0; index < 2; ++index) {
            if (index) diagnostics << ',';
            const std::string source = index == 0 ? "A" : "B";
            const key_observation observation = evaluate_key(ctx, vocab, collector, source, source,
                                                             "neutral", "neutral_" + source, 20000 + (int32_t) index);
            ++decode_calls;
            const auto logits = summarize_logits(copy_logits(vocab, ctx));
            const llama_token associated_token = source == "A" ? token_d : token_e;
            const llama_token shuffled_token = source == "A" ? token_e : token_d;
            diagnostics << "{\"source\":\"" << source
                        << "\",\"prompt\":\"" << source
                        << "\",\"associated_logit\":" << logits.logits[associated_token]
                        << ",\"shuffled_logit\":" << logits.logits[shuffled_token]
                        << ",\"S_neutral\":" << (double) logits.logits[associated_token] - logits.logits[shuffled_token]
                        << ",\"greedy_token\":" << logits.greedy
                        << ",\"k1_norm\":" << vector_norm(observation.k1) << "}";
        }
        diagnostics << "]}\n";

        std::ofstream metadata(output_root / "probe_metadata.json", std::ios::trunc);
        metadata << "{\"experiment\":\"step25_candidate_memory_predictive_compatibility\","
                 << "\"model\":\"Qwen3.5-4B UD-Q3_K_XL\","
                 << "\"model_path\":\"" << json_escape(opts.model) << "\","
                 << "\"gpu_layers\":" << opts.gpu_layers << ",\"context_size\":" << opts.ctx_size
                 << ",\"add_special\":true,\"parse_special\":true,\"clean_kv_per_forward\":true,"
                 << "\"stored_k1_artifact\":\"" << json_escape(stored_root.string()) << "\","
                 << "\"mapping\":{\"A\":\"D\",\"B\":\"E\"},"
                 << "\"shuffled_mapping\":{\"A\":\"E\",\"B\":\"D\"},"
                 << "\"heldout_count\":" << specs.size() << ",\"destination_token_ids\":{"
                 << "\"D\":" << token_d << ",\"E\":" << token_e << "},"
                 << "\"required_target_logits_persisted\":true}\n";

        std::ofstream validation(output_root / "validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"runtime_preflight_required\":true,\"qwen_weights_modified\":false,"
                   << "\"sidecar_injection\":false,\"plastic_write\":false,\"learning\":false,"
                   << "\"gradients\":false,\"optimizer\":false,\"clean_kv_per_forward\":true,"
                   << "\"destination_D_single_token\":true,\"destination_E_single_token\":true,"
                   << "\"heldout_count\":" << specs.size()
                   << ",\"zero_reset_max_hidden_difference\":" << zero_reset_diff
                   << ",\"zero_reset_exact\":" << (zero_reset_diff == 0.0 ? "true" : "false") << "}\n";

        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream runtime(output_root / "runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17) << "{\"decode_calls\":" << decode_calls
                << ",\"wall_seconds\":" << elapsed << ",\"gpu_layers\":" << opts.gpu_layers
                << ",\"context_size\":" << opts.ctx_size << "}\n";
        std::ofstream memory(output_root / "memory_metrics.json", std::ios::trunc);
        memory << "{\"mutable_model_bytes\":0,\"sidecar_bytes\":0,\"plastic_components\":0}\n";

        llama_free(ctx);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "candidate-memory-compatibility: " << error.what() << '\n';
        return 1;
    }
}
