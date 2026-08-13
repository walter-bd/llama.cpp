#include "../injection-sensitivity-allocation/multi-injection-support.inc"

#include <cassert>
#include <set>

namespace {

constexpr double ETA = 0.05;
constexpr double AUTHORITY = 0.01;
constexpr double FP_TOLERANCE = 1e-6;

const std::array<std::string, 8> UPDATE_PROMPTS = {
    "A synthetic diagnostic asks for one arbitrary marker. Marker:",
    "Complete this neutral token probe with one marker:",
    "Return the assigned code symbol:",
    "Output one arbitrary control token:",
    "Neutral probe alpha. Response:",
    "Neutral probe beta. Response:",
    "Neutral probe gamma. Response:",
    "Neutral probe delta. Response:",
};

const std::array<std::string, 16> PROTECTIVE_PROMPTS = {
    "The capital of France is",
    "Two plus two equals",
    "A common week has this many days:",
    "Water freezes at zero degrees",
    "The season after spring is",
    "A triangle has this many sides:",
    "The opposite of narrow is",
    "Monday, Tuesday,",
    "The clear daytime sky is usually",
    "The largest ocean on Earth is the",
    "A baby cat is called a",
    "The chemical symbol for water is",
    "Ten divided by two equals",
    "The first month of the year is",
    "Plants use sunlight during",
    "The opposite of cold is",
};

struct bank_component {
    std::vector<float> value;
    std::vector<float> key;
    double coefficient = 0;
};

struct one_component_bank {
    std::vector<bank_component> components;

    void clear() { components.clear(); }

    std::vector<float> apply(const std::vector<float> & key) const {
        std::vector<float> out(WIDTH, 0.0f);
        for (const auto & component : components) {
            double activation = 0;
            for (int32_t i = 0; i < WIDTH; ++i) activation += (double) component.key[i]*key[i];
            const double scale = component.coefficient*activation;
            for (int32_t i = 0; i < WIDTH; ++i) out[i] += scale*component.value[i];
        }
        return out;
    }
};

struct prompt_state {
    int32_t id = 0;
    std::string prompt;
    std::vector<llama_token> prompt_tokens;
    llama_token target = -1;
    std::string target_text;
    eval_result baseline;
    std::vector<float> key;
};

struct applied_probe {
    std::vector<float> response;
    std::vector<float> direction;
    std::vector<float> projected_direction;
    multi_probe_result probe;
    double raw_response_norm = 0;
};

double vector_dot(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("dot-product dimension mismatch");
    double out = 0;
    for (size_t i = 0; i < left.size(); ++i) out += (double) left[i]*right[i];
    return out;
}

double row_dot(const output_matrix & output, llama_token token, const std::vector<float> & vector) {
    if ((int64_t) vector.size() != output.width) throw std::runtime_error("output-row dimension mismatch");
    double out = 0;
    const float * row = output.row(token);
    for (int64_t i = 0; i < output.width; ++i) out += (double) row[i]*vector[i];
    return out;
}

std::vector<float> normalized_output_row_sanity(const output_matrix & output, llama_token token) {
    std::vector<float> row(output.width);
    std::copy(output.row(token), output.row(token) + output.width, row.begin());
    return normalized(row);
}

std::string token_piece_sanity(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize(-count);
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    return count > 0 ? std::string(buffer.data(), count) : std::string();
}

void write_token_array(std::ostream & out, const std::vector<llama_token> & tokens) {
    out << '[';
    for (size_t i = 0; i < tokens.size(); ++i) out << (i ? "," : "") << tokens[i];
    out << ']';
}

void write_metrics_object(std::ostream & out, const std::vector<float> & logits, llama_token token) {
    const auto metrics = metrics_from_logits(logits, token);
    out << std::setprecision(17)
        << "{\"logit\":" << logits[token]
        << ",\"probability\":" << metrics.probability
        << ",\"nll\":" << metrics.nll
        << ",\"rank\":" << metrics.rank
        << ",\"margin\":" << metrics.margin
        << ",\"greedy_token\":" << metrics.greedy << '}';
}

void write_rho_fields_sanity(std::ostream & out, const multi_probe_result & probe) {
    auto get = [&](int32_t site) {
        const auto found = probe.actual_rhos.find(site);
        return found == probe.actual_rhos.end() ? 0.0 : found->second;
    };
    out << std::setprecision(17)
        << "\"B\":" << probe.actual_budget
        << ",\"rho24\":" << get(23)
        << ",\"rho28\":" << get(27)
        << ",\"rho_final\":" << get(-1);
}

applied_probe apply_component(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const one_component_bank & bank,
        const prompt_state & state) {
    applied_probe out;
    out.response = bank.apply(state.key);
    out.raw_response_norm = l2(out.response);
    if (out.raw_response_norm <= 1e-12) throw std::runtime_error("rank-one update produced zero same-key response");
    out.direction = normalized(out.response);
    out.projected_direction = output_projection(output, out.direction);
    item definition{state.id, state.prompt, state.target, state.target_text};
    case_state case_value{definition, state.baseline, out.direction, out.projected_direction};
    const multi_configuration config{"sanity", "block24_block28_final", {23, 27, -1}};
    const auto weights = normalized_weights({0.23072, 0.38322, 0.89438});
    out.probe = probe_multi(ctx, vocab, collector, case_value, config, weights, AUTHORITY);
    clear_internal_injection(ctx);
    if (out.probe.actual_budget > AUTHORITY + FP_TOLERANCE) {
        throw std::runtime_error("authority limit exceeded");
    }
    return out;
}

prompt_state make_state(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        int32_t id,
        const std::string & prompt,
        llama_token target,
        const std::string & target_text) {
    clear_internal_injection(ctx);
    auto baseline = evaluate(ctx, vocab, collector, prompt, target, true);
    baseline.metrics = metrics_from_logits(baseline.logits, target);
    return {id, prompt, tokenize(vocab, prompt, true), target, target_text,
            std::move(baseline), normalized(baseline.final_hidden)};
}

} // namespace

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
        if (model.hparams.n_embd != WIDTH || model.hparams.n_layer() != N_LAYERS) {
            throw std::runtime_error("diagnostic requires Qwen3.5-4B dimensions 32x2560");
        }

        trace_collector collector;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.ctx_size;
        context_params.n_batch = opts.ctx_size;
        context_params.n_ubatch = opts.ctx_size;
        context_params.n_seq_max = 1;
        context_params.cb_eval = eval_callback;
        context_params.cb_eval_user_data = &collector;
        llama_context * ctx = llama_init_from_model(public_model, context_params);
        if (!ctx) throw std::runtime_error("context creation failed");
        const llama_vocab * vocab = llama_model_get_vocab(public_model);
        const auto output = load_output_matrix(model);
        const auto target_candidates = choose_single_tokens(vocab);

        std::vector<prompt_state> updates;
        for (size_t index = 0; index < UPDATE_PROMPTS.size(); ++index) {
            clear_internal_injection(ctx);
            auto initial = evaluate(ctx, vocab, collector, UPDATE_PROMPTS[index], 0, true);
            auto candidate = target_candidates[(3*index + 1) % target_candidates.size()];
            for (size_t offset = 0; candidate.first == initial.metrics.greedy; ++offset) {
                candidate = target_candidates[(3*index + 2 + offset) % target_candidates.size()];
            }
            initial.metrics = metrics_from_logits(initial.logits, candidate.first);
            updates.push_back({(int32_t) index, UPDATE_PROMPTS[index], tokenize(vocab, UPDATE_PROMPTS[index], true),
                               candidate.first, candidate.second, std::move(initial), {}});
            updates.back().key = normalized(updates.back().baseline.final_hidden);
        }

        std::vector<prompt_state> protective;
        for (size_t index = 0; index < PROTECTIVE_PROMPTS.size(); ++index) {
            clear_internal_injection(ctx);
            auto baseline = evaluate(ctx, vocab, collector, PROTECTIVE_PROMPTS[index], 0, true);
            const llama_token target = baseline.metrics.greedy;
            baseline.metrics = metrics_from_logits(baseline.logits, target);
            protective.push_back({(int32_t) index, PROTECTIVE_PROMPTS[index],
                                  tokenize(vocab, PROTECTIVE_PROMPTS[index], true), target,
                                  token_piece_sanity(vocab, target), std::move(baseline), {}});
            protective.back().key = normalized(protective.back().baseline.final_hidden);
        }

        std::ofstream mechanics(root/"mechanics_results.jsonl", std::ios::trunc);
        std::ofstream protection(root/"protective_results.jsonl", std::ios::trunc);
        std::ofstream cross_key(root/"cross_key_results.jsonl", std::ios::trunc);
        std::ofstream resets(root/"reset_results.jsonl", std::ios::trunc);
        for (std::ofstream * stream : {&mechanics, &protection, &cross_key, &resets}) {
            stream->setf(std::ios::unitbuf);
        }

        double zero_state_max = 0;
        double reset_max = 0;
        double max_budget = 0;
        int32_t positive_margin_passes = 0;
        int32_t positive_nll_passes = 0;
        int32_t negative_logit_passes = 0;
        int32_t negative_dot_passes = 0;

        for (const auto & state : updates) {
            clear_internal_injection(ctx);
            const auto zero = evaluate(ctx, vocab, collector, state.prompt, state.target, false);
            zero_state_max = std::max(zero_state_max, max_logit_difference(zero.logits, state.baseline.logits));

            const llama_token winner = state.baseline.metrics.greedy;
            const auto positive_direction = normalized(output_row_difference(output, state.target, winner));
            const auto negative_behavior_direction = normalized_output_row_sanity(output, winner);
            std::vector<float> direction_difference(WIDTH);
            for (int32_t i = 0; i < WIDTH; ++i) {
                direction_difference[i] = positive_direction[i] - negative_behavior_direction[i];
            }
            const double direction_distance = l2(direction_difference);
            if (!(ETA > 0)) throw std::runtime_error("positive update coefficient is not positive");
            if (!(-ETA < 0)) throw std::runtime_error("negative update coefficient is not negative");
            if (direction_distance <= 1e-6) throw std::runtime_error("positive and negative directions are indistinguishable");

            for (const char * update_type : {"positive", "negative"}) {
                one_component_bank bank;
                const bool positive = std::string_view(update_type) == "positive";
                const std::vector<float> & value = positive ? positive_direction : negative_behavior_direction;
                const double coefficient = positive ? ETA : -ETA;
                bank.components.push_back({value, state.key, coefficient});
                const auto applied = apply_component(ctx, vocab, collector, output, bank, state);
                max_budget = std::max(max_budget, applied.probe.actual_budget);

                const auto target_before = metrics_from_logits(state.baseline.logits, state.target);
                const auto target_after = metrics_from_logits(applied.probe.logits, state.target);
                const double target_margin_delta = target_after.margin - target_before.margin;
                const double target_nll_delta = target_after.nll - target_before.nll;
                const double winner_logit_delta = applied.probe.logits[winner] - state.baseline.logits[winner];
                const double raw_dot_winner = row_dot(output, winner, applied.response);
                const double final_dot_winner = applied.probe.alphas.at(-1)*applied.projected_direction[winner];
                if (positive) {
                    positive_margin_passes += target_margin_delta > 0;
                    positive_nll_passes += target_nll_delta < 0;
                } else {
                    negative_logit_passes += winner_logit_delta < 0;
                    negative_dot_passes += final_dot_winner < 0;
                    if (!(raw_dot_winner < 0 && final_dot_winner < 0)) {
                        throw std::runtime_error("negative update is not suppressive in immutable LM-head space");
                    }
                }

                mechanics << std::setprecision(17)
                    << "{\"case_id\":" << state.id
                    << ",\"update_type\":\"" << update_type << "\""
                    << ",\"prompt\":\"" << json_escape(state.prompt) << "\""
                    << ",\"tokenized_prompt\":";
                write_token_array(mechanics, state.prompt_tokens);
                mechanics << ",\"baseline_greedy_token\":" << winner
                    << ",\"baseline_greedy_text\":\"" << json_escape(token_piece_sanity(vocab, winner)) << "\""
                    << ",\"target_token\":" << state.target
                    << ",\"target_text\":\"" << json_escape(state.target_text) << "\""
                    << ",\"key_norm\":" << l2(state.key)
                    << ",\"positive_direction_norm\":" << l2(positive_direction)
                    << ",\"negative_direction_norm\":" << l2(negative_behavior_direction)
                    << ",\"positive_negative_direction_distance\":" << direction_distance
                    << ",\"positive_coefficient\":" << ETA
                    << ",\"negative_coefficient\":" << -ETA
                    << ",\"applied_coefficient\":" << coefficient
                    << ",\"raw_Pz_norm\":" << applied.raw_response_norm
                    << ",\"raw_Pz_dot_winner_row\":" << raw_dot_winner
                    << ",\"final_injection_dot_winner_row\":" << final_dot_winner
                    << ",\"fixed_authority_after_nonzero_response\":true,";
                write_rho_fields_sanity(mechanics, applied.probe);
                mechanics << ",\"target_before\":";
                write_metrics_object(mechanics, state.baseline.logits, state.target);
                mechanics << ",\"target_after\":";
                write_metrics_object(mechanics, applied.probe.logits, state.target);
                mechanics << ",\"winner_before\":";
                write_metrics_object(mechanics, state.baseline.logits, winner);
                mechanics << ",\"winner_after\":";
                write_metrics_object(mechanics, applied.probe.logits, winner);
                mechanics << ",\"target_margin_delta\":" << target_margin_delta
                    << ",\"target_nll_delta\":" << target_nll_delta
                    << ",\"winner_logit_delta\":" << winner_logit_delta
                    << ",\"positive_margin_assertion\":" << (target_margin_delta > 0 ? "true" : "false")
                    << ",\"positive_nll_assertion\":" << (target_nll_delta < 0 ? "true" : "false")
                    << ",\"negative_logit_assertion\":" << (winner_logit_delta < 0 ? "true" : "false")
                    << ",\"negative_dot_assertion\":" << (final_dot_winner < 0 ? "true" : "false") << "}\n";

                for (const auto & heldout : updates) {
                    const auto response = bank.apply(heldout.key);
                    const double raw_norm = l2(response);
                    cross_key << std::setprecision(17)
                        << "{\"source_case_id\":" << state.id << ",\"update_type\":\"" << update_type
                        << "\",\"probe_category\":\"" << (heldout.id == state.id ? "same_key" : "other_update_key")
                        << "\",\"probe_id\":" << heldout.id
                        << ",\"key_cosine\":" << vector_dot(state.key, heldout.key)
                        << ",\"raw_Pz_norm\":" << raw_norm
                        << ",\"raw_activation_scalar\":" << vector_dot(state.key, heldout.key)
                        << ",\"post_normalization_B\":" << (raw_norm > 1e-12 ? AUTHORITY : 0.0)
                        << ",\"weak_retrieval_promoted_to_fixed_authority\":" << (raw_norm > 1e-12 ? "true" : "false") << "}\n";
                }
                for (const auto & heldout : protective) {
                    const auto response = bank.apply(heldout.key);
                    const double raw_norm = l2(response);
                    cross_key << std::setprecision(17)
                        << "{\"source_case_id\":" << state.id << ",\"update_type\":\"" << update_type
                        << "\",\"probe_category\":\"protective_key\",\"probe_id\":" << heldout.id
                        << ",\"key_cosine\":" << vector_dot(state.key, heldout.key)
                        << ",\"raw_Pz_norm\":" << raw_norm
                        << ",\"raw_activation_scalar\":" << vector_dot(state.key, heldout.key)
                        << ",\"post_normalization_B\":" << (raw_norm > 1e-12 ? AUTHORITY : 0.0)
                        << ",\"weak_retrieval_promoted_to_fixed_authority\":" << (raw_norm > 1e-12 ? "true" : "false") << "}\n";
                }

                for (const auto & heldout : protective) {
                    const auto plastic = apply_component(ctx, vocab, collector, output, bank, heldout);
                    const auto base_winner_metrics = metrics_from_logits(heldout.baseline.logits, heldout.target);
                    const auto plastic_winner_metrics = metrics_from_logits(plastic.probe.logits, heldout.target);
                    protection << std::setprecision(17)
                        << "{\"source_case_id\":" << state.id << ",\"update_type\":\"" << update_type
                        << "\",\"protective_prompt_id\":" << heldout.id
                        << ",\"protective_prompt\":\"" << json_escape(heldout.prompt) << "\""
                        << ",\"raw_Pz_norm\":" << plastic.raw_response_norm
                        << ",\"key_cosine\":" << vector_dot(state.key, heldout.key)
                        << ",\"kl_plastic_base\":" << kl_divergence(plastic.probe.logits, heldout.baseline.logits)
                        << ",\"top1_probability_before\":" << base_winner_metrics.probability
                        << ",\"top1_probability_after\":" << plastic_winner_metrics.probability
                        << ",\"top1_probability_change\":" << plastic_winner_metrics.probability - base_winner_metrics.probability
                        << ",\"baseline_greedy_token\":" << heldout.target
                        << ",\"plastic_greedy_token\":" << plastic.probe.metrics.greedy
                        << ",\"greedy_changed\":" << (plastic.probe.metrics.greedy != heldout.target ? "true" : "false") << ',';
                    write_rho_fields_sanity(protection, plastic.probe);
                    protection << "}\n";
                }

                bank.clear();
                clear_internal_injection(ctx);
                const auto restored_source = evaluate(ctx, vocab, collector, state.prompt, state.target, false);
                double update_reset_max = max_logit_difference(restored_source.logits, state.baseline.logits);
                for (int offset = 0; offset < 2; ++offset) {
                    const auto & heldout = protective[(state.id + offset) % protective.size()];
                    const auto restored = evaluate(ctx, vocab, collector, heldout.prompt, heldout.target, false);
                    update_reset_max = std::max(update_reset_max,
                                                max_logit_difference(restored.logits, heldout.baseline.logits));
                }
                reset_max = std::max(reset_max, update_reset_max);
                resets << std::setprecision(17)
                    << "{\"case_id\":" << state.id << ",\"update_type\":\"" << update_type
                    << "\",\"component_count_after_clear\":" << bank.components.size()
                    << ",\"max_logit_difference\":" << update_reset_max
                    << ",\"reset_exact\":" << (update_reset_max <= FP_TOLERANCE ? "true" : "false") << "}\n";
                if (update_reset_max > FP_TOLERANCE) throw std::runtime_error("post-update reset is not exact");
            }
        }

        clear_internal_injection(ctx);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"raw_validation.json", std::ios::trunc);
        validation << std::setprecision(17)
            << "{\"zero_state_max_logit_difference\":" << zero_state_max
            << ",\"reset_max_logit_difference\":" << reset_max
            << ",\"positive_coefficients_valid\":true"
            << ",\"negative_coefficients_valid\":true"
            << ",\"positive_negative_directions_distinct\":true"
            << ",\"positive_margin_passes\":" << positive_margin_passes
            << ",\"positive_nll_passes\":" << positive_nll_passes
            << ",\"negative_logit_passes\":" << negative_logit_passes
            << ",\"negative_dot_passes\":" << negative_dot_passes
            << ",\"max_global_authority\":" << max_budget
            << ",\"authority_limit\":" << AUTHORITY
            << ",\"base_model_mutated\":false,\"backpropagation\":false,\"optimizer\":false"
            << ",\"dense_plastic_matrix_allocated\":false,\"persistent_state_after_run\":false}\n";

        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
            << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":" << decode_calls
            << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size
            << ",\"update_cases\":8,\"protective_prompts\":16}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"active_rank_components\":1,\"rank_one_factor_bytes_fp32\":20480,"
               << "\"correction_direction_bytes_fp32\":10240,\"boundary_trace_bytes_fp32\":81920,"
               << "\"temporary_sparse_cvec_host_bytes\":317440,\"dense_plastic_matrix_bytes\":0,"
               << "\"model_and_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        manifest << "{\"experiment\":\"qwen35_teacher_update_sanity\",\"model\":\""
                 << json_escape(opts.model) << "\",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                 << ",\"eta\":0.05,\"authority\":0.01,\"injection_sites\":[\"block_24\",\"block_28\",\"final\"],"
                 << "\"raw_allocation_weights\":[0.23072,0.38322,0.89438],"
                 << "\"positive_update\":\"+eta*normalize(w_t-w_a)*zhat^T\","
                 << "\"negative_update\":\"-eta*normalize(w_a)*zhat^T\","
                 << "\"response_policy\":\"normalize_every_nonzero_Pz_then_apply_fixed_B\","
                 << "\"base_model_mutated\":false,\"backpropagation\":false,\"autograd\":false,\"optimizer\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
