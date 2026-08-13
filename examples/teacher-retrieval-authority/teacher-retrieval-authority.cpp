#include "../injection-sensitivity-allocation/multi-injection-support.inc"

#include <array>

namespace {

constexpr double ETA = 0.05;
constexpr double B_MAX = 0.01;
constexpr double S_REF = 0.05;
constexpr double FP_TOLERANCE = 1e-6;

const std::array<std::string, 4> UPDATE_PROMPTS = {
    "A synthetic diagnostic asks for one arbitrary marker. Marker:",
    "Complete this neutral token probe with one marker:",
    "Return the assigned code symbol:",
    "Output one arbitrary control token:",
};

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

struct rank_one_component {
    std::vector<float> value;
    std::vector<float> key;
    double coefficient = ETA;

    std::vector<float> apply(const std::vector<float> & probe_key) const {
        double activation = 0;
        for (int32_t i = 0; i < WIDTH; ++i) activation += (double) key[i]*probe_key[i];
        std::vector<float> out(WIDTH);
        const double scale = coefficient*activation;
        for (int32_t i = 0; i < WIDTH; ++i) out[i] = scale*value[i];
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

struct reader_probe {
    std::string reader;
    double raw_response_norm = 0;
    double retrieval_ratio = 0;
    double alpha = 0;
    double effective_budget = 0;
    multi_probe_result probe;
};

std::string token_piece_gate(const llama_vocab * vocab, llama_token token) {
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

void write_metrics(std::ostream & out, const std::vector<float> & logits, llama_token token) {
    const auto metrics = metrics_from_logits(logits, token);
    out << std::setprecision(17)
        << "{\"logit\":" << logits[token]
        << ",\"probability\":" << metrics.probability
        << ",\"nll\":" << metrics.nll
        << ",\"rank\":" << metrics.rank
        << ",\"margin\":" << metrics.margin
        << ",\"greedy_token\":" << metrics.greedy << '}';
}

void write_rhos(std::ostream & out, const multi_probe_result & probe) {
    auto rho = [&](int32_t site) {
        const auto found = probe.actual_rhos.find(site);
        return found == probe.actual_rhos.end() ? 0.0 : found->second;
    };
    out << std::setprecision(17)
        << "\"B\":" << probe.actual_budget
        << ",\"rho24\":" << rho(23)
        << ",\"rho28\":" << rho(27)
        << ",\"rho_final\":" << rho(-1);
}

reader_probe apply_reader(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const rank_one_component & component,
        const prompt_state & state,
        const std::string & reader) {
    const auto response = component.apply(state.key);
    const double raw_norm = l2(response);
    const double retrieval_ratio = raw_norm/S_REF;
    const double alpha = reader == "fixed" ? (raw_norm > 1e-12 ? 1.0 : 0.0)
                                            : std::min(1.0, retrieval_ratio);
    const double effective_budget = B_MAX*alpha;
    std::vector<float> direction(WIDTH, 0.0f);
    if (raw_norm > 1e-12) direction = normalized(response);
    const auto projected = output_projection(output, direction);
    item definition{state.id, state.prompt, state.target, state.target_text};
    case_state case_value{definition, state.baseline, direction, projected};
    const multi_configuration config{"reader", "block24_block28_final", {23, 27, -1}};
    const auto weights = normalized_weights({0.23072, 0.38322, 0.89438});
    auto probe = probe_multi(ctx, vocab, collector, case_value, config, weights, effective_budget);
    clear_internal_injection(ctx);
    if (probe.actual_budget > B_MAX + FP_TOLERANCE || effective_budget > B_MAX + FP_TOLERANCE) {
        throw std::runtime_error("retrieval reader exceeded authority limit");
    }
    return {reader, raw_norm, retrieval_ratio, alpha, effective_budget, std::move(probe)};
}

prompt_state make_prompt_state(
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
                                  token_piece_gate(vocab, target), std::move(baseline), {}});
            protective.back().key = normalized(protective.back().baseline.final_hidden);
        }

        std::ofstream targets(root/"target_results.jsonl", std::ios::trunc);
        std::ofstream protection(root/"protective_results.jsonl", std::ios::trunc);
        std::ofstream resets(root/"reset_results.jsonl", std::ios::trunc);
        for (std::ofstream * stream : {&targets, &protection, &resets}) stream->setf(std::ios::unitbuf);

        double zero_state_max = 0;
        double reset_max = 0;
        double max_effective_budget = 0;
        for (const auto & state : updates) {
            clear_internal_injection(ctx);
            const auto zero = evaluate(ctx, vocab, collector, state.prompt, state.target, false);
            zero_state_max = std::max(zero_state_max, max_logit_difference(zero.logits, state.baseline.logits));

            const llama_token winner = state.baseline.metrics.greedy;
            const auto positive_direction = normalized(output_row_difference(output, state.target, winner));
            if (!(ETA > 0)) throw std::runtime_error("positive update coefficient must be positive");
            rank_one_component component{positive_direction, state.key, ETA};

            for (const std::string reader : {"fixed", "gated"}) {
                const auto target_probe = apply_reader(ctx, vocab, collector, output, component, state, reader);
                max_effective_budget = std::max(max_effective_budget, target_probe.probe.actual_budget);
                const auto before = metrics_from_logits(state.baseline.logits, state.target);
                const auto after = metrics_from_logits(target_probe.probe.logits, state.target);
                targets << std::setprecision(17)
                    << "{\"case_id\":" << state.id
                    << ",\"reader\":\"" << reader << "\""
                    << ",\"prompt\":\"" << json_escape(state.prompt) << "\""
                    << ",\"tokenized_prompt\":";
                write_token_array(targets, state.prompt_tokens);
                targets << ",\"baseline_greedy_token\":" << winner
                    << ",\"baseline_greedy_text\":\"" << json_escape(token_piece_gate(vocab, winner)) << "\""
                    << ",\"target_token\":" << state.target
                    << ",\"target_text\":\"" << json_escape(state.target_text) << "\""
                    << ",\"key_norm\":" << l2(state.key)
                    << ",\"positive_direction_norm\":" << l2(positive_direction)
                    << ",\"positive_coefficient\":" << ETA
                    << ",\"raw_Pz_norm\":" << target_probe.raw_response_norm
                    << ",\"raw_retrieval_ratio\":" << target_probe.retrieval_ratio
                    << ",\"reader_alpha\":" << target_probe.alpha
                    << ",\"effective_B_requested\":" << target_probe.effective_budget << ',';
                write_rhos(targets, target_probe.probe);
                targets << ",\"target_before\":";
                write_metrics(targets, state.baseline.logits, state.target);
                targets << ",\"target_after\":";
                write_metrics(targets, target_probe.probe.logits, state.target);
                targets << ",\"target_margin_gain\":" << after.margin - before.margin
                    << ",\"target_nll_gain\":" << before.nll - after.nll << "}\n";

                for (const auto & heldout : protective) {
                    const auto plastic = apply_reader(ctx, vocab, collector, output, component, heldout, reader);
                    max_effective_budget = std::max(max_effective_budget, plastic.probe.actual_budget);
                    const auto base_top1 = metrics_from_logits(heldout.baseline.logits, heldout.target);
                    const auto plastic_top1 = metrics_from_logits(plastic.probe.logits, heldout.target);
                    protection << std::setprecision(17)
                        << "{\"source_case_id\":" << state.id
                        << ",\"reader\":\"" << reader << "\""
                        << ",\"protective_prompt_id\":" << heldout.id
                        << ",\"protective_prompt\":\"" << json_escape(heldout.prompt) << "\""
                        << ",\"raw_Pz_norm\":" << plastic.raw_response_norm
                        << ",\"raw_retrieval_ratio\":" << plastic.retrieval_ratio
                        << ",\"reader_alpha\":" << plastic.alpha
                        << ",\"effective_B_requested\":" << plastic.effective_budget
                        << ",\"kl_plastic_base\":" << kl_divergence(plastic.probe.logits, heldout.baseline.logits)
                        << ",\"top1_probability_before\":" << base_top1.probability
                        << ",\"top1_probability_after\":" << plastic_top1.probability
                        << ",\"top1_probability_change\":" << plastic_top1.probability - base_top1.probability
                        << ",\"baseline_greedy_token\":" << heldout.target
                        << ",\"plastic_greedy_token\":" << plastic.probe.metrics.greedy
                        << ",\"greedy_changed\":" << (plastic.probe.metrics.greedy != heldout.target ? "true" : "false") << ',';
                    write_rhos(protection, plastic.probe);
                    protection << "}\n";
                }
            }

            clear_internal_injection(ctx);
            const auto restored_source = evaluate(ctx, vocab, collector, state.prompt, state.target, false);
            double case_reset_max = max_logit_difference(restored_source.logits, state.baseline.logits);
            for (int32_t offset = 0; offset < 2; ++offset) {
                const auto & heldout = protective[(state.id + offset) % protective.size()];
                const auto restored = evaluate(ctx, vocab, collector, heldout.prompt, heldout.target, false);
                case_reset_max = std::max(case_reset_max,
                    max_logit_difference(restored.logits, heldout.baseline.logits));
            }
            reset_max = std::max(reset_max, case_reset_max);
            resets << std::setprecision(17)
                << "{\"case_id\":" << state.id
                << ",\"component_count_after_clear\":0"
                << ",\"max_logit_difference\":" << case_reset_max
                << ",\"reset_exact\":" << (case_reset_max <= FP_TOLERANCE ? "true" : "false") << "}\n";
            if (case_reset_max > FP_TOLERANCE) throw std::runtime_error("post-case reset is not exact");
        }

        clear_internal_injection(ctx);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"raw_validation.json", std::ios::trunc);
        validation << std::setprecision(17)
            << "{\"zero_state_max_logit_difference\":" << zero_state_max
            << ",\"reset_max_logit_difference\":" << reset_max
            << ",\"positive_coefficient_valid\":true"
            << ",\"max_effective_B\":" << max_effective_budget
            << ",\"authority_limit\":" << B_MAX
            << ",\"base_model_mutated\":false,\"gradients\":false,\"backpropagation\":false"
            << ",\"optimizer\":false,\"dense_plastic_matrix_allocated\":false"
            << ",\"persistent_state_after_run\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
            << "{\"wall_seconds\":" << elapsed
            << ",\"model_decode_calls\":" << decode_calls
            << ",\"gpu_layers\":" << opts.gpu_layers
            << ",\"ctx_size\":" << opts.ctx_size
            << ",\"update_cases\":4,\"protective_prompts\":8"
            << ",\"readers\":2}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"active_rank_components\":1,\"rank_one_factor_bytes_fp32\":20480,"
               << "\"correction_direction_bytes_fp32\":10240,\"boundary_trace_bytes_fp32\":81920,"
               << "\"temporary_sparse_cvec_host_bytes\":317440,\"dense_plastic_matrix_bytes\":0,"
               << "\"model_and_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        manifest << "{\"experiment\":\"qwen35_retrieval_authority_gate\",\"model\":\""
                 << json_escape(opts.model) << "\",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                 << ",\"eta\":0.05,\"B_max\":0.01,\"s_ref\":0.05,"
                 << "\"injection_sites\":[\"block_24\",\"block_28\",\"final\"],"
                 << "\"raw_allocation_weights\":[0.23072,0.38322,0.89438],"
                 << "\"positive_update\":\"+0.05*normalize(w_t-w_a)*zhat^T\","
                 << "\"fixed_reader\":\"B_eff=0.01 for every nonzero Pz\","
                 << "\"gated_reader\":\"B_eff=0.01*min(1,||Pz||/0.05)\","
                 << "\"base_model_mutated\":false,\"gradients\":false,\"optimizer\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
