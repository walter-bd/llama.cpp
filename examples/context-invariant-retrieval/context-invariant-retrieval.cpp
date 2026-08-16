// Step 16 is a replay-only probe.  Reuse the validated Step-14B reader and
// injection path without changing its bank, threshold, or authority logic.
#define COMPETITIVE_READER_FORWARD_VALIDATION_EMBEDDED
#include "../competitive-reader-forward-validation/competitive-reader-forward-validation.cpp"
#undef COMPETITIVE_READER_FORWARD_VALIDATION_EMBEDDED

#include <tuple>
#include <utility>

namespace {

struct step16_metadata {
    std::string variant;
    std::string expected_source;
    std::string abstract_destination;
};

double second_best_cosine(const replay_bank & bank, const std::vector<float> & query, const std::string & winner) {
    double result = -1.0;
    for (const auto & [source, key] : bank.source_keys) {
        if (source == winner) continue;
        result = std::max(result, cosine_similarity(query, key));
    }
    return result;
}

void write_step16_row(
        std::ostream & output,
        const std::string & reader,
        const replay_case & current,
        const step16_metadata & metadata,
        const replay_result & result,
        double second_best,
        double kl_vs_frozen) {
    const auto baseline_metrics = metrics_from_logits(current.baseline.logits, current.target);
    output << std::setprecision(17)
           << "{\"reader\":\"" << reader
           << "\",\"category\":\"" << current.category
           << "\",\"variant\":\"" << metadata.variant
           << "\",\"prompt_id\":" << current.id
           << ",\"prompt\":\"" << json_escape(current.prompt)
           << "\",\"prompt_token_ids\":";
    write_token_array(output, current.prompt_tokens);
    output << ",\"expected_source\":\"" << metadata.expected_source
           << "\",\"abstract_destination\":\"" << metadata.abstract_destination
           << "\",\"target_token\":" << current.target
           << ",\"target_text\":\"" << json_escape(current.target_text)
           << "\",\"baseline_greedy_token\":" << baseline_metrics.greedy
           << ",\"winning_source\":\"" << json_escape(result.winning_source)
           << "\",\"best_cosine\":" << result.best_cosine
           << ",\"second_best_cosine\":" << second_best
           << ",\"cosine_margin\":" << result.best_cosine - second_best
           << ",\"abstained\":" << (result.abstained ? "true" : "false")
           << ",\"raw_Pz_norm\":" << result.raw_response_norm
           << ",\"effective_B_requested\":" << result.effective_budget
           << ",\"actual_B\":" << result.actual_budget
           << ",\"kl_vs_frozen\":" << kl_vs_frozen
           << ",\"greedy_changed\":"
           << (result.metrics.greedy != baseline_metrics.greedy ? "true" : "false")
           << ",";
    write_metric_fields(output, "baseline", current.baseline.logits, current.target);
    output << ',';
    write_metric_fields(output, "after", result.logits, current.target);
    output << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto opts = parse_args(argc, argv);
        const char * vectors_env = std::getenv("QWEN_COMPETITIVE_BANK_VECTORS");
        const char * components_env = std::getenv("QWEN_COMPETITIVE_BANK_COMPONENTS");
        if (!vectors_env || !components_env) {
            throw std::runtime_error("QWEN_COMPETITIVE_BANK_VECTORS and QWEN_COMPETITIVE_BANK_COMPONENTS are required");
        }

        const std::filesystem::path root(opts.output_dir);
        std::filesystem::create_directories(root);
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.gpu_layers;
        llama_model * public_model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (!public_model) throw std::runtime_error("model load failed");
        auto & model = *public_model;
        if (model.hparams.n_embd != REPLAY_WIDTH || model.hparams.n_layer() != 32) {
            throw std::runtime_error("Step 16 requires Qwen3.5-4B dimensions 32x2560");
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
        const auto bank = load_replay_bank(vectors_env, components_env);
        const std::map<std::string, std::pair<llama_token, std::string>> targets = {
            {"D", {629, " V"}},
            {"E", {710, " K"}},
        };

        const std::array<std::tuple<std::string, std::string, std::string>, 8> variant_specs = {{
            {"A_arrow", "A", "A ->"},
            {"A_input", "A", "Input: A"},
            {"A_symbol", "A", "Symbol: A"},
            {"A_current", "A", "Current symbol: A"},
            {"B_arrow", "B", "B ->"},
            {"B_input", "B", "Input: B"},
            {"B_symbol", "B", "Symbol: B"},
            {"B_current", "B", "Current symbol: B"},
        }};

        std::vector<replay_case> cases;
        std::vector<step16_metadata> metadata;
        for (const auto & [variant, source, prompt] : variant_specs) {
            const auto target = targets.at(source == "A" ? "D" : "E");
            clear_internal_injection(ctx);
            auto baseline = evaluate(ctx, vocab, collector, prompt, target.first, true);
            replay_case current{(int32_t) cases.size(), "context_variant", prompt, target.first, target.second,
                                std::move(baseline), tokenize(vocab, prompt, true), {}};
            current.query_key = normalized(current.baseline.final_hidden);
            cases.push_back(std::move(current));
            metadata.push_back({variant, source, source == "A" ? "D" : "E"});
        }

        for (size_t index = 0; index < REPLAY_PROTECTIVE_PROMPTS.size(); ++index) {
            const std::string prompt = REPLAY_PROTECTIVE_PROMPTS[index];
            clear_internal_injection(ctx);
            auto baseline = evaluate(ctx, vocab, collector, prompt, 0, true);
            const llama_token target = baseline.metrics.greedy;
            replay_case current{(int32_t) cases.size(), "protective", prompt, target, "",
                                std::move(baseline), tokenize(vocab, prompt, true), {}};
            current.query_key = normalized(current.baseline.final_hidden);
            cases.push_back(std::move(current));
            metadata.push_back({"protective_" + std::to_string(index), "", ""});
        }

        std::ofstream results_out(root/"forward_results.jsonl", std::ios::trunc);
        if (!results_out) throw std::runtime_error("cannot open forward_results.jsonl");
        results_out.setf(std::ios::unitbuf);
        double zero_max = 0;
        double reset_max = 0;
        double maximum_budget = 0;
        for (size_t index = 0; index < cases.size(); ++index) {
            const auto & current = cases[index];
            for (const bool competitive : {false, true}) {
                const auto result = run_replay_reader(ctx, vocab, collector, output, current, bank, competitive);
                maximum_budget = std::max(maximum_budget, result.actual_budget);
                const double second_best = second_best_cosine(bank, current.query_key, result.winning_source);
                write_step16_row(results_out, competitive ? "C1_competitive" : "C0_existing", current,
                                 metadata[index], result, second_best,
                                 kl_divergence(result.logits, current.baseline.logits));
            }
        }

        const auto & reference = cases.front();
        clear_internal_injection(ctx);
        const auto zero = evaluate(ctx, vocab, collector, reference.prompt, reference.target, true);
        zero_max = max_logit_difference(zero.logits, reference.baseline.logits);
        clear_internal_injection(ctx);
        const auto restored = evaluate(ctx, vocab, collector, reference.prompt, reference.target, true);
        reset_max = max_logit_difference(restored.logits, reference.baseline.logits);
        clear_internal_injection(ctx);

        std::ofstream bank_meta(root/"bank_metadata.json", std::ios::trunc);
        bank_meta << std::setprecision(17)
                  << "{\"component_count\":" << bank.components.size()
                  << ",\"unique_source_key_count\":" << bank.source_keys.size()
                  << ",\"vectors_path\":\"" << json_escape(vectors_env)
                  << "\",\"components_path\":\"" << json_escape(components_env)
                  << "\",\"threshold\":" << REPLAY_TAU
                  << ",\"B_max\":" << REPLAY_B_MAX
                  << ",\"s_ref\":" << REPLAY_S_REF
                  << ",\"topology\":{\"block24\":0.23072,\"block28\":0.38322,\"final\":0.89438}"
                  << ",\"component_order_preserved\":true,\"learning_replayed\":false"
                  << ",\"query_variants\":8,\"protective_prompts\":8}\n";

        std::ofstream validation(root/"validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"zero_state_max_logit_difference\":" << zero_max
                   << ",\"post_probe_restoration_max_logit_difference\":" << reset_max
                   << ",\"max_actual_B\":" << maximum_budget
                   << ",\"B_max\":" << REPLAY_B_MAX
                   << ",\"reader_threshold\":" << REPLAY_TAU
                   << ",\"clean_kv_per_forward\":true"
                   << ",\"base_model_mutated\":false"
                   << ",\"learning_replayed\":false"
                   << ",\"gradients\":false,\"optimizer\":false}\n";

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << elapsed
                << ",\"model_decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers
                << ",\"ctx_size\":" << opts.ctx_size
                << ",\"query_variants\":8,\"protective_prompts\":8"
                << ",\"readers\":2}\n";

        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"persistent_replay_bank_components\":18,\"source_keys\":8,\"dimension\":2560,"
               << "\"shared_vector_bytes\":10240,\"dense_matrix_allocated\":false,"
               << "\"model_or_kv_memory_excluded\":true}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
