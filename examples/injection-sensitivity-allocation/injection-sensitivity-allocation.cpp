int multi_injection_threshold_support_main(int argc, char ** argv);

#include "multi-injection-support.inc"

#include <functional>

namespace {

constexpr double SENS_SEARCH_MAX = 0.15;
constexpr double SENS_SEARCH_TOLERANCE = 1e-4;

const std::array<double, 5> SENSITIVITY_RHOS = {0.0, 0.0001, 0.0003, 0.001, 0.003};
const std::array<double, 9> SENSITIVITY_BUDGETS = {0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15};
const std::array<double, 3> PROTECTIVE_BUDGETS = {0.01, 0.03, 0.05};

struct sensitivity_fit {
    double slope = 0;
    double r2 = 0;
    double low_slope = 0;
    double high_slope = 0;
    double relative_drift = 0;
};

sensitivity_fit fit_sensitivity(const std::vector<double> & x, const std::vector<double> & y) {
    if (x.size() != y.size() || x.size() < 3) throw std::runtime_error("invalid sensitivity fit data");
    double xy = 0;
    double xx = 0;
    double mean_y = std::accumulate(y.begin(), y.end(), 0.0)/y.size();
    for (size_t i = 0; i < x.size(); ++i) {
        xy += x[i]*y[i];
        xx += x[i]*x[i];
    }
    sensitivity_fit out;
    out.slope = xx > 0 ? xy/xx : 0;
    double residual = 0;
    double total = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double error = y[i] - out.slope*x[i];
        residual += error*error;
        const double centered = y[i] - mean_y;
        total += centered*centered;
    }
    out.r2 = total > 0 ? 1.0 - residual/total : (residual == 0 ? 1.0 : 0.0);

    double low_xy = 0;
    double low_xx = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] <= 0.001) {
            low_xy += x[i]*y[i];
            low_xx += x[i]*x[i];
        }
    }
    out.low_slope = low_xx > 0 ? low_xy/low_xx : 0;
    const auto find_y = [&](double rho) {
        for (size_t i = 0; i < x.size(); ++i) if (std::abs(x[i] - rho) < 1e-12) return y[i];
        throw std::runtime_error("missing sensitivity rho");
    };
    out.high_slope = (find_y(0.003) - find_y(0.001))/0.002;
    out.relative_drift = (out.high_slope - out.low_slope)/std::max(1e-12, std::abs(out.low_slope));
    return out;
}

std::vector<double> positive_normalized_weights(const std::vector<double> & slopes, bool * fallback = nullptr) {
    std::vector<double> weights(slopes.size());
    double norm_sq = 0;
    for (size_t i = 0; i < slopes.size(); ++i) {
        weights[i] = std::max(0.0, slopes[i]);
        norm_sq += weights[i]*weights[i];
    }
    const bool used_fallback = norm_sq <= 0;
    if (used_fallback) {
        weights.assign(slopes.size(), 0.0);
        weights.back() = 1.0;
    } else {
        const double norm = std::sqrt(norm_sq);
        for (double & weight : weights) weight /= norm;
    }
    if (fallback) *fallback = used_fallback;
    return weights;
}

void write_number_array(std::ostream & out, const std::vector<double> & values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << std::setprecision(17) << values[i];
    out << "]";
}

struct allocation_condition {
    std::string id;
    std::string name;
    multi_configuration config;
    std::vector<double> weights;
    bool per_target = false;
};

struct sensitivity_threshold {
    bool reached = false;
    double budget = SENS_SEARCH_MAX;
    multi_probe_result probe;
};

sensitivity_threshold find_sensitivity_threshold(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const case_state & state,
        const multi_configuration & config,
        const std::vector<double> & weights,
        const std::vector<multi_probe_result> & fixed) {
    sensitivity_threshold out;
    out.probe = fixed.back();
    int32_t crossing = -1;
    for (size_t i = 1; i < fixed.size(); ++i) {
        if (fixed[i].metrics.greedy == state.definition.target) {
            crossing = (int32_t) i;
            break;
        }
    }
    if (crossing < 0) return out;
    out.reached = true;
    double low = SENSITIVITY_BUDGETS[crossing - 1];
    double high = SENSITIVITY_BUDGETS[crossing];
    while (high - low > SENS_SEARCH_TOLERANCE) {
        const double middle = 0.5*(low + high);
        const auto probe = probe_multi(ctx, vocab, collector, state, config, weights, middle);
        if (probe.metrics.greedy == state.definition.target) {
            high = middle;
            out.probe = probe;
        } else {
            low = middle;
        }
    }
    out.budget = high;
    out.probe = probe_multi(ctx, vocab, collector, state, config, weights, high);
    return out;
}

std::vector<double> condition_weights(
        const allocation_condition & condition,
        int32_t prompt_id,
        const std::map<int32_t, std::vector<double>> & per_target) {
    return condition.per_target ? per_target.at(prompt_id) : condition.weights;
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

        std::vector<case_state> cases;
        for (const auto & definition : stage_a_items(vocab)) {
            clear_internal_injection(ctx);
            auto baseline = evaluate(ctx, vocab, collector, definition.prompt, definition.target, true);
            if (baseline.metrics.greedy == definition.target) throw std::runtime_error("reused target unexpectedly top-1");
            auto direction = normalized(output_row_difference(output, definition.target, baseline.metrics.greedy));
            cases.push_back({definition, std::move(baseline), direction, output_projection(output, direction)});
        }

        const auto boundaries = all_boundaries();
        const multi_configuration all_sites{"sensitivity_sites", "all_eight_boundaries", boundaries};
        std::map<int32_t, std::vector<double>> slopes_by_target;
        std::map<std::pair<int32_t, int32_t>, sensitivity_fit> fit_by_target_boundary;

        const uint64_t sensitivity_calls_start = decode_calls;
        std::ofstream measurements(root/"sensitivity_measurements.jsonl", std::ios::trunc);
        std::ofstream fits(root/"raw_sensitivity_fits.jsonl", std::ios::trunc);
        for (const auto & state : cases) {
            std::vector<double> target_slopes;
            for (int32_t boundary : boundaries) {
                const multi_configuration single{"single_" + site_name(boundary), site_name(boundary), {boundary}};
                std::vector<double> x;
                std::vector<double> y;
                for (double rho : SENSITIVITY_RHOS) {
                    const auto probe = probe_multi(ctx, vocab, collector, state, single, {1.0}, rho);
                    const double delta = probe.metrics.margin - state.baseline.metrics.margin;
                    x.push_back(rho);
                    y.push_back(delta);
                    measurements << "{";
                    write_probe_fields(measurements, state, single, {1.0}, probe);
                    measurements << ",\"boundary\":\"" << site_name(boundary)
                                 << "\",\"zero_based_block_index\":" << boundary
                                 << ",\"baseline_margin\":" << state.baseline.metrics.margin
                                 << ",\"margin_delta\":" << delta << "}\n";
                }
                const auto fit = fit_sensitivity(x, y);
                fit_by_target_boundary[{state.definition.id, boundary}] = fit;
                target_slopes.push_back(fit.slope);
                fits << std::setprecision(17)
                     << "{\"prompt_id\":" << state.definition.id << ",\"boundary\":\"" << site_name(boundary)
                     << "\",\"zero_based_block_index\":" << boundary << ",\"estimated_g\":" << fit.slope
                     << ",\"r2\":" << fit.r2 << ",\"low_range_slope\":" << fit.low_slope
                     << ",\"high_range_slope\":" << fit.high_slope << ",\"relative_slope_drift\":"
                     << fit.relative_drift << ",\"positive\":" << (fit.slope > 0 ? "true" : "false") << "}\n";
            }
            slopes_by_target[state.definition.id] = target_slopes;
        }
        const uint64_t sensitivity_decode_calls = decode_calls - sensitivity_calls_start;

        std::map<int32_t, std::vector<double>> per_target_weights;
        std::ofstream per_target_out(root/"per_target_allocations.jsonl", std::ios::trunc);
        for (const auto & state : cases) {
            bool fallback = false;
            const auto weights = positive_normalized_weights(slopes_by_target.at(state.definition.id), &fallback);
            per_target_weights[state.definition.id] = weights;
            per_target_out << "{\"prompt_id\":" << state.definition.id << ",\"slopes\":";
            write_number_array(per_target_out, slopes_by_target.at(state.definition.id));
            per_target_out << ",\"positive_normalized_weights\":";
            write_number_array(per_target_out, weights);
            per_target_out << ",\"all_nonpositive_fallback_to_final\":" << (fallback ? "true" : "false") << "}\n";
        }

        std::vector<double> mean_positive(boundaries.size());
        std::vector<double> median_positive(boundaries.size());
        for (size_t index = 0; index < boundaries.size(); ++index) {
            std::vector<double> values;
            for (const auto & state : cases) values.push_back(std::max(0.0, slopes_by_target.at(state.definition.id)[index]));
            mean_positive[index] = std::accumulate(values.begin(), values.end(), 0.0)/values.size();
            median_positive[index] = median(values);
        }
        const auto mean_weights = positive_normalized_weights(mean_positive);
        const auto median_weights = positive_normalized_weights(median_positive);
        std::vector<double> top3_source = mean_positive;
        std::vector<size_t> order(boundaries.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t left, size_t right) { return top3_source[left] > top3_source[right]; });
        std::vector<double> top3_values(boundaries.size(), 0.0);
        for (size_t i = 0; i < std::min<size_t>(3, order.size()); ++i) top3_values[order[i]] = top3_source[order[i]];
        const auto top3_weights = positive_normalized_weights(top3_values);

        std::ofstream raw_fixed(root/"raw_fixed_allocations.json", std::ios::trunc);
        raw_fixed << "{\"boundaries\":[";
        for (size_t i = 0; i < boundaries.size(); ++i) raw_fixed << (i ? ",\"" : "\"") << site_name(boundaries[i]) << "\"";
        raw_fixed << "],\"mean_positive_sensitivity\":";
        write_number_array(raw_fixed, mean_positive);
        raw_fixed << ",\"median_positive_sensitivity\":";
        write_number_array(raw_fixed, median_positive);
        raw_fixed << ",\"mean_weights\":";
        write_number_array(raw_fixed, mean_weights);
        raw_fixed << ",\"median_weights\":";
        write_number_array(raw_fixed, median_weights);
        raw_fixed << ",\"top3_weights\":";
        write_number_array(raw_fixed, top3_weights);
        raw_fixed << ",\"top3_boundaries\":[";
        bool first_top = true;
        for (size_t i = 0; i < boundaries.size(); ++i) if (top3_weights[i] > 0) {
            raw_fixed << (first_top ? "\"" : ",\"") << site_name(boundaries[i]) << "\"";
            first_top = false;
        }
        raw_fixed << "]}\n";

        const auto equal_pair = normalized_weights({1, 1});
        const auto one_three = normalized_weights({1, 3});
        const auto one_two_four = normalized_weights({1, 2, 4});
        const std::vector<allocation_condition> conditions = {
            {"C1", "final_only", {"C1", "final_only", {-1}}, {1.0}, false},
            {"C2", "block28_final_equal", {"C2", "block28_final_equal", {27, -1}}, equal_pair, false},
            {"C3", "block28_final_1_to_3", {"C3", "block28_final_1_to_3", {27, -1}}, one_three, false},
            {"C4", "block16_block28_final_1_2_4", {"C4", "block16_block28_final_1_2_4", {15, 27, -1}}, one_two_four, false},
            {"C5", "fixed_mean_sensitivity", {"C5", "fixed_mean_sensitivity", boundaries}, mean_weights, false},
            {"C6", "fixed_median_sensitivity", {"C6", "fixed_median_sensitivity", boundaries}, median_weights, false},
            {"C7", "top3_fixed_sensitivity", {"C7", "top3_fixed_sensitivity", boundaries}, top3_weights, false},
            {"C8", "per_target_sensitivity_oracle", {"C8", "per_target_sensitivity_oracle", boundaries}, {}, true},
        };

        const uint64_t threshold_calls_start = decode_calls;
        std::ofstream fixed_out(root/"fixed_budget_results.jsonl", std::ios::trunc);
        std::ofstream threshold_out(root/"threshold_results.jsonl", std::ios::trunc);
        for (const auto & state : cases) {
            for (const auto & condition : conditions) {
                const auto weights = condition_weights(condition, state.definition.id, per_target_weights);
                std::vector<multi_probe_result> fixed;
                for (double budget : SENSITIVITY_BUDGETS) {
                    auto probe = probe_multi(ctx, vocab, collector, state, condition.config, weights, budget);
                    fixed.push_back(probe);
                    fixed_out << "{";
                    write_probe_fields(fixed_out, state, condition.config, weights, probe);
                    fixed_out << ",\"condition_id\":\"" << condition.id << "\"}\n";
                }
                const auto threshold = find_sensitivity_threshold(ctx, vocab, collector, state, condition.config, weights, fixed);
                threshold_out << "{";
                write_probe_fields(threshold_out, state, condition.config, weights, threshold.probe);
                threshold_out << ",\"condition_id\":\"" << condition.id
                              << "\",\"threshold_global_budget\":" << threshold.budget
                              << ",\"not_reached\":" << (threshold.reached ? "false" : "true") << "}\n";
                std::fprintf(stderr, "case=%d condition=%s reached=%d B=%.6f\n", state.definition.id,
                             condition.id.c_str(), threshold.reached, threshold.budget);
            }
        }
        const uint64_t threshold_decode_calls = decode_calls - threshold_calls_start;

        double zero_max = 0;
        for (const auto & condition : conditions) {
            const auto weights = condition_weights(condition, cases[0].definition.id, per_target_weights);
            const auto zero = probe_multi(ctx, vocab, collector, cases[0], condition.config, weights, 0.0);
            zero_max = std::max(zero_max, max_logit_difference(zero.logits, cases[0].baseline.logits));
        }
        (void) probe_multi(ctx, vocab, collector, cases[0], conditions[6].config, conditions[6].weights, 0.03);
        clear_internal_injection(ctx);
        const auto restored = evaluate(ctx, vocab, collector, cases[0].definition.prompt, cases[0].definition.target, false);
        const double restoration_max = max_logit_difference(restored.logits, cases[0].baseline.logits);

        const uint64_t protective_calls_start = decode_calls;
        const std::array<std::string, 4> protective_prompts = {
            "The capital of France is", "Two plus two equals", "Water freezes at", "The opposite of hot is",
        };
        const std::set<std::string> protective_condition_ids = {"C1", "C4", "C5", "C7"};
        std::ofstream protective_out(root/"protective_results.jsonl", std::ios::trunc);
        for (size_t index = 0; index < protective_prompts.size(); ++index) {
            clear_internal_injection(ctx);
            const auto baseline = evaluate(ctx, vocab, collector, protective_prompts[index], 0, true);
            item definition{(int32_t) index, protective_prompts[index], baseline.metrics.greedy, ""};
            case_state protective_state{definition, baseline, cases[index].direction, cases[index].projected_direction};
            for (const auto & condition : conditions) {
                if (!protective_condition_ids.count(condition.id)) continue;
                for (double budget : PROTECTIVE_BUDGETS) {
                    const auto probe = probe_multi(ctx, vocab, collector, protective_state, condition.config,
                                                   condition.weights, budget);
                    protective_out << std::setprecision(17)
                        << "{\"prompt_id\":" << index << ",\"prompt\":\"" << json_escape(protective_prompts[index])
                        << "\",\"direction_source_prompt_id\":" << index << ",\"condition_id\":\"" << condition.id
                        << "\",\"configuration\":\"" << condition.name << "\",\"global_budget\":" << budget
                        << ",\"kl_adapted_base\":" << kl_divergence(probe.logits, baseline.logits)
                        << ",\"greedy_changed\":" << (probe.metrics.greedy != baseline.metrics.greedy ? "true" : "false")
                        << ",\"kl_per_budget\":" << kl_divergence(probe.logits, baseline.logits)/budget << "}\n";
                }
            }
        }
        const uint64_t protective_decode_calls = decode_calls - protective_calls_start;

        clear_internal_injection(ctx);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"rho_zero_max_logit_difference\":" << zero_max
                   << ",\"post_probe_restoration_max_logit_difference\":" << restoration_max
                   << ",\"local_rho_tolerance\":" << LOCAL_RHO_TOLERANCE
                   << ",\"clean_kv_per_probe\":true,\"base_model_mutated\":false,\"persistent_state_after_run\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":" << decode_calls
                << ",\"sensitivity_decode_calls\":" << sensitivity_decode_calls
                << ",\"threshold_decode_calls\":" << threshold_decode_calls
                << ",\"protective_decode_calls\":" << protective_decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size << "}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"persistent_experiment_state_bytes\":0,\"shared_correction_vector_bytes\":10240,"
               << "\"sensitivity_vector_bytes_per_target\":64,\"allocation_vector_bytes\":64,"
               << "\"boundary_trace_bytes\":81920,\"temporary_sparse_cvec_host_bytes\":317440,"
               << "\"dense_matrix_allocated\":false,\"model_or_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        manifest << "{\"experiment\":\"qwen35_injection_sensitivity_allocation\",\"model\":\""
                 << json_escape(opts.model) << "\",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                 << ",\"sparse_residual_file\":\""
                 << (sparse ? json_escape(std::filesystem::path(sparse).filename().string()) : "")
                 << "\",\"prior_cases_reconstructed_exactly\":true,\"case_count\":16,"
                 << "\"boundary_count\":8,\"condition_count\":8,\"direction_type\":\"target_vs_winner\","
                 << "\"search_max_global_budget\":0.15,\"search_tolerance\":0.0001,"
                 << "\"base_model_mutated\":false,\"backpropagation\":false,\"autograd\":false,"
                 << "\"optimizer\":false,\"persistent_state\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
