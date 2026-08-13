int injection_depth_threshold_original_main(int argc, char ** argv);
#define main injection_depth_threshold_original_main
#include "../injection-depth-threshold/injection-depth-threshold.cpp"
#undef main

#include <set>
#include <tuple>

namespace {

constexpr double MULTI_SEARCH_MAX = 0.20;
constexpr double MULTI_SEARCH_TOLERANCE = 1e-4;
constexpr double LOCAL_RHO_TOLERANCE = 2e-6;
constexpr int32_t MAX_NORM_ITERATIONS = 6;

const std::array<double, 10> FIXED_BUDGETS = {
    0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20,
};
const std::array<double, 3> SMALL_BUDGETS = {0.001, 0.003, 0.01};
const std::array<double, 3> SYNERGY_BUDGETS = {0.01, 0.03, 0.05};

struct multi_configuration {
    std::string id;
    std::string name;
    std::vector<int32_t> sites; // -1 denotes final pre-LM-head injection.
};

std::vector<multi_configuration> configurations() {
    return {
        {"C01", "final",                              {-1}},
        {"C02", "block_28",                           {27}},
        {"C03", "block_16",                           {15}},
        {"C04", "block_24",                           {23}},
        {"C05", "block_8+final",                       {7, -1}},
        {"C06", "block_16+final",                     {15, -1}},
        {"C07", "block_24+final",                     {23, -1}},
        {"C08", "block_28+final",                     {27, -1}},
        {"C09", "block_16+block_24",                  {15, 23}},
        {"C10", "block_8+block_16+final",              {7, 15, -1}},
        {"C11", "block_16+block_24+final",            {15, 23, -1}},
        {"C12", "block_16+block_28+final",            {15, 27, -1}},
        {"C13", "block_8+block_16+block_24+final",     {7, 15, 23, -1}},
    };
}

std::string site_name(int32_t site) {
    return site < 0 ? "final" : "block_" + std::to_string(site + 1);
}

std::vector<double> equal_weights(size_t count) {
    return std::vector<double>(count, 1.0/std::sqrt((double) count));
}

std::vector<double> normalized_weights(const std::vector<double> & raw) {
    double sum = 0;
    for (double value : raw) sum += value*value;
    if (sum <= 0) throw std::runtime_error("allocation profile has zero norm");
    std::vector<double> out(raw);
    for (double & value : out) value /= std::sqrt(sum);
    return out;
}

void set_multi_internal_injection(
        llama_context * ctx,
        const multi_configuration & config,
        const std::vector<float> & direction,
        const std::map<int32_t, double> & alphas) {
    std::vector<int32_t> internal;
    for (int32_t site : config.sites) if (site >= 0) internal.push_back(site);
    if (internal.empty()) {
        clear_internal_injection(ctx);
        return;
    }
    std::vector<float> data((N_LAYERS - 1)*WIDTH, 0.0f);
    for (int32_t layer : internal) {
        const size_t offset = (size_t) (layer - 1)*WIDTH;
        const double alpha = alphas.at(layer);
        for (int32_t i = 0; i < WIDTH; ++i) data[offset + i] = alpha*direction[i];
    }
    const auto [minimum, maximum] = std::minmax_element(internal.begin(), internal.end());
    if (llama_set_adapter_cvec(ctx, data.data(), data.size(), WIDTH, *minimum, *maximum) != 0) {
        throw std::runtime_error("failed to install multi-site control vector");
    }
}

struct multi_probe_result {
    token_metrics metrics;
    std::vector<float> logits;
    std::map<int32_t, double> requested_rhos;
    std::map<int32_t, double> actual_rhos;
    std::map<int32_t, double> local_norms;
    std::map<int32_t, double> alphas;
    double requested_budget = 0;
    double actual_budget = 0;
    double max_local_rho_error = 0;
    int32_t norm_iterations = 0;
};

multi_probe_result probe_multi(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const case_state & state,
        const multi_configuration & config,
        const std::vector<double> & weights,
        double budget) {
    if (weights.size() != config.sites.size()) throw std::runtime_error("site/weight count mismatch");

    multi_probe_result out;
    out.requested_budget = budget;
    std::map<int32_t, double> norms;
    std::map<int32_t, double> alphas;
    for (size_t index = 0; index < config.sites.size(); ++index) {
        const int32_t site = config.sites[index];
        out.requested_rhos[site] = budget*weights[index];
        if (site >= 0) norms[site] = l2(state.baseline.boundaries.at(site));
    }

    eval_result current;
    bool has_internal = !norms.empty();
    if (!has_internal) {
        current = state.baseline;
    } else {
        bool converged = false;
        for (int32_t iteration = 1; iteration <= MAX_NORM_ITERATIONS; ++iteration) {
            for (const auto & [site, norm] : norms) alphas[site] = out.requested_rhos.at(site)*norm;
            set_multi_internal_injection(ctx, config, state.direction, alphas);
            try {
                current = evaluate(ctx, vocab, collector, state.definition.prompt, state.definition.target, true);
                clear_internal_injection(ctx);
            } catch (...) {
                clear_internal_injection(ctx);
                throw;
            }

            double max_error = 0;
            std::map<int32_t, double> measured_norms;
            for (const auto & [site, alpha] : alphas) {
                std::vector<float> pre = current.boundaries.at(site);
                for (int32_t i = 0; i < WIDTH; ++i) pre[i] -= alpha*state.direction[i];
                const double pre_norm = l2(pre);
                measured_norms[site] = pre_norm;
                const double actual = pre_norm > 0 ? std::abs(alpha)/pre_norm : 0;
                max_error = std::max(max_error, std::abs(actual - out.requested_rhos.at(site)));
            }
            norms = std::move(measured_norms);
            out.norm_iterations = iteration;
            out.max_local_rho_error = max_error;
            if (max_error <= LOCAL_RHO_TOLERANCE) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            // A simultaneous fixed-point solve can oscillate at large budgets.  The
            // graph is causal, so calibrate each site from shallow to deep: later
            // additions cannot affect an earlier site's pre-injection norm.
            std::vector<int32_t> ordered_sites;
            for (const auto & [site, _] : norms) ordered_sites.push_back(site);
            std::sort(ordered_sites.begin(), ordered_sites.end());
            alphas.clear();
            for (int32_t site : ordered_sites) alphas[site] = 0.0;

            for (size_t index = 0; index < ordered_sites.size(); ++index) {
                const int32_t site = ordered_sites[index];
                double pre_norm = 0;
                if (index == 0) {
                    pre_norm = l2(state.baseline.boundaries.at(site));
                } else {
                    set_multi_internal_injection(ctx, config, state.direction, alphas);
                    try {
                        current = evaluate(ctx, vocab, collector, state.definition.prompt, state.definition.target, true);
                        clear_internal_injection(ctx);
                    } catch (...) {
                        clear_internal_injection(ctx);
                        throw;
                    }
                    pre_norm = l2(current.boundaries.at(site));
                }
                norms[site] = pre_norm;
                alphas[site] = out.requested_rhos.at(site)*pre_norm;
            }

            set_multi_internal_injection(ctx, config, state.direction, alphas);
            try {
                current = evaluate(ctx, vocab, collector, state.definition.prompt, state.definition.target, true);
                clear_internal_injection(ctx);
            } catch (...) {
                clear_internal_injection(ctx);
                throw;
            }

            out.max_local_rho_error = 0;
            for (const auto & [site, alpha] : alphas) {
                std::vector<float> pre = current.boundaries.at(site);
                for (int32_t i = 0; i < WIDTH; ++i) pre[i] -= alpha*state.direction[i];
                const double measured_norm = l2(pre);
                norms[site] = measured_norm;
                const double actual = measured_norm > 0 ? std::abs(alpha)/measured_norm : 0;
                out.max_local_rho_error = std::max(
                    out.max_local_rho_error,
                    std::abs(actual - out.requested_rhos.at(site)));
            }
            if (out.max_local_rho_error > LOCAL_RHO_TOLERANCE) {
                throw std::runtime_error("causal local-rho calibration exceeded tolerance");
            }
        }
    }

    out.logits = current.logits;
    out.alphas = alphas;
    for (const auto & [site, norm] : norms) {
        out.local_norms[site] = norm;
        out.actual_rhos[site] = norm > 0 ? std::abs(alphas.at(site))/norm : 0;
    }
    if (out.requested_rhos.count(-1)) {
        const double norm = l2(current.final_hidden);
        const double rho = out.requested_rhos.at(-1);
        const double alpha = rho*norm;
        out.local_norms[-1] = norm;
        out.alphas[-1] = alpha;
        out.actual_rhos[-1] = rho;
        out.logits = add_final_residual(current.logits, state.projected_direction, alpha);
    }
    double budget_sq = 0;
    for (const auto & [site, rho] : out.actual_rhos) budget_sq += rho*rho;
    out.actual_budget = std::sqrt(budget_sq);
    out.metrics = metrics_from_logits(out.logits, state.definition.target);
    return out;
}

void write_sites(std::ostream & out, const multi_configuration & config) {
    out << "[";
    for (size_t i = 0; i < config.sites.size(); ++i) {
        out << (i ? ",\"" : "\"") << site_name(config.sites[i]) << "\"";
    }
    out << "]";
}

void write_number_map(std::ostream & out, const std::map<int32_t, double> & values) {
    out << "{";
    bool first = true;
    for (const auto & [site, value] : values) {
        out << (first ? "\"" : ",\"") << site_name(site) << "\":" << std::setprecision(17) << value;
        first = false;
    }
    out << "}";
}

void write_probe_fields(
        std::ostream & out,
        const case_state & state,
        const multi_configuration & config,
        const std::vector<double> & weights,
        const multi_probe_result & probe) {
    out << std::setprecision(17)
        << "\"prompt_id\":" << state.definition.id
        << ",\"target_token\":" << state.definition.target
        << ",\"target_text\":\"" << json_escape(state.definition.target_text) << "\""
        << ",\"baseline_greedy_token\":" << state.baseline.metrics.greedy
        << ",\"configuration_id\":\"" << config.id << "\""
        << ",\"configuration\":\"" << config.name << "\""
        << ",\"sites\":";
    write_sites(out, config);
    out << ",\"allocation_weights\":[";
    for (size_t i = 0; i < weights.size(); ++i) out << (i ? "," : "") << weights[i];
    out << "]"
        << ",\"requested_global_budget\":" << probe.requested_budget
        << ",\"actual_global_budget\":" << probe.actual_budget
        << ",\"requested_local_rhos\":";
    write_number_map(out, probe.requested_rhos);
    out << ",\"actual_local_rhos\":";
    write_number_map(out, probe.actual_rhos);
    out << ",\"local_residual_norms\":";
    write_number_map(out, probe.local_norms);
    out << ",\"raw_alphas\":";
    write_number_map(out, probe.alphas);
    out << ",\"norm_iterations\":" << probe.norm_iterations
        << ",\"max_local_rho_error\":" << probe.max_local_rho_error;
    write_metric_fields(out, probe.metrics, state.definition.target);
}

struct threshold_record {
    bool reached = false;
    double budget = MULTI_SEARCH_MAX;
    multi_probe_result probe;
};

threshold_record find_threshold(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const case_state & state,
        const multi_configuration & config,
        const std::vector<double> & weights,
        const std::vector<multi_probe_result> & fixed) {
    threshold_record out;
    out.probe = fixed.back();
    int32_t crossing = -1;
    for (size_t index = 1; index < fixed.size(); ++index) {
        if (fixed[index].metrics.greedy == state.definition.target) {
            crossing = (int32_t) index;
            break;
        }
    }
    if (crossing < 0) return out;
    out.reached = true;
    double low = FIXED_BUDGETS[crossing - 1];
    double high = FIXED_BUDGETS[crossing];
    while (high - low > MULTI_SEARCH_TOLERANCE) {
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

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::infinity();
    std::sort(values.begin(), values.end());
    const size_t middle = values.size()/2;
    return values.size()%2 ? values[middle] : 0.5*(values[middle - 1] + values[middle]);
}

struct profile_definition {
    std::string id;
    std::vector<double> raw;
};

std::vector<profile_definition> profiles_for(const multi_configuration & config) {
    if (config.sites.size() == 2) {
        return {{"internal3_final1", {3, 1}}, {"internal1_final3", {1, 3}}, {"internal1_final9", {1, 9}}};
    }
    if (config.sites.size() == 3) {
        return {{"one_one_two", {1, 1, 2}}, {"one_two_four", {1, 2, 4}}};
    }
    return {};
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
        const auto configs = configurations();

        std::ofstream config_meta(root/"configuration_metadata.json", std::ios::trunc);
        config_meta << "{\"global_budget_norm\":\"l2_over_local_rhos\",\"equal_allocation\":true,\"configurations\":[";
        for (size_t i = 0; i < configs.size(); ++i) {
            config_meta << (i ? ",{" : "{") << "\"id\":\"" << configs[i].id << "\",\"name\":\""
                        << configs[i].name << "\",\"sites\":";
            write_sites(config_meta, configs[i]);
            config_meta << "}";
        }
        config_meta << "],\"internal_boundary_semantics\":{\"block_8\":7,\"block_16\":15,\"block_24\":23,\"block_28\":27},"
                    << "\"final_boundary\":\"after final RMSNorm before immutable LM head\","
                    << "\"correction_direction\":\"normalized immutable LM-head target row minus baseline-winner row\"}\n";

        double zero_max = 0;
        for (const auto & config : configs) {
            const auto zero = probe_multi(ctx, vocab, collector, cases[0], config, equal_weights(config.sites.size()), 0.0);
            zero_max = std::max(zero_max, max_logit_difference(zero.logits, cases[0].baseline.logits));
        }
        (void) probe_multi(ctx, vocab, collector, cases[0], configs.back(), equal_weights(configs.back().sites.size()), 0.03);
        clear_internal_injection(ctx);
        const auto restored = evaluate(ctx, vocab, collector, cases[0].definition.prompt, cases[0].definition.target, false);
        const double restore_diff = max_logit_difference(restored.logits, cases[0].baseline.logits);

        std::ofstream fixed_out(root/"fixed_budget_results.jsonl", std::ios::trunc);
        std::ofstream threshold_out(root/"threshold_results.jsonl", std::ios::trunc);
        std::ofstream small_out(root/"small_signal_results.jsonl", std::ios::trunc);
        std::map<std::pair<int32_t, std::string>, std::map<int64_t, multi_probe_result>> fixed_cache;
        std::map<std::pair<int32_t, std::string>, threshold_record> thresholds;

        for (const auto & state : cases) {
            for (const auto & config : configs) {
                const auto weights = equal_weights(config.sites.size());
                std::vector<multi_probe_result> fixed;
                for (double budget : FIXED_BUDGETS) {
                    auto probe = probe_multi(ctx, vocab, collector, state, config, weights, budget);
                    fixed.push_back(probe);
                    fixed_cache[{state.definition.id, config.id}][std::llround(budget*1000000)] = probe;
                    fixed_out << "{";
                    write_probe_fields(fixed_out, state, config, weights, probe);
                    fixed_out << "}\n";
                }
                auto threshold = find_threshold(ctx, vocab, collector, state, config, weights, fixed);
                thresholds[{state.definition.id, config.id}] = threshold;
                threshold_out << "{";
                write_probe_fields(threshold_out, state, config, weights, threshold.probe);
                threshold_out << ",\"threshold_global_budget\":" << threshold.budget
                              << ",\"not_reached\":" << (threshold.reached ? "false" : "true") << "}\n";
                std::fprintf(stderr, "case=%d config=%s reached=%d B=%.6f\n", state.definition.id,
                             config.id.c_str(), threshold.reached, threshold.budget);

                for (double budget : SMALL_BUDGETS) {
                    multi_probe_result probe;
                    const int64_t key = std::llround(budget*1000000);
                    if (fixed_cache[{state.definition.id, config.id}].count(key)) {
                        probe = fixed_cache[{state.definition.id, config.id}].at(key);
                    } else {
                        probe = probe_multi(ctx, vocab, collector, state, config, weights, budget);
                    }
                    small_out << "{";
                    write_probe_fields(small_out, state, config, weights, probe);
                    small_out << ",\"baseline_margin\":" << state.baseline.metrics.margin
                              << ",\"margin_delta\":" << probe.metrics.margin - state.baseline.metrics.margin
                              << ",\"margin_delta_per_budget\":"
                              << (budget > 0 ? (probe.metrics.margin - state.baseline.metrics.margin)/budget : 0) << "}\n";
                }
            }
        }

        std::map<int32_t, multi_configuration> single_site_configs = {
            {-1, {"single_final", "final_component", {-1}}},
            {7,  {"single_8", "block_8_component", {7}}},
            {15, {"single_16", "block_16_component", {15}}},
            {23, {"single_24", "block_24_component", {23}}},
            {27, {"single_28", "block_28_component", {27}}},
        };
        const std::set<std::string> synergy_ids = {"C05", "C06", "C07", "C08", "C09"};
        std::ofstream synergy_out(root/"synergy_results.jsonl", std::ios::trunc);
        std::map<std::tuple<int32_t, int32_t, int64_t>, multi_probe_result> component_cache;
        for (const auto & state : cases) {
            for (const auto & config : configs) {
                if (!synergy_ids.count(config.id)) continue;
                for (double budget : SYNERGY_BUDGETS) {
                    const auto & joint = fixed_cache.at({state.definition.id, config.id}).at(std::llround(budget*1000000));
                    const double component_budget = budget/std::sqrt(2.0);
                    double component_sum = 0;
                    std::vector<double> component_deltas;
                    for (int32_t site : config.sites) {
                        const auto key = std::make_tuple(state.definition.id, site, std::llround(component_budget*100000000));
                        if (!component_cache.count(key)) {
                            component_cache[key] = probe_multi(ctx, vocab, collector, state, single_site_configs.at(site), {1.0}, component_budget);
                        }
                        const double delta = component_cache.at(key).metrics.margin - state.baseline.metrics.margin;
                        component_deltas.push_back(delta);
                        component_sum += delta;
                    }
                    const double joint_delta = joint.metrics.margin - state.baseline.metrics.margin;
                    synergy_out << std::setprecision(17)
                        << "{\"prompt_id\":" << state.definition.id << ",\"configuration_id\":\"" << config.id
                        << "\",\"configuration\":\"" << config.name << "\",\"global_budget\":" << budget
                        << ",\"component_local_rho\":" << component_budget
                        << ",\"joint_margin_delta\":" << joint_delta << ",\"component_margin_deltas\":["
                        << component_deltas[0] << "," << component_deltas[1] << "]"
                        << ",\"independent_component_sum\":" << component_sum
                        << ",\"interaction\":" << joint_delta - component_sum << "}\n";
                }
            }
        }

        std::vector<const multi_configuration *> profile_candidates;
        for (const auto & config : configs) {
            if ((config.id >= "C05" && config.id <= "C08") ||
                (config.id >= "C10" && config.id <= "C12")) {
                profile_candidates.push_back(&config);
            }
        }
        std::sort(profile_candidates.begin(), profile_candidates.end(), [&](const auto * left, const auto * right) {
            auto score = [&](const auto * config) {
                int reached_count = 0;
                std::vector<double> values;
                for (const auto & state : cases) {
                    const auto & value = thresholds.at({state.definition.id, config->id});
                    if (value.reached) { ++reached_count; values.push_back(value.budget); }
                }
                return std::make_pair(-reached_count, median(values));
            };
            return score(left) < score(right);
        });
        profile_candidates.resize(std::min<size_t>(2, profile_candidates.size()));

        std::ofstream profile_out(root/"allocation_profile_results.jsonl", std::ios::trunc);
        for (const auto * config : profile_candidates) {
            for (const auto & profile : profiles_for(*config)) {
                const auto weights = normalized_weights(profile.raw);
                for (const auto & state : cases) {
                    std::vector<multi_probe_result> fixed;
                    for (double budget : FIXED_BUDGETS) {
                        auto probe = probe_multi(ctx, vocab, collector, state, *config, weights, budget);
                        fixed.push_back(probe);
                        profile_out << "{\"record_type\":\"fixed\",\"profile_id\":\"" << profile.id << "\",";
                        write_probe_fields(profile_out, state, *config, weights, probe);
                        profile_out << "}\n";
                    }
                    const auto threshold = find_threshold(ctx, vocab, collector, state, *config, weights, fixed);
                    profile_out << "{\"record_type\":\"threshold\",\"profile_id\":\"" << profile.id << "\",";
                    write_probe_fields(profile_out, state, *config, weights, threshold.probe);
                    profile_out << ",\"threshold_global_budget\":" << threshold.budget
                                << ",\"not_reached\":" << (threshold.reached ? "false" : "true") << "}\n";
                }
            }
        }

        const std::array<std::string, 4> protective_prompts = {
            "The capital of France is", "Two plus two equals", "Water freezes at", "The opposite of hot is",
        };
        const std::set<std::string> protective_ids = {"C01", "C06", "C07", "C08", "C11", "C13"};
        std::ofstream protective_out(root/"protective_results.jsonl", std::ios::trunc);
        for (size_t index = 0; index < protective_prompts.size(); ++index) {
            clear_internal_injection(ctx);
            auto baseline = evaluate(ctx, vocab, collector, protective_prompts[index], 0, true);
            item definition{(int32_t) index, protective_prompts[index], baseline.metrics.greedy, ""};
            case_state protective_state{definition, baseline, cases[index].direction, cases[index].projected_direction};
            for (const auto & config : configs) {
                if (!protective_ids.count(config.id)) continue;
                const auto weights = equal_weights(config.sites.size());
                for (double budget : SYNERGY_BUDGETS) {
                    const auto probe = probe_multi(ctx, vocab, collector, protective_state, config, weights, budget);
                    protective_out << std::setprecision(17)
                        << "{\"prompt_id\":" << index << ",\"prompt\":\"" << json_escape(protective_prompts[index])
                        << "\",\"direction_source_prompt_id\":" << index << ",\"configuration_id\":\"" << config.id
                        << "\",\"configuration\":\"" << config.name << "\",\"global_budget\":" << budget
                        << ",\"kl_adapted_base\":" << kl_divergence(probe.logits, baseline.logits)
                        << ",\"baseline_nll\":" << metrics_from_logits(baseline.logits, definition.target).nll
                        << ",\"plastic_nll\":" << probe.metrics.nll
                        << ",\"greedy_changed\":" << (probe.metrics.greedy == definition.target ? "false" : "true") << "}\n";
                }
            }
        }

        clear_internal_injection(ctx);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"rho_zero_max_logit_difference\":" << zero_max
                   << ",\"post_probe_restoration_max_logit_difference\":" << restore_diff
                   << ",\"local_rho_tolerance\":" << LOCAL_RHO_TOLERANCE
                   << ",\"clean_kv_per_probe\":true,\"causal_single_forward_per_norm_iteration\":true,"
                   << "\"persistent_state_after_run\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17) << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":"
                << decode_calls << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size << "}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"persistent_experiment_state_bytes\":0,\"shared_correction_vector_bytes\":10240,"
               << "\"boundary_trace_bytes\":81920,\"temporary_sparse_cvec_host_bytes\":317440,"
               << "\"dense_matrix_allocated\":false,\"model_or_kv_memory_excluded\":true}\n";
        std::ofstream raw_manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        raw_manifest << "{\"experiment\":\"qwen35_multi_injection_threshold\",\"model\":\""
                     << json_escape(opts.model) << "\",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                     << ",\"sparse_residual_file\":\""
                     << (sparse ? json_escape(std::filesystem::path(sparse).filename().string()) : "")
                     << "\",\"prior_cases_reconstructed_exactly\":true,\"case_count\":16,"
                     << "\"configuration_count\":13,\"direction_type\":\"target_vs_winner\","
                     << "\"search_max_global_budget\":0.20,\"search_tolerance\":0.0001,"
                     << "\"profile_followup_configurations\":[\"" << profile_candidates[0]->id << "\",\""
                     << profile_candidates[1]->id << "\"],\"base_model_mutated\":false,"
                     << "\"backpropagation\":false,\"autograd\":false,\"optimizer\":false,"
                     << "\"persistent_state\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
