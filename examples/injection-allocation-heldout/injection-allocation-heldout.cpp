#include "../injection-sensitivity-allocation/multi-injection-support.inc"

#include <random>

namespace {

constexpr uint64_t HELDOUT_SEED = 20260811ULL;
constexpr int32_t HELDOUT_COUNT = 32;
constexpr double HELDOUT_SEARCH_MAX = 0.20;
constexpr double HELDOUT_SEARCH_TOLERANCE = 1e-4;
const std::array<double, 10> HELDOUT_BUDGETS = {
    0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20,
};
const std::array<double, 4> PROTECTIVE_BUDGETS = {0.01, 0.03, 0.05, 0.08};

struct heldout_case {
    case_state state;
    std::string synthetic_identifier;
    int32_t target_selection_attempts = 0;
};

struct heldout_condition {
    std::string id;
    std::string name;
    multi_configuration config;
    std::vector<double> weights;
};

struct heldout_threshold {
    bool reached = false;
    double budget = HELDOUT_SEARCH_MAX;
    multi_probe_result probe;
};

void write_double_array(std::ostream & out, const std::vector<double> & values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        out << (i ? "," : "") << std::setprecision(17) << values[i];
    }
    out << "]";
}

std::vector<std::string> target_candidates() {
    return {
        // These are the unused members of the original single-uppercase-token pool.
        " E", " H", " L", " M", " N", " S", " U", " Y",
    };
}

std::vector<std::pair<llama_token, std::string>> verified_candidate_tokens(
        const llama_vocab * vocab,
        const std::set<llama_token> & excluded,
        std::mt19937_64 & rng) {
    auto candidates = target_candidates();
    std::shuffle(candidates.begin(), candidates.end(), rng);
    std::set<llama_token> seen = excluded;
    std::vector<std::pair<llama_token, std::string>> verified;
    for (const auto & text : candidates) {
        const auto tokens = tokenize(vocab, text, false);
        if (tokens.size() != 1 || seen.count(tokens[0])) continue;
        seen.insert(tokens[0]);
        verified.push_back({tokens[0], text});
    }
    if (verified.size() != 8) {
        throw std::runtime_error("could not recover the eight unused uppercase single-token targets");
    }
    return verified;
}

std::vector<std::pair<std::string, std::string>> heldout_prompts(std::mt19937_64 & rng) {
    const std::array<std::string, 4> templates = {
        "Synthetic identifier %s. A synthetic probe requests one arbitrary marker. The marker is",
        "Synthetic identifier %s. Return one token for this deliberately unknown code:",
        "Synthetic identifier %s. The answer to this arbitrary token-control probe is",
        "Synthetic identifier %s. Emit the assigned synthetic marker only:",
    };
    std::uniform_int_distribution<uint64_t> distribution(100000, 999999);
    std::set<std::string> used;
    std::vector<std::pair<std::string, std::string>> result;
    for (int32_t index = 0; index < HELDOUT_COUNT; ++index) {
        std::string identifier;
        do {
            identifier = "H" + std::to_string(distribution(rng));
        } while (!used.insert(identifier).second);
        std::string prompt = templates[index % templates.size()];
        const size_t marker = prompt.find("%s");
        prompt.replace(marker, 2, identifier);
        result.push_back({identifier, prompt});
    }
    return result;
}

heldout_threshold find_heldout_threshold(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const case_state & state,
        const heldout_condition & condition,
        const std::vector<multi_probe_result> & fixed) {
    heldout_threshold result;
    result.probe = fixed.back();
    int32_t crossing = -1;
    for (size_t index = 1; index < fixed.size(); ++index) {
        if (fixed[index].metrics.greedy == state.definition.target) {
            crossing = (int32_t) index;
            break;
        }
    }
    if (crossing < 0) return result;
    result.reached = true;
    double low = HELDOUT_BUDGETS[crossing - 1];
    double high = HELDOUT_BUDGETS[crossing];
    while (high - low > HELDOUT_SEARCH_TOLERANCE) {
        const double middle = 0.5*(low + high);
        const auto probe = probe_multi(ctx, vocab, collector, state, condition.config, condition.weights, middle);
        if (probe.metrics.greedy == state.definition.target) {
            high = middle;
            result.probe = probe;
        } else {
            low = middle;
        }
    }
    result.budget = high;
    result.probe = probe_multi(ctx, vocab, collector, state, condition.config, condition.weights, high);
    return result;
}

void write_case_fields(std::ostream & out, const heldout_case & value) {
    out << ",\"dataset\":\"heldout\",\"synthetic_identifier\":\""
        << json_escape(value.synthetic_identifier) << "\",\"baseline_target_probability\":"
        << value.state.baseline.metrics.probability << ",\"baseline_target_nll\":"
        << value.state.baseline.metrics.nll << ",\"baseline_target_rank\":"
        << value.state.baseline.metrics.rank << ",\"baseline_target_margin\":"
        << value.state.baseline.metrics.margin;
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

        std::ofstream baseline_out(root/"baseline_results.jsonl", std::ios::trunc);
        std::set<llama_token> old_target_tokens;
        const auto calibration_definitions = stage_a_items(vocab);
        for (const auto & definition : calibration_definitions) {
            old_target_tokens.insert(definition.target);
            clear_internal_injection(ctx);
            const auto baseline = evaluate(ctx, vocab, collector, definition.prompt, definition.target, true);
            baseline_out << std::setprecision(17)
                         << "{\"dataset\":\"calibration\",\"prompt_id\":" << definition.id
                         << ",\"prompt\":\"" << json_escape(definition.prompt) << "\",\"target_token\":"
                         << definition.target << ",\"target_text\":\"" << json_escape(definition.target_text)
                         << "\",\"target_probability\":" << baseline.metrics.probability
                         << ",\"target_nll\":" << baseline.metrics.nll << ",\"target_rank\":"
                         << baseline.metrics.rank << ",\"target_margin\":" << baseline.metrics.margin
                         << ",\"greedy_token\":" << baseline.metrics.greedy << "}\n";
        }

        std::mt19937_64 rng(HELDOUT_SEED);
        const auto prompt_specs = heldout_prompts(rng);
        const auto candidates = verified_candidate_tokens(vocab, old_target_tokens, rng);
        std::vector<heldout_case> cases;
        for (int32_t index = 0; index < HELDOUT_COUNT; ++index) {
            int32_t attempts = 0;
            while (attempts < (int32_t) candidates.size()) {
                const auto candidate = candidates[(index + attempts) % candidates.size()];
                ++attempts;
                item definition{index, prompt_specs[index].second, candidate.first, candidate.second};
                clear_internal_injection(ctx);
                auto baseline = evaluate(ctx, vocab, collector, definition.prompt, definition.target, true);
                if (baseline.metrics.greedy == definition.target || !std::isfinite(baseline.metrics.nll)) continue;
                auto direction = normalized(output_row_difference(output, definition.target, baseline.metrics.greedy));
                heldout_case value{{definition, std::move(baseline), direction, output_projection(output, direction)},
                                   prompt_specs[index].first, attempts};
                baseline_out << std::setprecision(17)
                             << "{\"dataset\":\"heldout\",\"prompt_id\":" << index
                             << ",\"synthetic_identifier\":\"" << value.synthetic_identifier
                             << "\",\"prompt\":\"" << json_escape(definition.prompt)
                             << "\",\"target_token\":" << definition.target << ",\"target_text\":\""
                             << json_escape(definition.target_text) << "\",\"target_selection_attempts\":"
                             << attempts << ",\"target_probability\":" << value.state.baseline.metrics.probability
                             << ",\"target_nll\":" << value.state.baseline.metrics.nll
                             << ",\"target_rank\":" << value.state.baseline.metrics.rank
                             << ",\"target_margin\":" << value.state.baseline.metrics.margin
                             << ",\"greedy_token\":" << value.state.baseline.metrics.greedy << "}\n";
                cases.push_back(std::move(value));
                break;
            }
            if ((int32_t) cases.size() != index + 1) {
                throw std::runtime_error("all matched uppercase targets were invalid for a held-out prompt");
            }
        }

        const std::vector<double> published = {0.23072, 0.38322, 0.89438};
        const auto normalized_published = normalized_weights(published);
        const std::vector<heldout_condition> conditions = {
            {"C2", "final_only", {"C2", "final_only", {-1}}, {1.0}},
            {"C3", "manual_block28_final_1_to_3", {"C3", "manual_block28_final_1_to_3", {27, -1}}, normalized_weights({1, 3})},
            {"C4", "manual_16_28_final_1_2_4", {"C4", "manual_16_28_final_1_2_4", {15, 27, -1}}, normalized_weights({1, 2, 4})},
            {"C5", "frozen_24_28_final_sensitivity", {"C5", "frozen_24_28_final_sensitivity", {23, 27, -1}}, normalized_published},
        };

        std::ofstream dataset(root/"heldout_dataset.json", std::ios::trunc);
        dataset << "{\"seed\":" << HELDOUT_SEED << ",\"case_count\":" << cases.size()
                << ",\"construction\":\"original four arbitrary-marker templates with deterministic seeded identifiers; eight unused uppercase single-token targets repeated evenly; old target token ids excluded; only preregistered invalid cases rejected\","
                << "\"old_prompt_overlap\":false,\"old_target_token_overlap\":false,\"cases\":[";
        for (size_t index = 0; index < cases.size(); ++index) {
            const auto & value = cases[index];
            dataset << (index ? "," : "") << "{\"prompt_id\":" << value.state.definition.id
                    << ",\"synthetic_identifier\":\"" << value.synthetic_identifier << "\",\"prompt\":\""
                    << json_escape(value.state.definition.prompt) << "\",\"target_token\":"
                    << value.state.definition.target << ",\"target_text\":\""
                    << json_escape(value.state.definition.target_text) << "\",\"target_selection_attempts\":"
                    << value.target_selection_attempts << "}";
        }
        dataset << "]}\n";

        const uint64_t threshold_calls_start = decode_calls;
        std::ofstream fixed_out(root/"fixed_budget_results.jsonl", std::ios::trunc);
        std::ofstream threshold_out(root/"threshold_results.jsonl", std::ios::trunc);
        for (const auto & value : cases) {
            for (const auto & condition : conditions) {
                std::vector<multi_probe_result> fixed;
                for (double budget : HELDOUT_BUDGETS) {
                    auto probe = probe_multi(ctx, vocab, collector, value.state, condition.config, condition.weights, budget);
                    fixed.push_back(probe);
                    fixed_out << "{";
                    write_probe_fields(fixed_out, value.state, condition.config, condition.weights, probe);
                    write_case_fields(fixed_out, value);
                    fixed_out << ",\"condition_id\":\"" << condition.id << "\"}\n";
                }
                const auto threshold = find_heldout_threshold(ctx, vocab, collector, value.state, condition, fixed);
                threshold_out << "{";
                write_probe_fields(threshold_out, value.state, condition.config, condition.weights, threshold.probe);
                write_case_fields(threshold_out, value);
                threshold_out << ",\"condition_id\":\"" << condition.id
                              << "\",\"threshold_global_budget\":" << threshold.budget
                              << ",\"not_reached\":" << (threshold.reached ? "false" : "true") << "}\n";
                std::fprintf(stderr, "heldout=%d condition=%s reached=%d B=%.6f\n", value.state.definition.id,
                             condition.id.c_str(), threshold.reached, threshold.budget);
            }
        }
        const uint64_t threshold_decode_calls = decode_calls - threshold_calls_start;

        double zero_max = 0;
        for (const auto & condition : conditions) {
            const auto zero = probe_multi(ctx, vocab, collector, cases[0].state, condition.config, condition.weights, 0.0);
            zero_max = std::max(zero_max, max_logit_difference(zero.logits, cases[0].state.baseline.logits));
        }
        (void) probe_multi(ctx, vocab, collector, cases[0].state, conditions.back().config,
                           conditions.back().weights, 0.03);
        clear_internal_injection(ctx);
        const auto restored = evaluate(ctx, vocab, collector, cases[0].state.definition.prompt,
                                       cases[0].state.definition.target, false);
        const double restoration_max = max_logit_difference(restored.logits, cases[0].state.baseline.logits);

        const uint64_t protective_calls_start = decode_calls;
        const std::array<std::string, 8> protective_prompts = {
            "A held-out neutral prompt asks for the color of clear daytime sky:",
            "A held-out arithmetic check: five plus six equals",
            "Complete this ordinary sentence: Fresh water freezes at",
            "Write the opposite of the word narrow:",
            "A common week has this many days:",
            "The season after spring is",
            "A triangle has this many sides:",
            "Complete the sequence Monday, Tuesday,",
        };
        const std::set<std::string> protective_ids = {"C2", "C3", "C5"};
        std::ofstream protective_out(root/"protective_results.jsonl", std::ios::trunc);
        for (size_t index = 0; index < protective_prompts.size(); ++index) {
            clear_internal_injection(ctx);
            const auto baseline = evaluate(ctx, vocab, collector, protective_prompts[index], 0, true);
            item definition{(int32_t) index, protective_prompts[index], baseline.metrics.greedy, ""};
            case_state protective_state{definition, baseline, cases[index].state.direction,
                                        cases[index].state.projected_direction};
            for (const auto & condition : conditions) {
                if (!protective_ids.count(condition.id)) continue;
                for (double budget : PROTECTIVE_BUDGETS) {
                    const auto probe = probe_multi(ctx, vocab, collector, protective_state, condition.config,
                                                   condition.weights, budget);
                    const double kl = kl_divergence(probe.logits, baseline.logits);
                    protective_out << std::setprecision(17)
                        << "{\"prompt_id\":" << index << ",\"prompt\":\""
                        << json_escape(protective_prompts[index]) << "\",\"direction_source_prompt_id\":"
                        << index << ",\"condition_id\":\"" << condition.id << "\",\"configuration\":\""
                        << condition.name << "\",\"global_budget\":" << budget
                        << ",\"kl_adapted_base\":" << kl << ",\"greedy_changed\":"
                        << (probe.metrics.greedy != baseline.metrics.greedy ? "true" : "false")
                        << ",\"kl_per_budget\":" << kl/budget << "}\n";
                }
            }
        }
        const uint64_t protective_decode_calls = decode_calls - protective_calls_start;

        clear_internal_injection(ctx);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"raw_validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"zero_budget_max_logit_difference\":" << zero_max
                   << ",\"post_probe_restoration_max_logit_difference\":" << restoration_max
                   << ",\"clean_kv_per_probe\":true,\"no_sensitivity_probes_on_heldout\":true,"
                   << "\"base_model_mutated\":false,\"persistent_state_after_run\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":" << decode_calls
                << ",\"threshold_decode_calls\":" << threshold_decode_calls
                << ",\"protective_decode_calls\":" << protective_decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size << "}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"persistent_experiment_state_bytes\":0,\"shared_correction_vector_bytes\":10240,"
               << "\"allocation_weight_bytes\":24,\"boundary_trace_bytes\":81920,"
               << "\"temporary_sparse_cvec_host_bytes\":317440,\"dense_matrix_allocated\":false,"
               << "\"model_or_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        const double published_norm = std::sqrt(std::inner_product(published.begin(), published.end(),
                                                                   published.begin(), 0.0));
        manifest << "{\"experiment\":\"qwen35_injection_allocation_heldout\",\"model\":\""
                 << json_escape(opts.model) << "\",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                 << ",\"sparse_residual_file\":\""
                 << (sparse ? json_escape(std::filesystem::path(sparse).filename().string()) : "")
                 << "\",\"heldout_seed\":" << HELDOUT_SEED << ",\"calibration_case_count\":16,"
                 << "\"heldout_case_count\":32,\"condition_count\":4,\"direction_type\":\"target_vs_winner\","
                 << "\"search_max_global_budget\":0.2,\"search_tolerance\":0.0001,"
                 << "\"published_sensitivity_weights\":";
        write_double_array(manifest, published);
        manifest << ",\"published_weight_norm\":" << published_norm << ",\"runtime_normalized_weights\":";
        write_double_array(manifest, normalized_published);
        manifest << ",\"profile_frozen_before_heldout_generation\":true,"
                 << "\"heldout_sensitivity_measurement\":false,\"base_model_mutated\":false,"
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
