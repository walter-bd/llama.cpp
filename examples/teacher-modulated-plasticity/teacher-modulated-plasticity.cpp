#include "../injection-sensitivity-allocation/multi-injection-support.inc"

#include <random>

namespace {

constexpr int32_t FAST_RANK = 8;
constexpr double ETA_POSITIVE = 0.05;
constexpr double ETA_NEGATIVE = 0.05;
constexpr double PRIMARY_BUDGET = 0.03;
constexpr double SECONDARY_BUDGET = 0.05;
constexpr uint64_t BENCHMARK_SEED = 20260812ULL;
constexpr int32_t SEEDS_PER_FAMILY = 3;

const std::array<int32_t, 3> FAMILY_OFFSETS = {1, 3, 6};
const std::array<int32_t, 4> CHECKPOINTS = {0, 1, 2, 4};
const std::array<const char *, 6> CONDITION_NAMES = {
    "frozen", "blind_positive_repetition", "reward_only",
    "teacher_explicit_correction", "shuffled_feedback", "random_direction",
};

struct symbol_info {
    std::string label;
    std::string token_text;
    llama_token token = -1;
};

struct task_example {
    int32_t input_index = 0;
    int32_t output_index = 0;
    std::string prompt;
    llama_token target = -1;
};

struct episode {
    int32_t family = 0;
    int32_t offset = 0;
    int32_t seed = 0;
    std::vector<task_example> training;
    std::vector<task_example> heldout;
};

struct component {
    std::vector<float> q;
    std::vector<float> z;
    double coefficient = 0;
    int32_t age = 0;
    int32_t source_input = 0;
    std::string origin;
};

struct fast_bank {
    std::vector<component> components;

    void clear() { components.clear(); }

    void add(component value) {
        if ((int32_t) components.size() >= FAST_RANK) {
            throw std::runtime_error("rank-8 fast bank exhausted");
        }
        components.push_back(std::move(value));
    }

    std::vector<float> apply(const std::vector<float> & key) const {
        std::vector<float> out(WIDTH, 0.0f);
        for (const auto & term : components) {
            double activation = 0;
            for (int32_t i = 0; i < WIDTH; ++i) activation += (double) term.z[i]*key[i];
            const double scale = term.coefficient*activation;
            for (int32_t i = 0; i < WIDTH; ++i) out[i] += scale*term.q[i];
        }
        return out;
    }

    double frobenius_norm() const {
        double squared = 0;
        for (const auto & a : components) {
            for (const auto & b : components) {
                double q_dot = 0;
                double z_dot = 0;
                for (int32_t i = 0; i < WIDTH; ++i) {
                    q_dot += (double) a.q[i]*b.q[i];
                    z_dot += (double) a.z[i]*b.z[i];
                }
                squared += a.coefficient*b.coefficient*q_dot*z_dot;
            }
        }
        return std::sqrt(std::max(0.0, squared));
    }
};

struct plastic_eval {
    eval_result frozen;
    token_metrics metrics;
    std::vector<float> logits;
    std::vector<float> key;
    double raw_response_norm = 0;
    double budget = 0;
    std::map<int32_t, double> rhos;
};

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize(-count);
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    return count > 0 ? std::string(buffer.data(), count) : std::string();
}

std::vector<float> normalized_output_row(const output_matrix & output, llama_token token) {
    std::vector<float> row(output.width);
    std::copy(output.row(token), output.row(token) + output.width, row.begin());
    return normalized(row);
}

std::vector<float> deterministic_random_direction(uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::normal_distribution<float> normal;
    std::vector<float> values(WIDTH);
    for (float & value : values) value = normal(generator);
    return normalized(values);
}

std::vector<symbol_info> validate_symbols(const llama_vocab * vocab) {
    std::vector<symbol_info> result;
    for (char label = 'A'; label <= 'H'; ++label) {
        const std::string text = std::string(" ") + label;
        const auto tokens = tokenize(vocab, text, false);
        if (tokens.size() != 1) throw std::runtime_error("symbol is not one token: " + text);
        if (std::any_of(result.begin(), result.end(), [&](const auto & x) { return x.token == tokens[0]; })) {
            throw std::runtime_error("symbol token collision");
        }
        result.push_back({std::string(1, label), text, tokens[0]});
    }
    return result;
}

std::string task_prompt(const symbol_info & input) {
    return "Rule task.\n\nInput symbol: " + input.label + "\nOutput symbol:";
}

std::string corrected_prompt(const symbol_info & input, const symbol_info & target) {
    return "Rule task.\n\nInput symbol: " + input.label +
           "\nOutput symbol:\n\nTeacher correction: The correct output symbol is " + target.label +
           ".\nNow answer the original task.\nInput symbol: " + input.label + "\nOutput symbol:";
}

std::vector<episode> make_episodes(const std::vector<symbol_info> & symbols) {
    const std::array<int32_t, 4> base = {0, 2, 5, 7};
    std::vector<episode> result;
    for (int32_t family = 0; family < (int32_t) FAMILY_OFFSETS.size(); ++family) {
        for (int32_t seed = 0; seed < SEEDS_PER_FAMILY; ++seed) {
            std::set<int32_t> training_indices;
            for (int32_t value : base) training_indices.insert((value + seed) % 8);
            episode value;
            value.family = family;
            value.offset = FAMILY_OFFSETS[family];
            value.seed = seed;
            for (int32_t input = 0; input < 8; ++input) {
                const int32_t output_index = (input + value.offset) % 8;
                task_example example{input, output_index, task_prompt(symbols[input]), symbols[output_index].token};
                (training_indices.count(input) ? value.training : value.heldout).push_back(std::move(example));
            }
            result.push_back(std::move(value));
        }
    }
    return result;
}

plastic_eval evaluate_bank(
        llama_context * ctx, const llama_vocab * vocab, trace_collector & collector,
        const output_matrix & output, const fast_bank & bank, const std::string & prompt,
        llama_token target, double budget) {
    clear_internal_injection(ctx);
    plastic_eval out;
    out.frozen = evaluate(ctx, vocab, collector, prompt, target, true);
    out.key = normalized(out.frozen.final_hidden);
    out.logits = out.frozen.logits;
    out.metrics = out.frozen.metrics;
    if (bank.components.empty()) return out;
    const auto response = bank.apply(out.key);
    out.raw_response_norm = l2(response);
    if (out.raw_response_norm <= 1e-12) return out;
    const auto direction = normalized(response);
    item definition{0, prompt, target, ""};
    case_state state{definition, out.frozen, direction, output_projection(output, direction)};
    const multi_configuration config{"plastic", "block24_block28_final", {23, 27, -1}};
    const auto weights = normalized_weights({0.23072, 0.38322, 0.89438});
    const double bounded_budget = std::min(SECONDARY_BUDGET, std::max(0.0, budget));
    const auto probe = probe_multi(ctx, vocab, collector, state, config, weights, bounded_budget);
    out.metrics = probe.metrics;
    out.logits = probe.logits;
    out.budget = probe.actual_budget;
    out.rhos = probe.actual_rhos;
    clear_internal_injection(ctx);
    return out;
}

void write_metrics(std::ostream & out, const token_metrics & m) {
    out << std::setprecision(17)
        << "\"target_probability\":" << m.probability
        << ",\"target_nll\":" << m.nll
        << ",\"target_rank\":" << m.rank
        << ",\"target_margin\":" << m.margin
        << ",\"greedy_token\":" << m.greedy;
}

void write_rhos(std::ostream & out, const plastic_eval & value) {
    auto get = [&](int32_t site) { auto it = value.rhos.find(site); return it == value.rhos.end() ? 0.0 : it->second; };
    out << std::setprecision(17) << "\"B\":" << value.budget
        << ",\"rho24\":" << get(23) << ",\"rho28\":" << get(27)
        << ",\"rho_final\":" << get(-1) << ",\"authority_clipped\":"
        << (value.budget > SECONDARY_BUDGET + 1e-9 ? "true" : "false")
        << ",\"raw_sidecar_response_norm\":" << value.raw_response_norm;
}

double coefficient_norm(const std::vector<component> & terms, bool negative) {
    double total = 0;
    for (const auto & term : terms) {
        const bool is_negative = term.coefficient < 0 || term.origin.find("negative") != std::string::npos;
        if (is_negative == negative) total += term.coefficient*term.coefficient;
    }
    return std::sqrt(total);
}

void append_component(
        std::vector<component> & pending, std::vector<float> q, const std::vector<float> & z,
        double coefficient, int32_t age, int32_t input, const std::string & origin) {
    pending.push_back({normalized(q), normalized(z), coefficient, age, input, origin});
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
            throw std::runtime_error("experiment requires Qwen3.5-4B dimensions 32x2560");
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
        const auto symbols = validate_symbols(vocab);
        const auto episodes = make_episodes(symbols);

        std::ofstream benchmark(root/"benchmark.json", std::ios::trunc);
        benchmark << "{\"benchmark_seed\":" << BENCHMARK_SEED << ",\"families\":[";
        for (size_t i = 0; i < FAMILY_OFFSETS.size(); ++i) {
            benchmark << (i ? "," : "") << "{\"family\":" << i << ",\"hidden_offset\":" << FAMILY_OFFSETS[i] << "}";
        }
        benchmark << "],\"seeds_per_family\":" << SEEDS_PER_FAMILY
                  << ",\"training_examples_per_episode\":4,\"heldout_examples_per_episode\":4,"
                  << "\"rule\":\"y=(x+k) mod 8\",\"family_identity_in_prompt\":false,"
                  << "\"heldout_exact_prompt_seen\":false,\"heldout_exact_pair_seen\":false,"
                  << "\"same_latent_rule\":true,\"episodes\":[";
        for (size_t e = 0; e < episodes.size(); ++e) {
            benchmark << (e ? "," : "") << "{\"family\":" << episodes[e].family << ",\"seed\":"
                      << episodes[e].seed << ",\"training_inputs\":[";
            for (size_t i = 0; i < episodes[e].training.size(); ++i) benchmark << (i ? "," : "") << episodes[e].training[i].input_index;
            benchmark << "],\"heldout_inputs\":[";
            for (size_t i = 0; i < episodes[e].heldout.size(); ++i) benchmark << (i ? "," : "") << episodes[e].heldout[i].input_index;
            benchmark << "]}";
        }
        benchmark << "]}\n";

        std::ofstream token_out(root/"token_validation.json", std::ios::trunc);
        token_out << "{\"all_targets_single_token\":true,\"symbols\":[";
        for (size_t i = 0; i < symbols.size(); ++i) {
            token_out << (i ? "," : "") << "{\"symbol\":\"" << symbols[i].label
                      << "\",\"token_text\":\"" << json_escape(symbols[i].token_text)
                      << "\",\"token_id\":" << symbols[i].token << "}";
        }
        token_out << "]}\n";

        std::ofstream events(root/"adaptation_events.jsonl", std::ios::trunc);
        std::ofstream traces(root/"eligibility_traces_summary.jsonl", std::ios::trunc);
        std::ofstream history(root/"plastic_state_history.jsonl", std::ios::trunc);
        std::ofstream training_out(root/"training_results.jsonl", std::ios::trunc);
        std::ofstream heldout_out(root/"heldout_results.jsonl", std::ios::trunc);
        std::ofstream specificity_out(root/"specificity_results.jsonl", std::ios::trunc);
        std::ofstream protective_out(root/"protective_results.jsonl", std::ios::trunc);

        const std::array<std::string, 8> protective_prompts = {
            "A common week has this many days:", "Five plus six equals",
            "The season after spring is", "A triangle has this many sides:",
            "The opposite of narrow is", "Water freezes at zero degrees",
            "Monday, Tuesday,", "The clear daytime sky is usually",
        };

        double zero_state_max = 0;
        double reset_max = 0;
        double max_budget_observed = 0;
        bool no_feedback_state_unchanged = true;
        uint64_t update_rows = 0;
        uint64_t measurement_rows = 0;
        uint64_t protective_rows = 0;

        for (int32_t condition = 0; condition < (int32_t) CONDITION_NAMES.size(); ++condition) {
            for (const auto & ep : episodes) {
                fast_bank bank;
                int32_t event_index = 0;

                auto evaluate_checkpoint = [&](int32_t checkpoint, double budget) {
                    history << std::setprecision(17) << "{\"condition\":\"" << CONDITION_NAMES[condition]
                            << "\",\"family\":" << ep.family << ",\"seed\":" << ep.seed
                            << ",\"checkpoint\":" << checkpoint << ",\"rank\":" << bank.components.size()
                            << ",\"plastic_frobenius_norm\":" << bank.frobenius_norm()
                            << ",\"evaluation_budget\":" << budget << ",\"compression_event\":false}\n";

                    for (const auto & example : ep.training) {
                        const auto value = evaluate_bank(ctx, vocab, collector, output, bank, example.prompt, example.target, budget);
                        max_budget_observed = std::max(max_budget_observed, value.budget);
                        training_out << "{\"condition\":\"" << CONDITION_NAMES[condition] << "\",\"family\":"
                                     << ep.family << ",\"seed\":" << ep.seed << ",\"checkpoint\":" << checkpoint
                << ",\"authority_condition\":\"" << (budget == PRIMARY_BUDGET ? "primary" : "secondary") << "\""
                                     << ",\"input_index\":" << example.input_index << ",\"target_token\":" << example.target << ",";
                        write_metrics(training_out, value.metrics);
                        training_out << ",\"correct\":" << (value.metrics.greedy == example.target ? "true" : "false") << ",";
                        write_rhos(training_out, value);
                        training_out << "}\n";
                        ++measurement_rows;
                    }
                    for (const auto & example : ep.heldout) {
                        const auto value = evaluate_bank(ctx, vocab, collector, output, bank, example.prompt, example.target, budget);
                        max_budget_observed = std::max(max_budget_observed, value.budget);
                        heldout_out << "{\"condition\":\"" << CONDITION_NAMES[condition] << "\",\"family\":"
                                    << ep.family << ",\"seed\":" << ep.seed << ",\"checkpoint\":" << checkpoint
                                    << ",\"authority_condition\":\"" << (budget == PRIMARY_BUDGET ? "primary" : "secondary")
                                    << "\",\"input_index\":" << example.input_index << ",\"target_token\":" << example.target
                                    << ",\"same_exact_prompt_seen\":false,\"same_exact_pair_seen\":false,\"same_latent_rule\":true,";
                        write_metrics(heldout_out, value.metrics);
                        heldout_out << ",\"correct\":" << (value.metrics.greedy == example.target ? "true" : "false") << ",";
                        write_rhos(heldout_out, value);
                        heldout_out << "}\n";
                        ++measurement_rows;
                    }

                    for (const auto & other : episodes) {
                        if (other.seed != ep.seed || other.family == ep.family) continue;
                        for (const auto & example : other.heldout) {
                            const auto value = evaluate_bank(ctx, vocab, collector, output, bank, example.prompt, example.target, budget);
                            specificity_out << "{\"condition\":\"" << CONDITION_NAMES[condition] << "\",\"adapted_family\":"
                                            << ep.family << ",\"evaluated_family\":" << other.family << ",\"seed\":" << ep.seed
                                            << ",\"checkpoint\":" << checkpoint << ",\"authority_condition\":\""
                                            << (budget == PRIMARY_BUDGET ? "primary" : "secondary") << "\",\"input_index\":"
                                            << example.input_index << ",";
                            write_metrics(specificity_out, value.metrics);
                            specificity_out << ",\"correct\":" << (value.metrics.greedy == example.target ? "true" : "false") << "}\n";
                            ++measurement_rows;
                        }
                    }

                    if ((condition == 1 || condition == 3) && budget == PRIMARY_BUDGET) {
                        for (size_t p = 0; p < protective_prompts.size(); ++p) {
                            clear_internal_injection(ctx);
                            const auto baseline = evaluate(ctx, vocab, collector, protective_prompts[p], 0, true);
                            const auto value = evaluate_bank(ctx, vocab, collector, output, bank, protective_prompts[p], baseline.metrics.greedy, budget);
                            protective_out << "{\"condition\":\"" << CONDITION_NAMES[condition] << "\",\"family\":"
                                           << ep.family << ",\"seed\":" << ep.seed << ",\"checkpoint\":" << checkpoint
                                           << ",\"prompt_id\":" << p << ",\"kl_adapted_base\":"
                                           << kl_divergence(value.logits, baseline.logits) << ",\"greedy_changed\":"
                                           << (value.metrics.greedy != baseline.metrics.greedy ? "true" : "false") << "}\n";
                            ++protective_rows;
                        }
                    }
                };

                evaluate_checkpoint(0, PRIMARY_BUDGET);
                evaluate_checkpoint(0, SECONDARY_BUDGET);
                for (const auto & example : ep.training) {
                    ++event_index;
                    const double before_norm = bank.frobenius_norm();

                    // ACT: generate with the current sidecar, then TRACE without modifying state.
                    const auto action = evaluate_bank(ctx, vocab, collector, output, bank, example.prompt, example.target, PRIMARY_BUDGET);
                    const llama_token produced = action.metrics.greedy;
                    const bool correct = produced == example.target;
                    const double reward = correct ? 1.0 : -1.0;
                    const auto z_minus = action.key;
                    const int32_t rank_at_trace = bank.components.size();
                    const double state_at_trace = bank.frobenius_norm();
                    traces << std::setprecision(17) << "{\"phase\":\"TRACE\",\"condition\":\""
                           << CONDITION_NAMES[condition] << "\",\"family\":" << ep.family << ",\"seed\":" << ep.seed
                           << ",\"event\":" << event_index << ",\"input_index\":" << example.input_index
                           << ",\"produced_token\":" << produced << ",\"target_token\":" << example.target
                           << ",\"z_negative_norm\":" << l2(z_minus) << ",\"rank_at_trace\":" << rank_at_trace
                           << ",\"state_norm_at_trace\":" << state_at_trace << ",\"update_before_feedback\":false}\n";
                    no_feedback_state_unchanged = no_feedback_state_unchanged &&
                        rank_at_trace == (int32_t) bank.components.size() && state_at_trace == bank.frobenius_norm();

                    std::vector<component> pending;
                    bool corrected_forward = false;
                    llama_token corrected_greedy = -1;
                    double corrected_target_probability = 0;
                    std::vector<float> z_plus;
                    llama_token feedback_target = example.target;
                    if (condition == 4 && !correct) {
                        for (size_t offset = 0; offset < ep.training.size(); ++offset) {
                            const llama_token candidate = ep.training[(event_index + offset) % ep.training.size()].target;
                            if (candidate != example.target && candidate != produced) {
                                feedback_target = candidate;
                                break;
                            }
                        }
                    }

                    // FEEDBACK: only this section is allowed to change the bank.
                    if (condition == 1) {
                        append_component(pending, normalized_output_row(output, produced), z_minus,
                                         ETA_POSITIVE, event_index, example.input_index, "blind_positive");
                    } else if (condition == 2) {
                        if (correct) {
                            append_component(pending, normalized_output_row(output, produced), z_minus,
                                             ETA_POSITIVE, event_index, example.input_index, "reward_positive");
                        } else {
                            const auto q_minus = output_row_difference(output, produced, example.target);
                            append_component(pending, q_minus, z_minus, -ETA_NEGATIVE,
                                             event_index, example.input_index, "reward_negative_wrong");
                        }
                    } else if (condition == 3 || condition == 4 || condition == 5) {
                        if (correct) {
                            auto q = condition == 5
                                ? deterministic_random_direction(BENCHMARK_SEED + 100000*ep.family + 1000*ep.seed + event_index)
                                : normalized_output_row(output, produced);
                            append_component(pending, q, z_minus, ETA_POSITIVE,
                                             event_index, example.input_index, "positive_correct");
                        } else {
                            const auto feedback_symbol = std::find_if(symbols.begin(), symbols.end(),
                                [&](const symbol_info & symbol) { return symbol.token == feedback_target; });
                            if (feedback_symbol == symbols.end()) throw std::runtime_error("feedback token is not a task symbol");
                            const auto corrected = evaluate(ctx, vocab, collector,
                                corrected_prompt(symbols[example.input_index], *feedback_symbol),
                                feedback_target, true);
                            z_plus = normalized(corrected.final_hidden);
                            corrected_forward = true;
                            corrected_greedy = corrected.metrics.greedy;
                            corrected_target_probability = corrected.metrics.probability;
                            if (condition == 5) {
                                append_component(pending,
                                    deterministic_random_direction(BENCHMARK_SEED + 200000*ep.family + 2000*ep.seed + 2*event_index),
                                    z_minus, ETA_NEGATIVE, event_index, example.input_index, "random_negative_slot");
                                append_component(pending,
                                    deterministic_random_direction(BENCHMARK_SEED + 200000*ep.family + 2000*ep.seed + 2*event_index + 1),
                                    z_plus, ETA_POSITIVE, event_index, example.input_index, "random_positive_slot");
                            } else {
                                const auto q_plus = output_row_difference(output, feedback_target, produced);
                                const auto q_minus = output_row_difference(output, produced, feedback_target);
                                append_component(pending, q_minus, z_minus, -ETA_NEGATIVE,
                                                 event_index, example.input_index, "negative_wrong");
                                append_component(pending, q_plus, z_plus, ETA_POSITIVE,
                                                 event_index, example.input_index, "positive_corrected");
                            }
                        }
                    }

                    if (condition != 0) {
                        for (auto & term : pending) bank.add(std::move(term));
                    }
                    const double after_norm = bank.frobenius_norm();
                    events << std::setprecision(17) << "{\"phase\":\"FEEDBACK\",\"condition\":\""
                           << CONDITION_NAMES[condition] << "\",\"family\":" << ep.family << ",\"seed\":" << ep.seed
                           << ",\"event\":" << event_index << ",\"input_index\":" << example.input_index
                           << ",\"produced_token\":" << produced << ",\"produced_text\":\""
                           << json_escape(token_piece(vocab, produced)) << "\",\"correct_token\":" << example.target
                           << ",\"correct_text\":\"" << json_escape(token_piece(vocab, example.target))
                           << "\",\"feedback_target\":" << feedback_target << ",\"correct\":"
                           << (correct ? "true" : "false") << ",\"reward\":" << reward
                           << ",\"reward_prediction\":0,\"delta\":" << reward
                           << ",\"reward_mode\":\"raw_reward_primary\",\"corrected_forward\":"
                           << (corrected_forward ? "true" : "false") << ",\"corrected_greedy_token\":"
                           << corrected_greedy << ",\"corrected_target_probability\":"
                           << corrected_target_probability << ",\"corrected_trajectory_correct\":"
                           << (corrected_forward && corrected_greedy == feedback_target ? "true" : "false")
                           << ",\"positive_update_norm\":" << coefficient_norm(pending, false)
                           << ",\"negative_update_norm\":" << coefficient_norm(pending, true)
                           << ",\"plastic_state_norm_before\":" << before_norm
                           << ",\"plastic_state_norm_after\":" << after_norm << ",\"rank_before\":" << rank_at_trace
                           << ",\"rank_after\":" << bank.components.size() << ",\"compression_event\":false,"
                           << "\"B_before\":" << action.budget << ",\"B_after_limit\":" << PRIMARY_BUDGET << "}\n";
                    ++update_rows;

                    if (std::find(CHECKPOINTS.begin(), CHECKPOINTS.end(), event_index) != CHECKPOINTS.end()) {
                        evaluate_checkpoint(event_index, PRIMARY_BUDGET);
                        evaluate_checkpoint(event_index, SECONDARY_BUDGET);
                    }
                }

                // Reset must recover the same frozen logits exactly.
                const auto reset_prompt = ep.heldout.front().prompt;
                clear_internal_injection(ctx);
                const auto before_reset = evaluate(ctx, vocab, collector, reset_prompt, ep.heldout.front().target, false);
                bank.clear();
                clear_internal_injection(ctx);
                const auto after_reset = evaluate(ctx, vocab, collector, reset_prompt, ep.heldout.front().target, false);
                reset_max = std::max(reset_max, max_logit_difference(before_reset.logits, after_reset.logits));
            }
        }

        // Explicit zero-sidecar equivalence check with instrumentation enabled.
        fast_bank empty;
        const auto & validation_example = episodes.front().heldout.front();
        clear_internal_injection(ctx);
        const auto direct = evaluate(ctx, vocab, collector, validation_example.prompt, validation_example.target, true);
        const auto zero = evaluate_bank(ctx, vocab, collector, output, empty, validation_example.prompt,
                                        validation_example.target, PRIMARY_BUDGET);
        zero_state_max = max_logit_difference(direct.logits, zero.logits);

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"raw_validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"zero_state_max_logit_difference\":" << zero_state_max
                   << ",\"reset_max_logit_difference\":" << reset_max
                   << ",\"no_update_before_feedback\":" << (no_feedback_state_unchanged ? "true" : "false")
                   << ",\"max_global_authority\":" << max_budget_observed
                   << ",\"authority_limit\":0.05,\"base_model_mutated\":false,\"backpropagation\":false,"
                   << "\"optimizer\":false,\"dense_plastic_matrix\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17) << "{\"wall_seconds\":" << elapsed
                << ",\"model_decode_calls\":" << decode_calls << ",\"update_rows\":" << update_rows
                << ",\"measurement_rows\":" << measurement_rows << ",\"protective_rows\":" << protective_rows
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size << "}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"fast_rank\":8,\"persistent_sidecar_bytes_fp32\":163840,"
               << "\"eligibility_trace_bytes_fp32\":20480,\"reward_predictor_bytes\":0,"
               << "\"temporary_correction_vector_bytes_fp32\":10240,\"component_metadata_bytes_estimate\":1024,"
               << "\"full_dense_matrix_bytes\":0,\"model_and_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        manifest << "{\"experiment\":\"qwen35_teacher_modulated_plasticity\",\"model\":\""
                 << json_escape(opts.model) << "\",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                 << ",\"sparse_residual_file\":\""
                 << (sparse ? json_escape(std::filesystem::path(sparse).filename().string()) : "")
                 << "\",\"family_count\":3,\"seeds_per_family\":3,\"episode_count\":9,"
                 << "\"condition_count\":6,\"primary_budget\":0.03,\"secondary_budget\":0.05,"
                 << "\"fast_rank\":8,\"eta_positive\":0.05,\"eta_negative\":0.05,"
                 << "\"reward_mode\":\"raw_reward_primary\",\"eligibility_lambda\":1.0,"
                 << "\"injection_sites\":[\"block_24\",\"block_28\",\"final\"],"
                 << "\"published_weights\":[0.23072,0.38322,0.89438],\"base_model_mutated\":false,"
                 << "\"backpropagation\":false,\"autograd\":false,\"optimizer\":false}\n";

        clear_internal_injection(ctx);
        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
