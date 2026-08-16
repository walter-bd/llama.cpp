#include "../injection-sensitivity-allocation/multi-injection-support.inc"

#include <array>
#include <set>

namespace {

constexpr double GATE_ETA = 0.05;
constexpr double B_MAX = 0.01;
constexpr double S_REF = 0.05;
constexpr double FP_TOLERANCE = 1e-6;
constexpr double AUTHORITY_TOLERANCE = 1e-6;

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
    double coefficient = GATE_ETA;
    std::string source;
    std::string destination;
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

std::string token_piece_gate(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize(-count);
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    return count > 0 ? std::string(buffer.data(), count) : std::string();
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

struct stream_event {
    std::string source;
    std::string destination;
    std::string category;
};

const std::array<stream_event, 18> EVENT_STREAM = {{
    {"A", "D", "deterministic"},
    {"B", "E", "deterministic"},
    {"H", "A", "conflicting"},
    {"G", "D", "conflicting"},
    {"C", "F", "one_off"},
    {"D", "G", "one_off"},
    {"A", "D", "deterministic"},
    {"B", "E", "deterministic"},
    {"H", "B", "conflicting"},
    {"G", "E", "conflicting"},
    {"E", "H", "one_off"},
    {"F", "A", "one_off"},
    {"A", "D", "deterministic"},
    {"B", "E", "deterministic"},
    {"A", "D", "deterministic"},
    {"B", "E", "deterministic"},
    {"H", "C", "conflicting"},
    {"G", "F", "conflicting"},
}};

const std::array<std::pair<int32_t, const char *>, 4> CHECKPOINTS = {{
    {0, "0"}, {2, "after_first_exposure"},
    {8, "after_second_confirmed_exposure"}, {16, "after_fourth_confirmed_exposure"},
}};

struct online_bank {
    std::vector<rank_one_component> components;

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

    void clear() { components.clear(); }

    double association_strength(const std::string & source, const std::string & destination) const {
        double strength = 0;
        for (const auto & component : components) {
            if (component.source == source && component.destination == destination) {
                strength += component.coefficient;
            }
        }
        return strength;
    }

    void consolidate(
            const std::string & source,
            const std::string & destination,
            const std::vector<float> & value,
            const std::vector<float> & key,
            double increment) {
        for (auto & component : components) {
            if (component.source == source && component.destination == destination) {
                component.coefficient = std::min(GATE_ETA, component.coefficient + increment);
                return;
            }
        }
        components.push_back({value, key, std::min(GATE_ETA, increment), source, destination});
    }
};

struct source_state {
    int32_t id = 0;
    std::string symbol;
    std::string prompt;
    std::vector<llama_token> prompt_tokens;
    eval_result baseline;
    std::vector<float> key;
};

struct bank_probe {
    double raw_response_norm = 0;
    double retrieval_ratio = 0;
    double effective_budget = 0;
    multi_probe_result probe;
};

struct target_symbol {
    llama_token token = -1;
    std::string text;
};

using destination_counts = std::map<std::string, int32_t>;
using source_counts = std::map<std::string, destination_counts>;

std::string source_prompt(const std::string & symbol) {
    return "Observed transition memory.\nSource symbol: " + symbol + "\nDestination symbol:";
}

int32_t count_total(const destination_counts & counts) {
    int32_t total = 0;
    for (const auto & [_, count] : counts) total += count;
    return total;
}

std::string predicted_destination(const destination_counts & counts) {
    std::string result;
    int32_t best = 0;
    for (const auto & [destination, count] : counts) {
        if (count > best) {
            best = count;
            result = destination;
        }
    }
    return result;
}

void write_counts(std::ostream & out, const destination_counts & counts) {
    out << '{';
    bool first = true;
    for (const auto & [destination, count] : counts) {
        out << (first ? "\"" : ",\"") << destination << "\":" << count;
        first = false;
    }
    out << '}';
}

void write_probe_rhos(std::ostream & out, const multi_probe_result & probe) {
    auto value = [&](int32_t site) {
        const auto found = probe.actual_rhos.find(site);
        return found == probe.actual_rhos.end() ? 0.0 : found->second;
    };
    out << std::setprecision(17)
        << "\"global_B\":" << probe.actual_budget
        << ",\"rho24\":" << value(23)
        << ",\"rho28\":" << value(27)
        << ",\"rho_final\":" << value(-1);
}

bank_probe apply_bank(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const online_bank & bank,
        const source_state & source,
        llama_token target,
        const std::string & target_text) {
    bank_probe result;
    const auto response = bank.apply(source.key);
    result.raw_response_norm = l2(response);
    result.retrieval_ratio = result.raw_response_norm/S_REF;
    result.effective_budget = B_MAX*std::min(1.0, result.retrieval_ratio);

    result.probe.logits = source.baseline.logits;
    result.probe.metrics = metrics_from_logits(source.baseline.logits, target);
    if (result.raw_response_norm <= 1e-12) return result;

    const auto direction = normalized(response);
    eval_result baseline = source.baseline;
    baseline.metrics = metrics_from_logits(baseline.logits, target);
    item definition{source.id, source.prompt, target, target_text};
    case_state state{definition, std::move(baseline), direction, output_projection(output, direction)};
    const multi_configuration config{"reader", "block24_block28_final", {23, 27, -1}};
    const auto weights = normalized_weights({0.23072, 0.38322, 0.89438});
    result.probe = probe_multi(ctx, vocab, collector, state, config, weights, result.effective_budget);
    clear_internal_injection(ctx);
    if (result.probe.actual_budget > B_MAX + AUTHORITY_TOLERANCE) {
        throw std::runtime_error("autonomous reader exceeded authority limit");
    }
    return result;
}

std::vector<std::pair<std::string, std::string>> unique_pairs() {
    std::vector<std::pair<std::string, std::string>> out;
    std::set<std::pair<std::string, std::string>> seen;
    for (const auto & event : EVENT_STREAM) {
        if (seen.emplace(event.source, event.destination).second) {
            out.emplace_back(event.source, event.destination);
        }
    }
    return out;
}

std::string pair_category(const std::string & source, const std::string & destination) {
    for (const auto & event : EVENT_STREAM) {
        if (event.source == source && event.destination == destination) return event.category;
    }
    throw std::runtime_error("unknown stream pair");
}

std::map<std::string, target_symbol> assign_targets(
        const std::vector<std::pair<llama_token, std::string>> & candidates,
        const std::map<std::string, source_state> & sources) {
    std::map<std::string, target_symbol> assigned;
    std::set<llama_token> used;
    const std::array<std::string, 8> symbols = {"A", "B", "C", "D", "E", "F", "G", "H"};
    for (const auto & destination : symbols) {
        bool found = false;
        for (const auto & candidate : candidates) {
            if (used.count(candidate.first)) continue;
            bool valid = true;
            for (const auto & event : EVENT_STREAM) {
                if (event.destination == destination &&
                    sources.at(event.source).baseline.metrics.greedy == candidate.first) {
                    valid = false;
                }
            }
            if (!valid) continue;
            assigned[destination] = {candidate.first, candidate.second};
            used.insert(candidate.first);
            found = true;
            break;
        }
        if (!found) throw std::runtime_error("could not assign distinct non-greedy target token");
    }
    return assigned;
}

} // namespace

#ifndef CLOSED_LOOP_SIDECAR_REASONING_EMBED
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

        std::map<std::string, source_state> sources;
        const std::array<std::string, 8> symbols = {"A", "B", "C", "D", "E", "F", "G", "H"};
        for (size_t index = 0; index < symbols.size(); ++index) {
            const auto prompt = source_prompt(symbols[index]);
            clear_internal_injection(ctx);
            auto baseline = evaluate(ctx, vocab, collector, prompt, 0, true);
            sources.emplace(symbols[index], source_state{
                (int32_t) index, symbols[index], prompt, tokenize(vocab, prompt, true),
                std::move(baseline), {}});
            sources.at(symbols[index]).key = normalized(sources.at(symbols[index]).baseline.final_hidden);
        }
        const auto targets = assign_targets(choose_single_tokens(vocab), sources);

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

        std::ofstream events_out(root/"event_write_results.jsonl", std::ios::trunc);
        std::ofstream probes_out(root/"behavior_probe_results.jsonl", std::ios::trunc);
        std::ofstream protective_out(root/"protective_results.jsonl", std::ios::trunc);
        std::ofstream reset_out(root/"reset_results.jsonl", std::ios::trunc);
        for (auto * stream : {&events_out, &probes_out, &protective_out, &reset_out}) {
            stream->setf(std::ios::unitbuf);
        }

        double zero_state_max = 0;
        online_bank empty_bank;
        for (const auto & [symbol, source] : sources) {
            const auto & target = targets.at(symbol);
            const auto zero = apply_bank(ctx, vocab, collector, output, empty_bank, source, target.token, target.text);
            zero_state_max = std::max(zero_state_max,
                max_logit_difference(zero.probe.logits, source.baseline.logits));
        }

        double maximum_budget = 0;
        double reset_max = 0;
#ifdef RECURRENCE_CONSOLIDATION_EXPERIMENT
        const std::array<std::string, 2> conditions = {"C0_append_every_write", "C1_consolidated_repeated_write"};
#else
        const std::array<std::string, 3> conditions = {"C0_frozen", "C1_autonomous_gate", "C2_constant_0.25"};
#endif
        const auto pairs = unique_pairs();
        size_t maximum_rank_components = 0;

        for (const auto & condition : conditions) {
            online_bank bank;
            source_counts counts;

            auto write_checkpoint = [&](int32_t event_count, const std::string & checkpoint) {
                for (const auto & [source_symbol, destination_symbol] : pairs) {
                    const auto & source = sources.at(source_symbol);
                    const auto & target = targets.at(destination_symbol);
                    const auto result = apply_bank(ctx, vocab, collector, output, bank, source, target.token, target.text);
                    maximum_budget = std::max(maximum_budget, result.probe.actual_budget);
                    probes_out << std::setprecision(17)
                        << "{\"condition\":\"" << condition << "\""
                        << ",\"checkpoint\":\"" << checkpoint << "\""
                        << ",\"event_count\":" << event_count
                        << ",\"source\":\"" << source_symbol << "\""
                        << ",\"destination\":\"" << destination_symbol << "\""
                        << ",\"category\":\"" << pair_category(source_symbol, destination_symbol) << "\""
                        << ",\"prompt\":\"" << json_escape(source.prompt) << "\""
                        << ",\"target_token\":" << target.token
                        << ",\"target_text\":\"" << json_escape(target.text) << "\""
                        << ",\"component_count\":" << bank.components.size()
                        << ",\"association_strength\":" << bank.association_strength(source_symbol, destination_symbol)
                        << ",\"raw_Pz_norm\":" << result.raw_response_norm
                        << ",\"retrieval_ratio\":" << result.retrieval_ratio
                        << ",\"effective_B_requested\":" << result.effective_budget << ',';
                    write_probe_rhos(probes_out, result.probe);
                    probes_out << ",\"before\":";
                    write_metrics(probes_out, source.baseline.logits, target.token);
                    probes_out << ",\"after\":";
                    write_metrics(probes_out, result.probe.logits, target.token);
                    probes_out << "}\n";
                }
            };

            write_checkpoint(0, "0");
            for (size_t event_index = 0; event_index < EVENT_STREAM.size(); ++event_index) {
                const auto & event = EVENT_STREAM[event_index];
                const auto & source = sources.at(event.source);
                const auto & target = targets.at(event.destination);
                const auto before_counts = counts[event.source];
                const int32_t n_x = count_total(before_counts);
                const int32_t n_xy = before_counts.count(event.destination) ? before_counts.at(event.destination) : 0;
                const std::string prediction = predicted_destination(before_counts);
                const int32_t prediction_count = prediction.empty() ? 0 : before_counts.at(prediction);
                const double confidence = n_x > 0 ? (double) prediction_count/n_x : 0.0;
                const double novelty = n_x == 0 ? 1.0 : 0.0;
                const double probability = n_x > 0 ? (double) n_xy/n_x : 0.0;
                const double surprise = 1.0 - probability;
                const double recurrence = std::min(1.0, n_x/3.0);
                const double consistency = probability;
                const double autonomous_gate = std::clamp(
                    0.25*std::max(novelty, surprise) + recurrence*consistency, 0.0, 1.0);
#ifdef RECURRENCE_CONSOLIDATION_EXPERIMENT
                const double applied_gate = autonomous_gate;
#else
                const double applied_gate = condition == "C1_autonomous_gate" ? autonomous_gate
                    : condition == "C2_constant_0.25" ? 0.25 : 0.0;
#endif

                const auto behavior = apply_bank(ctx, vocab, collector, output, bank, source, target.token, target.text);
                maximum_budget = std::max(maximum_budget, behavior.probe.actual_budget);
                const llama_token reference_token = source.baseline.metrics.greedy;
                const auto direction = normalized(output_row_difference(output, target.token, reference_token));
                const double coefficient = GATE_ETA*applied_gate;
                const size_t rank_before = bank.components.size();
                const double strength_before = bank.association_strength(event.source, event.destination);
                if (coefficient > 0) {
#ifdef RECURRENCE_CONSOLIDATION_EXPERIMENT
                    if (condition == "C1_consolidated_repeated_write") {
                        bank.consolidate(event.source, event.destination, direction, source.key, coefficient);
                    } else {
                        bank.components.push_back({direction, source.key, coefficient, event.source, event.destination});
                    }
#else
                    bank.components.push_back({direction, source.key, coefficient, event.source, event.destination});
#endif
                }
                maximum_rank_components = std::max(maximum_rank_components, bank.components.size());
                const double strength_after = bank.association_strength(event.source, event.destination);
                counts[event.source][event.destination] += 1;

                events_out << std::setprecision(17)
                    << "{\"condition\":\"" << condition << "\""
                    << ",\"event_index\":" << event_index + 1
                    << ",\"source\":\"" << event.source << "\""
                    << ",\"destination\":\"" << event.destination << "\""
                    << ",\"category\":\"" << event.category << "\""
                    << ",\"prediction_before\":\"" << prediction << "\""
                    << ",\"prediction_confidence\":" << confidence
                    << ",\"prediction_correct\":" << (!prediction.empty() && prediction == event.destination ? "true" : "false")
                    << ",\"n_x_before\":" << n_x
                    << ",\"n_xy_before\":" << n_xy
                    << ",\"candidate_counts_before\":";
                write_counts(events_out, before_counts);
                events_out << ",\"N\":" << novelty
                    << ",\"S\":" << surprise
                    << ",\"R\":" << recurrence
                    << ",\"C\":" << consistency
                    << ",\"M\":0"
                    << ",\"autonomous_G\":" << autonomous_gate
                    << ",\"applied_G\":" << applied_gate
                    << ",\"eta\":" << GATE_ETA
                    << ",\"candidate_write_magnitude\":" << coefficient
                    << ",\"write_applied\":" << (coefficient > 0 ? "true" : "false")
                    << ",\"key_norm\":" << l2(source.key)
                    << ",\"direction_norm\":" << l2(direction)
                    << ",\"direction_reference_token\":" << reference_token
                    << ",\"observed_target_token\":" << target.token
                    << ",\"behavior_greedy_before_write\":" << behavior.probe.metrics.greedy
                    << ",\"raw_Pz_norm_before_write\":" << behavior.raw_response_norm
                    << ",\"B_eff_before_write\":" << behavior.effective_budget
                    << ",\"rank_before\":" << rank_before
                    << ",\"rank_after\":" << bank.components.size()
                    << ",\"association_strength_before\":" << strength_before
                    << ",\"association_strength_after\":" << strength_after
                    << ",\"write_behavior\":\""
#ifdef RECURRENCE_CONSOLIDATION_EXPERIMENT
                    << (condition == "C1_consolidated_repeated_write" ? "consolidate_exact_association" : "append_every_write")
#else
                    << "append_every_write"
#endif
                    << "\""
                    << ",\"candidate_counts_after\":";
                write_counts(events_out, counts[event.source]);
                events_out << "}\n";

                const int32_t processed = (int32_t) event_index + 1;
                for (const auto & [checkpoint_count, checkpoint_name] : CHECKPOINTS) {
                    if (processed == checkpoint_count && checkpoint_count > 0) {
                        write_checkpoint(processed, checkpoint_name);
                    }
                }
            }

            for (const auto & heldout : protective) {
                source_state source{heldout.id, "protective_" + std::to_string(heldout.id), heldout.prompt,
                    heldout.prompt_tokens, heldout.baseline, heldout.key};
                const auto result = apply_bank(ctx, vocab, collector, output, bank, source,
                    heldout.target, heldout.target_text);
                maximum_budget = std::max(maximum_budget, result.probe.actual_budget);
                protective_out << std::setprecision(17)
                    << "{\"condition\":\"" << condition << "\""
                    << ",\"prompt_id\":" << heldout.id
                    << ",\"prompt\":\"" << json_escape(heldout.prompt) << "\""
                    << ",\"raw_Pz_norm\":" << result.raw_response_norm
                    << ",\"retrieval_ratio\":" << result.retrieval_ratio
                    << ",\"effective_B_requested\":" << result.effective_budget
                    << ",\"kl_plastic_base\":" << kl_divergence(result.probe.logits, heldout.baseline.logits)
                    << ",\"baseline_greedy_token\":" << heldout.target
                    << ",\"plastic_greedy_token\":" << result.probe.metrics.greedy
                    << ",\"greedy_changed\":" << (result.probe.metrics.greedy != heldout.target ? "true" : "false") << ',';
                write_probe_rhos(protective_out, result.probe);
                protective_out << "}\n";
            }

            bank.clear();
            clear_internal_injection(ctx);
            double condition_reset_max = 0;
            for (const auto & symbol : {std::string("A"), std::string("B")}) {
                const auto & source = sources.at(symbol);
                const auto restored = evaluate(ctx, vocab, collector, source.prompt, 0, false);
                condition_reset_max = std::max(condition_reset_max,
                    max_logit_difference(restored.logits, source.baseline.logits));
            }
            for (size_t index = 0; index < 2; ++index) {
                const auto restored = evaluate(ctx, vocab, collector, protective[index].prompt, 0, false);
                condition_reset_max = std::max(condition_reset_max,
                    max_logit_difference(restored.logits, protective[index].baseline.logits));
            }
            reset_max = std::max(reset_max, condition_reset_max);
            reset_out << std::setprecision(17)
                << "{\"condition\":\"" << condition << "\""
                << ",\"component_count_after_clear\":0"
                << ",\"max_logit_difference\":" << condition_reset_max
                << ",\"reset_exact\":" << (condition_reset_max <= FP_TOLERANCE ? "true" : "false") << "}\n";
            if (condition_reset_max > FP_TOLERANCE) throw std::runtime_error("condition reset not exact");
        }

        clear_internal_injection(ctx);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream dataset(root/"stream_definition.json", std::ios::trunc);
        dataset << "{\"events\":[";
        for (size_t index = 0; index < EVENT_STREAM.size(); ++index) {
            const auto & event = EVENT_STREAM[index];
            dataset << (index ? ",{" : "{") << "\"index\":" << index + 1
                << ",\"source\":\"" << event.source << "\",\"destination\":\""
                << event.destination << "\",\"category\":\"" << event.category << "\"}";
        }
        dataset << "],\"target_token_map\":{";
        bool first_target = true;
        for (const auto & [symbol, target] : targets) {
            dataset << (first_target ? "\"" : ",\"") << symbol << "\":{\"token\":"
                << target.token << ",\"text\":\"" << json_escape(target.text) << "\"}";
            first_target = false;
        }
        dataset << "},\"checkpoints\":[0,2,8,16]}\n";

        std::ofstream validation(root/"raw_validation.json", std::ios::trunc);
        validation << std::setprecision(17)
            << "{\"zero_state_max_logit_difference\":" << zero_state_max
            << ",\"reset_max_logit_difference\":" << reset_max
            << ",\"max_effective_B\":" << maximum_budget
            << ",\"authority_limit\":" << B_MAX
            << ",\"pre_update_gate_statistics\":true"
            << ",\"base_model_mutated\":false,\"gradients\":false,\"optimizer\":false"
            << ",\"backpropagation\":false,\"persistent_state_after_run\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
            << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":" << decode_calls
            << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size
            << ",\"event_count_per_condition\":18,\"condition_count\":" << conditions.size()
            << ",\"checkpoint_count\":4,\"protective_prompts\":8}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"maximum_rank_components\":" << maximum_rank_components
            << ",\"bytes_per_rank_one_component_fp32\":20480,"
            << "\"maximum_rank_bank_bytes_fp32\":" << maximum_rank_components*20480
            << ",\"source_count_memory_bytes_upper_bound\":1024,"
            << "\"boundary_trace_bytes_fp32\":81920,\"temporary_sparse_cvec_host_bytes\":317440,"
            << "\"dense_plastic_matrix_bytes\":0,\"model_and_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
#ifdef RECURRENCE_CONSOLIDATION_EXPERIMENT
        manifest << "{\"experiment\":\"qwen35_recurrence_write_consolidation\",\"model\":\""
#else
        manifest << "{\"experiment\":\"qwen35_autonomous_sidecar_integration\",\"model\":\""
#endif
            << json_escape(opts.model) << "\",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
            << ",\"eta\":0.05,\"B_max\":0.01,\"s_ref\":0.05"
            << ",\"gate\":\"clip01(0.25*max(N,S)+R*C)\""
#ifdef RECURRENCE_CONSOLIDATION_EXPERIMENT
            << ",\"conditions\":[\"C0_append_every_write\",\"C1_consolidated_repeated_write\"]"
            << ",\"consolidated_strength_rule\":\"min(0.05,s+0.05*G)\""
            << ",\"consolidated_key_and_direction\":\"fixed_from_first_observation\""
#else
            << ",\"conditions\":[\"C0_frozen\",\"C1_autonomous_gate\",\"C2_constant_0.25\"]"
#endif
            << ",\"injection_sites\":[\"block_24\",\"block_28\",\"final\"]"
            << ",\"raw_allocation_weights\":[0.23072,0.38322,0.89438]"
            << ",\"reader\":\"B_eff=0.01*min(1,||Pz||/0.05)\""
            << ",\"write\":\"eta*G*normalize(w_t-w_a)*zhat^T\""
            << ",\"direction_reference\":\"frozen baseline greedy token for source prompt\""
            << ",\"base_model_mutated\":false,\"gradients\":false,\"optimizer\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
#endif
