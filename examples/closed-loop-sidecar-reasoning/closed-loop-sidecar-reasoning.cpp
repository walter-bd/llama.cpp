#define CLOSED_LOOP_SIDECAR_REASONING_EMBED
#include "../autonomous-sidecar-integration/autonomous-sidecar-integration.cpp"
#undef CLOSED_LOOP_SIDECAR_REASONING_EMBED

#include "chat.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double STEP27_B_MAX = 0.01;
constexpr double STEP27_S_REF = 0.05;
constexpr double STEP27_ETA = 0.05;
constexpr double STEP27_FIRST_GATE = 0.25;
constexpr int32_t STEP27_MAX_TOKENS = 32;
constexpr double STEP27_TOLERANCE = 1e-6;

struct step27_prompt {
    int32_t id = 0;
    std::string name;
    std::string source;
    std::string user_prompt;
};

struct source_memory {
    std::string source;
    std::string destination;
    llama_token target = -1;
    std::string target_text;
    std::string prompt;
    std::vector<float> key;
    std::vector<float> direction;
};

struct step27_bank {
    std::vector<rank_one_component> components;

    std::vector<float> apply_source(const std::string & source, const std::vector<float> & key) const {
        std::vector<float> response(WIDTH, 0.0f);
        for (const auto & component : components) {
            if (component.source != source) continue;
            double activation = 0.0;
            for (int32_t i = 0; i < WIDTH; ++i) activation += (double) component.key[i]*key[i];
            const double scale = component.coefficient*activation;
            for (int32_t i = 0; i < WIDTH; ++i) response[i] += (float) (scale*component.value[i]);
        }
        return response;
    }
};

struct step27_counts {
    int32_t total = 0;
    int32_t matching = 0;
};

struct step27_episode_result {
    std::vector<llama_token> tokens;
    std::string text;
    eval_result readout;
};

std::string render_step27_chat(
        const common_chat_templates * templates,
        const std::string & user,
        bool & supports_thinking) {
    common_chat_msg message;
    message.role = "user";
    message.content = user;
    common_chat_templates_inputs inputs;
    inputs.messages.push_back(message);
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = true;
    const auto params = common_chat_templates_apply(templates, inputs);
    supports_thinking = params.supports_thinking;
    return params.prompt;
}

std::vector<step27_prompt> make_prompts() {
    return {
        {0, "A1", "A", "Source A has a learned destination in memory. Think through the retrieval before answering."},
        {1, "A2", "A", "Reason about the stored association whose source is A, then determine its destination."},
        {2, "B1", "B", "Source B has a learned destination in memory. Think through the retrieval before answering."},
        {3, "B2", "B", "Reason about the stored association whose source is B, then determine its destination."},
    };
}

uint64_t checksum64_step27(const std::vector<float> & values) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(values.data());
    const size_t size = values.size()*sizeof(float);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hex64_step27(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::string piece_step27(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize(-count);
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    return count > 0 ? std::string(buffer.data(), count) : std::string();
}

void write_metric_fields(std::ostream & out, const std::vector<float> & logits, llama_token target) {
    const auto metrics = metrics_from_logits(logits, target);
    out << std::setprecision(17)
        << "\"target_logit\":" << logits[target]
        << ",\"target_nll\":" << metrics.nll
        << ",\"target_probability\":" << metrics.probability
        << ",\"target_rank\":" << metrics.rank
        << ",\"target_margin\":" << metrics.margin
        << ",\"greedy_token\":" << metrics.greedy
        << ",\"target_is_top1\":" << (metrics.greedy == target ? "true" : "false");
}

void write_vector_binary(std::ofstream & out, const std::vector<float> & values) {
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size()*sizeof(float)));
    if (!out) throw std::runtime_error("failed writing Step 27 vector");
}

void write_bank_snapshot(const std::filesystem::path & directory, const step27_bank & bank) {
    std::filesystem::create_directories(directory);
    std::ofstream metadata(directory/"components.jsonl", std::ios::trunc);
    std::ofstream vectors(directory/"vectors.f32", std::ios::binary | std::ios::trunc);
    if (!metadata || !vectors) throw std::runtime_error("cannot write bank snapshot");
    size_t index = 0;
    for (const auto & component : bank.components) {
        const auto key_checksum = hex64_step27(checksum64_step27(component.key));
        const auto value_checksum = hex64_step27(checksum64_step27(component.value));
        write_vector_binary(vectors, component.key);
        write_vector_binary(vectors, component.value);
        metadata << std::setprecision(17)
            << "{\"component\":" << index++
            << ",\"source\":\"" << json_escape(component.source)
            << "\",\"destination\":\"" << json_escape(component.destination)
            << "\",\"coefficient\":" << component.coefficient
            << ",\"key_norm\":" << l2(component.key)
            << ",\"value_norm\":" << l2(component.value)
            << ",\"key_checksum\":\"" << key_checksum
            << "\",\"value_checksum\":\"" << value_checksum << "\"}\n";
    }
    std::ofstream summary(directory/"summary.json", std::ios::trunc);
    summary << "{\"component_count\":" << bank.components.size()
            << ",\"vector_dimension\":2560,";
    if (bank.components.empty()) {
        summary << "\"first_key_checksum\":null,\"first_value_checksum\":null}";
    } else {
        summary << "\"first_key_checksum\":\""
                << hex64_step27(checksum64_step27(bank.components.front().key))
                << "\",\"first_value_checksum\":\""
                << hex64_step27(checksum64_step27(bank.components.front().value)) << "\"}";
    }
}

double max_logit_diff_step27(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("logit size mismatch");
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) result = std::max(result, std::abs((double) a[i] - b[i]));
    return result;
}

struct active_probe {
    multi_probe_result probe;
    double raw_norm = 0.0;
    double requested_budget = 0.0;
    bool active = false;
};

active_probe probe_step27(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const step27_bank & bank,
        const source_memory & memory,
        const std::string & prompt,
        llama_token target,
        const std::string & target_text,
        int32_t prompt_id,
        bool active,
        const eval_result & baseline) {
    active_probe result;
    result.probe.logits = baseline.logits;
    result.probe.metrics = metrics_from_logits(baseline.logits, target);
    if (!active) return result;

    const auto query_key = normalized(baseline.final_hidden);
    const auto response = bank.apply_source(memory.source, query_key);
    result.raw_norm = l2(response);
    if (result.raw_norm <= 1e-12) return result;

    result.active = true;
    result.requested_budget = STEP27_B_MAX*std::min(1.0, result.raw_norm/STEP27_S_REF);
    const auto direction = normalized(response);
    const item definition{prompt_id, prompt, target, target_text};
    const case_state state{definition, baseline, direction, output_projection(output, direction)};
    const multi_configuration config{"step27", "block24_block28_final", {23, 27, -1}};
    const auto weights = normalized_weights({0.23072, 0.38322, 0.89438});
    result.probe = probe_multi(ctx, vocab, collector, state, config, weights, result.requested_budget);
    clear_internal_injection(ctx);
    if (result.probe.actual_budget > STEP27_B_MAX + STEP27_TOLERANCE) {
        throw std::runtime_error("Step 27 exceeded authority limit");
    }
    return result;
}

void write_injection_fields(std::ostream & out, const active_probe & probe) {
    auto actual = [&](int32_t site) {
        const auto found = probe.probe.actual_rhos.find(site);
        return found == probe.probe.actual_rhos.end() ? 0.0 : found->second;
    };
    auto alpha = [&](int32_t site) {
        const auto found = probe.probe.alphas.find(site);
        return found == probe.probe.alphas.end() ? 0.0 : found->second;
    };
    out << std::setprecision(17)
        << ",\"raw_retrieval_norm\":" << probe.raw_norm
        << ",\"requested_B\":" << probe.requested_budget
        << ",\"actual_B\":" << probe.probe.actual_budget
        << ",\"rho24\":" << actual(23)
        << ",\"rho28\":" << actual(27)
        << ",\"rho_final\":" << actual(-1)
        << ",\"alpha24\":" << alpha(23)
        << ",\"alpha28\":" << alpha(27)
        << ",\"alpha_final\":" << alpha(-1)
        << ",\"norm_iterations\":" << probe.probe.norm_iterations;
}

step27_episode_result run_episode(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const common_chat_templates * templates,
        const step27_prompt & query,
        const source_memory & memory,
        const step27_bank & bank,
        const std::string & condition,
        int32_t episode,
        int32_t cycle,
        bool only_first_step,
        std::ostream & trajectory_out,
        std::ostream & readout_out,
        bool & supports_thinking) {
    const std::string rendered = render_step27_chat(templates, query.user_prompt, supports_thinking);
    if (!supports_thinking) throw std::runtime_error("Step 27 chat template does not support thinking");
    std::string prefix = rendered;
    std::vector<llama_token> generated;
    std::string generated_text;

    for (int32_t step = 0; step < STEP27_MAX_TOKENS; ++step) {
        clear_internal_injection(ctx);
        const auto baseline = evaluate(ctx, vocab, collector, prefix, memory.target, true);
        const bool active = condition != "C0_Qwen_only" && (!only_first_step || step == 0);
        const auto probed = probe_step27(ctx, vocab, collector, output, bank, memory,
                                         prefix, memory.target, memory.target_text,
                                         query.id, active, baseline);
        const auto & logits = probed.probe.logits;
        const llama_token next = probed.probe.metrics.greedy;
        if (next < 0 || llama_vocab_is_eog(vocab, next)) break;
        const std::string piece = piece_step27(vocab, next);
        const auto & actual_final = probed.active ? collector.final_hidden : baseline.final_hidden;
        const auto & actual_boundaries = probed.active ? collector.boundaries : baseline.boundaries;

        trajectory_out << std::setprecision(17)
            << "{\"condition\":\"" << condition
            << "\",\"episode\":" << episode
            << ",\"cycle\":" << cycle
            << ",\"prompt_id\":" << query.id
            << ",\"prompt_name\":\"" << query.name
            << "\",\"source\":\"" << query.source
            << "\",\"step\":" << step
            << ",\"generated_token_id\":" << next
            << ",\"generated_token_piece\":\"" << json_escape(piece)
            << "\",\"active\":" << (probed.active ? "true" : "false")
            << ",\"target_token\":" << memory.target;
        write_injection_fields(trajectory_out, probed);
        trajectory_out << ",\"final_hidden_checksum\":\"" << hex64_step27(checksum64_step27(actual_final))
            << "\",\"block24_checksum\":\"" << hex64_step27(checksum64_step27(actual_boundaries.at(23)))
            << "\",\"block28_checksum\":\"" << hex64_step27(checksum64_step27(actual_boundaries.at(27)))
            << "\",\"block24_norm\":" << l2(actual_boundaries.at(23))
            << ",\"block28_norm\":" << l2(actual_boundaries.at(27))
            << ",\"final_norm\":" << l2(actual_final)
            << ",\"target_metrics\":{";
        write_metric_fields(trajectory_out, logits, memory.target);
        trajectory_out << "}}\n";

        generated.push_back(next);
        generated_text += piece;
        prefix += piece;
    }

    const std::string readout_prompt = prefix + "\nFinal destination:";
    clear_internal_injection(ctx);
    const auto readout = evaluate(ctx, vocab, collector, readout_prompt, memory.target, false);
    readout_out << std::setprecision(17)
        << "{\"condition\":\"" << condition
        << "\",\"episode\":" << episode
        << ",\"cycle\":" << cycle
        << ",\"prompt_id\":" << query.id
        << ",\"prompt_name\":\"" << query.name
        << "\",\"source\":\"" << query.source
        << "\",\"target_token\":" << memory.target
        << ",\"generated_token_count\":" << generated.size()
        << ",\"generated_text\":\"" << json_escape(generated_text)
        << "\",\"readout_suffix\":\"Final destination:\",\"target_metrics\":{";
    write_metric_fields(readout_out, readout.logits, memory.target);
    readout_out << "}}\n";
    return {std::move(generated), std::move(generated_text), readout};
}

double update_c3(
        step27_bank & bank,
        step27_counts & counts,
        const source_memory & memory,
        std::ostream & updates_out,
        int32_t episode,
        int32_t cycle) {
    const double novelty = counts.total == 0 ? 1.0 : 0.0;
    const double p = counts.total == 0 ? 0.0 : (double) counts.matching/counts.total;
    const double surprise = 1.0 - p;
    const double recurrence = std::min(1.0, counts.total/3.0);
    const double consistency = p;
    const double gate = std::clamp(0.25*std::max(novelty, surprise) + recurrence*consistency, 0.0, 1.0);
    const double coefficient = STEP27_ETA*gate;
    const size_t rank_before = bank.components.size();
    if (coefficient > 0) {
        bank.components.push_back({memory.direction, memory.key, coefficient, memory.source, memory.destination});
    }
    updates_out << std::setprecision(17)
        << "{\"episode\":" << episode
        << ",\"cycle\":" << cycle
        << ",\"source\":\"" << memory.source
        << "\",\"destination\":\"" << memory.destination
        << "\",\"count_total_before\":" << counts.total
        << ",\"count_matching_before\":" << counts.matching
        << ",\"N\":" << novelty << ",\"S\":" << surprise
        << ",\"R\":" << recurrence << ",\"C\":" << consistency
        << ",\"G\":" << gate << ",\"eta\":" << STEP27_ETA
        << ",\"coefficient\":" << coefficient
        << ",\"rank_before\":" << rank_before
        << ",\"rank_after\":" << bank.components.size() << "}\n";
    ++counts.total;
    ++counts.matching;
    return gate;
}

void write_preregistration(const std::filesystem::path & root) {
    std::ofstream out(root/"preregistration.json", std::ios::trunc);
    out << R"JSON({
"experiment":"step27_closed_loop_sidecar_reasoning",
"runtime_baseline":"rtx3090_20260814",
"model_immutable":true,
"source_partition":{"A":"D","B":"E"},
"injection_topology":{"block24":0.23072,"block28":0.38322,"final":0.89438},
"reader":"B_eff=0.01*min(1,||r||/0.05)",
"eta":0.05,"B_max":0.01,"s_ref":0.05,
"max_reasoning_tokens":32,
"conditions":["C0_Qwen_only","C1_one_shot","C2_continuous_frozen","C3_continuous_online_plasticity"],
"c3_schedule":["A1","B1","A2","B2","A1","B1","A2","B2","A1","B1","A2","B2","A1","B1","A2","B2"],
"learning_only_after_final_readout":true,
"generated_tokens_never_update":true,
"no_k1_addressing":true,
"final_readout_suffix":"Final destination:"
})JSON";
}

} // namespace

#ifdef STATEFUL_RECURSIVE_SIDECAR_EMBED
int step27_embedded_main(int argc, char ** argv) {
#else
int main(int argc, char ** argv) {
#endif
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto opts = parse_args(argc, argv);
        const std::filesystem::path root(opts.output_dir);
        std::filesystem::create_directories(root/"initial_bank");
        std::filesystem::create_directories(root/"c3_bank_checkpoints");
        write_preregistration(root);
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.gpu_layers;
        llama_model * public_model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (!public_model) throw std::runtime_error("model load failed");
        auto & model = *public_model;
        if (model.hparams.n_embd != WIDTH || model.hparams.n_layer() != 32) {
            throw std::runtime_error("Step 27 requires Qwen3.5-4B dimensions 32x2560");
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
        auto templates = common_chat_templates_init(public_model, "");
        if (!templates) throw std::runtime_error("chat template initialization failed");

        const std::array<std::string, 8> symbols = {"A","B","C","D","E","F","G","H"};
        std::map<std::string, source_state> source_states;
        for (size_t i = 0; i < symbols.size(); ++i) {
            const std::string prompt = source_prompt(symbols[i]);
            clear_internal_injection(ctx);
            auto baseline = evaluate(ctx, vocab, collector, prompt, 0, true);
            source_states.emplace(symbols[i], source_state{
                (int32_t) i, symbols[i], prompt, tokenize(vocab, prompt, true),
                std::move(baseline), {}});
            source_states.at(symbols[i]).key = normalized(source_states.at(symbols[i]).baseline.final_hidden);
        }
        const auto targets = assign_targets(choose_single_tokens(vocab), source_states);
        source_memory memory_a{"A", "D", targets.at("D").token, targets.at("D").text,
                               source_states.at("A").prompt, source_states.at("A").key,
                               normalized(output_row_difference(output, targets.at("D").token,
                                                               source_states.at("A").baseline.metrics.greedy))};
        source_memory memory_b{"B", "E", targets.at("E").token, targets.at("E").text,
                               source_states.at("B").prompt, source_states.at("B").key,
                               normalized(output_row_difference(output, targets.at("E").token,
                                                               source_states.at("B").baseline.metrics.greedy))};

        step27_bank initial_bank;
        initial_bank.components.push_back({memory_a.direction, memory_a.key, STEP27_ETA*STEP27_FIRST_GATE, "A", "D"});
        initial_bank.components.push_back({memory_b.direction, memory_b.key, STEP27_ETA*STEP27_FIRST_GATE, "B", "E"});
        write_bank_snapshot(root/"initial_bank", initial_bank);

        std::ofstream c0(root/"c0_trajectories.jsonl", std::ios::trunc);
        std::ofstream c1(root/"c1_trajectories.jsonl", std::ios::trunc);
        std::ofstream c2(root/"c2_trajectories.jsonl", std::ios::trunc);
        std::ofstream c3(root/"c3_trajectories.jsonl", std::ios::trunc);
        std::ofstream readouts(root/"final_readouts.jsonl", std::ios::trunc);
        std::ofstream updates(root/"c3_updates.jsonl", std::ios::trunc);
        std::ofstream prompts_out(root/"prompt_metadata.jsonl", std::ios::trunc);
        for (const auto & query : make_prompts()) {
            prompts_out << "{\"id\":" << query.id << ",\"name\":\"" << query.name
                        << "\",\"source\":\"" << query.source
                        << "\",\"user_prompt\":\"" << json_escape(query.user_prompt) << "\"}\n";
        }
        for (auto * stream : {&c0, &c1, &c2, &c3, &readouts, &updates, &prompts_out}) stream->setf(std::ios::unitbuf);

        bool supports_thinking = false;
        std::map<std::string, std::vector<llama_token>> c0_tokens;
        std::map<std::string, eval_result> c0_readouts;
        const auto prompts = make_prompts();
        auto run_condition = [&](const std::string & condition, const step27_bank & bank,
                                 std::ostream & trajectories, bool first_only) {
            for (const auto & query : prompts) {
                const auto & memory = query.source == "A" ? memory_a : memory_b;
                const auto episode = run_episode(ctx, vocab, collector, output, templates.get(), query,
                                                 memory, bank, condition, 0, 0, first_only,
                                                 trajectories, readouts, supports_thinking);
                if (condition == "C0_Qwen_only") {
                    c0_tokens[query.name] = episode.tokens;
                    c0_readouts[query.name] = episode.readout;
                }
            }
        };

        const step27_bank empty_bank;
        run_condition("C0_Qwen_only", empty_bank, c0, false);
        run_condition("C1_one_shot", initial_bank, c1, true);
        run_condition("C2_continuous_frozen", initial_bank, c2, false);

        step27_bank c3_bank = initial_bank;
        std::map<std::string, step27_counts> counts = {{"A", {1, 1}}, {"B", {1, 1}}};
        const std::array<int32_t, 16> schedule = {0, 2, 1, 3, 0, 2, 1, 3,
                                                   0, 2, 1, 3, 0, 2, 1, 3};
        for (int32_t episode = 0; episode < (int32_t) schedule.size(); ++episode) {
            const auto & query = prompts.at(schedule[episode]);
            const auto & memory = query.source == "A" ? memory_a : memory_b;
            const int32_t cycle = episode/4 + 1;
            (void) run_episode(ctx, vocab, collector, output, templates.get(), query, memory,
                               c3_bank, "C3_continuous_online_plasticity", episode, cycle, false,
                               c3, readouts, supports_thinking);
            const double gate = update_c3(c3_bank, counts[memory.source], memory, updates, episode, cycle);
            write_bank_snapshot(root/"c3_bank_checkpoints"/("episode_" + std::to_string(episode + 1)), c3_bank);
            std::ofstream checkpoint(root/"c3_bank_checkpoints"/("episode_" + std::to_string(episode + 1) + ".json"), std::ios::trunc);
            checkpoint << std::setprecision(17) << "{\"episode\":" << episode
                       << ",\"cycle\":" << cycle << ",\"source\":\"" << memory.source
                       << "\",\"gate\":" << gate << ",\"component_count\":"
                       << c3_bank.components.size() << "}\n";
        }

        double shadow_max = 0.0;
        step27_bank shadow_bank = initial_bank;
        std::map<std::string, step27_counts> shadow_counts = {{"A", {1, 1}}, {"B", {1, 1}}};
        std::ostringstream shadow_updates;
        for (int32_t episode = 0; episode < (int32_t) schedule.size(); ++episode) {
            const auto & query = prompts.at(schedule[episode]);
            const auto & memory = query.source == "A" ? memory_a : memory_b;
            const auto before = evaluate(ctx, vocab, collector, render_step27_chat(templates.get(), query.user_prompt, supports_thinking), memory.target, false);
            const auto after = evaluate(ctx, vocab, collector, render_step27_chat(templates.get(), query.user_prompt, supports_thinking), memory.target, false);
            shadow_max = std::max(shadow_max, max_logit_diff_step27(before.logits, after.logits));
            update_c3(shadow_bank, shadow_counts[memory.source], memory, shadow_updates, episode, episode/4 + 1);
        }
        std::ofstream shadow(root/"shadow_invariant.json", std::ios::trunc);
        shadow << std::setprecision(17) << "{\"no_injection_update_max_logit_difference\":" << shadow_max
               << ",\"pass\":" << (shadow_max <= STEP27_TOLERANCE ? "true" : "false")
               << ",\"base_model_mutated\":false}\n";
        if (shadow_max > STEP27_TOLERANCE) throw std::runtime_error("Step 27 no-injection shadow invariant failed");

        std::ofstream bank_reference(root/"bank_reference.json", std::ios::trunc);
        bank_reference << "{\"initial_component_count\":2,\"initial_components\":["
                       << "{\"source\":\"A\",\"destination\":\"D\",\"coefficient\":0.0125,\"key_checksum\":\""
                       << hex64_step27(checksum64_step27(memory_a.key)) << "\",\"value_checksum\":\""
                       << hex64_step27(checksum64_step27(memory_a.direction)) << "\"},"
                       << "{\"source\":\"B\",\"destination\":\"E\",\"coefficient\":0.0125,\"key_checksum\":\""
                       << hex64_step27(checksum64_step27(memory_b.key)) << "\",\"value_checksum\":\""
                       << hex64_step27(checksum64_step27(memory_b.direction)) << "\"}],"
                       << "\"target_tokens\":{\"A\":" << memory_a.target << ",\"B\":" << memory_b.target << "},"
                       << "\"target_texts\":{\"A\":\"" << json_escape(memory_a.target_text)
                       << "\",\"B\":\"" << json_escape(memory_b.target_text) << "\"}}\n";

        std::ofstream validation(root/"validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"runtime_baseline\":\"rtx3090_20260814\",\"qwen_weights_modified\":false"
                   << ",\"gradients\":false,\"optimizer\":false,\"generated_tokens_update\":false"
                   << ",\"source_partition_fixed\":true,\"authority_limit\":" << STEP27_B_MAX
                   << ",\"max_shadow_logit_difference\":" << shadow_max
                   << ",\"shadow_exact\":" << (shadow_max <= STEP27_TOLERANCE ? "true" : "false")
                   << ",\"reasoning_max_tokens\":" << STEP27_MAX_TOKENS << "}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":"
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ",\"decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size
                << ",\"conditions\":4,\"prompts_per_fixed_condition\":4"
                << ",\"c3_episodes\":16,\"clean_kv_per_forward\":true}\n";
        std::ofstream manifest(root/"manifest.json", std::ios::trunc);
        manifest << "{\"experiment\":\"qwen35_closed_loop_sidecar_reasoning\",\"model\":\""
                 << json_escape(opts.model) << "\",\"sparse_residual_enabled\":true"
                 << ",\"conditions\":[\"C0_Qwen_only\",\"C1_one_shot\",\"C2_continuous_frozen\",\"C3_continuous_online_plasticity\"]"
                 << ",\"injection_sites\":[\"block_24\",\"block_28\",\"final\"]"
                 << ",\"allocation_weights\":[0.23072,0.38322,0.89438]"
                 << ",\"reader\":\"source-partitioned Pz with B_eff=0.01*min(1,||Pz||/0.05)\""
                 << ",\"base_model_mutated\":false,\"no_gradients\":true,\"no_optimizer\":true}\n";

        clear_internal_injection(ctx);
        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
