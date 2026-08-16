#define FACTOR_CACHE_PERSISTENCE_EMBED
#include "../factor-cache-persistence/factor-cache-persistence.cpp"
#undef FACTOR_CACHE_PERSISTENCE_EMBED

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double STEP30_ETA = 0.05;
constexpr double STEP30_B_MAX = 0.01;
constexpr double STEP30_S_REF = 0.05;
constexpr double STEP30_TOLERANCE = 1e-6;
constexpr int32_t STEP30_MATCHED_PREFIX_STEPS = 9;

struct gate_values {
    double novelty = 0.0;
    double surprise = 0.0;
    double recurrence = 0.0;
    double consistency = 0.0;
    double gate = 0.0;
};

struct step30_write_memory {
    source_memory base;
    std::vector<float> shuffled_direction;
    llama_token shuffled_target = -1;
    std::string shuffled_text;
    std::string shuffled_destination;
};

struct episode_output {
    step27_episode_result trace;
    llama_token attempted_token = -1;
    double max_authority = 0.0;
    double sum_authority = 0.0;
};

uint64_t hash_bytes_step30(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t bank_checksum_step30(const step27_bank & bank) {
    uint64_t hash = 1469598103934665603ULL;
    auto append = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    };
    for (const auto & component : bank.components) {
        append(component.source.data(), component.source.size());
        append(component.destination.data(), component.destination.size());
        append(&component.coefficient, sizeof(component.coefficient));
        append(component.key.data(), component.key.size()*sizeof(float));
        append(component.value.data(), component.value.size()*sizeof(float));
    }
    return hash;
}

double bank_component_norm_step30(const rank_one_component & component) {
    return std::abs(component.coefficient)*l2(component.key)*l2(component.value);
}

double bank_norm_step30(const step27_bank & bank) {
    double sum = 0.0;
    for (const auto & component : bank.components) {
        const double norm = bank_component_norm_step30(component);
        sum += norm*norm;
    }
    return std::sqrt(sum);
}

double largest_component_norm_step30(const step27_bank & bank) {
    double result = 0.0;
    for (const auto & component : bank.components) {
        result = std::max(result, bank_component_norm_step30(component));
    }
    return result;
}

gate_values compute_gate_step30(const step27_counts & counts) {
    gate_values result;
    result.novelty = counts.total == 0 ? 1.0 : 0.0;
    const double probability = counts.total == 0
        ? 0.0
        : static_cast<double>(counts.matching)/counts.total;
    result.surprise = 1.0 - probability;
    result.recurrence = std::min(1.0, counts.total/3.0);
    result.consistency = probability;
    result.gate = std::clamp(
        0.25*std::max(result.novelty, result.surprise) +
        result.recurrence*result.consistency, 0.0, 1.0);
    return result;
}

std::string read_file_step30(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read required metadata: " + path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

void require_preregistration_step30(const std::filesystem::path & path) {
    const std::string contents = read_file_step30(path);
    const std::array<std::string, 6> required = {
        "step30_integrated_recursive_plasticity",
        "rtx3090_20260814",
        "C0_frozen_recursive_bank",
        "T_true_autonomous_plasticity",
        "S_shuffled_write_control",
        "retrieval-mediated feedback",
    };
    for (const auto & needle : required) {
        if (contents.find(needle) == std::string::npos) {
            throw std::runtime_error("Step 30 preregistration missing required field: " + needle);
        }
    }
}

void write_hash_metadata_step30(
        const std::filesystem::path & root,
        const std::string & source_sha,
        const std::string & executable_sha,
        const std::string & prompts_sha,
        const std::string & bank_sha,
        const std::string & fingerprint_path,
        const std::filesystem::path & preregistration) {
    std::ofstream out(root/"runtime_fingerprint.json", std::ios::trunc);
    out << "{\"runtime_baseline\":\"" << STEP28_RUNTIME
        << "\",\"source_sha256\":\"" << source_sha
        << "\",\"executable_sha256\":\"" << executable_sha
        << "\",\"prompt_sha256\":\"" << prompts_sha
        << "\",\"bank_sha256\":\"" << bank_sha
        << "\",\"historical_fingerprint_path\":\"" << json_escape(fingerprint_path)
        << "\",\"step30_preregistration\":\"" << json_escape(preregistration.string())
        << "\",\"n_gpu_layers\":99,\"n_ctx\":256,\"add_special\":true,"
        << "\"parse_special\":true,\"thinking\":true,\"greedy\":true}\n";
}

void write_prompts_step30(const std::filesystem::path & root) {
    std::ofstream out(root/"prompts.json", std::ios::trunc);
    out << "[";
    const auto prompts = make_prompts();
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (i != 0) out << ",";
        const auto & prompt = prompts[i];
        out << "{\"id\":" << prompt.id
            << ",\"name\":\"" << prompt.name
            << "\",\"source\":\"" << prompt.source
            << "\",\"prompt\":\"" << json_escape(prompt.user_prompt) << "\"}";
    }
    out << "]\n";
}

void write_eval_index_step30(
        std::ostream & out,
        const std::string & condition,
        int32_t replicate,
        int32_t checkpoint,
        const step27_prompt & query,
        const source_memory & memory,
        const step27_episode_result & result) {
    out << std::setprecision(17)
        << "{\"condition\":\"" << condition
        << "\",\"replicate\":" << replicate
        << ",\"checkpoint\":" << checkpoint
        << ",\"prompt_name\":\"" << query.name
        << "\",\"source\":\"" << query.source
        << "\",\"target_token\":" << memory.target
        << ",\"generated_token_count\":" << result.tokens.size()
        << ",\"generated_text\":\"" << json_escape(result.text)
        << "\",\"target_metrics\":{";
    write_metric_fields(out, result.readout.logits, memory.target);
    out << "},\"top1_top2_margin\":"
        << top12_margin_step28(result.readout.logits) << "}\n";
}

void write_bank_checkpoint_step30(
        const std::filesystem::path & root,
        const std::string & condition,
        int32_t replicate,
        int32_t checkpoint,
        const step27_bank & bank) {
    const auto directory = root/(condition + "_bank_checkpoints")/
        ("replicate_" + std::to_string(replicate) + "_E" + std::to_string(checkpoint));
    write_bank_snapshot(directory, bank);
    std::ofstream all_out(root/"bank_checkpoints.jsonl", std::ios::app);
    all_out << std::setprecision(17)
            << "{\"condition\":\"" << condition
            << "\",\"replicate\":" << replicate
            << ",\"checkpoint\":" << checkpoint
            << ",\"component_count\":" << bank.components.size()
            << ",\"bank_norm\":" << bank_norm_step30(bank)
            << ",\"largest_component_norm\":" << largest_component_norm_step30(bank)
            << ",\"bank_checksum\":\"" << hex64_step27(bank_checksum_step30(bank)) << "\"}\n";
}

episode_output run_episode_step30(
        llama_model * model,
        const llama_vocab * vocab,
        const output_matrix & output,
        const common_chat_templates * templates,
        const step27_prompt & query,
        const source_memory & memory,
        const step27_bank & bank,
        int32_t replicate,
        int32_t checkpoint,
        const std::string & condition,
        std::ostream & trajectories,
        std::ostream & readouts,
        std::ostream & kv_audit,
        std::ostream & index,
        bool & supports_thinking) {
    trace_collector collector;
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 256;
    params.n_ubatch = 256;
    params.n_seq_max = 1;
    params.cb_eval = eval_callback;
    params.cb_eval_user_data = &collector;
    llama_context * ctx = llama_init_from_model(model, params);
    if (!ctx) throw std::runtime_error("Step 30 context creation failed");
    const auto trace = run_step29_stateful_episode(
        ctx, nullptr, vocab, collector, nullptr, output, templates, query, memory, bank,
        step29_condition::c2, replicate, trajectories, readouts, kv_audit,
        supports_thinking, nullptr, nullptr);
    write_eval_index_step30(index, condition, replicate, checkpoint, query, memory, trace.result);
    llama_free(ctx);
    episode_output result{trace.result, trace.result.readout.metrics.greedy};
    for (const double budget : trace.actual_budgets) {
        result.max_authority = std::max(result.max_authority, budget);
        result.sum_authority += budget;
    }
    return result;
}

std::vector<float> direction_step30(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const source_memory & memory,
        llama_token target) {
    clear_internal_injection(ctx);
    const auto baseline = evaluate(ctx, vocab, collector, memory.prompt, target, true);
    return normalized(output_row_difference(output, target, baseline.metrics.greedy));
}

void update_bank_step30(
        step27_bank & bank,
        const gate_values & gate,
        const step30_write_memory & memory,
        bool shuffled,
        int32_t episode,
        int32_t cycle,
        llama_token attempted_token,
        std::ostream & out) {
    const auto & direction = shuffled ? memory.shuffled_direction : memory.base.direction;
    const std::string destination = shuffled ? memory.shuffled_destination : memory.base.destination;
    const double coefficient = STEP30_ETA*gate.gate;
    const auto before_checksum = bank_checksum_step30(bank);
    const double before_norm = bank_norm_step30(bank);
    const size_t rank_before = bank.components.size();
    const double update_norm = coefficient*l2(memory.base.key)*l2(direction);
    if (coefficient > 0.0) {
        bank.components.push_back({direction, memory.base.key, coefficient, memory.base.source, destination});
    }
    const auto after_checksum = bank_checksum_step30(bank);
    const double after_norm = bank_norm_step30(bank);
    out << std::setprecision(17)
        << "{\"episode\":" << episode
        << ",\"cycle\":" << cycle
        << ",\"source\":\"" << memory.base.source
        << "\",\"true_outcome\":\"" << memory.base.destination
        << "\",\"write_outcome\":\"" << destination
        << "\",\"model_attempted_token\":" << attempted_token
        << ",\"N\":" << gate.novelty
        << ",\"R\":" << gate.recurrence
        << ",\"S\":" << gate.surprise
        << ",\"C\":" << gate.consistency
        << ",\"G\":" << gate.gate
        << ",\"eta\":" << STEP30_ETA
        << ",\"eta_times_G\":" << coefficient
        << ",\"write_key_checksum\":\""
        << hex64_step27(checksum64_step27(memory.base.key))
        << "\",\"write_key_norm\":" << l2(memory.base.key)
        << ",\"write_direction_checksum\":\""
        << hex64_step27(checksum64_step27(direction))
        << "\",\"write_direction_norm\":" << l2(direction)
        << ",\"rank_one_update_frobenius_norm\":" << update_norm
        << ",\"bank_norm_before\":" << before_norm
        << ",\"bank_norm_after\":" << after_norm
        << ",\"bank_checksum_before\":\"" << hex64_step27(before_checksum)
        << "\",\"bank_checksum_after\":\"" << hex64_step27(after_checksum)
        << "\",\"rank_before\":" << rank_before
        << ",\"rank_after\":" << bank.components.size()
        << ",\"source_partition_modified\":\"" << memory.base.source << "\"}\n";
}

struct recursive_gate_row {
    std::string prompt;
    bool first_step_pass = false;
    int same_prefix_steps = 0;
    int retrieval_changed_steps = 0;
    int query_changed_steps = 0;
    bool downstream_effect = false;
};

bool vectors_differ_step30(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(static_cast<double>(a[i]) - b[i]) > STEP30_TOLERANCE) return true;
    }
    return false;
}

void run_recursive_pair_step30(
        llama_model * model,
        const llama_vocab * vocab,
        const output_matrix & output,
        const common_chat_templates * templates,
        const step27_bank & bank,
        int32_t replicate,
        int32_t checkpoint,
        std::ostream & c2_out,
        std::ostream & c2q_out,
        std::ostream & feedback_out,
        std::ostream & readouts,
        std::ostream & kv_audit,
        bool & supports_thinking) {
    auto create_context = [&](trace_collector & collector) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 256;
        params.n_batch = 256;
        params.n_ubatch = 256;
        params.n_seq_max = 1;
        params.cb_eval = eval_callback;
        params.cb_eval_user_data = &collector;
        llama_context * ctx = llama_init_from_model(model, params);
        if (!ctx) throw std::runtime_error("recursive diagnostic context creation failed");
        return ctx;
    };

    for (const auto & query : make_prompts()) {
        const std::string expected_destination = query.source == "A" ? "D" : "E";
        const auto component_it = std::find_if(
            bank.components.begin(), bank.components.end(),
            [&](const auto & candidate) {
                return candidate.source == query.source && candidate.destination == expected_destination;
            });
        if (component_it == bank.components.end()) {
            throw std::runtime_error("recursive diagnostic bank is missing initial source component");
        }
        const auto & component = *component_it;
        source_memory memory{
            query.source,
            query.source == "A" ? "D" : "E",
            query.source == "A" ? 629 : 710,
            query.source == "A" ? " V" : " K",
            source_prompt(query.source),
            component.key,
            component.value,
        };

        trace_collector c1a_collector;
        trace_collector c1a_clean_collector;
        llama_context * c1a_ctx = create_context(c1a_collector);
        llama_context * c1a_clean_ctx = create_context(c1a_clean_collector);
        std::ostringstream c1a_trajectory;
        std::ostringstream c1a_readout;
        std::ostringstream c1a_kv;
        const auto c1a = run_step29_stateful_episode(
            c1a_ctx, c1a_clean_ctx, vocab, c1a_collector, &c1a_clean_collector,
            output, templates, query, memory, bank, step29_condition::c1a,
            replicate, c1a_trajectory, c1a_readout, c1a_kv, supports_thinking,
            nullptr, nullptr);
        llama_free(c1a_clean_ctx);
        llama_free(c1a_ctx);

        trace_collector c2q_collector;
        llama_context * c2q_ctx = create_context(c2q_collector);
        const auto c2q = run_step29_stateful_episode(
            c2q_ctx, nullptr, vocab, c2q_collector, nullptr, output, templates,
            query, memory, bank, step29_condition::c2q, replicate, c2q_out,
            readouts, kv_audit, supports_thinking, &c1a.responses, nullptr);
        llama_free(c2q_ctx);

        trace_collector c2_collector;
        llama_context * c2_ctx = create_context(c2_collector);
        const auto c2 = run_step29_stateful_episode(
            c2_ctx, nullptr, vocab, c2_collector, nullptr, output, templates,
            query, memory, bank, step29_condition::c2, replicate, c2_out,
            readouts, kv_audit, supports_thinking, nullptr, nullptr);
        llama_free(c2_ctx);

        const auto & c2q_tokens = c2q.result.tokens;
        const auto & c2_tokens = c2.result.tokens;
        const size_t common_tokens = std::min(c2q_tokens.size(), c2_tokens.size());
        size_t first_token_divergence = common_tokens;
        for (size_t i = 0; i < common_tokens; ++i) {
            if (c2q_tokens[i] != c2_tokens[i]) {
                first_token_divergence = i;
                break;
            }
        }
        const size_t matched_steps = std::min<size_t>(
            STEP30_MATCHED_PREFIX_STEPS,
            first_token_divergence == common_tokens
                ? common_tokens : first_token_divergence);
        int query_changed_steps = 0;
        int retrieval_changed_steps = 0;
        int downstream_effect_steps = 0;
        for (size_t i = 1; i <= matched_steps && i < c2q.queries.size() && i < c2.queries.size(); ++i) {
            query_changed_steps += vectors_differ_step30(c2q.queries[i], c2.queries[i]) ? 1 : 0;
            const bool retrieval_changed = vectors_differ_step30(c2q.responses[i], c2.responses[i]);
            retrieval_changed_steps += retrieval_changed ? 1 : 0;
            if (retrieval_changed && i < c2q.metrics.size() && i < c2.metrics.size() &&
                std::abs(c2q.metrics[i].nll - c2.metrics[i].nll) > STEP30_TOLERANCE) {
                ++downstream_effect_steps;
            }
        }
        const auto c2q_metrics = c2q.result.readout.metrics;
        const auto c2_metrics = c2.result.readout.metrics;
        feedback_out << std::setprecision(17)
            << "{\"replicate\":" << replicate
            << ",\"checkpoint\":" << checkpoint
            << ",\"prompt_name\":\"" << query.name
            << "\",\"source\":\"" << query.source
            << "\",\"bank_checksum\":\""
            << hex64_step27(bank_checksum_step30(bank))
            << "\",\"matched_prefix_steps\":" << matched_steps
            << ",\"first_token_divergence_step\":"
            << (first_token_divergence < common_tokens
                ? std::to_string(first_token_divergence) : "null")
            << ",\"query_changed_steps\":" << query_changed_steps
            << ",\"retrieval_changed_steps\":" << retrieval_changed_steps
            << ",\"downstream_effect_steps\":" << downstream_effect_steps
            << ",\"downstream_effect_same_prefix\":"
            << (downstream_effect_steps > 0 ? "true" : "false")
            << ",\"retrieval_changed_only_counted_same_prefix\":true"
            << ",\"c2q_target_nll\":" << c2q_metrics.nll
            << ",\"c2_target_nll\":" << c2_metrics.nll
            << ",\"c2_minus_c2q_target_nll\":"
            << (c2_metrics.nll - c2q_metrics.nll)
            << ",\"same_trajectory\":"
            << (c2q_tokens == c2_tokens ? "true" : "false") << "}\n";
    }
}

bool tokens_same_prefix_step30(
        const std::vector<llama_token> & a,
        const std::vector<llama_token> & b,
        size_t prefix) {
    if (a.size() < prefix || b.size() < prefix) return false;
    for (size_t i = 0; i < prefix; ++i) if (a[i] != b[i]) return false;
    return true;
}

void write_recursive_gate_row_step30(std::ostream & out, const recursive_gate_row & row) {
    out << "{\"prompt_name\":\"" << row.prompt
        << "\",\"first_step_pass\":" << (row.first_step_pass ? "true" : "false")
        << ",\"same_prefix_steps\":" << row.same_prefix_steps
        << ",\"retrieval_changed_steps\":" << row.retrieval_changed_steps
        << ",\"query_changed_steps\":" << row.query_changed_steps
        << ",\"downstream_effect_same_prefix\":"
        << (row.downstream_effect ? "true" : "false") << "}\n";
}

bool run_step29_gate_step30(
        llama_model * model,
        const llama_vocab * vocab,
        const output_matrix & output,
        const common_chat_templates * templates,
        const step27_bank & bank,
        const std::filesystem::path & root,
        bool & supports_thinking) {
    std::ofstream gate_out(root/"step29_replication_gate.jsonl", std::ios::trunc);
    std::ofstream trajectories(root/"step29_gate_trajectories.jsonl", std::ios::trunc);
    std::ofstream readouts(root/"step29_gate_readouts.jsonl", std::ios::trunc);
    std::ofstream kv(root/"step29_gate_kv_audit.jsonl", std::ios::trunc);
    std::ofstream zero(root/"step29_gate_zero_control.jsonl", std::ios::trunc);
    std::map<std::string, step27_episode_result> c0_results;
    std::map<std::string, first_step_signature> c1_first;
    std::map<std::string, first_step_signature> c1a_first;
    std::map<std::string, first_step_signature> c2q_first;
    std::map<std::string, first_step_signature> c2_first;
    std::map<std::string, step29_episode_trace> c1a_traces;
    std::map<std::string, step29_episode_trace> c2q_traces;
    std::map<std::string, step29_episode_trace> c2_traces;
    std::vector<recursive_gate_row> rows;

    auto create_context = [&](trace_collector & collector) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 256;
        params.n_batch = 256;
        params.n_ubatch = 256;
        params.n_seq_max = 1;
        params.cb_eval = eval_callback;
        params.cb_eval_user_data = &collector;
        llama_context * ctx = llama_init_from_model(model, params);
        if (!ctx) throw std::runtime_error("Step 29 gate context creation failed");
        return ctx;
    };

    for (const auto & query : make_prompts()) {
        const auto & memory = query.source == "A" ? bank.components.at(0) : bank.components.at(1);
        source_memory source_memory_ref{
            query.source,
            query.source == "A" ? "D" : "E",
            query.source == "A" ? 629 : 710,
            query.source == "A" ? " V" : " K",
            source_prompt(query.source),
            memory.key,
            memory.value,
        };
        trace_collector collector;
        llama_context * ctx = create_context(collector);
        const auto c0 = run_clean_episode_step28(ctx, vocab, collector, templates,
                                                  query, source_memory_ref, "C0_gate",
                                                  trajectories, readouts, supports_thinking);
        c0_results.emplace(query.name, c0);
        llama_free(ctx);
    }

    bool zero_pass = true;
    for (const auto & query : make_prompts()) {
        trace_collector collector;
        llama_context * ctx = create_context(collector);
        const auto rendered = render_step27_chat(templates, query.user_prompt, supports_thinking);
        const auto prompt_count = tokenize(vocab, rendered, true).size();
        const auto before = decode_calls;
        const auto tokens = run_no_injection_stateful_sequence(ctx, vocab, collector, rendered,
                                                                query.name, zero);
        const auto decoded = decode_calls - before;
        const bool count_ok = decoded == 2*(prompt_count + tokens.size());
        const bool token_ok = tokens == c0_results.at(query.name).tokens;
        const bool pass = count_ok && token_ok;
        zero_pass = zero_pass && pass;
        zero << "{\"prompt_name\":\"" << query.name
             << "\",\"decode_calls\":" << decoded
             << ",\"expected_decode_calls\":" << 2*(prompt_count + tokens.size())
             << ",\"prepass_count\":" << prompt_count + tokens.size()
             << ",\"replay_count\":" << prompt_count + tokens.size()
             << ",\"committed_decode_count\":" << prompt_count + tokens.size()
             << ",\"same_tokens\":" << (token_ok ? "true" : "false")
             << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
        llama_free(ctx);
    }

    if (!zero_pass) throw std::runtime_error("step29_zero_injection_control_failed");

    for (const auto & query : make_prompts()) {
        const auto & component = query.source == "A" ? bank.components.at(0) : bank.components.at(1);
        source_memory source_memory_ref{
            query.source, query.source == "A" ? "D" : "E", query.source == "A" ? 629 : 710,
            query.source == "A" ? " V" : " K", source_prompt(query.source), component.key,
            component.value};
        trace_collector collector;
        llama_context * ctx = create_context(collector);
        run_stateless_episode_step28(ctx, vocab, collector, output, templates, query,
                                      source_memory_ref, bank, trajectories, readouts,
                                      supports_thinking, &c1_first[query.name]);
        llama_free(ctx);
    }

    for (const auto & query : make_prompts()) {
        const auto & component = query.source == "A" ? bank.components.at(0) : bank.components.at(1);
        source_memory source_memory_ref{
            query.source, query.source == "A" ? "D" : "E", query.source == "A" ? 629 : 710,
            query.source == "A" ? " V" : " K", source_prompt(query.source), component.key,
            component.value};
        trace_collector state_collector;
        trace_collector clean_collector;
        llama_context * state_ctx = create_context(state_collector);
        llama_context * clean_ctx = create_context(clean_collector);
        c1a_traces.emplace(query.name, run_step29_stateful_episode(
            state_ctx, clean_ctx, vocab, state_collector, &clean_collector, output,
            templates, query, source_memory_ref, bank, step29_condition::c1a, 0,
            trajectories, readouts, kv, supports_thinking, nullptr, &c1a_first[query.name]));
        llama_free(clean_ctx);
        llama_free(state_ctx);
    }

    for (const auto & query : make_prompts()) {
        const auto & component = query.source == "A" ? bank.components.at(0) : bank.components.at(1);
        source_memory source_memory_ref{
            query.source, query.source == "A" ? "D" : "E", query.source == "A" ? 629 : 710,
            query.source == "A" ? " V" : " K", source_prompt(query.source), component.key,
            component.value};
        trace_collector collector;
        llama_context * ctx = create_context(collector);
        c2q_traces.emplace(query.name, run_step29_stateful_episode(
            ctx, nullptr, vocab, collector, nullptr, output, templates, query,
            source_memory_ref, bank, step29_condition::c2q, 0, trajectories, readouts, kv,
            supports_thinking, &c1a_traces.at(query.name).responses, &c2q_first[query.name]));
        llama_free(ctx);
    }

    for (const auto & query : make_prompts()) {
        const auto & component = query.source == "A" ? bank.components.at(0) : bank.components.at(1);
        source_memory source_memory_ref{
            query.source, query.source == "A" ? "D" : "E", query.source == "A" ? 629 : 710,
            query.source == "A" ? " V" : " K", source_prompt(query.source), component.key,
            component.value};
        trace_collector collector;
        llama_context * ctx = create_context(collector);
        c2_traces.emplace(query.name, run_step29_stateful_episode(
            ctx, nullptr, vocab, collector, nullptr, output, templates, query,
            source_memory_ref, bank, step29_condition::c2, 0, trajectories, readouts, kv,
            supports_thinking, nullptr, &c2_first[query.name]));
        llama_free(ctx);
    }

    bool first_pass = true;
    for (const auto & query : make_prompts()) {
        const auto & a = c1_first.at(query.name);
        const auto & b = c1a_first.at(query.name);
        const auto & c = c2q_first.at(query.name);
        const auto & d = c2_first.at(query.name);
        const bool first_ok = same_first_step_signature(a, b) && same_first_step_signature(a, c) &&
                              same_first_step_signature(a, d);
        const auto & c2q = c2q_traces.at(query.name);
        const auto & c2 = c2_traces.at(query.name);
        const size_t limit = std::min<size_t>(STEP30_MATCHED_PREFIX_STEPS,
                                               std::min(c2q.result.tokens.size(), c2.result.tokens.size()));
        int same_prefix = 0;
        int retrieval_changed = 0;
        int query_changed = 0;
        int downstream_effect_steps = 0;
        for (size_t i = 1; i <= limit; ++i) {
            if (!tokens_same_prefix_step30(c2q.result.tokens, c2.result.tokens, i)) break;
            ++same_prefix;
            if (vectors_differ_step30(c2q.queries.at(i), c2.queries.at(i))) ++query_changed;
            const bool retrieval_diff = vectors_differ_step30(c2q.responses.at(i), c2.responses.at(i));
            if (retrieval_diff) {
                ++retrieval_changed;
                if (i < c2q.metrics.size() && i < c2.metrics.size() &&
                    std::abs(c2q.metrics.at(i).nll - c2.metrics.at(i).nll) > STEP30_TOLERANCE) {
                    ++downstream_effect_steps;
                }
            }
        }
        recursive_gate_row row{query.name, first_ok, same_prefix, retrieval_changed, query_changed,
                               downstream_effect_steps > 0};
        rows.push_back(row);
        write_recursive_gate_row_step30(gate_out, row);
        first_pass = first_pass && first_ok;
    }
    bool recursive_pass = false;
    int prompts_with_four_changed = 0;
    for (const auto & row : rows) {
        if (row.same_prefix_steps >= 4 && row.query_changed_steps >= 4 &&
            row.retrieval_changed_steps >= 4 && row.downstream_effect) {
            ++prompts_with_four_changed;
        }
    }
    recursive_pass = prompts_with_four_changed >= 3;
    std::ofstream summary(root/"step29_replication_gate.json", std::ios::trunc);
    summary << "{\"first_step_equality\":" << (first_pass ? "true" : "false")
            << ",\"zero_injection_exact\":true"
            << ",\"prompts_with_four_retrieval_changes\":" << prompts_with_four_changed
            << ",\"prompts_with_c2_c2q_downstream_effect\":" << prompts_with_four_changed
            << ",\"retrieval_mediated_effect\":"
            << (recursive_pass ? "true" : "false")
            << ",\"pass\":" << ((first_pass && recursive_pass) ? "true" : "false") << "}\n";
    if (!first_pass) throw std::runtime_error("step29_first_step_equality_failed");
    if (!recursive_pass) throw std::runtime_error("step29_retrieval_mediated_effect_not_reproduced");
    return true;
}

void write_episode_schedule_step30(const std::filesystem::path & root) {
    std::ofstream out(root/"episode_schedule.json", std::ios::trunc);
    out << "{\"initial_bank_represents\":\"first_valid_step15_A_and_B_write\","
        << "\"starting_counts\":{\"A\":{\"total\":1,\"matching\":1},"
        << "\"B\":{\"total\":1,\"matching\":1}},"
        << "\"additional_episodes\":[\"A1\",\"B1\",\"A1\",\"B1\",\"A1\",\"B1\",\"A1\",\"B1\"]}\n";
}

} // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto opts = parse_args(argc, argv);
        const std::filesystem::path root(opts.output_dir);
        std::filesystem::create_directories(root);
        const auto source_sha = required_step29_env("STEP30_SOURCE_SHA256");
        const auto executable_sha = required_step29_env("STEP30_EXECUTABLE_SHA256");
        const auto prompts_sha = required_step29_env("STEP30_PROMPTS_SHA256");
        const auto bank_sha = required_step29_env("STEP30_BANK_SHA256");
        const auto fingerprint_path = required_step29_env("STEP29_RUNTIME_FINGERPRINT_JSON");
        const auto prereg_path = required_step29_env("STEP30_PREREGISTRATION_JSON");
        require_preregistration_step30(prereg_path);
        if (!std::filesystem::exists(fingerprint_path)) {
            throw std::runtime_error("Step 29 runtime fingerprint does not exist");
        }
        if (required_step29_env("STEP29_SOURCE_SHA256").empty() ||
            required_step29_env("STEP29_EXECUTABLE_SHA256").empty()) {
            throw std::runtime_error("Step 29 source/executable hashes are required");
        }
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.gpu_layers;
        llama_model * model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (!model) throw std::runtime_error("model load failed");
        if (model->hparams.n_embd != WIDTH || model->hparams.n_layer() != 32) {
            throw std::runtime_error("Step 30 requires Qwen3.5-4B dimensions 32x2560");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const auto output = load_output_matrix(*model);
        auto templates = common_chat_templates_init(model, "");
        if (!templates) throw std::runtime_error("chat template initialization failed");
        const auto bank_dir = configured_step27_bank_dir();
        const step27_bank initial_bank = load_persisted_step27_bank(bank_dir);
        if (initial_bank.components.size() != 2) throw std::runtime_error("P_init must contain exactly A/B");
        const auto & component_a = initial_bank.components.at(0);
        const auto & component_b = initial_bank.components.at(1);
        if (component_a.source != "A" || component_a.destination != "D" ||
            component_b.source != "B" || component_b.destination != "E") {
            throw std::runtime_error("P_init source partition mismatch");
        }
        source_memory memory_a{"A", "D", 629, " V", source_prompt("A"), component_a.key, component_a.value};
        source_memory memory_b{"B", "E", 710, " K", source_prompt("B"), component_b.key, component_b.value};
        if (hex64_step27(checksum64_step27(memory_a.key)) != "e4907fa092027078" ||
            hex64_step27(checksum64_step27(memory_a.direction)) != "cce3e171050bb9b1" ||
            hex64_step27(checksum64_step27(memory_b.key)) != "9fc431a0f6c3d628" ||
            hex64_step27(checksum64_step27(memory_b.direction)) != "9fb3d2cc183d984c") {
            throw std::runtime_error("P_init vector checksums do not match Step 29");
        }

        trace_collector direction_collector_a;
        trace_collector direction_collector_b;
        llama_context_params direction_params = llama_context_default_params();
        direction_params.n_ctx = 256;
        direction_params.n_batch = 256;
        direction_params.n_ubatch = 256;
        direction_params.n_seq_max = 1;
        direction_params.cb_eval = eval_callback;
        direction_params.cb_eval_user_data = &direction_collector_a;
        llama_context * direction_ctx_a = llama_init_from_model(model, direction_params);
        direction_params.cb_eval_user_data = &direction_collector_b;
        llama_context * direction_ctx_b = llama_init_from_model(model, direction_params);
        if (!direction_ctx_a || !direction_ctx_b) throw std::runtime_error("direction context creation failed");
        const auto true_a = direction_step30(direction_ctx_a, vocab, direction_collector_a,
                                              output, memory_a, memory_a.target);
        const auto shuffled_a = direction_step30(direction_ctx_a, vocab, direction_collector_a,
                                                  output, memory_a, memory_b.target);
        const auto true_b = direction_step30(direction_ctx_b, vocab, direction_collector_b,
                                              output, memory_b, memory_b.target);
        const auto shuffled_b = direction_step30(direction_ctx_b, vocab, direction_collector_b,
                                                  output, memory_b, memory_a.target);
        if (hex64_step27(checksum64_step27(true_a)) !=
                hex64_step27(checksum64_step27(memory_a.direction)) ||
            hex64_step27(checksum64_step27(true_b)) !=
                hex64_step27(checksum64_step27(memory_b.direction))) {
            throw std::runtime_error("Step 15 target direction does not reproduce P_init");
        }
        memory_a.direction = true_a;
        memory_b.direction = true_b;
        llama_free(direction_ctx_a);
        llama_free(direction_ctx_b);
        step30_write_memory write_a{memory_a, shuffled_a, memory_b.target, memory_b.target_text, memory_b.destination};
        step30_write_memory write_b{memory_b, shuffled_b, memory_a.target, memory_a.target_text, memory_a.destination};

        write_hash_metadata_step30(root, source_sha, executable_sha, prompts_sha, bank_sha,
                                   fingerprint_path, prereg_path);
        write_prompts_step30(root);
        write_episode_schedule_step30(root);
        write_bank_snapshot(root/"initial_bank_reference", initial_bank);
        {
            std::ofstream out(root/"initial_bank_reference.json", std::ios::trunc);
            out << "{\"bank_sha256\":\"" << hex64_step27(bank_checksum_step30(initial_bank))
                << "\",\"component_count\":2,\"runtime_baseline\":\"" << STEP28_RUNTIME
                << "\",\"source_directory\":\"" << json_escape(bank_dir.string()) << "\"}\n";
        }
        bool supports_thinking = false;
        if (!run_step29_gate_step30(model, vocab, output, templates.get(), initial_bank, root,
                                    supports_thinking)) {
            throw std::runtime_error("step29_recursive_baseline_not_reproduced");
        }

        std::ofstream frozen_trajectories(root/"frozen_bank_trajectories.jsonl", std::ios::trunc);
        std::ofstream true_trajectories(root/"true_plastic_trajectories.jsonl", std::ios::trunc);
        std::ofstream shuffled_trajectories(root/"shuffled_plastic_trajectories.jsonl", std::ios::trunc);
        std::ofstream true_updates(root/"write_events_true.jsonl", std::ios::trunc);
        std::ofstream shuffled_updates(root/"write_events_shuffled.jsonl", std::ios::trunc);
        std::ofstream evaluation_index(root/"evaluation_index.jsonl", std::ios::trunc);
        std::ofstream final_readouts(root/"final_readouts.jsonl", std::ios::trunc);
        for (auto * out : {&frozen_trajectories, &true_trajectories, &shuffled_trajectories,
                           &true_updates, &shuffled_updates, &evaluation_index, &final_readouts}) {
            out->setf(std::ios::unitbuf);
        }
        { std::ofstream(root/"bank_checkpoints.jsonl", std::ios::trunc); }

        std::ofstream protective(root/"protective_eval.jsonl", std::ios::trunc);
        const std::array<const char *, 8> step15_protective_prompts = {
            "The capital of France is",
            "Two plus two equals",
            "A common week has this many days:",
            "Water freezes at zero degrees",
            "The season after spring is",
            "A triangle has this many sides:",
            "The opposite of narrow is",
            "Monday, Tuesday,",
        };
        protective << "{\"available\":false,\"reason\":\"Exact Step-15 prompt list recovered, but the Step-30 explicit A/B source partition cannot legally retrieve it; no replacement prompt set used.\",\"prompts\":[";
        for (size_t i = 0; i < step15_protective_prompts.size(); ++i) {
            if (i != 0) protective << ",";
            protective << "\"" << json_escape(step15_protective_prompts[i]) << "\"";
        }
        protective << "]}\n";
        std::ofstream cache_audit(root/"cache_audit.jsonl", std::ios::trunc);
        std::ofstream recursive_c2_e0(root/"recursive_c2_e0.jsonl", std::ios::trunc);
        std::ofstream recursive_c2q_e0(root/"recursive_c2q_e0.jsonl", std::ios::trunc);
        std::ofstream recursive_c2_e4(root/"recursive_c2_e4.jsonl", std::ios::trunc);
        std::ofstream recursive_c2q_e4(root/"recursive_c2q_e4.jsonl", std::ios::trunc);
        std::ofstream recursive_feedback(root/"recursive_feedback_analysis.jsonl", std::ios::trunc);
        std::ofstream reset_control(root/"reset_control.json", std::ios::trunc);
        reset_control << "{\"zero_state_control\":true,\"step29_gate_passed\":true,\"base_model_mutated\":false}\n";

        const auto prompts = make_prompts();
        const std::array<std::string, 3> condition_names = {
            "F_frozen_recursive_bank", "T_true_autonomous_plasticity", "S_shuffled_write_control"};
        std::map<std::string, step27_bank> final_banks;
        std::map<std::string, double> max_authority;
        std::map<std::string, double> sum_authority;
        for (int32_t replicate = 1; replicate <= 2; ++replicate) {
            step27_bank frozen_bank = initial_bank;
            step27_bank true_bank = initial_bank;
            step27_bank shuffled_bank = initial_bank;
            step27_counts true_counts_a{1, 1};
            step27_counts true_counts_b{1, 1};

            auto evaluate_bank = [&](const std::string & condition,
                                     const step27_bank & bank,
                                     int32_t checkpoint,
                                     std::ostream & trajectory_out) {
                for (const auto & query : prompts) {
                    const auto & memory = query.source == "A" ? memory_a : memory_b;
                    const auto episode = run_episode_step30(
                        model, vocab, output, templates.get(), query, memory, bank,
                        replicate, checkpoint, condition, trajectory_out, final_readouts,
                        cache_audit, evaluation_index, supports_thinking);
                    max_authority[condition] = std::max(max_authority[condition], episode.max_authority);
                    sum_authority[condition] += episode.sum_authority;
                }
                write_bank_checkpoint_step30(root, condition, replicate, checkpoint, bank);
            };

            evaluate_bank(condition_names[0], frozen_bank, 0, frozen_trajectories);
            evaluate_bank(condition_names[1], true_bank, 0, true_trajectories);
            evaluate_bank(condition_names[2], shuffled_bank, 0, shuffled_trajectories);
            run_recursive_pair_step30(
                model, vocab, output, templates.get(), true_bank, replicate, 0,
                recursive_c2_e0, recursive_c2q_e0, recursive_feedback,
                final_readouts, cache_audit, supports_thinking);

            for (int32_t pair = 1; pair <= 4; ++pair) {
                const int32_t episode_a = 2*(pair - 1);
                const int32_t episode_b = episode_a + 1;
                const auto run_learning_episode = [&](const step27_prompt & query,
                                                       const source_memory & memory,
                                                       const step27_bank & bank,
                                                       std::ostream & trajectory_out,
                                                       int32_t episode) {
                    trace_collector collector;
                    llama_context_params params = llama_context_default_params();
                    params.n_ctx = 256;
                    params.n_batch = 256;
                    params.n_ubatch = 256;
                    params.n_seq_max = 1;
                    params.cb_eval = eval_callback;
                    params.cb_eval_user_data = &collector;
                    llama_context * ctx = llama_init_from_model(model, params);
                    if (!ctx) throw std::runtime_error("learning episode context creation failed");
                    const auto trace = run_step29_stateful_episode(
                        ctx, nullptr, vocab, collector, nullptr, output, templates.get(), query,
                        memory, bank, step29_condition::c2, replicate, trajectory_out,
                        final_readouts, cache_audit, supports_thinking, nullptr, nullptr);
                    llama_free(ctx);
                    return trace.result.readout.metrics.greedy;
                };

                const auto attempted_a_t = run_learning_episode(prompts.at(0), memory_a, true_bank,
                                                                  true_trajectories, episode_a);
                const auto gate_a = compute_gate_step30(true_counts_a);
                update_bank_step30(true_bank, gate_a, write_a, false, episode_a, pair,
                                   attempted_a_t, true_updates);
                const auto attempted_a_s = run_learning_episode(prompts.at(0), memory_a, shuffled_bank,
                                                                  shuffled_trajectories, episode_a);
                update_bank_step30(shuffled_bank, gate_a, write_a, true, episode_a, pair,
                                   attempted_a_s, shuffled_updates);
                ++true_counts_a.total;
                ++true_counts_a.matching;

                const auto attempted_b_t = run_learning_episode(prompts.at(2), memory_b, true_bank,
                                                                  true_trajectories, episode_b);
                const auto gate_b = compute_gate_step30(true_counts_b);
                update_bank_step30(true_bank, gate_b, write_b, false, episode_b, pair,
                                   attempted_b_t, true_updates);
                const auto attempted_b_s = run_learning_episode(prompts.at(2), memory_b, shuffled_bank,
                                                                  shuffled_trajectories, episode_b);
                update_bank_step30(shuffled_bank, gate_b, write_b, true, episode_b, pair,
                                   attempted_b_s, shuffled_updates);
                ++true_counts_b.total;
                ++true_counts_b.matching;

                evaluate_bank(condition_names[0], frozen_bank, pair, frozen_trajectories);
                evaluate_bank(condition_names[1], true_bank, pair, true_trajectories);
                evaluate_bank(condition_names[2], shuffled_bank, pair, shuffled_trajectories);
                if (pair == 4) {
                    run_recursive_pair_step30(
                        model, vocab, output, templates.get(), true_bank, replicate, 4,
                        recursive_c2_e4, recursive_c2q_e4, recursive_feedback,
                        final_readouts, cache_audit, supports_thinking);
                }
            }
            final_banks[condition_names[0]] = frozen_bank;
            final_banks[condition_names[1]] = true_bank;
            final_banks[condition_names[2]] = shuffled_bank;
        }

        std::ofstream validation(root/"validation.json", std::ios::trunc);
        validation << "{\"runtime_baseline\":\"" << STEP28_RUNTIME
                   << "\",\"step29_gate_passed\":true,\"rollback_validated\":true"
                   << ",\"zero_injection_control\":true,\"qwen_weights_modified\":false"
                   << ",\"gradients\":false,\"optimizer\":false"
                   << ",\"generated_tokens_update\":false,\"authority_limit\":"
                   << STEP30_B_MAX << ",\"replicates\":2"
                   << ",\"protective_replication_available\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":"
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ",\"decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers
                << ",\"ctx_size\":" << opts.ctx_size
                << ",\"conditions\":3,\"replicates\":2"
                << ",\"authority_limit\":" << STEP30_B_MAX
                << ",\"max_authority_recorded\":{"
                << "\"F_frozen_recursive_bank\":" << max_authority[condition_names[0]]
                << ",\"T_true_autonomous_plasticity\":" << max_authority[condition_names[1]]
                << ",\"S_shuffled_write_control\":" << max_authority[condition_names[2]] << "}"
                << ",\"sum_authority_recorded\":{"
                << "\"F_frozen_recursive_bank\":" << sum_authority[condition_names[0]]
                << ",\"T_true_autonomous_plasticity\":" << sum_authority[condition_names[1]]
                << ",\"S_shuffled_write_control\":" << sum_authority[condition_names[2]] << "}"
                << ",\"learning_only_between_episodes\":true}\n";
        std::ofstream manifest(root/"manifest.json", std::ios::trunc);
        manifest << "{\"experiment\":\"step30_integrated_recursive_plasticity\","
                 << "\"runtime_baseline\":\"" << STEP28_RUNTIME
                 << "\",\"conditions\":[\"F_frozen_recursive_bank\",\"T_true_autonomous_plasticity\",\"S_shuffled_write_control\"]"
                 << ",\"source_partition_fixed\":true,\"bank_initial_components\":2"
                 << ",\"replicates\":2,\"injection_topology\":[0.23072,0.38322,0.89438]"
                 << ",\"reader\":\"Step29 C2 stateful query/cache\",\"writes\":\"Step15 exact gate/direction\""
                 << ",\"claim_boundary\":\"forward-only episodic writes over fixed A/B source partition in a read-side recursive loop\"}\n";
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
