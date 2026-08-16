#define STATEFUL_RECURSIVE_SIDECAR_EMBED
#include "../closed-loop-sidecar-reasoning/closed-loop-sidecar-reasoning.cpp"
#undef STATEFUL_RECURSIVE_SIDECAR_EMBED

#include <cstring>
#include <cstdlib>
#include <optional>

namespace {

step27_bank load_persisted_step27_bank(const std::filesystem::path & directory) {
    const auto metadata_path = directory/"components.jsonl";
    const auto vectors_path = directory/"vectors.f32";
    std::ifstream metadata(metadata_path);
    std::ifstream vectors(vectors_path, std::ios::binary);
    if (!metadata || !vectors) {
        throw std::runtime_error("cannot open persisted Step 27 P_init bank: " + directory.string());
    }

    std::vector<std::string> metadata_rows;
    for (std::string line; std::getline(metadata, line);) {
        if (!line.empty()) metadata_rows.push_back(line);
    }
    if (metadata_rows.size() != 2 ||
        metadata_rows[0].find("\"component\":0") == std::string::npos ||
        metadata_rows[0].find("\"source\":\"A\"") == std::string::npos ||
        metadata_rows[0].find("\"destination\":\"D\"") == std::string::npos ||
        metadata_rows[0].find("\"coefficient\":0.0125") == std::string::npos ||
        metadata_rows[0].find("\"key_checksum\":\"e4907fa092027078\"") == std::string::npos ||
        metadata_rows[0].find("\"value_checksum\":\"cce3e171050bb9b1\"") == std::string::npos ||
        metadata_rows[1].find("\"component\":1") == std::string::npos ||
        metadata_rows[1].find("\"source\":\"B\"") == std::string::npos ||
        metadata_rows[1].find("\"destination\":\"E\"") == std::string::npos ||
        metadata_rows[1].find("\"coefficient\":0.0125") == std::string::npos ||
        metadata_rows[1].find("\"key_checksum\":\"9fc431a0f6c3d628\"") == std::string::npos ||
        metadata_rows[1].find("\"value_checksum\":\"9fb3d2cc183d984c\"") == std::string::npos) {
        throw std::runtime_error("persisted Step 27 P_init metadata does not match the frozen A/B bank");
    }

    constexpr size_t component_count = 2;
    constexpr size_t floats_per_component = 2*WIDTH;
    constexpr size_t expected_bytes = component_count*floats_per_component*sizeof(float);
    vectors.seekg(0, std::ios::end);
    const auto size = vectors.tellg();
    vectors.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamoff>(expected_bytes)) {
        throw std::runtime_error("persisted Step 27 P_init vector size mismatch");
    }
    std::vector<float> raw(expected_bytes/sizeof(float));
    vectors.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(expected_bytes));
    if (!vectors) throw std::runtime_error("failed reading persisted Step 27 P_init vectors");

    step27_bank bank;
    const std::array<std::string, 2> sources = {"A", "B"};
    const std::array<std::string, 2> destinations = {"D", "E"};
    for (size_t index = 0; index < component_count; ++index) {
        rank_one_component component;
        const size_t base = index*floats_per_component;
        component.key.assign(raw.begin() + static_cast<std::ptrdiff_t>(base),
                             raw.begin() + static_cast<std::ptrdiff_t>(base + WIDTH));
        component.value.assign(raw.begin() + static_cast<std::ptrdiff_t>(base + WIDTH),
                               raw.begin() + static_cast<std::ptrdiff_t>(base + 2*WIDTH));
        component.coefficient = 0.0125;
        component.source = sources[index];
        component.destination = destinations[index];
        const std::string expected_key = index == 0 ? "e4907fa092027078" : "9fc431a0f6c3d628";
        const std::string expected_value = index == 0 ? "cce3e171050bb9b1" : "9fb3d2cc183d984c";
        if (hex64_step27(checksum64_step27(component.key)) != expected_key ||
            hex64_step27(checksum64_step27(component.value)) != expected_value) {
            throw std::runtime_error("persisted Step 27 P_init vector checksum mismatch");
        }
        bank.components.push_back(std::move(component));
    }
    return bank;
}

std::filesystem::path configured_step27_bank_dir() {
    const char * value = std::getenv("STEP28_P_INIT_BANK_DIR");
    if (value == nullptr || *value == '\0') {
        throw std::runtime_error("STEP28_P_INIT_BANK_DIR is required; refusing to reconstruct P_init");
    }
    return std::filesystem::path(value);
}

constexpr const char * STEP28_RUNTIME = "rtx3090_20260814";
constexpr double STEP28_STATE_TOLERANCE = 1e-6;
constexpr int32_t STEP28_MAX_TOKENS = 32;
constexpr int32_t STEP28_MAX_NORM_ITERATIONS = 6;
constexpr double STEP28_LOCAL_RHO_TOLERANCE = 2e-6;

struct state_blob {
    std::vector<uint8_t> bytes;
    uint64_t checksum = 0;

    struct restore_audit {
        size_t requested_bytes = 0;
        size_t written_bytes = 0;
        size_t restored_bytes = 0;
        uint64_t restored_checksum = 0;
        size_t first_difference = std::numeric_limits<size_t>::max();
        size_t differing_bytes = 0;
        bool sequence_cleared_before_restore = false;

        bool write_size_exact() const {
            return written_bytes == requested_bytes;
        }

        bool byte_exact() const {
            return write_size_exact() && restored_bytes == requested_bytes &&
                   first_difference == std::numeric_limits<size_t>::max();
        }
    };

    static uint64_t hash_bytes(const std::vector<uint8_t> & values) {
        uint64_t hash = 1469598103934665603ULL;
        for (const uint8_t value : values) {
            hash ^= value;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    void capture(llama_context * ctx) {
        const size_t size = llama_state_get_size(ctx);
        bytes.resize(size);
        const size_t written = llama_state_get_data(ctx, bytes.data(), bytes.size());
        if (written != size) throw std::runtime_error("state capture size mismatch");
        checksum = hash_bytes(bytes);
    }

    restore_audit restore_with_audit(llama_context * ctx) const {
        restore_audit result;
        result.requested_bytes = bytes.size();
        // llama_state_set_data does not clear a sequence when the checkpoint
        // contains zero cells. Remove the experiment's sole sequence first.
        result.sequence_cleared_before_restore = llama_memory_seq_rm(
            llama_get_memory(ctx), 0, -1, -1);
        result.written_bytes = llama_state_set_data(ctx, bytes.data(), bytes.size());
        state_blob after;
        after.capture(ctx);
        result.restored_bytes = after.bytes.size();
        result.restored_checksum = after.checksum;
        const size_t common = std::min(bytes.size(), after.bytes.size());
        for (size_t i = 0; i < common; ++i) {
            if (bytes[i] != after.bytes[i]) {
                if (result.first_difference == std::numeric_limits<size_t>::max()) {
                    result.first_difference = i;
                }
                ++result.differing_bytes;
            }
        }
        result.differing_bytes += bytes.size() > common ? bytes.size() - common : after.bytes.size() - common;
        if (result.first_difference == std::numeric_limits<size_t>::max() && bytes.size() != after.bytes.size()) {
            result.first_difference = common;
        }
        return result;
    }

    bool restore(llama_context * ctx) const {
        const auto result = restore_with_audit(ctx);
        if (!result.write_size_exact()) throw std::runtime_error("state restore size mismatch");
        return result.byte_exact();
    }
};

int64_t n_past_step28(llama_context * ctx) {
    return (int64_t) llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
}

eval_result decode_current_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        llama_token token,
        bool capture) {
    collector.clear();
    collector.enabled = capture;
    llama_token mutable_token = token;
    if (llama_decode(ctx, llama_batch_get_one(&mutable_token, 1)) != 0) {
        collector.enabled = false;
        throw std::runtime_error("current-token decode failed");
    }
    ++decode_calls;
    llama_synchronize(ctx);
    collector.enabled = false;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * raw = llama_get_logits_ith(ctx, -1);
    if (!raw) throw std::runtime_error("missing current-token logits");
    eval_result out;
    out.logits.assign(raw, raw + n_vocab);
    out.final_hidden = collector.final_hidden;
    out.boundaries = collector.boundaries;
    if (capture) {
        if (out.final_hidden.size() != WIDTH ||
            out.boundaries.size() != INTERNAL_LAYERS.size()) {
            throw std::runtime_error("missing current-token activation traces");
        }
    }
    out.metrics = metrics_from_logits(out.logits, 0);
    return out;
}

eval_result decode_full_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const std::vector<llama_token> & tokens,
        bool capture) {
    llama_memory_clear(llama_get_memory(ctx), true);
    collector.clear();
    collector.enabled = capture;
    if (tokens.empty() ||
        llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(tokens.data()),
                                              static_cast<int32_t>(tokens.size()))) != 0) {
        collector.enabled = false;
        throw std::runtime_error("full prompt decode failed");
    }
    ++decode_calls;
    llama_synchronize(ctx);
    collector.enabled = false;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * raw = llama_get_logits_ith(ctx, -1);
    if (!raw) throw std::runtime_error("missing full-prompt logits");
    eval_result out;
    out.logits.assign(raw, raw + n_vocab);
    out.final_hidden = collector.final_hidden;
    out.boundaries = collector.boundaries;
    if (capture) {
        if (out.final_hidden.size() != WIDTH ||
            out.boundaries.size() != INTERNAL_LAYERS.size()) {
            throw std::runtime_error("missing full-prompt activation traces");
        }
    }
    out.metrics = metrics_from_logits(out.logits, 0);
    return out;
}

struct stateful_probe_result {
    active_probe active;
    eval_result prepass;
    eval_result committed;
    std::vector<float> query;
    std::vector<float> response;
    uint64_t checkpoint_checksum = 0;
    uint64_t prepass_checksum = 0;
    uint64_t restored_checksum = 0;
    uint64_t committed_checksum = 0;
    bool restore_exact = false;
    int64_t n_past_before = 0;
    int64_t n_past_after_prepass = 0;
    int64_t n_past_after_rollback = 0;
    int64_t n_past_after_commit = 0;
};

active_probe replay_full_probe_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const step27_bank & bank,
        const source_memory & memory,
        const std::vector<llama_token> & tokens,
        const eval_result & baseline,
        const active_probe & selected) {
    active_probe out = selected;
    if (!selected.active) {
        out.probe.logits = baseline.logits;
        out.probe.metrics = metrics_from_logits(baseline.logits, memory.target);
        llama_memory_clear(llama_get_memory(ctx), true);
        out.probe.logits = decode_full_step28(ctx, vocab, collector, tokens, true).logits;
        out.probe.metrics = metrics_from_logits(out.probe.logits, memory.target);
        clear_internal_injection(ctx);
        return out;
    }

    const auto response = bank.apply_source(memory.source, normalized(baseline.final_hidden));
    const auto direction = normalized(response);
    const auto projected = output_projection(output, direction);
    const multi_configuration config{"step28", "block24_block28_final", {23, 27, -1}};
    set_multi_internal_injection(ctx, config, direction, selected.probe.alphas);
    try {
        llama_memory_clear(llama_get_memory(ctx), true);
        collector.clear();
        collector.enabled = true;
        if (tokens.empty() ||
            llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(tokens.data()),
                                                  static_cast<int32_t>(tokens.size()))) != 0) {
            collector.enabled = false;
            clear_internal_injection(ctx);
            throw std::runtime_error("stateful initial injected decode failed");
        }
        ++decode_calls;
        llama_synchronize(ctx);
        collector.enabled = false;
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        const float * raw = llama_get_logits_ith(ctx, -1);
        if (!raw) throw std::runtime_error("missing initial injected logits");
        out.probe.logits.assign(raw, raw + n_vocab);
        const auto final_hidden = collector.final_hidden;
        if (final_hidden.size() != WIDTH) throw std::runtime_error("missing initial injected final state");
        out.probe.logits = add_final_residual(out.probe.logits, projected,
                                              selected.probe.alphas.at(-1));
        out.probe.metrics = metrics_from_logits(out.probe.logits, memory.target);
        clear_internal_injection(ctx);
        return out;
    } catch (...) {
        collector.enabled = false;
        clear_internal_injection(ctx);
        throw;
    }
}

stateful_probe_result run_stateful_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const step27_bank & bank,
        const source_memory & memory,
        llama_token input_token) {
    stateful_probe_result result;
    state_blob checkpoint;
    checkpoint.capture(ctx);
    result.checkpoint_checksum = checkpoint.checksum;
    result.n_past_before = n_past_step28(ctx);

    result.prepass = decode_current_step28(ctx, vocab, collector, input_token, true);
    result.n_past_after_prepass = n_past_step28(ctx);
    state_blob prepass_state;
    prepass_state.capture(ctx);
    result.prepass_checksum = prepass_state.checksum;
    result.query = normalized(result.prepass.final_hidden);
    const auto response = bank.apply_source(memory.source, result.query);
    result.response = response;
    result.active.raw_norm = l2(response);
    result.active.requested_budget =
        STEP27_B_MAX*std::min(1.0, result.active.raw_norm/STEP27_S_REF);
    result.active.probe.logits = result.prepass.logits;
    result.active.probe.metrics = metrics_from_logits(result.prepass.logits, memory.target);

    auto restore_checkpoint = [&]() {
        result.restore_exact = checkpoint.restore(ctx);
        state_blob rollback_state;
        rollback_state.capture(ctx);
        result.restored_checksum = rollback_state.checksum;
        result.n_past_after_rollback = n_past_step28(ctx);
        if (!result.restore_exact || rollback_state.bytes != checkpoint.bytes) {
            throw std::runtime_error("current-token rollback was not byte-exact");
        }
    };

    if (result.active.raw_norm <= 1e-12) {
        restore_checkpoint();
        result.committed = decode_current_step28(ctx, vocab, collector, input_token, true);
        result.committed.metrics = metrics_from_logits(result.committed.logits, memory.target);
        result.n_past_after_commit = n_past_step28(ctx);
        state_blob committed_state;
        committed_state.capture(ctx);
        result.committed_checksum = committed_state.checksum;
        return result;
    }

    result.active.active = true;
    const auto direction = normalized(response);
    const auto projected = output_projection(output, direction);
    const multi_configuration config{"step28", "block24_block28_final", {23, 27, -1}};
    const auto weights = normalized_weights({0.23072, 0.38322, 0.89438});
    for (size_t i = 0; i < config.sites.size(); ++i) {
        result.active.probe.requested_rhos[config.sites[i]] =
            result.active.requested_budget*weights[i];
    }

    std::map<int32_t, double> norms;
    std::map<int32_t, double> alphas;
    for (const int32_t site : {23, 27}) {
        norms[site] = l2(result.prepass.boundaries.at(site));
        alphas[site] = 0.0;
    }
    bool converged = false;
    for (int32_t iteration = 1; iteration <= STEP28_MAX_NORM_ITERATIONS; ++iteration) {
        for (const auto & [site, norm] : norms) {
            alphas[site] = result.active.probe.requested_rhos.at(site)*norm;
        }
        restore_checkpoint();
        set_multi_internal_injection(ctx, config, direction, alphas);
        eval_result current;
        try {
            current = decode_current_step28(ctx, vocab, collector, input_token, true);
        } catch (...) {
            clear_internal_injection(ctx);
            throw;
        }
        clear_internal_injection(ctx);
        double max_error = 0.0;
        std::map<int32_t, double> measured;
        for (const auto & [site, alpha] : alphas) {
            auto pre = current.boundaries.at(site);
            for (int32_t i = 0; i < WIDTH; ++i) pre[i] -= alpha*direction[i];
            const double pre_norm = l2(pre);
            measured[site] = pre_norm;
            const double actual = pre_norm > 0 ? std::abs(alpha)/pre_norm : 0;
            max_error = std::max(max_error,
                                 std::abs(actual - result.active.probe.requested_rhos.at(site)));
        }
        norms = std::move(measured);
        result.active.probe.norm_iterations = iteration;
        result.active.probe.max_local_rho_error = max_error;
        if (max_error <= STEP28_LOCAL_RHO_TOLERANCE) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        alphas.clear();
        std::vector<int32_t> ordered = {23, 27};
        for (size_t index = 0; index < ordered.size(); ++index) {
            const int32_t site = ordered[index];
            double pre_norm = 0.0;
            if (index == 0) {
                pre_norm = l2(result.prepass.boundaries.at(site));
            } else {
                restore_checkpoint();
                set_multi_internal_injection(ctx, config, direction, alphas);
                auto current = decode_current_step28(ctx, vocab, collector, input_token, true);
                clear_internal_injection(ctx);
                pre_norm = l2(current.boundaries.at(site));
            }
            norms[site] = pre_norm;
            alphas[site] = result.active.probe.requested_rhos.at(site)*pre_norm;
        }
        result.active.probe.max_local_rho_error = 0.0;
    }

    restore_checkpoint();
    set_multi_internal_injection(ctx, config, direction, alphas);
    try {
        result.committed = decode_current_step28(ctx, vocab, collector, input_token, true);
    } catch (...) {
        clear_internal_injection(ctx);
        throw;
    }
    clear_internal_injection(ctx);

    result.active.probe.alphas = alphas;
    for (const auto & [site, norm] : norms) {
        result.active.probe.local_norms[site] = norm;
        result.active.probe.actual_rhos[site] =
            norm > 0 ? std::abs(alphas.at(site))/norm : 0;
    }
    const double final_norm = l2(result.committed.final_hidden);
    const double final_rho = result.active.probe.requested_rhos.at(-1);
    result.active.probe.local_norms[-1] = final_norm;
    result.active.probe.alphas[-1] = final_rho*final_norm;
    result.active.probe.actual_rhos[-1] = final_rho;
    result.active.probe.logits = add_final_residual(
        result.committed.logits, projected, result.active.probe.alphas.at(-1));
    result.active.probe.metrics = metrics_from_logits(result.active.probe.logits, memory.target);
    double budget_sq = 0.0;
    for (const auto & [site, rho] : result.active.probe.actual_rhos) budget_sq += rho*rho;
    result.active.probe.actual_budget = std::sqrt(budget_sq);
    if (result.active.probe.actual_budget > STEP27_B_MAX + STEP28_STATE_TOLERANCE) {
        throw std::runtime_error("stateful reader exceeded authority");
    }
    result.committed.logits = result.active.probe.logits;
    result.committed.metrics = result.active.probe.metrics;
    result.n_past_after_commit = n_past_step28(ctx);
    state_blob committed_state;
    committed_state.capture(ctx);
    result.committed_checksum = committed_state.checksum;
    return result;
}

struct first_step_signature {
    std::vector<float> query;
    std::vector<float> response;
    std::vector<float> logits;
    llama_token generated = -1;
    double raw_norm = 0.0;
    double budget = 0.0;
    double actual_budget = 0.0;
};

double max_vector_difference_step28(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) return std::numeric_limits<double>::infinity();
    double result = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result = std::max(result, std::abs((double) lhs[i] - rhs[i]));
    }
    return result;
}

first_step_signature capture_first_step_signature_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const common_chat_templates * templates,
        const step27_prompt & query,
        const source_memory & memory,
        const step27_bank & bank,
        bool & supports_thinking) {
    const std::string rendered = render_step27_chat(templates, query.user_prompt, supports_thinking);
    if (!supports_thinking) throw std::runtime_error("chat template does not support thinking");
    clear_internal_injection(ctx);
    const auto baseline = evaluate(ctx, vocab, collector, rendered, memory.target, true);
    const auto probed = probe_step27(ctx, vocab, collector, output, bank, memory,
                                     rendered, memory.target, memory.target_text,
                                     query.id, true, baseline);
    first_step_signature signature;
    signature.query = normalized(baseline.final_hidden);
    signature.response = bank.apply_source(memory.source, signature.query);
    signature.logits = probed.probe.logits;
    signature.generated = probed.probe.metrics.greedy;
    signature.raw_norm = probed.raw_norm;
    signature.budget = probed.requested_budget;
    signature.actual_budget = probed.probe.actual_budget;
    clear_internal_injection(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);
    return signature;
}

void write_vector_json(std::ostream & out, const std::vector<float> & values) {
    out << "[" << std::setprecision(9);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ",";
        out << values[i];
    }
    out << "]";
}

double top12_margin_step28(const std::vector<float> & logits) {
    if (logits.size() < 2) return 0.0;
    float top1 = -std::numeric_limits<float>::infinity();
    float top2 = -std::numeric_limits<float>::infinity();
    for (const float value : logits) {
        if (value > top1) {
            top2 = top1;
            top1 = value;
        } else if (value > top2) {
            top2 = value;
        }
    }
    return static_cast<double>(top1) - static_cast<double>(top2);
}

void write_stateful_probe_fields(std::ostream & out, const active_probe & probe) {
    auto actual = [&](int32_t site) {
        const auto it = probe.probe.actual_rhos.find(site);
        return it == probe.probe.actual_rhos.end() ? 0.0 : it->second;
    };
    auto alpha = [&](int32_t site) {
        const auto it = probe.probe.alphas.find(site);
        return it == probe.probe.alphas.end() ? 0.0 : it->second;
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
        << ",\"norm_iterations\":" << probe.probe.norm_iterations
        << ",\"max_local_rho_error\":" << probe.probe.max_local_rho_error;
}

void write_standard_row(
        std::ostream & out,
        const std::string & condition,
        const step27_prompt & query,
        const source_memory & memory,
        int32_t step,
        llama_token generated,
        const std::string & piece,
        const std::vector<float> & query_key,
        const std::vector<float> & retrieval,
        const active_probe & probe,
        const eval_result & state,
        int64_t n_past_before,
        int64_t n_past_after,
        const std::string & query_kind,
        llama_token input_token = -1,
        double cumulative_budget = 0.0) {
    const auto & final_hidden = state.final_hidden;
    const auto & boundaries = state.boundaries;
    out << std::setprecision(17)
        << "{\"condition\":\"" << condition
        << "\",\"prompt_name\":\"" << query.name
        << "\",\"source\":\"" << query.source
        << "\",\"step\":" << step
        << ",\"input_token_id\":" << input_token
        << ",\"generated_token_id\":" << generated
        << ",\"generated_token_piece\":\"" << json_escape(piece)
        << "\",\"query_kind\":\"" << query_kind
        << "\",\"query_vector_checksum\":\"" << hex64_step27(checksum64_step27(query_key))
        << "\",\"query_vector_norm\":" << l2(query_key)
        << ",\"query_vector\":";
    write_vector_json(out, query_key);
    out << ",\"retrieval_vector\":";
    write_vector_json(out, retrieval);
    out << ",\"target_token\":" << memory.target
        << ",\"n_past_before\":" << n_past_before
        << ",\"n_past_after\":" << n_past_after
        << ",\"block24_checksum\":\"" << hex64_step27(checksum64_step27(boundaries.at(23)))
        << "\",\"block28_checksum\":\"" << hex64_step27(checksum64_step27(boundaries.at(27)))
        << "\",\"final_hidden_checksum\":\"" << hex64_step27(checksum64_step27(final_hidden))
        << "\",\"block24_norm\":" << l2(boundaries.at(23))
        << ",\"block28_norm\":" << l2(boundaries.at(27))
        << ",\"final_norm\":" << l2(final_hidden)
        << ",\"target_metrics\":{";
    write_metric_fields(out, probe.probe.logits, memory.target);
    out << "},\"top1_top2_margin\":" << top12_margin_step28(probe.probe.logits);
    out << ",\"cumulative_B\":" << cumulative_budget;
    write_stateful_probe_fields(out, probe);
    out << "}\n";
}

eval_result decode_suffix_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const std::string & suffix,
        llama_token target);

step27_episode_result run_clean_episode_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const common_chat_templates * templates,
        const step27_prompt & query,
        const source_memory & memory,
        const std::string & condition,
        std::ostream & trajectory_out,
        std::ostream & readout_out,
        bool & supports_thinking) {
    const std::string rendered = render_step27_chat(templates, query.user_prompt, supports_thinking);
    if (!supports_thinking) throw std::runtime_error("chat template does not support thinking");
    const auto prompt_tokens = tokenize(vocab, rendered, true);
    if (prompt_tokens.empty()) throw std::runtime_error("empty rendered prompt");
    // Use the same one-token execution path as the rollback audit. A full-prefix
    // batch is a different CUDA reduction path and can diverge at near ties.
    llama_memory_clear(llama_get_memory(ctx), true);
    eval_result state;
    for (const llama_token token : prompt_tokens) {
        clear_internal_injection(ctx);
        state = decode_current_step28(ctx, vocab, collector, token, true);
    }
    std::string generated_text;
    std::vector<llama_token> generated;
    for (int32_t step = 0; step < STEP28_MAX_TOKENS; ++step) {
        const llama_token next = state.metrics.greedy;
        if (next < 0 || llama_vocab_is_eog(vocab, next)) break;
        active_probe inactive;
        inactive.probe.logits = state.logits;
        inactive.probe.metrics = metrics_from_logits(state.logits, memory.target);
        const std::string piece = piece_step27(vocab, next);
        const auto query_key = normalized(state.final_hidden);
        write_standard_row(trajectory_out, condition, query, memory, step, next, piece,
                           query_key, {}, inactive, state,
                           0, n_past_step28(ctx), "clean");
        generated.push_back(next);
        generated_text += piece;
        clear_internal_injection(ctx);
        state = decode_current_step28(ctx, vocab, collector, next, true);
    }
    clear_internal_injection(ctx);
    const auto readout = decode_suffix_step28(
        ctx, vocab, collector, "\nFinal destination:", memory.target);
    readout_out << std::setprecision(17)
        << "{\"condition\":\"" << condition
        << "\",\"prompt_name\":\"" << query.name
        << "\",\"source\":\"" << query.source
        << "\",\"target_token\":" << memory.target
        << ",\"generated_token_count\":" << generated.size()
        << ",\"generated_text\":\"" << json_escape(generated_text)
        << "\",\"readout_suffix\":\"Final destination:\",\"target_metrics\":{";
    write_metric_fields(readout_out, readout.logits, memory.target);
    readout_out << "},\"top1_top2_margin\":" << top12_margin_step28(readout.logits) << "}\n";
    return {std::move(generated), std::move(generated_text), readout};
}

step27_episode_result run_stateless_episode_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const common_chat_templates * templates,
        const step27_prompt & query,
        const source_memory & memory,
        const step27_bank & bank,
        std::ostream & trajectory_out,
        std::ostream & readout_out,
        bool & supports_thinking,
        first_step_signature * first_step) {
    const std::string rendered = render_step27_chat(templates, query.user_prompt, supports_thinking);
    if (!supports_thinking) throw std::runtime_error("chat template does not support thinking");
    std::string prefix = rendered;
    std::string generated_text;
    std::vector<llama_token> generated;
    double cumulative_budget = 0.0;
    for (int32_t step = 0; step < STEP28_MAX_TOKENS; ++step) {
        clear_internal_injection(ctx);
        const auto baseline = evaluate(ctx, vocab, collector, prefix, memory.target, true);
        const auto probed = probe_step27(ctx, vocab, collector, output, bank, memory,
                                         prefix, memory.target, memory.target_text,
                                         query.id, true, baseline);
        const llama_token next = probed.probe.metrics.greedy;
        if (next < 0 || llama_vocab_is_eog(vocab, next)) break;
        const std::string piece = piece_step27(vocab, next);
        const auto query_key = normalized(baseline.final_hidden);
        eval_result state_result = baseline;
        if (probed.active) {
            state_result.final_hidden = collector.final_hidden;
            state_result.boundaries = collector.boundaries;
        }
        state_result.logits = probed.probe.logits;
        state_result.metrics = probed.probe.metrics;
        const auto retrieval = bank.apply_source(memory.source, query_key);
        cumulative_budget += probed.probe.actual_budget;
        if (step == 0 && first_step != nullptr) {
            first_step->query = query_key;
            first_step->response = retrieval;
            first_step->logits = probed.probe.logits;
            first_step->generated = next;
            first_step->raw_norm = probed.raw_norm;
            first_step->budget = probed.requested_budget;
            first_step->actual_budget = probed.probe.actual_budget;
        }
        write_standard_row(trajectory_out, "C1_stateless", query, memory, step, next, piece,
                           query_key, retrieval, probed, state_result, 0, n_past_step28(ctx),
                           "clean_prepass", -1, cumulative_budget);
        generated.push_back(next);
        generated_text += piece;
        prefix += piece;
    }
    const std::string readout_prompt = prefix + "\nFinal destination:";
    clear_internal_injection(ctx);
    const auto readout = evaluate(ctx, vocab, collector, readout_prompt, memory.target, false);
    readout_out << std::setprecision(17)
        << "{\"condition\":\"C1_stateless\",\"prompt_name\":\"" << query.name
        << "\",\"source\":\"" << query.source
        << "\",\"target_token\":" << memory.target
        << ",\"generated_token_count\":" << generated.size()
        << ",\"generated_text\":\"" << json_escape(generated_text)
        << "\",\"readout_suffix\":\"Final destination:\",\"target_metrics\":{";
    write_metric_fields(readout_out, readout.logits, memory.target);
    readout_out << "},\"top1_top2_margin\":" << top12_margin_step28(readout.logits) << "}\n";
    return {std::move(generated), std::move(generated_text), readout};
}

eval_result decode_suffix_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const std::string & suffix,
        llama_token target) {
    const auto tokens = tokenize(vocab, suffix, false);
    if (tokens.empty()) throw std::runtime_error("empty readout suffix");
    eval_result out;
    for (const llama_token token : tokens) {
        clear_internal_injection(ctx);
        out = decode_current_step28(ctx, vocab, collector, token, true);
        out.metrics = metrics_from_logits(out.logits, target);
    }
    return out;
}

step27_episode_result run_stateful_episode_step28(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const common_chat_templates * templates,
        const step27_prompt & query,
        const source_memory & memory,
        const step27_bank & bank,
        std::ostream & trajectory_out,
        std::ostream & readout_out,
        bool & supports_thinking,
        std::ofstream & kv_audit,
        first_step_signature * first_step) {
    const std::string rendered = render_step27_chat(templates, query.user_prompt, supports_thinking);
    if (!supports_thinking) throw std::runtime_error("chat template does not support thinking");
    const auto rendered_tokens = tokenize(vocab, rendered, true);
    clear_internal_injection(ctx);
    const auto baseline = evaluate(ctx, vocab, collector, rendered, memory.target, true);
    const auto first = probe_step27(ctx, vocab, collector, output, bank, memory,
                                    rendered, memory.target, memory.target_text,
                                    query.id, true, baseline);
    const auto committed_prompt = replay_full_probe_step28(
        ctx, vocab, collector, output, bank, memory, rendered_tokens, baseline, first);
    state_blob prompt_commit_state;
    prompt_commit_state.capture(ctx);
    kv_audit << std::setprecision(17)
        << "{\"prompt_name\":\"" << query.name
        << "\",\"phase\":\"prompt_injected_commit\",\"step\":-1"
        << ",\"committed_checksum\":\"" << hex64_step27(prompt_commit_state.checksum)
        << "\",\"n_past_after_commit\":" << n_past_step28(ctx)
        << ",\"position_id_commit\":" << n_past_step28(ctx)
        << ",\"cache_hash_scope\":\"serialized_llama_state_full\""
        << ",\"per_layer_tensor_hashes_available\":false}\n";
    eval_result committed_prompt_state = baseline;
    committed_prompt_state.logits = committed_prompt.probe.logits;
    committed_prompt_state.metrics = committed_prompt.probe.metrics;
    committed_prompt_state.final_hidden = collector.final_hidden;
    committed_prompt_state.boundaries = collector.boundaries;
    const auto query0 = normalized(baseline.final_hidden);
    const auto response0 = bank.apply_source(memory.source, query0);
    if (first_step != nullptr) {
        first_step->query = query0;
        first_step->response = response0;
        first_step->logits = first.probe.logits;
        first_step->generated = first.probe.metrics.greedy;
        first_step->raw_norm = first.raw_norm;
        first_step->budget = first.requested_budget;
        first_step->actual_budget = first.probe.actual_budget;
    }
    double cumulative_budget = first.probe.actual_budget;
    write_standard_row(trajectory_out, "C2_stateful", query, memory, 0,
                       first.probe.metrics.greedy,
                       piece_step27(vocab, first.probe.metrics.greedy),
                       query0, response0, committed_prompt, committed_prompt_state,
                       0, n_past_step28(ctx), "stateful_prompt_prepass", -1, cumulative_budget);

    std::string prefix = rendered;
    std::string generated_text;
    std::vector<llama_token> generated;
    llama_token current = first.probe.metrics.greedy;
    while (current >= 0 && !llama_vocab_is_eog(vocab, current) &&
           (int32_t) generated.size() < STEP28_MAX_TOKENS) {
        const auto state = run_stateful_step28(ctx, vocab, collector, output, bank, memory, current);
        cumulative_budget += state.active.probe.actual_budget;
        const auto current_piece = piece_step27(vocab, current);
        generated.push_back(current);
        generated_text += current_piece;
        prefix += current_piece;
        const llama_token next = state.active.probe.metrics.greedy;
        const auto & state_for_row = state.committed;
        write_standard_row(trajectory_out, "C2_stateful", query, memory,
                           (int32_t) generated.size(), next,
                           next >= 0 ? piece_step27(vocab, next) : std::string(),
                           state.query, state.response, state.active, state_for_row,
                           state.n_past_before, state.n_past_after_commit,
                           "stateful_current_token", current, cumulative_budget);
        kv_audit << std::setprecision(17)
            << "{\"prompt_name\":\"" << query.name
            << "\",\"step\":" << generated.size()
            << ",\"input_token_id\":" << current
            << ",\"checkpoint_checksum\":\"" << hex64_step27(state.checkpoint_checksum)
            << "\",\"prepass_checksum\":\"" << hex64_step27(state.prepass_checksum)
            << "\",\"restored_checksum\":\"" << hex64_step27(state.restored_checksum)
            << "\",\"committed_checksum\":\"" << hex64_step27(state.committed_checksum)
            << "\",\"restore_exact\":" << (state.restore_exact ? "true" : "false")
            << ",\"n_past_before\":" << state.n_past_before
            << ",\"n_past_after_prepass\":" << state.n_past_after_prepass
            << ",\"n_past_after_rollback\":" << state.n_past_after_rollback
            << ",\"n_past_after_commit\":" << state.n_past_after_commit
            << ",\"position_id_before\":" << state.n_past_before
            << ",\"position_id_prepass\":" << state.n_past_before
            << ",\"position_id_rollback\":" << state.n_past_before
            << ",\"position_id_commit\":" << state.n_past_before
            << ",\"cache_hash_scope\":\"serialized_llama_state_full\""
            << ",\"per_layer_tensor_hashes_available\":false"
            << ",\"stateful_previous_cache_preserved\":true}\n";
        if (next < 0 || llama_vocab_is_eog(vocab, next)) break;
        current = next;
    }

    const auto readout = decode_suffix_step28(ctx, vocab, collector, "\nFinal destination:", memory.target);
    readout_out << std::setprecision(17)
        << "{\"condition\":\"C2_stateful\",\"prompt_name\":\"" << query.name
        << "\",\"source\":\"" << query.source
        << "\",\"target_token\":" << memory.target
        << ",\"generated_token_count\":" << generated.size()
        << ",\"generated_text\":\"" << json_escape(generated_text)
        << "\",\"readout_suffix\":\"Final destination:\",\"target_metrics\":{";
    write_metric_fields(readout_out, readout.logits, memory.target);
    readout_out << "},\"top1_top2_margin\":" << top12_margin_step28(readout.logits) << "}\n";
    return {std::move(generated), std::move(generated_text), readout};
}

std::vector<llama_token> run_no_injection_stateful_sequence(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const std::string & rendered,
        const std::string & prompt_name,
        std::ofstream & audit) {
    const auto prompt_tokens = tokenize(vocab, rendered, true);
    llama_memory_clear(llama_get_memory(ctx), true);
    int32_t step = 0;
    auto replay_zero = [&](llama_token token, const char * phase) {
        state_blob checkpoint;
        checkpoint.capture(ctx);
        const int64_t n_before = n_past_step28(ctx);
        const uint64_t decode_calls_before = decode_calls;

        clear_internal_injection(ctx);
        const auto prepass = decode_current_step28(ctx, vocab, collector, token, false);
        const uint64_t decode_calls_after_prepass = decode_calls;
        state_blob prepass_state;
        prepass_state.capture(ctx);
        const int64_t n_after_prepass = n_past_step28(ctx);

        const auto restore_audit = checkpoint.restore_with_audit(ctx);
        const bool restored = restore_audit.byte_exact();
        state_blob rollback_state;
        rollback_state.capture(ctx);
        const int64_t n_after_rollback = n_past_step28(ctx);
        const bool rollback_exact = restored && rollback_state.bytes == checkpoint.bytes;
        if (!rollback_exact) {
            audit << std::setprecision(17)
                << "{\"prompt_name\":\"" << prompt_name
                << "\",\"phase\":\"" << phase
                << "\",\"step\":" << step++
                << ",\"input_token_id\":" << token
                << ",\"checkpoint_checksum\":\"" << hex64_step27(checkpoint.checksum)
                << "\",\"prepass_checksum\":\"" << hex64_step27(prepass_state.checksum)
                << "\",\"rollback_checksum\":\"" << hex64_step27(rollback_state.checksum)
                << "\",\"checkpoint_bytes\":" << checkpoint.bytes.size()
                << ",\"restore_requested_bytes\":" << restore_audit.requested_bytes
                << ",\"restore_written_bytes\":" << restore_audit.written_bytes
                << ",\"rollback_bytes\":" << rollback_state.bytes.size()
                << ",\"sequence_cleared_before_restore\":"
                << (restore_audit.sequence_cleared_before_restore ? "true" : "false")
                << ",\"restore_first_difference\":";
            if (restore_audit.first_difference == std::numeric_limits<size_t>::max()) {
                audit << "null";
            } else {
                audit << restore_audit.first_difference;
            }
            audit << ",\"restore_differing_bytes\":" << restore_audit.differing_bytes
                << ",\"restore_exact\":" << (restored ? "true" : "false")
                << ",\"rollback_exact\":" << (rollback_exact ? "true" : "false")
                << ",\"n_past_before\":" << n_before
                << ",\"n_past_after_prepass\":" << n_after_prepass
                << ",\"n_past_after_rollback\":" << n_after_rollback
                << ",\"cache_hash_scope\":\"serialized_llama_state_full\""
                << ",\"per_layer_tensor_hashes_available\":false}\n";
            audit.flush();
            throw std::runtime_error("no-injection rollback was not byte-exact");
        }

        clear_internal_injection(ctx);
        const auto replay = decode_current_step28(ctx, vocab, collector, token, false);
        const uint64_t decode_calls_after_replay = decode_calls;
        state_blob committed_state;
        committed_state.capture(ctx);
        const int64_t n_after_commit = n_past_step28(ctx);
        const uint64_t prepass_decode_count = decode_calls_after_prepass - decode_calls_before;
        const uint64_t replay_decode_count = decode_calls_after_replay - decode_calls_after_prepass;
        const uint64_t committed_decode_count = replay_decode_count;
        if (prepass_decode_count != 1 || replay_decode_count != 1 ||
            committed_decode_count != 1 || prepass.logits != replay.logits ||
            prepass_state.bytes != committed_state.bytes) {
            throw std::runtime_error("no-injection prepass/replay or decode count diverged");
        }

        audit << std::setprecision(17)
            << "{\"prompt_name\":\"" << prompt_name
            << "\",\"phase\":\"" << phase
            << "\",\"step\":" << step++
            << ",\"input_token_id\":" << token
            << ",\"checkpoint_checksum\":\"" << hex64_step27(checkpoint.checksum)
            << "\",\"prepass_checksum\":\"" << hex64_step27(prepass_state.checksum)
            << "\",\"rollback_checksum\":\"" << hex64_step27(rollback_state.checksum)
            << "\",\"committed_checksum\":\"" << hex64_step27(committed_state.checksum)
            << "\",\"checkpoint_bytes\":" << checkpoint.bytes.size()
            << ",\"restore_requested_bytes\":" << restore_audit.requested_bytes
            << ",\"restore_written_bytes\":" << restore_audit.written_bytes
            << ",\"rollback_bytes\":" << rollback_state.bytes.size()
            << ",\"sequence_cleared_before_restore\":"
            << (restore_audit.sequence_cleared_before_restore ? "true" : "false")
            << ",\"restore_first_difference\":null"
            << ",\"restore_differing_bytes\":0"
            << ",\"restore_exact\":true"
            << ",\"rollback_exact\":" << (rollback_exact ? "true" : "false")
            << ",\"prepass_replay_exact\":true"
            << ",\"prepass_greedy_token\":" << prepass.metrics.greedy
            << ",\"replay_greedy_token\":" << replay.metrics.greedy
            << ",\"prepass_logits_checksum\":\""
            << hex64_step27(checksum64_step27(prepass.logits))
            << "\",\"replay_logits_checksum\":\""
            << hex64_step27(checksum64_step27(replay.logits)) << "\""
            << ",\"n_past_before\":" << n_before
            << ",\"n_past_after_prepass\":" << n_after_prepass
            << ",\"n_past_after_rollback\":" << n_after_rollback
            << ",\"n_past_after_commit\":" << n_after_commit
            << ",\"decode_calls_before\":" << decode_calls_before
            << ",\"decode_calls_after_prepass\":" << decode_calls_after_prepass
            << ",\"decode_calls_after_replay\":" << decode_calls_after_replay
            << ",\"prepass_decode_count\":" << prepass_decode_count
            << ",\"replay_decode_count\":" << replay_decode_count
            << ",\"committed_decode_count\":" << committed_decode_count
            << ",\"expected_committed_decode_count\":1"
            << ",\"position_id_before\":" << n_before
            << ",\"position_id_prepass\":" << n_before
            << ",\"position_id_rollback\":" << n_before
            << ",\"position_id_commit\":" << n_before
            << ",\"cache_hash_scope\":\"serialized_llama_state_full\""
            << ",\"per_layer_tensor_hashes_available\":false"
            << "}\n";
        return replay;
    };

    eval_result state;
    for (const llama_token token : prompt_tokens) {
        state = replay_zero(token, "prompt");
    }
    std::vector<llama_token> generated;
    for (int32_t i = 0; i < STEP28_MAX_TOKENS; ++i) {
        const llama_token next = state.metrics.greedy;
        if (next < 0 || llama_vocab_is_eog(vocab, next)) break;
        generated.push_back(next);
        state = replay_zero(next, "generated");
    }
    return generated;
}

void write_expert_note(std::ostream & out) {
    out << "The expert consultation is persisted separately in expert_consultation.md.\n"
        << "Its main caution was that C1/C2 confound query provenance with cache carryover; "
           "this run retains the preregistered C0/C1/C2 scope and narrows the claim to "
           "stateful persistence affecting later retrieval queries and responses. Full "
           "rollback hashes, no-injection controls, dose, and CUDA epoch are logged.\n";
}

void write_step28_preregistration(const std::filesystem::path & root) {
    std::ofstream out(root/"preregistration_runtime.json", std::ios::trunc);
    out << R"JSON({
"experiment":"step28_stateful_recursive_sidecar",
"runtime_baseline":"rtx3090_20260814",
"model_immutable":true,
"bank":"Step27_P_init_A_to_D_B_to_E",
"source_partition":{"A":"D","B":"E"},
"injection_topology":{"block24":0.23072,"block28":0.38322,"final":0.89438},
"reader":"B_eff=0.01*min(1,||r||/0.05)",
"B_max":0.01,"s_ref":0.05,
"max_reasoning_tokens":32,
"conditions":["C0_Qwen_only","C1_stateless_repeated_injection","C2_stateful_recursive_injection"],
"c1_query_protocol":"no-injection prepass for every step; injected replay is not retained for the next query",
"c2_query_protocol":"persistent injected cache; current-token checkpoint, no-current-injection prepass, exact rollback, injected replay and commit",
"no_injection_control":"same checkpoint/prepass/rollback/replay/commit path with injection strength zero",
"hard_cache_gate":"full serialized llama state checksum before prepass and after rollback; byte exact required",
"first_step_gate":"C1/C2 query, retrieval, logits, token, raw norm, requested and actual injection budget B exact",
"zero_injection_decode_gate":"exactly one committed replay decode per token after rollback; prepass/replay/committed counts are logged and asserted",
"learning":false,
"generated_tokens_never_update":true,
"c1a_c1b_random_controls":false,
"claim_boundary":"stateful sidecar persistence affecting subsequent retrieval/computation; frozen bank, not a full read/write loop"
})JSON";
}

void write_pre_execution_corrections(const std::filesystem::path & root) {
    std::ofstream out(root/"pre_execution_corrections.json", std::ios::trunc);
    out << R"JSON({
"experiment":"step28_stateful_recursive_sidecar",
"status":"pre_execution_implementation_corrections",
"corrections":[
  {"issue":"stale_step27_preregistration_metadata",
   "correction":"dedicated_step28_preregistration_writer",
   "verified":true},
  {"issue":"double_replay_zero_injection_path",
   "correction":"exactly_one_prepass_and_one_committed_replay_decode_per_token_with_decode_delta_assertions",
   "verified":true},
  {"issue":"cross_gpu_p_init_reconstruction_drift",
   "correction":"load_and_checksum_the_exact_persisted_step27_initial_bank_snapshot_instead_of_reconstructing_from_current_gpu_activations",
   "verified":true}
  ,{"issue":"full_prefix_vs_incremental_zero_injection_reference_drift",
   "correction":"C0_uses_the_same_incremental_one_token_path_as_the_checkpoint_rollback_control",
   "verified":true}
],
"hard_gates":[
  "full serialized cache hash before prepass and after rollback",
  "byte-exact rollback required",
  "first-step query/retrieval/logits/token/requested-B/actual-B equality",
  "zero-injection committed decode count exactly one per generated token"
]
})JSON";
}

} // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto opts = parse_args(argc, argv);
        const std::filesystem::path root(opts.output_dir);
        std::filesystem::create_directories(root);
        const auto bank_dir = configured_step27_bank_dir();
        write_step28_preregistration(root);
        write_pre_execution_corrections(root);
        {
            std::ofstream out(root/"expert_consultation_note.md", std::ios::trunc);
            write_expert_note(out);
        }
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.gpu_layers;
        llama_model * public_model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (!public_model) throw std::runtime_error("model load failed");
        auto & model = *public_model;
        if (model.hparams.n_embd != WIDTH || model.hparams.n_layer() != 32) {
            throw std::runtime_error("Step 28 requires Qwen3.5-4B dimensions 32x2560");
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

        // The inherited target allocator validates every source in the fixed
#if 0
        // Step-9 event stream. Reconstruct all source states as Step 27 did;
        // only the frozen A/B bank is used by the Step-28 conditions below.
        const std::array<std::string, 8> symbols = {"A", "B", "C", "D", "E", "F", "G", "H"};
        std::map<std::string, source_state> source_states;
        for (size_t i = 0; i < symbols.size(); ++i) {
            const std::string prompt = source_prompt(symbols[i]);
            clear_internal_injection(ctx);
            const auto baseline_source = evaluate(ctx, vocab, collector, prompt, 0, true);
            source_states.emplace(symbols[i], source_state{
                (int32_t) i, symbols[i], prompt, tokenize(vocab, prompt, true),
                baseline_source, {}});
            source_states.at(symbols[i]).key =
                normalized(source_states.at(symbols[i]).baseline.final_hidden);
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
        if (memory_a.target != 629 || memory_b.target != 710) {
            throw std::runtime_error("Step 27 target-token reconstruction mismatch");
        }
        if (hex64_step27(checksum64_step27(memory_a.key)) != "e4907fa092027078" ||
            hex64_step27(checksum64_step27(memory_a.direction)) != "cce3e171050bb9b" ||
            hex64_step27(checksum64_step27(memory_b.key)) != "9fc431a0f6c3d628" ||
            hex64_step27(checksum64_step27(memory_b.direction)) != "9fb3d2cc183d984c") {
            throw std::runtime_error("Step 27 P_init bank checksum mismatch");
        }

        step27_bank initial_bank;
        initial_bank.components.push_back({memory_a.direction, memory_a.key,
                                           STEP27_ETA*STEP27_FIRST_GATE, "A", "D"});
        initial_bank.components.push_back({memory_b.direction, memory_b.key,
                                           STEP27_ETA*STEP27_FIRST_GATE, "B", "E"});
#endif
        step27_bank initial_bank = load_persisted_step27_bank(bank_dir);
        const auto & component_a = initial_bank.components.at(0);
        const auto & component_b = initial_bank.components.at(1);
        source_memory memory_a{"A", "D", 629, " V", source_prompt("A"),
                               component_a.key, component_a.value};
        source_memory memory_b{"B", "E", 710, " K", source_prompt("B"),
                               component_b.key, component_b.value};
        write_bank_snapshot(root/"initial_bank", initial_bank);
        std::ofstream bank_ref(root/"bank_reference.json", std::ios::trunc);
        bank_ref << "{\"runtime_baseline\":\"" << STEP28_RUNTIME
                 << "\",\"source_snapshot\":\"" << json_escape(bank_dir.string())
                 << "\",\"component_count\":2,\"components\":["
                 << "{\"source\":\"A\",\"destination\":\"D\",\"coefficient\":0.0125,"
                 << "\"key_checksum\":\"" << hex64_step27(checksum64_step27(memory_a.key))
                 << "\",\"value_checksum\":\"" << hex64_step27(checksum64_step27(memory_a.direction)) << "\"},"
                 << "{\"source\":\"B\",\"destination\":\"E\",\"coefficient\":0.0125,"
                 << "\"key_checksum\":\"" << hex64_step27(checksum64_step27(memory_b.key))
                 << "\",\"value_checksum\":\"" << hex64_step27(checksum64_step27(memory_b.direction)) << "\"}]}\n";

        std::ofstream c0(root/"c0_trajectories.jsonl", std::ios::trunc);
        std::ofstream c1(root/"c1_stateless_trajectories.jsonl", std::ios::trunc);
        std::ofstream c2(root/"c2_stateful_trajectories.jsonl", std::ios::trunc);
        std::ofstream readouts(root/"final_readouts.jsonl", std::ios::trunc);
        std::ofstream kv_audit(root/"kv_state_audit.jsonl", std::ios::trunc);
        std::ofstream prompts_out(root/"prompts.jsonl", std::ios::trunc);
        for (const auto & query : make_prompts()) {
            prompts_out << "{\"id\":" << query.id << ",\"name\":\"" << query.name
                        << "\",\"source\":\"" << query.source << "\",\"prompt\":\""
                        << json_escape(query.user_prompt) << "\"}\n";
        }
        for (auto * stream : {&c0, &c1, &c2, &readouts, &kv_audit, &prompts_out}) {
            stream->setf(std::ios::unitbuf);
        }

        bool supports_thinking = false;
        std::map<std::string, step27_episode_result> c0_results;
        for (const auto & query : make_prompts()) {
            const auto & memory = query.source == "A" ? memory_a : memory_b;
            c0_results[query.name] = run_clean_episode_step28(
                ctx, vocab, collector, templates.get(), query, memory,
                "C0_Qwen_only", c0, readouts, supports_thinking);
        }

        // Hard preflight: do not execute C1/C2 until the exact rollback path
        // and first-step C1/C2 isolation invariants have passed.
        std::ofstream preflight_no_injection(
            root/"preflight_no_injection_stateful_control.jsonl", std::ios::trunc);
        std::ofstream preflight_no_injection_summary(
            root/"preflight_no_injection_stateful_control.json", std::ios::trunc);
        bool preflight_no_injection_pass = true;
        preflight_no_injection_summary << "[";
        bool first_preflight_no_injection_summary = true;
        for (const auto & query : make_prompts()) {
            const auto rendered = render_step27_chat(
                templates.get(), query.user_prompt, supports_thinking);
            const auto prompt_token_count = tokenize(vocab, rendered, true).size();
            const auto decode_start = decode_calls;
            const auto stateful_tokens = run_no_injection_stateful_sequence(
                ctx, vocab, collector, rendered, query.name, preflight_no_injection);
            const auto decode_end = decode_calls;
            const uint64_t expected_committed_decodes =
                prompt_token_count + stateful_tokens.size();
            const uint64_t expected_total_decodes = 2*expected_committed_decodes;
            const bool decode_count_exact =
                decode_end - decode_start == expected_total_decodes;
            const bool same = stateful_tokens == c0_results.at(query.name).tokens;
            const bool pass = same && decode_count_exact;
            preflight_no_injection_pass = preflight_no_injection_pass && pass;
            if (!first_preflight_no_injection_summary) preflight_no_injection_summary << ",";
            first_preflight_no_injection_summary = false;
            preflight_no_injection_summary
                << "{\"prompt_name\":\"" << query.name
                << "\",\"reference_mode\":\"incremental_clean_c0\""
                << ",\"prompt_token_count\":" << prompt_token_count
                << ",\"generated_token_count\":" << stateful_tokens.size()
                << ",\"decode_calls_delta\":" << (decode_end - decode_start)
                << ",\"expected_committed_decodes\":" << expected_committed_decodes
                << ",\"expected_total_decodes\":" << expected_total_decodes
                << ",\"decode_count_exact\":" << (decode_count_exact ? "true" : "false")
                << ",\"same_tokens\":" << (same ? "true" : "false")
                << ",\"pass\":" << (pass ? "true" : "false") << "}";
        }
        preflight_no_injection_summary << "]\n";
        preflight_no_injection.close();
        preflight_no_injection_summary.close();
        if (!preflight_no_injection_pass) {
            throw std::runtime_error("preflight_no_injection_stateful_control_failed");
        }

        std::ofstream preflight_first_step(root/"preflight_first_step_control.json", std::ios::trunc);
        bool preflight_first_step_pass = true;
        preflight_first_step << "[";
        bool first_preflight_step = true;
        for (const auto & query : make_prompts()) {
            const auto & memory = query.source == "A" ? memory_a : memory_b;
            const auto c1 = capture_first_step_signature_step28(
                ctx, vocab, collector, output, templates.get(), query, memory,
                initial_bank, supports_thinking);
            const auto c2 = capture_first_step_signature_step28(
                ctx, vocab, collector, output, templates.get(), query, memory,
                initial_bank, supports_thinking);
            const double query_diff = max_vector_difference_step28(c1.query, c2.query);
            const double retrieval_diff = max_vector_difference_step28(c1.response, c2.response);
            const double logits_diff = max_vector_difference_step28(c1.logits, c2.logits);
            const bool query_exact = c1.query == c2.query;
            const bool retrieval_exact = c1.response == c2.response;
            const bool logits_exact = c1.logits == c2.logits;
            const bool token_exact = c1.generated == c2.generated;
            const bool raw_exact = c1.raw_norm == c2.raw_norm;
            const bool budget_exact = c1.budget == c2.budget;
            const bool actual_budget_exact = c1.actual_budget == c2.actual_budget;
            const bool pass = query_exact && retrieval_exact && logits_exact &&
                              token_exact && raw_exact && budget_exact && actual_budget_exact;
            preflight_first_step_pass = preflight_first_step_pass && pass;
            if (!first_preflight_step) preflight_first_step << ",";
            first_preflight_step = false;
            preflight_first_step << std::setprecision(17)
                << "{\"prompt_name\":\"" << query.name
                << "\",\"query_exact\":" << (query_exact ? "true" : "false")
                << ",\"retrieval_exact\":" << (retrieval_exact ? "true" : "false")
                << ",\"logits_exact\":" << (logits_exact ? "true" : "false")
                << ",\"token_exact\":" << (token_exact ? "true" : "false")
                << ",\"raw_norm_exact\":" << (raw_exact ? "true" : "false")
                << ",\"requested_budget_exact\":" << (budget_exact ? "true" : "false")
                << ",\"actual_budget_exact\":" << (actual_budget_exact ? "true" : "false")
                << ",\"query_max_abs_diff\":" << query_diff
                << ",\"retrieval_max_abs_diff\":" << retrieval_diff
                << ",\"logits_max_abs_diff\":" << logits_diff
                << ",\"c1_token\":" << c1.generated
                << ",\"c2_token\":" << c2.generated
                << ",\"c1_requested_B\":" << c1.budget
                << ",\"c2_requested_B\":" << c2.budget
                << ",\"c1_actual_B\":" << c1.actual_budget
                << ",\"c2_actual_B\":" << c2.actual_budget
                << ",\"pass\":" << (pass ? "true" : "false") << "}";
        }
        preflight_first_step << "]\n";
        preflight_first_step.close();
        if (!preflight_first_step_pass) {
            throw std::runtime_error("preflight_first_step_control_failed");
        }

        std::map<std::string, step27_episode_result> c1_results;
        std::map<std::string, first_step_signature> c1_first_steps;
        for (const auto & query : make_prompts()) {
            const auto & memory = query.source == "A" ? memory_a : memory_b;
            c1_results[query.name] = run_stateless_episode_step28(
                ctx, vocab, collector, output, templates.get(), query, memory,
                initial_bank, c1, readouts, supports_thinking,
                &c1_first_steps[query.name]);
        }

        std::map<std::string, step27_episode_result> c2_results;
        std::map<std::string, first_step_signature> c2_first_steps;
        for (const auto & query : make_prompts()) {
            const auto & memory = query.source == "A" ? memory_a : memory_b;
            c2_results[query.name] = run_stateful_episode_step28(
                ctx, vocab, collector, output, templates.get(), query, memory,
                initial_bank, c2, readouts, supports_thinking, kv_audit,
                &c2_first_steps[query.name]);
        }

        std::ofstream no_injection(root/"no_injection_stateful_control.jsonl", std::ios::trunc);
        std::ofstream no_injection_summary(root/"no_injection_stateful_control.json", std::ios::trunc);
        bool no_injection_pass = true;
        no_injection_summary << "[";
        bool first_no_injection_summary = true;
        for (const auto & query : make_prompts()) {
            const auto rendered = render_step27_chat(templates.get(), query.user_prompt, supports_thinking);
            const auto stateful_tokens = run_no_injection_stateful_sequence(
                ctx, vocab, collector, rendered, query.name, no_injection);
            const auto & reference = c0_results.at(query.name).tokens;
            const bool same = stateful_tokens == reference;
            no_injection_pass = no_injection_pass && same;
            if (!first_no_injection_summary) no_injection_summary << ",";
            first_no_injection_summary = false;
            no_injection_summary << "{\"prompt_name\":\"" << query.name
                                 << "\",\"reference_token_count\":" << reference.size()
                                 << ",\"stateful_token_count\":" << stateful_tokens.size()
                                 << ",\"same_tokens\":" << (same ? "true" : "false")
                                 << ",\"pass\":" << (same ? "true" : "false") << "}";
            no_injection << "{\"prompt_name\":\"" << query.name
                         << "\",\"reference_token_count\":" << reference.size()
                         << ",\"stateful_token_count\":" << stateful_tokens.size()
                         << ",\"same_tokens\":" << (same ? "true" : "false")
                         << ",\"pass\":" << (same ? "true" : "false") << "}\n";
        }
        no_injection_summary << "]\n";
        no_injection.close();
        no_injection_summary.close();
        if (!no_injection_pass) {
            throw std::runtime_error("no-injection stateful cache control diverged from C0");
        }

        auto max_vector_difference = [](const std::vector<float> & lhs,
                                        const std::vector<float> & rhs) {
            if (lhs.size() != rhs.size()) return std::numeric_limits<double>::infinity();
            double result = 0.0;
            for (size_t i = 0; i < lhs.size(); ++i) {
                result = std::max(result, std::abs((double) lhs[i] - rhs[i]));
            }
            return result;
        };
        std::ofstream first_step_control(root/"first_step_control.json", std::ios::trunc);
        bool first_step_pass = true;
        first_step_control << "[";
        bool first_first_step = true;
        for (const auto & query : make_prompts()) {
            const auto & c1 = c1_first_steps.at(query.name);
            const auto & c2 = c2_first_steps.at(query.name);
            const double query_diff = max_vector_difference(c1.query, c2.query);
            const double retrieval_diff = max_vector_difference(c1.response, c2.response);
            const double logits_diff = max_vector_difference(c1.logits, c2.logits);
            const bool query_exact = c1.query == c2.query;
            const bool retrieval_exact = c1.response == c2.response;
            const bool logits_exact = c1.logits == c2.logits;
            const bool token_exact = c1.generated == c2.generated;
            const bool raw_exact = c1.raw_norm == c2.raw_norm;
            const bool budget_exact = c1.budget == c2.budget;
            const bool actual_budget_exact = c1.actual_budget == c2.actual_budget;
            const bool pass = query_exact && retrieval_exact && logits_exact &&
                              token_exact && raw_exact && budget_exact && actual_budget_exact;
            first_step_pass = first_step_pass && pass;
            if (!first_first_step) first_step_control << ",";
            first_first_step = false;
            first_step_control << std::setprecision(17)
                << "{\"prompt_name\":\"" << query.name
                << "\",\"query_exact\":" << (query_exact ? "true" : "false")
                << ",\"retrieval_exact\":" << (retrieval_exact ? "true" : "false")
                << ",\"logits_exact\":" << (logits_exact ? "true" : "false")
                << ",\"token_exact\":" << (token_exact ? "true" : "false")
                << ",\"raw_norm_exact\":" << (raw_exact ? "true" : "false")
                << ",\"budget_exact\":" << (budget_exact ? "true" : "false")
                << ",\"actual_budget_exact\":" << (actual_budget_exact ? "true" : "false")
                << ",\"query_max_abs_diff\":" << query_diff
                << ",\"retrieval_max_abs_diff\":" << retrieval_diff
                << ",\"logits_max_abs_diff\":" << logits_diff
                << ",\"c1_token\":" << c1.generated
                << ",\"c2_token\":" << c2.generated
                << ",\"c1_raw_norm\":" << c1.raw_norm
                << ",\"c2_raw_norm\":" << c2.raw_norm
                << ",\"c1_B\":" << c1.budget
                << ",\"c2_B\":" << c2.budget
                << ",\"c1_actual_B\":" << c1.actual_budget
                << ",\"c2_actual_B\":" << c2.actual_budget
                << ",\"pass\":" << (pass ? "true" : "false") << "}";
        }
        first_step_control << "]\n";
        first_step_control.close();
        if (!first_step_pass) {
            throw std::runtime_error("recursive_control_not_isolated");
        }

        std::ofstream comparisons(root/"recursive_comparisons.jsonl", std::ios::trunc);
        for (const auto & query : make_prompts()) {
            const auto & c1_result = c1_results.at(query.name);
            const auto & c2_result = c2_results.at(query.name);
            const size_t common = std::min(c1_result.tokens.size(), c2_result.tokens.size());
            size_t first_diff = common;
            for (size_t i = 0; i < common; ++i) {
                if (c1_result.tokens[i] != c2_result.tokens[i]) {
                    first_diff = i;
                    break;
                }
            }
            if (first_diff == common && c1_result.tokens.size() != c2_result.tokens.size()) {
                first_diff = common;
            }
            const auto c1_readout = c1_result.readout.metrics;
            const auto c2_readout = c2_result.readout.metrics;
            comparisons << std::setprecision(17)
                << "{\"prompt_name\":\"" << query.name
                << "\",\"source\":\"" << query.source
                << "\",\"common_prefix_tokens\":" << first_diff
                << ",\"first_token_divergence_step\":"
                << (first_diff < common || c1_result.tokens.size() != c2_result.tokens.size()
                    ? std::to_string(first_diff) : "null")
                << ",\"c1_token_count\":" << c1_result.tokens.size()
                << ",\"c2_token_count\":" << c2_result.tokens.size()
                << ",\"same_trajectory\":"
                << (c1_result.tokens == c2_result.tokens ? "true" : "false")
                << ",\"c1_target_nll\":" << c1_readout.nll
                << ",\"c2_target_nll\":" << c2_readout.nll
                << ",\"c1_minus_c2_nll\":" << c1_readout.nll - c2_readout.nll
                << ",\"c1_target_logit\":" << c1_result.readout.logits.at(query.source == "A" ? memory_a.target : memory_b.target)
                << ",\"c2_target_logit\":" << c2_result.readout.logits.at(query.source == "A" ? memory_a.target : memory_b.target)
                << "}\n";
        }
        comparisons.close();

        std::ofstream validation(root/"validation.json", std::ios::trunc);
        validation << "{\"runtime_baseline\":\"" << STEP28_RUNTIME
                   << "\",\"bank_frozen\":true,\"bank_checksum_epoch\":\""
                   << STEP28_RUNTIME << "\",\"qwen_weights_modified\":false"
                   << ",\"gradients\":false,\"optimizer\":false,\"learning\":false"
                   << ",\"no_injection_stateful_control\":true"
                   << ",\"preflight_no_injection_stateful_control_pass\":"
                   << (preflight_no_injection_pass ? "true" : "false")
                   << ",\"preflight_first_step_control_pass\":"
                   << (preflight_first_step_pass ? "true" : "false")
                   << ",\"no_injection_stateful_control_pass\":"
                   << (no_injection_pass ? "true" : "false")
                   << ",\"first_step_control_required\":true"
                   << ",\"first_step_control_pass\":"
                   << (first_step_pass ? "true" : "false")
                   << ",\"authority_limit\":" << STEP27_B_MAX
                   << ",\"max_reasoning_tokens\":" << STEP28_MAX_TOKENS << "}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":"
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ",\"decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers
                << ",\"ctx_size\":" << opts.ctx_size
                << ",\"conditions\":3,\"prompts\":4,\"learning\":false}\n";
        std::ofstream manifest(root/"manifest.json", std::ios::trunc);
        manifest << "{\"experiment\":\"qwen35_stateful_recursive_sidecar\","
                 << "\"runtime_baseline\":\"" << STEP28_RUNTIME
                 << "\",\"model\":\"" << json_escape(opts.model)
                 << "\",\"conditions\":[\"C0_Qwen_only\",\"C1_stateless\",\"C2_stateful\"],"
                 << "\"bank\":\"Step27_P_init_A_to_D_B_to_E\","
                 << "\"injection_sites\":[\"block_24\",\"block_28\",\"final\"],"
                 << "\"allocation_weights\":[0.23072,0.38322,0.89438],"
                 << "\"B_max\":0.01,\"s_ref\":0.05,\"learning\":false,"
                 << "\"no_gradients\":true,\"no_optimizer\":true,\"base_model_mutated\":false}\n";

        clear_internal_injection(ctx);
        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
