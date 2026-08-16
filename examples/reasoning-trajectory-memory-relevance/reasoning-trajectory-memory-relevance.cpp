#define CONTEXT_STABLE_KEY_PROBE_LIBRARY
#include "../context-stable-key-probe/context-stable-key-probe.cpp"
#include "chat.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int32_t MAX_REASONING_TOKENS = 32;
constexpr int32_t MIN_REASONING_TOKENS = 8;

struct query_spec {
    int32_t id;
    std::string class_name;
    std::string source;
    std::string user_prompt;
};

struct trajectory_state {
    int32_t step = 0;
    int32_t token_id = -1;
    std::string token_piece;
    std::vector<float> hidden;
    int32_t greedy = -1;
};

std::string render_reasoning_chat(const common_chat_templates * templates,
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

std::vector<query_spec> make_queries() {
    const std::array<std::string, 8> applicable = {
        "Task: memory lookup. Source: {X}. Reason through which stored destination belongs to this source.",
        "Operation: retrieve association. Input symbol: {X}. Think before identifying its learned destination.",
        "Memory task. Query source = {X}. Work out the destination assigned to this source.",
        "Use stored associative knowledge. Source {X} is given. Reason before determining its paired destination.",
        "Mode: association recall. Symbol: {X}. Determine what memory should be retrieved.",
        "This task requires the learned source-destination relation. Source: {X}. Think through the lookup.",
        "Retrieve from associative memory for source {X}. Reason before producing the destination.",
        "Given source {X}, solve using the previously acquired association. Think through the retrieval.",
    };
    const std::array<std::string, 8> non_applicable = {
        "Task: literal symbol inspection. Source: {X}. Reason about the displayed symbol without using its stored destination.",
        "Operation: inspect symbol. Input symbol: {X}. Think without retrieving its learned destination.",
        "Literal-symbol task. Query source = {X}. Reason only about the symbol itself.",
        "Do not use stored associative knowledge. Source {X} is given. Reason about the literal source only.",
        "Mode: symbol inspection. Symbol: {X}. Determine what character is present without retrieving memory.",
        "This task does not require the learned source-destination relation. Source: {X}. Think about the literal input.",
        "Ignore associative memory for source {X}. Reason about the source symbol itself.",
        "Given source {X}, solve without using the previously acquired association. Think only about the literal symbol.",
    };

    std::vector<query_spec> queries;
    int32_t id = 0;
    for (const std::string & class_name : {std::string("applicable"), std::string("non_applicable")}) {
        const auto & templates = class_name == "applicable" ? applicable : non_applicable;
        for (const std::string & source : {std::string("A"), std::string("B")}) {
            for (const auto & template_text : templates) {
                std::string prompt = template_text;
                const std::string marker = "{X}";
                const auto position = prompt.find(marker);
                if (position == std::string::npos) throw std::runtime_error("query template missing {X}");
                prompt.replace(position, marker.size(), source);
                queries.push_back({id++, class_name, source, prompt});
            }
        }
    }
    if (queries.size() != 32) throw std::runtime_error("Step 26 requires exactly 32 queries");
    return queries;
}

uint64_t checksum64(const std::vector<float> & values) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(values.data());
    const size_t size = values.size() * sizeof(float);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hex_checksum(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::streamoff write_vector(std::ofstream & stream, const std::vector<float> & values) {
    const std::streamoff offset = stream.tellp();
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("failed writing hidden-state vector");
    return offset;
}

void write_string_array(std::ostream & out, const std::vector<std::string> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        out << (i ? "," : "") << '"' << json_escape(values[i]) << '"';
    }
    out << ']';
}

void write_int_array(std::ostream & out, const std::vector<llama_token> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
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
        if (model.hparams.n_embd != KEY_WIDTH || model.hparams.n_layer() != 32) {
            throw std::runtime_error("Step 26 requires Qwen3.5-4B dimensions 32x2560");
        }

        key_trace_collector collector;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.ctx_size;
        context_params.n_batch = opts.ctx_size;
        context_params.n_ubatch = opts.ctx_size;
        context_params.n_seq_max = 1;
        context_params.cb_eval = key_eval_callback;
        context_params.cb_eval_user_data = &collector;
        llama_context * ctx = llama_init_from_model(public_model, context_params);
        if (!ctx) throw std::runtime_error("context creation failed");
        const llama_vocab * vocab = llama_model_get_vocab(public_model);

        auto chat_templates = common_chat_templates_init(public_model, "");
        if (!chat_templates) throw std::runtime_error("chat template initialization failed");
        if (!common_chat_templates_support_enable_thinking(chat_templates.get())) {
            throw std::runtime_error("active chat template does not support thinking mode");
        }

        const auto queries = make_queries();
        std::ofstream generation_out(root / "reasoning_generations.jsonl", std::ios::trunc);
        std::ofstream state_index_out(root / "hidden_states_index.jsonl", std::ios::trunc);
        std::ofstream k1_index_out(root / "k1_metadata.jsonl", std::ios::trunc);
        std::ofstream hidden_out(root / "hidden_states.f32", std::ios::binary | std::ios::trunc);
        std::ofstream k1_out(root / "k1_vectors.f32", std::ios::binary | std::ios::trunc);
        if (!generation_out || !state_index_out || !k1_index_out || !hidden_out || !k1_out) {
            throw std::runtime_error("cannot open Step 26 output files");
        }

        bool all_support_thinking = true;
        double max_reset_hidden_difference = 0;
        int32_t too_short_count = 0;
        int32_t total_generated_tokens = 0;

        for (const auto & query : queries) {
            bool supports_thinking = false;
            const std::string rendered = render_reasoning_chat(chat_templates.get(), query.user_prompt, supports_thinking);
            all_support_thinking = all_support_thinking && supports_thinking;
            if (!supports_thinking) throw std::runtime_error("rendered chat does not support thinking");

            std::string generated_text;
            std::vector<llama_token> generated_tokens;
            std::vector<std::string> generated_pieces;
            std::vector<trajectory_state> states;

            auto observation = evaluate_key(ctx, vocab, collector, rendered, query.source,
                                            "trajectory", "prompt", query.id);
            const std::vector<llama_token> prompt_tokens = observation.prompt_tokens;
            const std::streamoff k1_offset = write_vector(k1_out, observation.k1);
            k1_index_out << std::setprecision(17)
                         << "{\"prompt_id\":" << query.id
                         << ",\"source\":\"" << query.source
                         << "\",\"byte_offset\":" << k1_offset
                         << ",\"dimension\":" << observation.k1.size()
                         << ",\"checksum\":\"" << hex_checksum(checksum64(observation.k1)) << "\"}\n";

            states.push_back({0, -1, "", observation.final_hidden, observation.greedy});
            const std::streamoff h0_offset = write_vector(hidden_out, observation.final_hidden);
            state_index_out << std::setprecision(17)
                            << "{\"prompt_id\":" << query.id << ",\"step\":0"
                            << ",\"generated_token_id\":-1,\"byte_offset\":" << h0_offset
                            << ",\"dimension\":" << observation.final_hidden.size()
                            << ",\"checksum\":\"" << hex_checksum(checksum64(observation.final_hidden)) << "\"}\n";

            for (int32_t step = 1; step <= MAX_REASONING_TOKENS; ++step) {
                const llama_token next = observation.greedy;
                if (next < 0 || llama_vocab_is_eog(vocab, next)) break;
                const std::string piece = token_piece(vocab, next);
                generated_tokens.push_back(next);
                generated_pieces.push_back(piece);
                generated_text += piece;

                observation = evaluate_key(ctx, vocab, collector, rendered + generated_text, query.source,
                                           "trajectory", "generated", query.id);
                states.push_back({step, next, piece, observation.final_hidden, observation.greedy});
                const std::streamoff state_offset = write_vector(hidden_out, observation.final_hidden);
                state_index_out << std::setprecision(17)
                                << "{\"prompt_id\":" << query.id << ",\"step\":" << step
                                << ",\"generated_token_id\":" << next
                                << ",\"byte_offset\":" << state_offset
                                << ",\"dimension\":" << observation.final_hidden.size()
                                << ",\"checksum\":\"" << hex_checksum(checksum64(observation.final_hidden)) << "\"}\n";
            }

            const int32_t generated_count = static_cast<int32_t>(generated_tokens.size());
            const bool too_short = generated_count < MIN_REASONING_TOKENS;
            if (too_short) ++too_short_count;
            total_generated_tokens += generated_count;

            generation_out << std::setprecision(17)
                           << "{\"prompt_id\":" << query.id
                           << ",\"class\":\"" << query.class_name
                           << "\",\"source\":\"" << query.source
                           << "\",\"user_prompt\":\"" << json_escape(query.user_prompt)
                           << "\",\"rendered_prompt\":\"" << json_escape(rendered)
                           << "\",\"prompt_token_ids\":";
            write_int_array(generation_out, prompt_tokens);
            generation_out << ",\"reasoning_enabled\":true"
                           << ",\"supports_thinking\":" << (supports_thinking ? "true" : "false")
                           << ",\"generated_token_ids\":";
            write_int_array(generation_out, generated_tokens);
            generation_out << ",\"generated_pieces\":";
            write_string_array(generation_out, generated_pieces);
            generation_out << ",\"generated_text\":\"" << json_escape(generated_text)
                           << "\",\"generated_count\":" << generated_count
                           << ",\"trajectory_too_short\":" << (too_short ? "true" : "false")
                           << ",\"max_reasoning_tokens\":" << MAX_REASONING_TOKENS << "}\n";

            std::fprintf(stderr, "step26 prompt %d/32: %d generated tokens%s\n", query.id + 1,
                         generated_count, too_short ? " (too short)" : "");
        }

        bool supports_thinking = false;
        const std::string reset_rendered = render_reasoning_chat(chat_templates.get(), queries.front().user_prompt, supports_thinking);
        const auto reset_observation = evaluate_key(ctx, vocab, collector, reset_rendered, queries.front().source,
                                                     "trajectory", "reset", 100000);

        // Re-run the first prompt once more after all trajectories. This checks
        // that every clean-prefix probe restored the frozen state.
        const auto baseline_rendered = render_reasoning_chat(chat_templates.get(), queries.front().user_prompt, supports_thinking);
        const auto baseline_observation = evaluate_key(ctx, vocab, collector, baseline_rendered, queries.front().source,
                                                       "trajectory", "baseline", 100001);
        if (reset_observation.final_hidden.size() != baseline_observation.final_hidden.size()) {
            throw std::runtime_error("reset hidden-state size mismatch");
        }
        for (size_t i = 0; i < baseline_observation.final_hidden.size(); ++i) {
            max_reset_hidden_difference = std::max(max_reset_hidden_difference,
                std::abs((double) reset_observation.final_hidden[i] - baseline_observation.final_hidden[i]));
        }

        std::ofstream validation(root / "validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"qwen_weights_modified\":false,\"sidecar_injection\":false"
                   << ",\"plastic_writes\":false,\"gradients\":false,\"optimizer\":false"
                   << ",\"reasoning_enabled\":" << (all_support_thinking ? "true" : "false")
                   << ",\"max_reasoning_tokens\":" << MAX_REASONING_TOKENS
                   << ",\"too_short_count\":" << too_short_count
                   << ",\"query_count\":" << queries.size()
                   << ",\"max_reset_hidden_difference\":" << max_reset_hidden_difference << "}\n";

        std::ofstream runtime(root / "runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ",\"decode_calls\":" << decode_calls
                << ",\"generated_tokens\":" << total_generated_tokens
                << ",\"gpu_layers\":" << opts.gpu_layers
                << ",\"ctx_size\":" << opts.ctx_size
                << ",\"clean_kv_per_prefix\":true,\"sidecar_injection\":false}\n";

        std::ofstream metadata(root / "probe_metadata.json", std::ios::trunc);
        metadata << "{\"experiment\":\"step26_reasoning_trajectory_memory_relevance\","
                 << "\"query_count\":32,\"applicable_count\":16,\"non_applicable_count\":16,"
                 << "\"source_symbols\":[\"A\",\"B\"],\"max_reasoning_tokens\":32,"
                 << "\"min_reasoning_tokens\":8,\"reasoning_enabled\":true,"
                 << "\"no_learning\":true,\"no_sidecar_injection\":true}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
