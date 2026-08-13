#include "../injection-sensitivity-allocation/multi-injection-support.inc"

#include <array>

namespace {

constexpr double FP_TOLERANCE = 1e-6;

struct stream_event {
    char x;
    char y;
    const char * offline_class;
};

// This is exactly the fixed event stream from the Step-3 controller dry run.
const std::array<stream_event, 20> STREAM = {{
    {'A', 'D', "recurring_rule"}, {'A', 'F', "random"},
    {'B', 'E', "recurring_rule"}, {'C', 'A', "random"},
    {'C', 'F', "recurring_rule"}, {'H', 'B', "random"},
    {'D', 'G', "recurring_rule"}, {'B', 'H', "random"},
    {'E', 'H', "recurring_rule"}, {'G', 'D', "random"},
    {'F', 'A', "recurring_rule"}, {'D', 'B', "random"},
    {'G', 'B', "recurring_rule"}, {'F', 'C', "random"},
    {'H', 'C', "recurring_rule"}, {'E', 'B', "random"},
    {'A', 'D', "recurring_rule"}, {'A', 'G', "random"},
    {'B', 'E', "recurring_rule"}, {'H', 'F', "random"},
}};

struct symbol_state {
    char symbol;
    llama_token token;
    std::vector<float> hidden;
};

std::vector<float> normalized_difference(
        const std::vector<float> & after,
        const std::vector<float> & before) {
    if (after.size() != before.size() || after.size() != WIDTH) {
        throw std::runtime_error("relation operands must both have dimension 2560");
    }
    std::vector<float> delta(WIDTH);
    for (int32_t i = 0; i < WIDTH; ++i) delta[i] = after[i] - before[i];
    return normalized(delta);
}

void write_vector(std::ostream & out, const std::vector<float> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        out << (i ? "," : "") << std::setprecision(9) << values[i];
    }
    out << ']';
}

std::string token_piece_probe(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize(-count);
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    return count > 0 ? std::string(buffer.data(), count) : std::string();
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

        std::map<char, symbol_state> symbols;
        std::ofstream token_log(root/"token_validation.json", std::ios::trunc);
        token_log << "{\"symbol_template\":\"Symbol:<space><symbol>\""
                  << ",\"transition_prefix_template\":\"Transition:<space><x><space>->\""
                  << ",\"full_transition_template\":\"Transition:<space><x><space>-><space><y>\""
                  << ",\"representation_location\":\"result_norm_final_prompt_token\""
                  << ",\"symbols\":[";
        bool first = true;
        for (char symbol = 'A'; symbol <= 'H'; ++symbol) {
            const std::string token_text = std::string(" ") + symbol;
            const auto tokenized = tokenize(vocab, token_text, false);
            if (tokenized.size() != 1) throw std::runtime_error("symbol is not exactly one token");
            const auto state = evaluate(ctx, vocab, collector, "Symbol:" + token_text, tokenized[0], true);
            if (state.final_hidden.size() != WIDTH) throw std::runtime_error("missing symbol representation");
            symbols.emplace(symbol, symbol_state{symbol, tokenized[0], state.final_hidden});
            token_log << (first ? "" : ",")
                      << "{\"symbol\":\"" << symbol << "\",\"token_text\":\""
                      << json_escape(token_text) << "\",\"token_id\":" << tokenized[0]
                      << ",\"token_piece\":\"" << json_escape(token_piece_probe(vocab, tokenized[0]))
                      << "\",\"single_token\":true}";
            first = false;
        }
        token_log << "]}\n";

        const std::string validation_prompt = "Transition: A -> D";
        const auto validation_a = evaluate(ctx, vocab, collector, validation_prompt, symbols.at('D').token, true);
        const auto validation_b = evaluate(ctx, vocab, collector, validation_prompt, symbols.at('D').token, true);
        const double zero_max_diff = max_logit_difference(validation_a.logits, validation_b.logits);
        if (zero_max_diff > FP_TOLERANCE) throw std::runtime_error("repeated frozen evaluation changed logits");

        std::ofstream vectors(root/"relation_vectors.jsonl", std::ios::trunc);
        for (size_t index = 0; index < STREAM.size(); ++index) {
            const auto & event = STREAM[index];
            const auto & x = symbols.at(event.x);
            const auto & y = symbols.at(event.y);
            const auto r0 = normalized_difference(y.hidden, x.hidden);

            const std::string prefix_prompt = std::string("Transition: ") + event.x + " ->";
            const std::string full_prompt = prefix_prompt + " " + event.y;
            const auto before = evaluate(ctx, vocab, collector, prefix_prompt, y.token, true);
            const auto after = evaluate(ctx, vocab, collector, full_prompt, y.token, true);
            if (before.final_hidden.size() != WIDTH || after.final_hidden.size() != WIDTH) {
                throw std::runtime_error("missing contextual transition representation");
            }
            const auto r1 = normalized(after.final_hidden);
            const auto r2 = normalized_difference(after.final_hidden, before.final_hidden);

            vectors << "{\"event_index\":" << index
                    << ",\"x\":\"" << event.x << "\",\"y\":\"" << event.y << "\""
                    << ",\"offline_class\":\"" << event.offline_class << "\""
                    << ",\"r0_symbol_difference_prompt_x\":\"Symbol: " << event.x << "\""
                    << ",\"r0_symbol_difference_prompt_y\":\"Symbol: " << event.y << "\""
                    << ",\"r1_full_transition_prompt\":\"" << json_escape(full_prompt) << "\""
                    << ",\"r2_prefix_prompt\":\"" << json_escape(prefix_prompt) << "\""
                    << ",\"r0\":";
            write_vector(vectors, r0);
            vectors << ",\"r1\":";
            write_vector(vectors, r1);
            vectors << ",\"r2\":";
            write_vector(vectors, r2);
            vectors << "}\n";
        }

        const auto restored = evaluate(ctx, vocab, collector, validation_prompt, symbols.at('D').token, true);
        const double reset_max_diff = max_logit_difference(validation_a.logits, restored.logits);
        if (reset_max_diff > FP_TOLERANCE) throw std::runtime_error("post-probe frozen logits changed");

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"raw_validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"zero_state_max_logit_difference\":" << zero_max_diff
                   << ",\"reset_max_logit_difference\":" << reset_max_diff
                   << ",\"base_model_mutated\":false,\"plasticity_gate_computed\":false"
                   << ",\"clustering_performed\":false,\"memory_writes_applied\":false"
                   << ",\"sidecar_injection_enabled\":false,\"teacher_feedback_used\":false"
                   << ",\"gradients\":false,\"backpropagation\":false,\"optimizer\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size
                << ",\"symbol_representation_calls\":8,\"transition_prefix_calls\":20"
                << ",\"full_transition_calls\":20,\"validation_calls\":3}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"relation_vector_dimension\":2560,\"events\":20,\"representations_per_event\":3"
               << ",\"raw_vector_bytes_fp32\":614400,\"plastic_state_bytes\":0"
               << ",\"model_and_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        manifest << "{\"experiment\":\"qwen35_relation_geometry_probe\""
                 << ",\"model\":\"" << json_escape(opts.model) << "\""
                 << ",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                 << ",\"stream_events\":20,\"representations\":[\"R0\",\"R1\",\"R2\"]"
                 << ",\"decision_auc_threshold\":0.80,\"base_model_mutated\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
