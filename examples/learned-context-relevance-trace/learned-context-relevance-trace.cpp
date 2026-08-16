#define main step20_included_main
#include "../injection-depth-threshold/injection-depth-threshold.cpp"
#undef main

#include <sstream>

namespace {

struct step20_probe {
    int32_t id = -1;
    std::string role;
    std::string variant;
    std::string source;
    std::string prompt;
};

void write_tokens_json(std::ostream & out, const std::vector<llama_token> & tokens) {
    out << '[';
    for (size_t i = 0; i < tokens.size(); ++i) {
        out << (i ? "," : "") << tokens[i];
    }
    out << ']';
}

void write_float_array_json(std::ostream & out, const std::vector<float> & values) {
    out << '[' << std::setprecision(9);
    for (size_t i = 0; i < values.size(); ++i) {
        out << (i ? "," : "") << values[i];
    }
    out << ']';
}

std::string prompt_bytes_hex(const std::string & prompt) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : prompt) out << std::setw(2) << static_cast<unsigned int>(byte);
    return out.str();
}

std::vector<step20_probe> make_probes() {
    const std::array<std::string, 8> symbols = {"A", "B", "C", "D", "E", "F", "G", "H"};
    const std::array<std::string, 3> training_templates = {
        "Input: {symbol}",
        "Symbol: {symbol}",
        "Current symbol: {symbol}",
    };
    const std::array<std::string, 8> irrelevant = {
        "The letter A comes before B.",
        "Grade A is considered excellent.",
        "The word A appears in this sentence.",
        "A is the first letter of the alphabet.",
        "Vitamin B is found in many foods.",
        "The letter B comes after A.",
        "Plan B is the backup option.",
        "The word B appears in this sentence.",
    };
    const std::array<std::string, 8> protective = {
        "The capital of France is",
        "Two plus two equals",
        "A common week has this many days:",
        "Water freezes at zero degrees",
        "The season after spring is",
        "A triangle has this many sides:",
        "The opposite of narrow is",
        "Monday, Tuesday,",
    };

    std::vector<step20_probe> probes;
    int32_t id = 0;
    for (const auto & symbol : symbols) {
        for (size_t variant_index = 0; variant_index < training_templates.size(); ++variant_index) {
            std::string prompt = training_templates[variant_index];
            const std::string needle = "{symbol}";
            const size_t position = prompt.find(needle);
            prompt.replace(position, needle.size(), symbol);
            probes.push_back({id++, "training", "training_" + std::to_string(variant_index + 1), symbol, prompt});
        }
    }
    for (const auto & symbol : symbols) {
        probes.push_back({id++, "heldout_relevant", "arrow", symbol, symbol + " ->"});
    }
    probes.push_back({id++, "irrelevant", "irrelevant_01", "A", irrelevant[0]});
    probes.push_back({id++, "irrelevant", "irrelevant_02", "A", irrelevant[1]});
    probes.push_back({id++, "irrelevant", "irrelevant_03", "A", irrelevant[2]});
    probes.push_back({id++, "irrelevant", "irrelevant_04", "A", irrelevant[3]});
    probes.push_back({id++, "irrelevant", "irrelevant_05", "B", irrelevant[4]});
    probes.push_back({id++, "irrelevant", "irrelevant_06", "B", irrelevant[5]});
    probes.push_back({id++, "irrelevant", "irrelevant_07", "B", irrelevant[6]});
    probes.push_back({id++, "irrelevant", "irrelevant_08", "B", irrelevant[7]});
    for (size_t index = 0; index < protective.size(); ++index) {
        probes.push_back({id++, "protective", "protective_" + std::to_string(index + 1), "", protective[index]});
    }
    return probes;
}

double max_vector_difference(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("vector size mismatch");
    double maximum = 0;
    for (size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs((double) left[index] - right[index]));
    }
    return maximum;
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
            throw std::runtime_error("Step 20 requires Qwen3.5-4B dimensions 32x2560");
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

        const auto probes = make_probes();
        if (probes.size() != 48) throw std::runtime_error("unexpected Step 20 probe count");

        std::ofstream vectors(root / "context_vectors.f32", std::ios::binary | std::ios::trunc);
        std::ofstream metadata(root / "context_metadata.jsonl", std::ios::trunc);
        if (!vectors || !metadata) throw std::runtime_error("cannot open Step 20 raw artifacts");

        std::vector<float> first_hidden;
        std::vector<float> first_logits;
        double maximum_prompt_reset_hidden_difference = 0;
        double maximum_prompt_reset_logit_difference = 0;

        for (const auto & probe : probes) {
            clear_internal_injection(ctx);
            const auto result = evaluate(ctx, vocab, collector, probe.prompt, 0, true);
            const auto tokens = tokenize(vocab, probe.prompt, true);
            if (result.final_hidden.size() != WIDTH) throw std::runtime_error("missing final hidden state");
            const auto normalized_hidden = normalized(result.final_hidden);
            const std::streamoff raw_offset = vectors.tellp();
            vectors.write(reinterpret_cast<const char *>(result.final_hidden.data()),
                          static_cast<std::streamsize>(result.final_hidden.size() * sizeof(float)));
            const std::streamoff normalized_offset = vectors.tellp();
            vectors.write(reinterpret_cast<const char *>(normalized_hidden.data()),
                          static_cast<std::streamsize>(normalized_hidden.size() * sizeof(float)));
            if (!vectors) throw std::runtime_error("failed writing context vectors");

            metadata << std::setprecision(17)
                     << "{\"probe_id\":" << probe.id
                     << ",\"role\":\"" << json_escape(probe.role)
                     << "\",\"variant\":\"" << json_escape(probe.variant)
                     << "\",\"source\":\"" << json_escape(probe.source)
                     << "\",\"prompt\":\"" << json_escape(probe.prompt)
                     << "\",\"prompt_bytes_hex\":\"" << prompt_bytes_hex(probe.prompt)
                     << "\",\"prompt_token_ids\":";
            write_tokens_json(metadata, tokens);
            metadata << ",\"greedy_token\":" << result.metrics.greedy
                     << ",\"raw_byte_offset\":" << raw_offset
                     << ",\"normalized_byte_offset\":" << normalized_offset
                     << ",\"dimension\":" << WIDTH
                     << ",\"add_special\":true,\"parse_special\":true}\n";

            if (probe.id == 0) {
                first_hidden = result.final_hidden;
                first_logits = result.logits;
            } else if (probe.id == 1) {
                const auto repeat = evaluate(ctx, vocab, collector, probes[0].prompt, 0, true);
                maximum_prompt_reset_hidden_difference = max_vector_difference(first_hidden, repeat.final_hidden);
                maximum_prompt_reset_logit_difference = max_vector_difference(first_logits, repeat.logits);
            }
        }

        std::ofstream validation(root / "validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"zero_state_max_hidden_difference\":" << maximum_prompt_reset_hidden_difference
                   << ",\"reset_max_hidden_difference\":" << maximum_prompt_reset_hidden_difference
                   << ",\"zero_state_max_logit_difference\":" << maximum_prompt_reset_logit_difference
                   << ",\"reset_max_logit_difference\":" << maximum_prompt_reset_logit_difference
                   << ",\"qwen_weights_modified\":false,\"sidecar_injection\":false"
                   << ",\"learning_replayed\":false,\"gradients\":false,\"optimizer\":false}\n";

        std::ofstream runtime(root / "runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":"
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ",\"model_decode_calls\":" << decode_calls
                << ",\"probe_count\":" << probes.size()
                << ",\"gpu_layers\":" << opts.gpu_layers
                << ",\"ctx_size\":" << opts.ctx_size
                << ",\"add_special\":true,\"parse_special\":true,\"clean_kv_per_forward\":true}\n";

        std::ofstream config(root / "probe_metadata.json", std::ios::trunc);
        config << "{\"experiment\":\"qwen35_learned_context_relevance_trace\","
               << "\"runtime_baseline_id\":\"rtx3090_20260814\","
               << "\"representation\":\"normalize(final result_norm hidden state)\","
               << "\"training_contexts\":24,\"heldout_relevant\":8,\"irrelevant\":8,\"protective\":8,"
               << "\"merge_threshold\":0.80,\"new_strength\":0.25,\"recurrence_strength\":\"min(1,count/4)\","
               << "\"heldout_arrows_in_training\":false,\"sidecar_injection\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
