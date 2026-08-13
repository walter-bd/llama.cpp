#include "../injection-sensitivity-allocation/multi-injection-support.inc"

#include <array>

namespace {

constexpr double TAU = 0.80;
constexpr double ETA = 0.05;
constexpr double FP_TOLERANCE = 1e-6;

struct stream_event {
    char x;
    char y;
    const char * offline_class;
};

// Labels are consumed only by offline analysis, never by the controller.
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
    std::string token_text;
    llama_token token;
    std::string representation_prompt;
    std::vector<float> hidden;
};

struct relation_cluster {
    int32_t id = -1;
    int32_t count = 0;
    std::vector<float> sum;
    std::vector<float> prototype;
};

double unit_dot(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("relation-vector dimension mismatch");
    double total = 0;
    for (size_t i = 0; i < a.size(); ++i) total += (double) a[i]*b[i];
    return total;
}

double clipped_unit_cosine(const std::vector<float> & a, const std::vector<float> & b) {
    return std::clamp(unit_dot(a, b), 0.0, 1.0);
}

std::vector<float> relation_vector(const symbol_state & x, const symbol_state & y) {
    std::vector<float> difference(WIDTH);
    for (int32_t i = 0; i < WIDTH; ++i) difference[i] = y.hidden[i] - x.hidden[i];
    return normalized(difference);
}

std::string token_piece_controller(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize(-count);
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    return count > 0 ? std::string(buffer.data(), count) : std::string();
}

void write_vector(std::ostream & out, const std::vector<float> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        out << (i ? "," : "") << std::setprecision(9) << values[i];
    }
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

        const std::string representation_prefix = "Symbol:";
        const std::string prediction_prefix = "Observed transition:";
        std::map<char, symbol_state> symbols;
        std::ofstream token_log(root/"token_validation.json", std::ios::trunc);
        token_log << "{\"representation_template\":\"Symbol:<SYMBOL>\","
                  << "\"prediction_template\":\"Observed transition:<X> ->\","
                  << "\"representation_location\":\"result_norm_final_prompt_token\","
                  << "\"symbols\":[";
        bool first_symbol = true;
        for (char symbol = 'A'; symbol <= 'H'; ++symbol) {
            const std::string token_text = std::string(" ") + symbol;
            const auto tokenized = tokenize(vocab, token_text, false);
            if (tokenized.size() != 1) throw std::runtime_error("symbol is not exactly one token");
            const std::string prompt = representation_prefix + token_text;
            const auto state = evaluate(ctx, vocab, collector, prompt, tokenized[0], true);
            if (state.final_hidden.size() != WIDTH) throw std::runtime_error("missing final representation");
            symbols.emplace(symbol, symbol_state{symbol, token_text, tokenized[0], prompt, state.final_hidden});
            token_log << (first_symbol ? "" : ",")
                      << "{\"symbol\":\"" << symbol << "\",\"token_text\":\""
                      << json_escape(token_text) << "\",\"token_id\":" << tokenized[0]
                      << ",\"token_piece\":\"" << json_escape(token_piece_controller(vocab, tokenized[0]))
                      << "\",\"single_token\":true}";
            first_symbol = false;
        }
        token_log << "]}\n";

        const std::string zero_prompt = prediction_prefix + symbols.at('A').token_text + " ->";
        const auto zero_a = evaluate(ctx, vocab, collector, zero_prompt, symbols.at('D').token, false);
        const auto zero_b = evaluate(ctx, vocab, collector, zero_prompt, symbols.at('D').token, false);
        const double zero_max_diff = max_logit_difference(zero_a.logits, zero_b.logits);
        if (zero_max_diff > FP_TOLERANCE) throw std::runtime_error("frozen repeated evaluation changed logits");

        std::vector<std::vector<float>> previous_relations;
        std::vector<relation_cluster> clusters;
        std::ofstream events(root/"controller_events.jsonl", std::ios::trunc);

        for (size_t event_index = 0; event_index < STREAM.size(); ++event_index) {
            const auto & event = STREAM[event_index];
            const auto & x = symbols.at(event.x);
            const auto & y = symbols.at(event.y);
            const auto phi = relation_vector(x, y);

            const std::string prediction_prompt = prediction_prefix + x.token_text + " ->";
            const auto prediction = evaluate(ctx, vocab, collector, prediction_prompt, y.token, false);
            const double surprise = 1.0 - prediction.metrics.probability;

            double maximum_prior_cosine = 0;
            for (const auto & prior : previous_relations) {
                maximum_prior_cosine = std::max(maximum_prior_cosine, clipped_unit_cosine(phi, prior));
            }
            const double novelty = previous_relations.empty() ? 1.0 : 1.0 - maximum_prior_cosine;

            int32_t assigned = -1;
            double best_cluster_cosine = -1;
            for (size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index) {
                const double similarity = unit_dot(phi, clusters[cluster_index].prototype);
                if (similarity > best_cluster_cosine) {
                    best_cluster_cosine = similarity;
                    assigned = static_cast<int32_t>(cluster_index);
                }
            }

            bool created = assigned < 0 || best_cluster_cosine < TAU;
            double consistency = 1.0;
            if (created) {
                relation_cluster cluster;
                cluster.id = static_cast<int32_t>(clusters.size());
                cluster.count = 1;
                cluster.sum = phi;
                cluster.prototype = phi;
                clusters.push_back(std::move(cluster));
                assigned = static_cast<int32_t>(clusters.size() - 1);
            } else {
                auto & cluster = clusters[assigned];
                consistency = std::clamp(best_cluster_cosine, 0.0, 1.0);
                ++cluster.count;
                for (int32_t i = 0; i < WIDTH; ++i) cluster.sum[i] += phi[i];
                cluster.prototype = normalized(cluster.sum);
            }

            const int32_t cluster_count = clusters[assigned].count;
            const double recurrence = std::min(1.0, cluster_count/3.0);
            const double modulation = 0.0;
            const double gate = std::clamp(
                surprise*(0.25*novelty + 0.75*recurrence*consistency), 0.0, 1.0);
            const double candidate_write = ETA*gate;

            events << std::setprecision(17)
                   << "{\"event_index\":" << event_index
                   << ",\"x\":\"" << event.x << "\",\"y\":\"" << event.y << "\""
                   << ",\"offline_class\":\"" << event.offline_class << "\""
                   << ",\"prediction_prompt\":\"" << json_escape(prediction_prompt) << "\""
                   << ",\"x_token\":" << x.token << ",\"y_token\":" << y.token
                   << ",\"observed_probability\":" << prediction.metrics.probability
                   << ",\"S\":" << surprise << ",\"N\":" << novelty
                   << ",\"R\":" << recurrence << ",\"C\":" << consistency
                   << ",\"M\":" << modulation << ",\"G\":" << gate
                   << ",\"assigned_cluster\":" << assigned
                   << ",\"cluster_created\":" << (created ? "true" : "false")
                   << ",\"cluster_count\":" << cluster_count
                   << ",\"maximum_prior_relation_cosine\":" << maximum_prior_cosine
                   << ",\"best_cluster_cosine_before_assignment\":" << best_cluster_cosine
                   << ",\"candidate_write_magnitude\":" << candidate_write
                   << ",\"eta\":" << ETA << ",\"phi\":";
            write_vector(events, phi);
            events << "}\n";
            previous_relations.push_back(phi);
        }

        std::ofstream cluster_log(root/"clusters.json", std::ios::trunc);
        cluster_log << "{\"tau\":" << TAU << ",\"cluster_count\":" << clusters.size()
                    << ",\"clusters\":[";
        for (size_t index = 0; index < clusters.size(); ++index) {
            cluster_log << (index ? "," : "") << "{\"cluster_id\":" << clusters[index].id
                        << ",\"count\":" << clusters[index].count << "}";
        }
        cluster_log << "]}\n";

        const auto reset = evaluate(ctx, vocab, collector, zero_prompt, symbols.at('D').token, false);
        const double reset_max_diff = max_logit_difference(zero_a.logits, reset.logits);
        if (reset_max_diff > FP_TOLERANCE) throw std::runtime_error("post-controller frozen logits changed");

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"raw_validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"zero_state_max_logit_difference\":" << zero_max_diff
                   << ",\"reset_max_logit_difference\":" << reset_max_diff
                   << ",\"base_model_mutated\":false,\"candidate_writes_applied\":false"
                   << ",\"sidecar_injection_enabled\":false,\"gradients\":false"
                   << ",\"backpropagation\":false,\"optimizer\":false"
                   << ",\"right_wrong_labels_visible_to_controller\":false"
                   << ",\"external_modulation\":0}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size
                << ",\"symbol_representation_calls\":8,\"event_probability_calls\":20"
                << ",\"validation_calls\":3}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"relation_vector_dimension\":2560,\"stored_relation_vectors\":20,"
               << "\"candidate_plastic_bytes\":0,\"model_and_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        manifest << "{\"experiment\":\"qwen35_autonomous_plasticity_controller_sanity\""
                 << ",\"model\":\"" << json_escape(opts.model) << "\""
                 << ",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                 << ",\"stream_events\":20,\"eta\":0.05,\"tau\":0.80"
                 << ",\"gate_formula\":\"clip(S*(0.25*N+0.75*R*C),0,1)\""
                 << ",\"offline_clear_selectivity_ratio\":1.25"
                 << ",\"candidate_writes_applied\":false,\"base_model_mutated\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
