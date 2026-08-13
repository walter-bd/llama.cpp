#include "../injection-sensitivity-allocation/multi-injection-support.inc"

#include <array>
#include <random>

namespace {

constexpr int32_t PROJECTED_DIMENSION = 64;
constexpr double ETA = 0.05;
constexpr uint64_t PROJECTION_SEED = 20260812;
constexpr double FP_TOLERANCE = 1e-6;

struct stream_event {
    char x;
    char y;
    const char * offline_class;
};

const std::array<stream_event, 24> STREAM = {{
    {'A', 'D', "recurring"}, {'A', 'F', "random"},
    {'B', 'E', "recurring"}, {'C', 'A', "random"},
    {'C', 'F', "recurring"}, {'H', 'B', "random"},
    {'A', 'D', "recurring"}, {'B', 'H', "random"},
    {'B', 'E', "recurring"}, {'G', 'D', "random"},
    {'C', 'F', "recurring"}, {'D', 'B', "random"},
    {'A', 'D', "recurring"}, {'F', 'C', "random"},
    {'B', 'E', "recurring"}, {'E', 'B', "random"},
    {'C', 'F', "recurring"}, {'A', 'G', "random"},
    {'A', 'D', "recurring"}, {'H', 'F', "random"},
    {'B', 'E', "recurring"}, {'D', 'H', "random"},
    {'C', 'F', "recurring"}, {'G', 'A', "random"},
}};

using vector64 = std::array<double, PROJECTED_DIMENSION>;
using matrix64 = std::array<double, PROJECTED_DIMENSION*PROJECTED_DIMENSION>;

double vector_norm(const vector64 & values) {
    double total = 0;
    for (double value : values) total += value*value;
    return std::sqrt(total);
}

vector64 normalize64(vector64 values) {
    const double norm = vector_norm(values);
    if (norm <= 0) throw std::runtime_error("cannot normalize zero projected vector");
    for (double & value : values) value /= norm;
    return values;
}

vector64 project_hidden(
        const std::vector<float> & hidden,
        const std::vector<float> & projection) {
    if (hidden.size() != WIDTH || projection.size() != (size_t) PROJECTED_DIMENSION*WIDTH) {
        throw std::runtime_error("projection dimension mismatch");
    }
    vector64 projected{};
    for (int32_t row = 0; row < PROJECTED_DIMENSION; ++row) {
        double total = 0;
        const size_t offset = (size_t) row*WIDTH;
        for (int32_t column = 0; column < WIDTH; ++column) {
            total += (double) projection[offset + column]*hidden[column];
        }
        projected[row] = total;
    }
    return normalize64(projected);
}

vector64 predict(const matrix64 & predictor, const vector64 & x) {
    vector64 output{};
    for (int32_t row = 0; row < PROJECTED_DIMENSION; ++row) {
        double total = 0;
        const size_t offset = (size_t) row*PROJECTED_DIMENSION;
        for (int32_t column = 0; column < PROJECTED_DIMENSION; ++column) {
            total += predictor[offset + column]*x[column];
        }
        output[row] = total;
    }
    return output;
}

double frobenius_norm(const matrix64 & matrix) {
    double total = 0;
    for (double value : matrix) total += value*value;
    return std::sqrt(total);
}

void write_vector(std::ostream & out, const vector64 & values) {
    out << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        out << (index ? "," : "") << std::setprecision(17) << values[index];
    }
    out << ']';
}

std::string token_piece_predictor(const llama_vocab * vocab, llama_token token) {
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

        // The projection is fixed before observing any stream event and is never updated.
        std::mt19937_64 generator(PROJECTION_SEED);
        std::normal_distribution<float> gaussian(0.0f, 1.0f/std::sqrt((float) WIDTH));
        std::vector<float> projection((size_t) PROJECTED_DIMENSION*WIDTH);
        for (float & value : projection) value = gaussian(generator);

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

        std::map<char, vector64> projected_symbols;
        std::map<char, llama_token> symbol_tokens;
        std::ofstream token_log(root/"token_validation.json", std::ios::trunc);
        token_log << "{\"representation_template\":\"Symbol:<space><symbol>\""
                  << ",\"representation_location\":\"result_norm_final_prompt_token\""
                  << ",\"projection_dimension\":64,\"symbols\":[";
        bool first = true;
        for (char symbol = 'A'; symbol <= 'H'; ++symbol) {
            const std::string token_text = std::string(" ") + symbol;
            const auto tokenized = tokenize(vocab, token_text, false);
            if (tokenized.size() != 1) throw std::runtime_error("symbol is not exactly one token");
            const auto state = evaluate(ctx, vocab, collector, "Symbol:" + token_text, tokenized[0], true);
            if (state.final_hidden.size() != WIDTH) throw std::runtime_error("missing symbol representation");
            projected_symbols[symbol] = project_hidden(state.final_hidden, projection);
            symbol_tokens[symbol] = tokenized[0];
            token_log << (first ? "" : ",")
                      << "{\"symbol\":\"" << symbol << "\",\"token_text\":\""
                      << json_escape(token_text) << "\",\"token_id\":" << tokenized[0]
                      << ",\"token_piece\":\"" << json_escape(token_piece_predictor(vocab, tokenized[0]))
                      << "\",\"single_token\":true}";
            first = false;
        }
        token_log << "]}\n";

        const std::string validation_prompt = "Symbol: A";
        const auto validation_a = evaluate(ctx, vocab, collector, validation_prompt, symbol_tokens.at('A'), false);
        const auto validation_b = evaluate(ctx, vocab, collector, validation_prompt, symbol_tokens.at('A'), false);
        const double zero_max_diff = max_logit_difference(validation_a.logits, validation_b.logits);
        if (zero_max_diff > FP_TOLERANCE) throw std::runtime_error("repeated frozen evaluation changed logits");

        matrix64 predictor{};
        std::map<std::string, int32_t> occurrences;
        std::ofstream events(root/"predictive_events.jsonl", std::ios::trunc);
        std::ofstream trajectory(root/"a_norm_trajectory.jsonl", std::ios::trunc);

        for (size_t event_index = 0; event_index < STREAM.size(); ++event_index) {
            const auto & event = STREAM[event_index];
            const auto & x = projected_symbols.at(event.x);
            const auto & y = projected_symbols.at(event.y);
            const std::string transition = std::string(1, event.x) + "->" + event.y;
            const int32_t occurrence = ++occurrences[transition];
            const vector64 prediction = predict(predictor, x);
            const double prediction_norm = vector_norm(prediction);

            vector64 error{};
            double prediction_dot_y = 0;
            for (int32_t row = 0; row < PROJECTED_DIMENSION; ++row) {
                error[row] = y[row] - prediction[row];
                prediction_dot_y += prediction[row]*y[row];
            }
            const double error_norm = vector_norm(error);
            const double cosine = prediction_norm > 0 ? prediction_dot_y/prediction_norm : 0.0;
            const double predictor_norm_before = frobenius_norm(predictor);

            double update_norm_squared = 0;
            for (int32_t row = 0; row < PROJECTED_DIMENSION; ++row) {
                const size_t offset = (size_t) row*PROJECTED_DIMENSION;
                for (int32_t column = 0; column < PROJECTED_DIMENSION; ++column) {
                    const double update = ETA*error[row]*x[column];
                    predictor[offset + column] += update;
                    update_norm_squared += update*update;
                }
            }
            const double update_norm = std::sqrt(update_norm_squared);
            const double expected_update_norm = ETA*error_norm;
            if (std::abs(update_norm - expected_update_norm) > 1e-10) {
                throw std::runtime_error("rank-one update norm does not match eta*error for unit key");
            }
            const double predictor_norm_after = frobenius_norm(predictor);

            events << std::setprecision(17)
                   << "{\"event_index\":" << event_index
                   << ",\"x\":\"" << event.x << "\",\"y\":\"" << event.y << "\""
                   << ",\"transition\":\"" << transition << "\""
                   << ",\"offline_class\":\"" << event.offline_class << "\""
                   << ",\"exact_transition_occurrence\":" << occurrence
                   << ",\"prediction_error_before_update\":" << error_norm
                   << ",\"prediction_norm_before_update\":" << prediction_norm
                   << ",\"prediction_cosine_before_update\":";
            if (prediction_norm > 0) events << cosine; else events << "null";
            events << ",\"update_frobenius_norm\":" << update_norm
                   << ",\"predictor_frobenius_norm_before\":" << predictor_norm_before
                   << ",\"predictor_frobenius_norm_after\":" << predictor_norm_after
                   << ",\"eta\":" << ETA << ",\"x_projected\":";
            write_vector(events, x);
            events << ",\"y_projected\":";
            write_vector(events, y);
            events << "}\n";
            trajectory << std::setprecision(17)
                       << "{\"event_index\":" << event_index
                       << ",\"transition\":\"" << transition << "\""
                       << ",\"predictor_frobenius_norm\":" << predictor_norm_after << "}\n";
        }

        std::ofstream final_predictor(root/"final_predictor.json", std::ios::trunc);
        final_predictor << "{\"rows\":64,\"columns\":64,\"values\":[";
        for (size_t index = 0; index < predictor.size(); ++index) {
            final_predictor << (index ? "," : "") << std::setprecision(17) << predictor[index];
        }
        final_predictor << "]}\n";

        const auto restored = evaluate(ctx, vocab, collector, validation_prompt, symbol_tokens.at('A'), false);
        const double reset_max_diff = max_logit_difference(validation_a.logits, restored.logits);
        if (reset_max_diff > FP_TOLERANCE) throw std::runtime_error("post-predictor frozen logits changed");

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"raw_validation.json", std::ios::trunc);
        validation << std::setprecision(17)
                   << "{\"zero_state_max_logit_difference\":" << zero_max_diff
                   << ",\"reset_max_logit_difference\":" << reset_max_diff
                   << ",\"base_model_mutated\":false,\"qwen_behavior_modified\":false"
                   << ",\"sidecar_injection_enabled\":false,\"plasticity_gate_computed\":false"
                   << ",\"teacher_feedback_used\":false,\"reasoning_used\":false"
                   << ",\"gradients\":false,\"backpropagation\":false,\"optimizer\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":" << decode_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size
                << ",\"symbol_representation_calls\":8,\"validation_calls\":3"
                << ",\"predictor_events\":24}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"input_dimension\":2560,\"projected_dimension\":64"
               << ",\"projection_bytes_fp32\":655360,\"predictor_bytes_fp64\":32768"
               << ",\"qwen_plastic_bytes\":0,\"model_and_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        manifest << "{\"experiment\":\"qwen35_forward_predictive_trace_sanity\""
                 << ",\"model\":\"" << json_escape(opts.model) << "\""
                 << ",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                 << ",\"stream_events\":24,\"eta\":0.05,\"projection_dimension\":64"
                 << ",\"projection_seed\":" << PROJECTION_SEED
                 << ",\"projection_distribution\":\"normal(0,1/sqrt(2560)) via mt19937_64\""
                 << ",\"recurring_transitions\":[\"A->D\",\"B->E\",\"C->F\"]"
                 << ",\"base_model_mutated\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
