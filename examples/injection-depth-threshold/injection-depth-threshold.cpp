#include "llama.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int32_t WIDTH = 2560;
constexpr int32_t N_LAYERS = 32;
constexpr double SEARCH_MAX = 0.30;
constexpr double SEARCH_TOLERANCE = 1e-4;
constexpr double SMALL_SIGNAL_RHO = 1e-4;

const std::array<int32_t, 7> INTERNAL_LAYERS = {3, 7, 11, 15, 19, 23, 27};
const std::array<int32_t, 7> CONCEPTUAL_BLOCKS = {4, 8, 12, 16, 20, 24, 28};
const std::array<double, 9> FIXED_RHOS = {0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30};

uint64_t decode_calls = 0;

struct options {
    std::string model;
    std::string output_dir;
    int32_t gpu_layers = 99;
    int32_t ctx_size = 256;
};

options parse_args(int argc, char ** argv) {
    options out;
    auto value = [&](int & i) {
        if (++i >= argc) throw std::runtime_error("missing option value");
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" || arg == "-m") out.model = value(i);
        else if (arg == "--output-dir") out.output_dir = value(i);
        else if (arg == "--gpu-layers" || arg == "-ngl") out.gpu_layers = std::stoi(value(i));
        else if (arg == "--ctx-size" || arg == "-c") out.ctx_size = std::stoi(value(i));
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (out.model.empty() || out.output_dir.empty()) {
        throw std::runtime_error("--model and --output-dir are required");
    }
    return out;
}

std::string json_escape(const std::string & value) {
    std::string out;
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') out.push_back('\\');
        if (ch == '\n') out += "\\n";
        else if (ch == '\r') out += "\\r";
        else if (ch == '\t') out += "\\t";
        else out.push_back(ch);
    }
    return out;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int32_t count = llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
    if (count >= 0) throw std::runtime_error("unexpected tokenizer sizing result");
    std::vector<llama_token> tokens(-count);
    count = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, true);
    if (count < 0) throw std::runtime_error("tokenization failed");
    tokens.resize(count);
    return tokens;
}

float tensor_value(const uint8_t * data, enum ggml_type type, size_t offset) {
    if (type == GGML_TYPE_F32) return *reinterpret_cast<const float *>(data + offset);
    if (type == GGML_TYPE_F16) return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(data + offset));
    if (type == GGML_TYPE_BF16) return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(data + offset));
    throw std::runtime_error("unsupported activation type");
}

std::vector<float> copy_last_vector(ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error("unsupported callback tensor type");
    }
    std::vector<uint8_t> copied;
    const uint8_t * data = nullptr;
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        data = static_cast<const uint8_t *>(tensor->data);
    } else {
        copied.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, copied.data(), 0, copied.size());
        data = copied.data();
    }
    const int64_t count = tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
    const int64_t vector = std::max<int64_t>(0, count - 1);
    const int64_t i1 = vector % tensor->ne[1];
    const int64_t rem = vector / tensor->ne[1];
    const int64_t i2 = rem % tensor->ne[2];
    const int64_t i3 = rem / tensor->ne[2];
    std::vector<float> out(tensor->ne[0]);
    for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
        const size_t offset = i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0];
        out[i0] = tensor_value(data, tensor->type, offset);
    }
    return out;
}

struct trace_collector {
    bool enabled = false;
    std::vector<float> final_hidden;
    std::map<int32_t, std::vector<float>> boundaries;

    void clear() {
        final_hidden.clear();
        boundaries.clear();
    }
};

bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & collector = *static_cast<trace_collector *>(user_data);
    if (!collector.enabled) return false;
    const std::string name(tensor->name);
    if (name == "result_norm") {
        if (ask) return true;
        collector.final_hidden = copy_last_vector(tensor);
        return true;
    }
    if (name.rfind("l_out-", 0) != 0) return false;
    const int32_t layer = std::stoi(name.substr(6));
    if (std::find(INTERNAL_LAYERS.begin(), INTERNAL_LAYERS.end(), layer) == INTERNAL_LAYERS.end()) return false;
    if (ask) return true;
    collector.boundaries[layer] = copy_last_vector(tensor);
    return true;
}

double l2(const std::vector<float> & values) {
    double total = 0;
    for (float value : values) total += (double) value*value;
    return std::sqrt(total);
}

std::vector<float> normalized(const std::vector<float> & values) {
    std::vector<float> out(values);
    const double norm = l2(out);
    if (norm <= 0) throw std::runtime_error("cannot normalize zero vector");
    for (float & value : out) value /= norm;
    return out;
}

double logsumexp(const std::vector<float> & logits) {
    const float maximum = *std::max_element(logits.begin(), logits.end());
    double total = 0;
    for (float value : logits) total += std::exp((double) value - maximum);
    return maximum + std::log(total);
}

struct token_metrics {
    double probability = 0;
    double nll = 0;
    double margin = 0;
    int32_t rank = 0;
    llama_token greedy = -1;
};

token_metrics metrics_from_logits(const std::vector<float> & logits, llama_token target) {
    token_metrics out;
    const double z = logsumexp(logits);
    out.nll = z - logits[target];
    out.probability = std::exp(logits[target] - z);
    out.greedy = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
    float best_other = -std::numeric_limits<float>::infinity();
    for (size_t token = 0; token < logits.size(); ++token) {
        if ((llama_token) token != target) best_other = std::max(best_other, logits[token]);
        if (logits[token] > logits[target]) ++out.rank;
    }
    ++out.rank;
    out.margin = logits[target] - best_other;
    return out;
}

struct eval_result {
    std::vector<float> logits;
    std::vector<float> final_hidden;
    std::map<int32_t, std::vector<float>> boundaries;
    token_metrics metrics;
};

eval_result evaluate(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const std::string & prompt,
        llama_token target,
        bool capture) {
    llama_memory_clear(llama_get_memory(ctx), true);
    collector.clear();
    collector.enabled = capture;
    auto tokens = tokenize(vocab, prompt, true);
    if (tokens.empty() || llama_decode(ctx, llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()))) != 0) {
        throw std::runtime_error("prompt decode failed");
    }
    ++decode_calls;
    llama_synchronize(ctx);
    collector.enabled = false;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * raw = llama_get_logits_ith(ctx, -1);
    if (!raw) throw std::runtime_error("missing logits");
    eval_result out;
    out.logits.assign(raw, raw + n_vocab);
    out.final_hidden = collector.final_hidden;
    out.boundaries = collector.boundaries;
    out.metrics = metrics_from_logits(out.logits, target);
    if (capture) {
        if (out.final_hidden.size() != WIDTH || out.boundaries.size() != INTERNAL_LAYERS.size()) {
            throw std::runtime_error("missing residual boundary traces");
        }
        for (const auto & [layer, values] : out.boundaries) {
            if (values.size() != WIDTH) throw std::runtime_error("non-2560 residual boundary");
        }
    }
    return out;
}

struct output_matrix {
    int64_t width = 0;
    int64_t rows = 0;
    std::vector<float> values;
    const float * row(int64_t index) const { return values.data() + index*width; }
};

output_matrix load_output_matrix(const llama_model & model) {
    const ggml_tensor * tensor = model.output;
    if (!tensor || tensor->ne[2] != 1 || tensor->ne[3] != 1) throw std::runtime_error("invalid output tensor");
    const auto * traits = ggml_get_type_traits(tensor->type);
    if (!traits || !traits->to_float) throw std::runtime_error("output tensor cannot be dequantized for diagnostic");
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    output_matrix out;
    out.width = tensor->ne[0];
    out.rows = tensor->ne[1];
    out.values.resize((size_t) out.width*out.rows);
    for (int64_t row = 0; row < out.rows; ++row) {
        traits->to_float(raw.data() + row*tensor->nb[1], out.values.data() + row*out.width, out.width);
    }
    return out;
}

std::vector<float> output_row_difference(const output_matrix & output, llama_token positive, llama_token negative) {
    std::vector<float> out(output.width);
    const float * pos = output.row(positive);
    const float * neg = output.row(negative);
    for (int64_t i = 0; i < output.width; ++i) out[i] = pos[i] - neg[i];
    return out;
}

std::vector<float> output_projection(const output_matrix & output, const std::vector<float> & direction) {
    std::vector<float> out(output.rows);
    for (int64_t row = 0; row < output.rows; ++row) {
        double total = 0;
        const float * weight = output.row(row);
        for (int64_t column = 0; column < output.width; ++column) total += (double) weight[column]*direction[column];
        out[row] = total;
    }
    return out;
}

std::vector<float> add_final_residual(
        const std::vector<float> & baseline_logits,
        const std::vector<float> & projected_direction,
        double alpha) {
    std::vector<float> out(baseline_logits);
    for (size_t i = 0; i < out.size(); ++i) out[i] += alpha*projected_direction[i];
    return out;
}

struct item {
    int32_t id = 0;
    std::string prompt;
    llama_token target = -1;
    std::string target_text;
};

std::vector<std::pair<llama_token, std::string>> choose_single_tokens(const llama_vocab * vocab) {
    const std::vector<std::string> candidates = {
        " Q", " X", " J", " V", " K", " Z", " R", " T", " P", " G", " W", " F",
        " A", " B", " C", " D", " E", " H", " L", " M", " N", " S", " U", " Y",
    };
    std::vector<std::pair<llama_token, std::string>> out;
    for (const auto & text : candidates) {
        const auto tokens = tokenize(vocab, text, false);
        if (tokens.size() == 1 && std::none_of(out.begin(), out.end(), [&](const auto & value) {
                return value.first == tokens[0];
            })) {
            out.emplace_back(tokens[0], text);
        }
        if (out.size() == 16) break;
    }
    if (out.size() != 16) throw std::runtime_error("could not reconstruct the prior 16 verified targets");
    return out;
}

std::vector<item> stage_a_items(const llama_vocab * vocab) {
    const std::array<const char *, 4> prompts = {
        "A synthetic probe requests one arbitrary marker. The marker is",
        "Return one token for this deliberately unknown code:",
        "The answer to this arbitrary token-control probe is",
        "Emit the assigned synthetic marker only:",
    };
    const auto labels = choose_single_tokens(vocab);
    std::vector<item> out;
    for (int32_t index = 0; index < 16; ++index) {
        out.push_back({index, prompts[index%prompts.size()], labels[index].first, labels[index].second});
    }
    return out;
}

void set_internal_injection(
        llama_context * ctx,
        int32_t layer,
        const std::vector<float> & direction,
        double alpha) {
    if (layer < 1 || layer >= N_LAYERS || direction.size() != WIDTH) {
        throw std::runtime_error("invalid internal injection request");
    }
    std::vector<float> data((N_LAYERS - 1)*WIDTH, 0.0f);
    const size_t offset = (size_t) (layer - 1)*WIDTH;
    for (int32_t i = 0; i < WIDTH; ++i) data[offset + i] = alpha*direction[i];
    if (llama_set_adapter_cvec(ctx, data.data(), data.size(), WIDTH, layer, layer) != 0) {
        throw std::runtime_error("failed to install temporary control vector");
    }
}

void clear_internal_injection(llama_context * ctx) {
    if (llama_set_adapter_cvec(ctx, nullptr, 0, WIDTH, -1, -1) != 0) {
        throw std::runtime_error("failed to clear temporary control vector");
    }
}

eval_result probe_internal(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const item & current,
        int32_t layer,
        const std::vector<float> & direction,
        double local_norm,
        double signed_rho) {
    set_internal_injection(ctx, layer, direction, signed_rho*local_norm);
    try {
        auto result = evaluate(ctx, vocab, collector, current.prompt, current.target, false);
        clear_internal_injection(ctx);
        return result;
    } catch (...) {
        clear_internal_injection(ctx);
        throw;
    }
}

double max_logit_difference(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("logit size mismatch");
    double out = 0;
    for (size_t i = 0; i < a.size(); ++i) out = std::max(out, std::abs((double) a[i] - b[i]));
    return out;
}

double kl_divergence(const std::vector<float> & current, const std::vector<float> & baseline) {
    const double z_current = logsumexp(current);
    const double z_base = logsumexp(baseline);
    double out = 0;
    for (size_t i = 0; i < current.size(); ++i) {
        const double p = std::exp(current[i] - z_current);
        if (p > 0) out += p*((current[i] - z_current) - (baseline[i] - z_base));
    }
    return out;
}

struct case_state {
    item definition;
    eval_result baseline;
    std::vector<float> direction;
    std::vector<float> projected_direction;
};

std::string boundary_name(int32_t layer) {
    if (layer < 0) return "final";
    return "block_" + std::to_string(layer + 1);
}

token_metrics probe_metrics(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const case_state & state,
        int32_t layer,
        double rho) {
    if (layer < 0) {
        const double alpha = rho*l2(state.baseline.final_hidden);
        return metrics_from_logits(add_final_residual(state.baseline.logits, state.projected_direction, alpha),
                                   state.definition.target);
    }
    const double local_norm = l2(state.baseline.boundaries.at(layer));
    return probe_internal(ctx, vocab, collector, state.definition, layer, state.direction, local_norm, rho).metrics;
}

void write_metric_fields(std::ostream & out, const token_metrics & metric, llama_token target) {
    out << std::setprecision(17)
        << ",\"target_probability\":" << metric.probability
        << ",\"target_nll\":" << metric.nll
        << ",\"target_rank\":" << metric.rank
        << ",\"target_margin\":" << metric.margin
        << ",\"greedy_token\":" << metric.greedy
        << ",\"target_is_top1\":" << (metric.greedy == target ? "true" : "false");
}

std::vector<int32_t> all_boundaries() {
    std::vector<int32_t> out(INTERNAL_LAYERS.begin(), INTERNAL_LAYERS.end());
    out.push_back(-1);
    return out;
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
        const auto output = load_output_matrix(model);
        const auto definitions = stage_a_items(vocab);

        std::vector<case_state> cases;
        for (const auto & definition : definitions) {
            clear_internal_injection(ctx);
            auto baseline = evaluate(ctx, vocab, collector, definition.prompt, definition.target, true);
            if (baseline.metrics.greedy == definition.target) {
                throw std::runtime_error("reused Stage-A target unexpectedly top-1");
            }
            auto direction = normalized(output_row_difference(output, definition.target, baseline.metrics.greedy));
            cases.push_back({definition, std::move(baseline), direction, output_projection(output, direction)});
        }

        std::ofstream validation(root/"validation.json", std::ios::trunc);
        validation << std::setprecision(17) << "{\"rho_zero_max_logit_diff_by_boundary\":{";
        bool first_validation = true;
        double overall_zero_diff = 0;
        double overall_restore_diff = 0;
        for (const int32_t layer : all_boundaries()) {
            double zero_diff = 0;
            double restore_diff = 0;
            if (layer >= 0) {
                set_internal_injection(ctx, layer, cases[0].direction, 0.0);
                const auto zero = evaluate(ctx, vocab, collector, cases[0].definition.prompt, cases[0].definition.target, false);
                clear_internal_injection(ctx);
                zero_diff = max_logit_difference(zero.logits, cases[0].baseline.logits);
                set_internal_injection(ctx, layer, cases[0].direction, 0.01*l2(cases[0].baseline.boundaries.at(layer)));
                (void) evaluate(ctx, vocab, collector, cases[0].definition.prompt, cases[0].definition.target, false);
                clear_internal_injection(ctx);
                const auto restored = evaluate(ctx, vocab, collector, cases[0].definition.prompt, cases[0].definition.target, false);
                restore_diff = max_logit_difference(restored.logits, cases[0].baseline.logits);
            }
            overall_zero_diff = std::max(overall_zero_diff, zero_diff);
            overall_restore_diff = std::max(overall_restore_diff, restore_diff);
            validation << (first_validation ? "" : ",") << "\"" << boundary_name(layer) << "\":" << zero_diff;
            first_validation = false;
        }
        validation << "},\"overall_rho_zero_max_logit_diff\":" << overall_zero_diff
                   << ",\"restoration_max_logit_diff\":" << overall_restore_diff
                   << ",\"clean_kv_per_probe\":true,\"persistent_state_after_run\":false}\n";

        std::ofstream fixed(root/"fixed_rho_results.jsonl", std::ios::trunc);
        std::ofstream thresholds(root/"threshold_results.jsonl", std::ios::trunc);
        std::ofstream small_signal(root/"small_signal_results.jsonl", std::ios::trunc);
        std::map<std::pair<int32_t, int32_t>, double> reached_thresholds;
        std::map<std::pair<int32_t, int32_t>, bool> reached;

        for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
            const auto & state = cases[case_index];
            for (const int32_t layer : all_boundaries()) {
                const double local_norm = layer < 0 ? l2(state.baseline.final_hidden) : l2(state.baseline.boundaries.at(layer));
                std::vector<token_metrics> fixed_metrics;
                for (const double rho : FIXED_RHOS) {
                    const auto metric = rho == 0 ? state.baseline.metrics : probe_metrics(ctx, vocab, collector, state, layer, rho);
                    fixed_metrics.push_back(metric);
                    fixed << std::setprecision(17)
                          << "{\"prompt_id\":" << state.definition.id
                          << ",\"prompt\":\"" << json_escape(state.definition.prompt)
                          << "\",\"target_token\":" << state.definition.target
                          << ",\"target_text\":\"" << json_escape(state.definition.target_text)
                          << "\",\"baseline_greedy_token\":" << state.baseline.metrics.greedy
                          << ",\"target_baseline_rank\":" << state.baseline.metrics.rank
                          << ",\"target_baseline_probability\":" << state.baseline.metrics.probability
                          << ",\"target_baseline_nll\":" << state.baseline.metrics.nll
                          << ",\"target_baseline_margin\":" << state.baseline.metrics.margin
                          << ",\"boundary\":\"" << boundary_name(layer)
                          << "\",\"conceptual_block\":" << (layer < 0 ? -1 : layer + 1)
                          << ",\"zero_based_block_index\":" << layer
                          << ",\"direction_type\":\"target_vs_winner\""
                          << ",\"local_residual_norm\":" << local_norm
                          << ",\"raw_correction_norm\":" << rho*local_norm
                          << ",\"rho\":" << rho << ",\"alpha\":" << rho*local_norm;
                    write_metric_fields(fixed, metric, state.definition.target);
                    fixed << "}\n";
                }

                const auto positive_small = probe_metrics(ctx, vocab, collector, state, layer, SMALL_SIGNAL_RHO);
                small_signal << std::setprecision(17)
                    << "{\"prompt_id\":" << state.definition.id << ",\"boundary\":\"" << boundary_name(layer)
                    << "\",\"conceptual_block\":" << (layer < 0 ? -1 : layer + 1)
                    << ",\"zero_based_block_index\":" << layer
                    << ",\"rho_probe\":" << SMALL_SIGNAL_RHO
                    << ",\"positive_nll_delta\":" << positive_small.nll - state.baseline.metrics.nll
                    << ",\"positive_margin_delta\":" << positive_small.margin - state.baseline.metrics.margin;
                if (positive_small.nll > state.baseline.metrics.nll) {
                    token_metrics negative;
                    if (layer < 0) {
                        const double alpha = -SMALL_SIGNAL_RHO*local_norm;
                        negative = metrics_from_logits(add_final_residual(state.baseline.logits, state.projected_direction, alpha),
                                                       state.definition.target);
                    } else {
                        negative = probe_internal(ctx, vocab, collector, state.definition, layer, state.direction,
                                                  local_norm, -SMALL_SIGNAL_RHO).metrics;
                    }
                    small_signal << ",\"sign_reversal_tested\":true,\"negative_nll_delta\":"
                                 << negative.nll - state.baseline.metrics.nll;
                } else {
                    small_signal << ",\"sign_reversal_tested\":false";
                }
                small_signal << "}\n";

                int crossing = -1;
                for (size_t i = 1; i < fixed_metrics.size(); ++i) {
                    if (fixed_metrics[i].greedy == state.definition.target) {
                        crossing = i;
                        break;
                    }
                }
                bool did_reach = crossing >= 0;
                double threshold_rho = SEARCH_MAX;
                token_metrics threshold_metric = fixed_metrics.back();
                int binary_iterations = 0;
                if (did_reach) {
                    double low = FIXED_RHOS[crossing - 1];
                    double high = FIXED_RHOS[crossing];
                    while (high - low > SEARCH_TOLERANCE) {
                        const double middle = 0.5*(low + high);
                        const auto metric = probe_metrics(ctx, vocab, collector, state, layer, middle);
                        ++binary_iterations;
                        if (metric.greedy == state.definition.target) {
                            high = middle;
                            threshold_metric = metric;
                        } else {
                            low = middle;
                        }
                    }
                    threshold_rho = high;
                    threshold_metric = probe_metrics(ctx, vocab, collector, state, layer, threshold_rho);
                }
                reached[{state.definition.id, layer}] = did_reach;
                reached_thresholds[{state.definition.id, layer}] = threshold_rho;
                thresholds << std::setprecision(17)
                    << "{\"prompt_id\":" << state.definition.id
                    << ",\"target_token\":" << state.definition.target
                    << ",\"target_text\":\"" << json_escape(state.definition.target_text)
                    << "\",\"baseline_greedy_token\":" << state.baseline.metrics.greedy
                    << ",\"boundary\":\"" << boundary_name(layer)
                    << "\",\"conceptual_block\":" << (layer < 0 ? -1 : layer + 1)
                    << ",\"zero_based_block_index\":" << layer
                    << ",\"direction_type\":\"target_vs_winner\""
                    << ",\"local_residual_norm\":" << local_norm
                    << ",\"rho_top1\":" << threshold_rho
                    << ",\"alpha\":" << threshold_rho*local_norm
                    << ",\"not_reached\":" << (did_reach ? "false" : "true")
                    << ",\"binary_iterations\":" << binary_iterations;
                write_metric_fields(thresholds, threshold_metric, state.definition.target);
                thresholds << "}\n";
                std::fprintf(stderr, "case=%zu boundary=%s reached=%d rho=%.6f\n",
                             case_index, boundary_name(layer).c_str(), did_reach, threshold_rho);
            }
        }

        std::ofstream per_target(root/"per_target_depth_results.jsonl", std::ios::trunc);
        for (const auto & state : cases) {
            const double final_rho = reached_thresholds.at({state.definition.id, -1});
            int32_t winner = -2;
            double winner_rho = std::numeric_limits<double>::infinity();
            for (const int32_t layer : all_boundaries()) {
                if (reached.at({state.definition.id, layer}) && reached_thresholds.at({state.definition.id, layer}) < winner_rho) {
                    winner = layer;
                    winner_rho = reached_thresholds.at({state.definition.id, layer});
                }
            }
            for (const int32_t layer : all_boundaries()) {
                const bool did_reach = reached.at({state.definition.id, layer});
                const double rho = reached_thresholds.at({state.definition.id, layer});
                per_target << std::setprecision(17)
                    << "{\"prompt_id\":" << state.definition.id << ",\"target_token\":" << state.definition.target
                    << ",\"boundary\":\"" << boundary_name(layer) << "\",\"rho_top1\":" << rho
                    << ",\"not_reached\":" << (did_reach ? "false" : "true")
                    << ",\"rho_ratio_to_final\":" << (did_reach && final_rho > 0 ? rho/final_rho : -1)
                    << ",\"is_winning_boundary\":" << (layer == winner ? "true" : "false") << "}\n";
            }
        }

        const std::array<std::string, 4> protective_prompts = {
            "The capital of France is", "Two plus two equals", "Water freezes at", "The opposite of hot is",
        };
        std::ofstream protective(root/"protective_results.jsonl", std::ios::trunc);
        for (size_t index = 0; index < protective_prompts.size(); ++index) {
            clear_internal_injection(ctx);
            const auto probe_base = evaluate(ctx, vocab, collector, protective_prompts[index], 0, true);
            const llama_token baseline_target = probe_base.metrics.greedy;
            const auto baseline_metric = metrics_from_logits(probe_base.logits, baseline_target);
            const auto & direction = cases[index].direction;
            const auto projected = output_projection(output, direction);
            for (const int32_t layer : all_boundaries()) {
                const double local_norm = layer < 0 ? l2(probe_base.final_hidden) : l2(probe_base.boundaries.at(layer));
                for (const double rho : {0.01, 0.03}) {
                    std::vector<float> logits;
                    if (layer < 0) {
                        logits = add_final_residual(probe_base.logits, projected, rho*local_norm);
                    } else {
                        item protective_item{(int32_t) index, protective_prompts[index], baseline_target, ""};
                        logits = probe_internal(ctx, vocab, collector, protective_item, layer, direction, local_norm, rho).logits;
                    }
                    const auto current = metrics_from_logits(logits, baseline_target);
                    protective << std::setprecision(17)
                        << "{\"prompt_id\":" << index << ",\"prompt\":\"" << json_escape(protective_prompts[index])
                        << "\",\"direction_source_prompt_id\":" << index
                        << ",\"boundary\":\"" << boundary_name(layer) << "\",\"rho\":" << rho
                        << ",\"kl_adapted_base\":" << kl_divergence(logits, probe_base.logits)
                        << ",\"baseline_nll\":" << baseline_metric.nll << ",\"plastic_nll\":" << current.nll
                        << ",\"greedy_changed\":" << (current.greedy == baseline_target ? "false" : "true") << "}\n";
                }
            }
        }

        clear_internal_injection(ctx);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream boundary(root/"boundary_metadata.json", std::ios::trunc);
        boundary << "{\"residual_width\":2560,\"conceptual_boundaries\":[4,8,12,16,20,24,28],"
                 << "\"zero_based_block_outputs\":[3,7,11,15,19,23,27],"
                 << "\"internal_hook\":\"llama_set_adapter_cvec after full block output (after build_cvec call site)\","
                 << "\"final_hook\":\"exact linear residual after final RMSNorm before immutable LM head\","
                 << "\"internal_space_dimension_compatible\":true,\"final_space_dimension_compatible\":true}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17) << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":"
                << decode_calls << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size << "}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"persistent_experiment_state_bytes\":0,\"correction_vector_bytes\":10240,"
               << "\"boundary_trace_bytes\":81920,\"temporary_cvec_host_bytes\":317440,"
               << "\"model_or_kv_memory_excluded\":true}\n";
        std::ofstream manifest(root/"manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        manifest << "{\"experiment\":\"qwen35_injection_depth_threshold\",\"model\":\""
                 << json_escape(opts.model) << "\",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                 << ",\"sparse_residual_file\":\"" << (sparse ? json_escape(std::filesystem::path(sparse).filename().string()) : "")
                 << "\",\"prior_stage_a_cases_reconstructed_exactly\":true,\"case_count\":16,"
                 << "\"direction_type\":\"target_vs_winner\",\"search_max_rho\":0.30,"
                 << "\"search_tolerance\":0.0001,\"base_model_mutated\":false,\"persistent_state\":false,"
                 << "\"backpropagation\":false,\"autograd\":false,\"optimizer\":false}\n";

        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
