#include "llama.h"
#include "llama-adapter.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct options {
    std::string model;
    std::string adapter;
    std::string output = "forward-plasticity-smoke.json";
    std::string exposure = "Zorbin's preferred symbol is Q7.";
    std::string query = "Zorbin's preferred symbol is";
    std::string answer = " Q7";
    std::string control_query = "The capital of France is";
    std::string control_answer = " Paris";
    std::string rule = "competitive";
    std::string adapter_tensor = "blk.20.ffn_down.weight";
    std::string input_tensor = "ffn_swiglu-20";
    std::string output_tensor = "ffn_out-20";
    std::string correction_contrast_dir;
    std::string correction_task_file;
    std::string dynamic_selection_dir;
    std::string dynamic_selection_task_file;
    std::string dynamic_condition = "dynamic_causal";
    int32_t gpu_layers = 99;
    int32_t ctx_size = 1024;
    int32_t rank = 4;
    int32_t repetitions = 1;
    uint32_t seed = 123;
    float eta = 1e-3f;
    float decay = 0.05f;
    float gamma = 2.0f;
    float g_clip_l2 = 50.5964426f;
    float ratio_cap = 0.03f;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model MODEL --adapter ZERO_LORA [options]\n"
        "  --rule hebbian|competitive|random|shuffled --rank 1|4|8|16 --eta FLOAT\n"
        "  --exposure TEXT --repetitions INT --query TEXT --answer TEXT --output FILE\n"
        "  --correction-contrast-dir DIR --correction-task-file TSV (isolated rank-8 pilot)\n"
        "  --dynamic-selection-dir DIR --dynamic-selection-task-file TSV\n"
        "  --dynamic-condition dynamic_causal|eligibility_only|random_module|fixed_module|shuffled_correction|frozen\n",
        argv0);
}

options parse_args(int argc, char ** argv) {
    options result;
    auto value = [&](int & i) -> const char * {
        if (++i >= argc) {
            throw std::runtime_error("missing value for command-line option");
        }
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" || arg == "-m") result.model = value(i);
        else if (arg == "--adapter") result.adapter = value(i);
        else if (arg == "--output") result.output = value(i);
        else if (arg == "--exposure") result.exposure = value(i);
        else if (arg == "--query") result.query = value(i);
        else if (arg == "--answer") result.answer = value(i);
        else if (arg == "--control-query") result.control_query = value(i);
        else if (arg == "--control-answer") result.control_answer = value(i);
        else if (arg == "--rule") result.rule = value(i);
        else if (arg == "--adapter-tensor") result.adapter_tensor = value(i);
        else if (arg == "--input-tensor") result.input_tensor = value(i);
        else if (arg == "--output-tensor") result.output_tensor = value(i);
        else if (arg == "--correction-contrast-dir") result.correction_contrast_dir = value(i);
        else if (arg == "--correction-task-file") result.correction_task_file = value(i);
        else if (arg == "--dynamic-selection-dir") result.dynamic_selection_dir = value(i);
        else if (arg == "--dynamic-selection-task-file") result.dynamic_selection_task_file = value(i);
        else if (arg == "--dynamic-condition") result.dynamic_condition = value(i);
        else if (arg == "--gpu-layers" || arg == "-ngl") result.gpu_layers = std::stoi(value(i));
        else if (arg == "--ctx-size" || arg == "-c") result.ctx_size = std::stoi(value(i));
        else if (arg == "--rank") result.rank = std::stoi(value(i));
        else if (arg == "--repetitions") result.repetitions = std::stoi(value(i));
        else if (arg == "--seed") result.seed = std::stoul(value(i));
        else if (arg == "--eta") result.eta = std::stof(value(i));
        else if (arg == "--decay") result.decay = std::stof(value(i));
        else if (arg == "--gamma") result.gamma = std::stof(value(i));
        else if (arg == "--g-clip-l2") result.g_clip_l2 = std::stof(value(i));
        else if (arg == "--ratio-cap") result.ratio_cap = std::stof(value(i));
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (result.model.empty() || result.adapter.empty()) {
        throw std::runtime_error("--model and --adapter are required");
    }
    if (result.rule != "hebbian" && result.rule != "competitive" && result.rule != "random" && result.rule != "shuffled") {
        throw std::runtime_error("--rule must be hebbian, competitive, random, or shuffled");
    }
    if (result.rank <= 0 || result.repetitions <= 0 || result.eta < 0 || result.decay < 0 || result.decay >= 1) {
        throw std::runtime_error("invalid rank, repetitions, eta, or decay");
    }
    if (!result.correction_contrast_dir.empty() && result.correction_task_file.empty()) {
        throw std::runtime_error("--correction-task-file is required with --correction-contrast-dir");
    }
    if (!result.dynamic_selection_dir.empty() && result.dynamic_selection_task_file.empty()) {
        throw std::runtime_error("--dynamic-selection-task-file is required with --dynamic-selection-dir");
    }
    const std::vector<std::string> dynamic_conditions = {
        "dynamic_causal", "eligibility_only", "random_module", "fixed_module",
        "shuffled_correction", "frozen",
    };
    if (std::find(dynamic_conditions.begin(), dynamic_conditions.end(), result.dynamic_condition) == dynamic_conditions.end()) {
        throw std::runtime_error("invalid --dynamic-condition");
    }
    return result;
}

float tensor_value(const uint8_t * data, enum ggml_type type, size_t offset) {
    if (type == GGML_TYPE_F32) return *reinterpret_cast<const float *>(data + offset);
    if (type == GGML_TYPE_F16) return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(data + offset));
    if (type == GGML_TYPE_BF16) return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(data + offset));
    throw std::runtime_error(std::string("unsupported activation type: ") + ggml_type_name(type));
}

struct activation_collector {
    std::string input_name;
    std::string output_name;
    bool enabled = false;
    std::vector<double> sum_x;
    std::vector<double> sum_y;
    std::vector<double> sum_y2;
    int64_t n_x = 0;
    int64_t n_y = 0;
    bool dynamic_mode = false;
    std::unordered_set<std::string> dynamic_names;
    std::unordered_map<std::string, std::vector<double>> dynamic_sums;
    std::unordered_map<std::string, int64_t> dynamic_counts;

    void reset() {
        sum_x.clear();
        sum_y.clear();
        sum_y2.clear();
        n_x = 0;
        n_y = 0;
        dynamic_sums.clear();
        dynamic_counts.clear();
    }

    void collect(ggml_tensor * tensor, bool input) {
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
            throw std::runtime_error("captured activation has unsupported type");
        }
        const int64_t width = tensor->ne[0];
        const int64_t vectors = tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
        if (width <= 0 || vectors <= 0) {
            throw std::runtime_error("captured activation has invalid dimensions");
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
        auto & sum = input ? sum_x : sum_y;
        if (sum.empty()) sum.assign(width, 0.0);
        if ((int64_t) sum.size() != width) throw std::runtime_error("activation width changed during turn");
        if (!input && sum_y2.empty()) sum_y2.assign(width, 0.0);
        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < width; ++i0) {
                        const size_t offset = i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0];
                        const float current = tensor_value(data, tensor->type, offset);
                        if (!std::isfinite(current)) throw std::runtime_error("non-finite captured activation");
                        sum[i0] += current;
                        if (!input) sum_y2[i0] += (double) current * current;
                    }
                }
            }
        }
        if (input) n_x += vectors;
        else n_y += vectors;
    }

    void collect_dynamic(ggml_tensor * tensor) {
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
            throw std::runtime_error("captured dynamic activation has unsupported type");
        }
        const int64_t width = tensor->ne[0];
        const int64_t vectors = tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
        std::vector<uint8_t> copied;
        const uint8_t * data = nullptr;
        if (ggml_backend_buffer_is_host(tensor->buffer)) data = static_cast<const uint8_t *>(tensor->data);
        else {
            copied.resize(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, copied.data(), 0, copied.size());
            data = copied.data();
        }
        auto & sum = dynamic_sums[tensor->name];
        if (sum.empty()) sum.assign(width, 0.0);
        if ((int64_t) sum.size() != width) throw std::runtime_error("dynamic activation width changed");
        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < width; ++i0) {
                        const size_t offset = i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0];
                        const float current = tensor_value(data, tensor->type, offset);
                        if (!std::isfinite(current)) throw std::runtime_error("non-finite dynamic activation");
                        sum[i0] += current;
                    }
                }
            }
        }
        dynamic_counts[tensor->name] += vectors;
    }
};

bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & collector = *static_cast<activation_collector *>(user_data);
    if (!collector.enabled) return false;
    if (collector.dynamic_mode) {
        const bool selected = collector.dynamic_names.count(tensor->name) != 0;
        if (ask) return selected;
        if (selected) collector.collect_dynamic(tensor);
        return true;
    }
    const bool is_input = collector.input_name == tensor->name;
    const bool is_output = collector.output_name == tensor->name;
    if (ask) return is_input || is_output;
    if (is_input || is_output) collector.collect(tensor, is_input);
    return true;
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

struct score_result {
    double mean_nll = 0;
    double mean_entropy = 0;
    double top1_rate = 0;
    std::vector<float> first_logits;
};

double logsumexp(const float * logits, int32_t size) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < size; ++i) maximum = std::max(maximum, logits[i]);
    double total = 0;
    for (int32_t i = 0; i < size; ++i) total += std::exp((double) logits[i] - maximum);
    return maximum + std::log(total);
}

score_result score_continuation(llama_context * ctx, const llama_vocab * vocab, const std::string & prompt, const std::string & answer) {
    llama_memory_clear(llama_get_memory(ctx), true);
    auto prompt_tokens = tokenize(vocab, prompt, true);
    auto answer_tokens = tokenize(vocab, answer, false);
    if (prompt_tokens.empty() || answer_tokens.empty()) throw std::runtime_error("empty prompt or answer tokenization");
    if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size())) != 0) {
        throw std::runtime_error("prompt decode failed");
    }
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    score_result result;
    int32_t top1_hits = 0;
    for (size_t position = 0; position < answer_tokens.size(); ++position) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) throw std::runtime_error("missing logits");
        if (position == 0) result.first_logits.assign(logits, logits + vocab_size);
        const double normalizer = logsumexp(logits, vocab_size);
        result.mean_nll += normalizer - logits[answer_tokens[position]];
        int32_t top1 = 0;
        double entropy = 0;
        for (int32_t token = 0; token < vocab_size; ++token) {
            if (logits[token] > logits[top1]) top1 = token;
            const double probability = std::exp(logits[token] - normalizer);
            if (probability > 0) entropy -= probability * std::log(probability);
        }
        result.mean_entropy += entropy;
        top1_hits += top1 == answer_tokens[position];
        if (position + 1 < answer_tokens.size() &&
            llama_decode(ctx, llama_batch_get_one(&answer_tokens[position], 1)) != 0) {
            throw std::runtime_error("answer decode failed");
        }
    }
    result.mean_nll /= answer_tokens.size();
    result.mean_entropy /= answer_tokens.size();
    result.top1_rate = (double) top1_hits / answer_tokens.size();
    return result;
}

void collect_exposure(llama_context * ctx, const llama_vocab * vocab, activation_collector & collector, const std::string & exposure) {
    llama_memory_clear(llama_get_memory(ctx), true);
    collector.reset();
    collector.enabled = true;
    auto tokens = tokenize(vocab, exposure, true);
    const int32_t result = llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    if (result != 0) throw std::runtime_error("exposure decode failed");
    llama_synchronize(ctx);
    collector.enabled = false;
    if (collector.n_x == 0 || collector.n_y == 0 || collector.n_x != collector.n_y) {
        throw std::runtime_error("activation capture did not produce matched token statistics");
    }
}

double l2_norm(const std::vector<float> & values) {
    double total = 0;
    for (float value : values) total += (double) value * value;
    return std::sqrt(total);
}

struct update_result {
    float effective_alpha = 0;
    double unclipped_ratio = 0;
    double applied_ratio = 0;
    bool ratio_clipped = false;
    double g_norm = 0;
    double v_norm = 0;
};

void set_bank(llama_adapter_lora_weight & weight, const std::vector<float> & a, const std::vector<float> & b) {
    if (weight.a->type != GGML_TYPE_F32 || weight.b->type != GGML_TYPE_F32) {
        throw std::runtime_error("PoC requires FP32 adapter tensors");
    }
    if (a.size() * sizeof(float) != ggml_nbytes(weight.a) || b.size() * sizeof(float) != ggml_nbytes(weight.b)) {
        throw std::runtime_error("adapter bank byte count mismatch");
    }
    ggml_backend_tensor_set(weight.a, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_set(weight.b, b.data(), 0, b.size() * sizeof(float));
}

update_result apply_update(
        const options & opts,
        const activation_collector & collector,
        llama_adapter_lora_weight & weight,
        std::vector<float> & bank_a,
        std::vector<float> & bank_b,
        int64_t slot,
        int32_t turn) {
    const int64_t d_in = weight.a->ne[0];
    const int64_t rank = weight.a->ne[1];
    const int64_t d_out = weight.b->ne[1];
    if (rank != opts.rank || (int64_t) collector.sum_x.size() != d_in || (int64_t) collector.sum_y.size() != d_out) {
        throw std::runtime_error("captured dimensions or requested rank do not match adapter");
    }
    std::vector<float> x_bar(d_in), y_bar(d_out), a_rms(d_out), g(d_out), v(d_in);
    for (int64_t i = 0; i < d_in; ++i) x_bar[i] = collector.sum_x[i] / collector.n_x;
    for (int64_t j = 0; j < d_out; ++j) {
        y_bar[j] = collector.sum_y[j] / collector.n_y;
        a_rms[j] = std::sqrt(collector.sum_y2[j] / collector.n_y);
    }
    const double x_norm = l2_norm(x_bar);
    if (!(x_norm > 0)) throw std::runtime_error("pooled input norm is zero");
    for (int64_t i = 0; i < d_in; ++i) v[i] = x_bar[i] / x_norm;

    if (opts.rule != "competitive") {
        const double y_norm = l2_norm(y_bar);
        if (!(y_norm > 0)) throw std::runtime_error("pooled output norm is zero");
        for (int64_t j = 0; j < d_out; ++j) g[j] = y_bar[j] / y_norm;
    } else {
        double denominator = 1e-12;
        for (float value : a_rms) denominator += std::pow(std::max(value, 1e-12f), opts.gamma);
        for (int64_t j = 0; j < d_out; ++j) {
            const double probability = std::pow(std::max(a_rms[j], 1e-12f), opts.gamma) / denominator;
            const float sign = (y_bar[j] > 0) - (y_bar[j] < 0);
            g[j] = sign * (d_out * probability - 1.0);
        }
    }
    std::mt19937 generator(opts.seed + turn);
    if (opts.rule == "random") {
        std::normal_distribution<float> normal;
        for (float & value : v) value = normal(generator);
        for (float & value : g) value = normal(generator);
        const double random_v_norm = l2_norm(v);
        const double random_g_norm = l2_norm(g);
        for (float & value : v) value /= random_v_norm;
        for (float & value : g) value /= random_g_norm;
    } else if (opts.rule == "shuffled") {
        std::shuffle(v.begin(), v.end(), generator);
        std::shuffle(g.begin(), g.end(), generator);
    }
    double g_norm = l2_norm(g);
    if (opts.g_clip_l2 > 0 && g_norm > opts.g_clip_l2) {
        const double scale = opts.g_clip_l2 / g_norm;
        for (float & value : g) value *= scale;
        g_norm = l2_norm(g);
    }

    double base_rms_norm2 = 0;
    for (double value : collector.sum_y2) base_rms_norm2 += value / collector.n_y;
    const double base_rms_norm = std::sqrt(base_rms_norm2);
    float alpha = opts.eta;
    const double raw_ratio = base_rms_norm > 0 ? alpha * x_norm * g_norm / base_rms_norm : 0;
    bool clipped = false;
    if (opts.ratio_cap > 0 && raw_ratio > opts.ratio_cap) {
        alpha *= opts.ratio_cap / raw_ratio;
        clipped = true;
    }
    const double applied_ratio = base_rms_norm > 0 ? alpha * x_norm * g_norm / base_rms_norm : 0;

    if ((int64_t) bank_a.size() != d_in * rank || (int64_t) bank_b.size() != rank * d_out) {
        throw std::runtime_error("in-memory bank dimensions do not match adapter");
    }
    for (float & value : bank_b) value *= 1.0f - opts.decay;
    for (int64_t i = 0; i < d_in; ++i) bank_a[i + d_in * slot] = v[i];
    for (int64_t j = 0; j < d_out; ++j) bank_b[rank * j + slot] = alpha * g[j];
    set_bank(weight, bank_a, bank_b);
    return { alpha, raw_ratio, applied_ratio, clipped, g_norm, l2_norm(v) };
}

void reset_bank(llama_adapter_lora_weight & weight) {
    std::vector<float> a(ggml_nelements(weight.a), 0.0f);
    std::vector<float> b(ggml_nelements(weight.b), 0.0f);
    set_bank(weight, a, b);
}

double maximum_difference(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) throw std::runtime_error("logit vector size mismatch");
    double result = 0;
    for (size_t i = 0; i < lhs.size(); ++i) result = std::max(result, std::abs((double) lhs[i] - rhs[i]));
    return result;
}

double kl_divergence(const std::vector<float> & base, const std::vector<float> & candidate) {
    if (base.size() != candidate.size()) throw std::runtime_error("logit vector size mismatch");
    const double z_base = logsumexp(base.data(), base.size());
    const double z_candidate = logsumexp(candidate.data(), candidate.size());
    double result = 0;
    for (size_t i = 0; i < base.size(); ++i) {
        const double probability = std::exp(base[i] - z_base);
        result += probability * ((base[i] - z_base) - (candidate[i] - z_candidate));
    }
    return result;
}

int32_t top1(const std::vector<float> & logits) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

void write_json(
        const options & opts,
        const activation_collector & collector,
        const std::vector<update_result> & updates,
        const score_result & baseline,
        const score_result & zero,
        const score_result & plastic,
        const score_result & reset,
        const score_result & control_baseline,
        const score_result & control_plastic) {
    std::ofstream output(opts.output);
    if (!output) throw std::runtime_error("cannot open output JSON");
    const update_result & update = updates.back();
    output << "{\n"
           << "  \"status\": \"completed\",\n"
           << "  \"rule\": \"" << opts.rule << "\",\n"
           << "  \"rank\": " << opts.rank << ",\n"
           << "  \"repetitions\": " << opts.repetitions << ",\n"
           << "  \"seed\": " << opts.seed << ",\n"
           << "  \"eta\": " << opts.eta << ",\n"
           << "  \"decay\": " << opts.decay << ",\n"
           << "  \"gamma\": " << opts.gamma << ",\n"
           << "  \"ratio_cap\": " << opts.ratio_cap << ",\n"
           << "  \"active_slots\": " << std::min(opts.rank, opts.repetitions) << ",\n"
           << "  \"captured_input_width\": " << collector.sum_x.size() << ",\n"
           << "  \"captured_output_width\": " << collector.sum_y.size() << ",\n"
           << "  \"captured_tokens\": " << collector.n_x << ",\n"
           << "  \"effective_alpha\": " << update.effective_alpha << ",\n"
           << "  \"unclipped_plastic_base_ratio\": " << update.unclipped_ratio << ",\n"
           << "  \"applied_plastic_base_ratio\": " << update.applied_ratio << ",\n"
           << "  \"ratio_cap_applied\": " << (update.ratio_clipped ? "true" : "false") << ",\n"
           << "  \"g_l2\": " << update.g_norm << ",\n"
           << "  \"v_l2\": " << update.v_norm << ",\n"
           << "  \"zero_adapter_max_logit_difference\": " << maximum_difference(baseline.first_logits, zero.first_logits) << ",\n"
           << "  \"reset_max_logit_difference\": " << maximum_difference(zero.first_logits, reset.first_logits) << ",\n"
           << "  \"query_baseline_mean_nll\": " << baseline.mean_nll << ",\n"
           << "  \"query_plastic_mean_nll\": " << plastic.mean_nll << ",\n"
           << "  \"query_nll_change\": " << plastic.mean_nll - baseline.mean_nll << ",\n"
           << "  \"query_baseline_entropy\": " << baseline.mean_entropy << ",\n"
           << "  \"query_plastic_entropy\": " << plastic.mean_entropy << ",\n"
           << "  \"query_first_position_kl\": " << kl_divergence(baseline.first_logits, plastic.first_logits) << ",\n"
           << "  \"query_first_position_top1_agreement\": " << (top1(baseline.first_logits) == top1(plastic.first_logits) ? "true" : "false") << ",\n"
           << "  \"control_baseline_mean_nll\": " << control_baseline.mean_nll << ",\n"
           << "  \"control_plastic_mean_nll\": " << control_plastic.mean_nll << ",\n"
           << "  \"control_nll_change\": " << control_plastic.mean_nll - control_baseline.mean_nll << ",\n"
           << "  \"updates\": [\n";
    for (size_t i = 0; i < updates.size(); ++i) {
        const auto & item = updates[i];
        output << "    {\"turn\": " << i + 1
               << ", \"slot\": " << i % opts.rank
               << ", \"effective_alpha\": " << item.effective_alpha
               << ", \"unclipped_ratio\": " << item.unclipped_ratio
               << ", \"applied_ratio\": " << item.applied_ratio
               << ", \"ratio_cap_applied\": " << (item.ratio_clipped ? "true" : "false")
               << ", \"g_l2\": " << item.g_norm << "}"
               << (i + 1 == updates.size() ? "\n" : ",\n");
    }
    output << "  ]\n"
           << "}\n";
}

#include "correction-contrast.inc"
#include "dynamic-selection.inc"

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse_args(argc, argv);
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.gpu_layers;
        llama_model * model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (!model) throw std::runtime_error("model load failed");

        llama_adapter_lora * adapter = llama_adapter_lora_init(model, opts.adapter.c_str());
        if (!adapter) throw std::runtime_error("zero adapter load failed");
        auto weight_it = adapter->ab_map.find(opts.adapter_tensor);
        if (weight_it == adapter->ab_map.end() && opts.dynamic_selection_dir.empty()) {
            throw std::runtime_error("target adapter tensor not found");
        }

        activation_collector collector;
        collector.input_name = opts.input_tensor;
        collector.output_name = opts.output_tensor;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.ctx_size;
        context_params.n_batch = opts.ctx_size;
        context_params.n_ubatch = std::min(opts.ctx_size, 512);
        context_params.n_seq_max = 1;
        context_params.cb_eval = eval_callback;
        context_params.cb_eval_user_data = &collector;
        llama_context * ctx = llama_init_from_model(model, context_params);
        if (!ctx) throw std::runtime_error("context creation failed");
        const llama_vocab * vocab = llama_model_get_vocab(model);

        if (!opts.dynamic_selection_dir.empty()) {
            run_dynamic_selection(opts, *model, ctx, vocab, collector, adapter);
            llama_free(ctx);
            llama_adapter_lora_free(adapter);
            llama_model_free(model);
            return 0;
        }

        if (!opts.correction_contrast_dir.empty()) {
            run_correction_contrast(opts, ctx, vocab, collector, adapter, weight_it->second);
            llama_free(ctx);
            llama_adapter_lora_free(adapter);
            llama_model_free(model);
            return 0;
        }

        const score_result baseline = score_continuation(ctx, vocab, opts.query, opts.answer);
        const score_result control_baseline = score_continuation(ctx, vocab, opts.control_query, opts.control_answer);

        llama_adapter_lora * adapters[] = { adapter };
        float scales[] = { 1.0f };
        if (llama_set_adapters_lora(ctx, adapters, 1, scales) != 0) throw std::runtime_error("adapter activation failed");
        const score_result zero = score_continuation(ctx, vocab, opts.query, opts.answer);

        const int64_t d_in = weight_it->second.a->ne[0];
        const int64_t rank = weight_it->second.a->ne[1];
        const int64_t d_out = weight_it->second.b->ne[1];
        std::vector<float> bank_a(d_in * rank, 0.0f);
        std::vector<float> bank_b(rank * d_out, 0.0f);
        std::vector<update_result> updates;
        updates.reserve(opts.repetitions);
        for (int32_t turn = 0; turn < opts.repetitions; ++turn) {
            collect_exposure(ctx, vocab, collector, opts.exposure);
            updates.push_back(apply_update(opts, collector, weight_it->second, bank_a, bank_b, turn % rank, turn));
        }
        llama_synchronize(ctx);
        const score_result plastic = score_continuation(ctx, vocab, opts.query, opts.answer);
        const score_result control_plastic = score_continuation(ctx, vocab, opts.control_query, opts.control_answer);

        reset_bank(weight_it->second);
        llama_synchronize(ctx);
        const score_result reset = score_continuation(ctx, vocab, opts.query, opts.answer);
        write_json(opts, collector, updates, baseline, zero, plastic, reset, control_baseline, control_plastic);

        std::fprintf(stderr,
            "completed rule=%s rank=%d repetitions=%d x=%zu y=%zu tokens=%lld ratio=%.6f output=%s\n",
            opts.rule.c_str(), opts.rank, opts.repetitions, collector.sum_x.size(), collector.sum_y.size(),
            (long long) collector.n_x, updates.back().applied_ratio, opts.output.c_str());

        llama_free(ctx);
        llama_adapter_lora_free(adapter);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "forward-plasticity error: %s\n", error.what());
        usage(argv[0]);
        return 1;
    }
}
