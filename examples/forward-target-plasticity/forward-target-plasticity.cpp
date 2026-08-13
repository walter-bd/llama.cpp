#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct options {
    std::string model;
    std::string benchmark;
    std::string output = "target-plasticity.jsonl";
    std::string memory_output_dir;
    std::string task_output_dir;
    std::string sleep_output_dir;
    std::string task_benchmark;
    std::string task_mode = "primary";
    std::string delta_schedule;
    std::string rule = "target_delta";
    int32_t gpu_layers = 99;
    int32_t ctx_size = 1024;
    int32_t rank = 4;
    int32_t exposures = 16;
    int32_t q_threads = 16;
    uint32_t seed = 123;
    float rho = 0.003f;
    float decay = 0.05f;
    float eta = 1.0f;
};

struct association {
    std::string id;
    std::string learn_prompt;
    std::string target;
    std::string query;
    std::string paraphrase_1;
    std::string paraphrase_2;
    std::string foil;
};

struct task_example {
    std::string episode_id;
    std::string task_family;
    uint32_t task_seed = 0;
    std::string split;
    std::string id;
    std::string prompt;
    std::string target;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model MODEL --benchmark TSV [options]\n"
        "  --rule target_delta|shuffled_target|random|hebbian|error_gated_hebbian\n"
        "         fixed_step_clipped|probability_decayed_incremental|raw_q_incremental\n"
        "         nlms_target_response|random_incremental|shuffled_incremental\n"
        "  --rho FLOAT --rank INT --exposures INT --decay FLOAT --eta FLOAT --seed INT\n"
        "  --delta-schedule TSV  required for stabilized random/shuffled controls\n"
        "  --memory-output-dir DIR  run the fixed rank-4 multi-memory benchmark\n"
        "  --task-benchmark TSV --task-output-dir DIR  run task-level adaptation benchmark\n"
        "  --sleep-output-dir DIR  run forward-only fast-to-slow consolidation PoC\n"
        "  --task-mode primary|unrelated  select primary matrix or cross-task control\n",
        argv0);
}

std::vector<std::string> split_tsv(const std::string & line);

options parse_args(int argc, char ** argv) {
    options result;
    auto value = [&](int & i) -> const char * {
        if (++i >= argc) throw std::runtime_error("missing option value");
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" || arg == "-m") result.model = value(i);
        else if (arg == "--benchmark") result.benchmark = value(i);
        else if (arg == "--output") result.output = value(i);
        else if (arg == "--memory-output-dir") result.memory_output_dir = value(i);
        else if (arg == "--task-output-dir") result.task_output_dir = value(i);
        else if (arg == "--sleep-output-dir") result.sleep_output_dir = value(i);
        else if (arg == "--task-benchmark") result.task_benchmark = value(i);
        else if (arg == "--task-mode") result.task_mode = value(i);
        else if (arg == "--delta-schedule") result.delta_schedule = value(i);
        else if (arg == "--rule") result.rule = value(i);
        else if (arg == "--gpu-layers" || arg == "-ngl") result.gpu_layers = std::stoi(value(i));
        else if (arg == "--ctx-size" || arg == "-c") result.ctx_size = std::stoi(value(i));
        else if (arg == "--rank") result.rank = std::stoi(value(i));
        else if (arg == "--exposures") result.exposures = std::stoi(value(i));
        else if (arg == "--q-threads") result.q_threads = std::stoi(value(i));
        else if (arg == "--seed") result.seed = std::stoul(value(i));
        else if (arg == "--rho") result.rho = std::stof(value(i));
        else if (arg == "--decay") result.decay = std::stof(value(i));
        else if (arg == "--eta") result.eta = std::stof(value(i));
        else if (arg == "--help" || arg == "-h") { usage(argv[0]); std::exit(0); }
        else throw std::runtime_error("unknown option: " + arg);
    }
    const std::vector<std::string> rules = {
        "target_delta", "shuffled_target", "random", "hebbian", "error_gated_hebbian",
        "fixed_step_clipped", "probability_decayed_incremental", "raw_q_incremental",
        "nlms_target_response", "random_incremental", "shuffled_incremental",
    };
    if (result.model.empty()) throw std::runtime_error("--model is required");
    if (result.task_output_dir.empty() && result.benchmark.empty()) {
        throw std::runtime_error("--benchmark is required outside task-adaptation mode");
    }
    if (!result.task_output_dir.empty() && result.task_benchmark.empty()) {
        throw std::runtime_error("--task-benchmark is required with --task-output-dir");
    }
    if (result.task_mode != "primary" && result.task_mode != "unrelated") {
        throw std::runtime_error("--task-mode must be primary or unrelated");
    }
    if (std::find(rules.begin(), rules.end(), result.rule) == rules.end()) throw std::runtime_error("invalid rule");
    if (result.memory_output_dir.empty() &&
            (result.rule == "random_incremental" || result.rule == "shuffled_incremental") && result.delta_schedule.empty()) {
        throw std::runtime_error("stabilized controls require --delta-schedule");
    }
    if (result.rank <= 0 || result.exposures <= 0 || result.q_threads <= 0 || result.rho <= 0 ||
            result.decay < 0 || result.decay >= 1 || result.eta <= 0 || result.eta > 1) {
        throw std::runtime_error("invalid numeric option");
    }
    return result;
}

std::vector<task_example> load_task_benchmark(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open task benchmark TSV");
    std::vector<task_example> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto fields = split_tsv(line);
        if (fields.size() != 7) throw std::runtime_error("task row must contain seven tab-separated fields");
        if (fields[3] != "adaptation" && fields[3] != "heldout") {
            throw std::runtime_error("task split must be adaptation or heldout");
        }
        result.push_back({fields[0], fields[1], (uint32_t) std::stoul(fields[2]),
            fields[3], fields[4], fields[5], fields[6]});
    }
    if (result.empty()) throw std::runtime_error("task benchmark is empty");
    return result;
}

std::vector<std::string> split_tsv(const std::string & line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) fields.push_back(field);
    return fields;
}

std::vector<association> load_benchmark(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open benchmark TSV");
    std::vector<association> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto fields = split_tsv(line);
        if (fields.size() != 7) throw std::runtime_error("benchmark row must contain seven tab-separated fields");
        result.push_back({fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6]});
    }
    if (result.size() < 2) throw std::runtime_error("benchmark requires at least two associations");
    return result;
}

using delta_schedule = std::map<std::pair<std::string, int32_t>, double>;

delta_schedule load_delta_schedule(const std::string & path) {
    delta_schedule result;
    if (path.empty()) return result;
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open delta schedule TSV");
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto fields = split_tsv(line);
        if (fields.size() != 3) throw std::runtime_error("delta schedule row must contain three fields");
        const auto key = std::make_pair(fields[0], std::stoi(fields[1]));
        if (!result.emplace(key, std::stod(fields[2])).second) {
            throw std::runtime_error("duplicate delta schedule row");
        }
    }
    return result;
}

std::string json_escape(const std::string & value) {
    std::string result;
    for (char ch : value) {
        if (ch == '\\' || ch == '"') result.push_back('\\');
        if (ch == '\n') result += "\\n";
        else if (ch == '\r') result += "\\r";
        else if (ch == '\t') result += "\\t";
        else result.push_back(ch);
    }
    return result;
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

struct final_state_collector {
    bool enabled = false;
    std::vector<float> hidden;

    void collect(ggml_tensor * tensor) {
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
            throw std::runtime_error("unsupported final hidden type");
        }
        std::vector<uint8_t> copied;
        const uint8_t * data = nullptr;
        if (ggml_backend_buffer_is_host(tensor->buffer)) data = static_cast<const uint8_t *>(tensor->data);
        else {
            copied.resize(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, copied.data(), 0, copied.size());
            data = copied.data();
        }
        const int64_t width = tensor->ne[0];
        const int64_t vector = tensor->ne[1] * tensor->ne[2] * tensor->ne[3] - 1;
        hidden.resize(width);
        const int64_t i1 = vector % tensor->ne[1];
        const int64_t rem = vector / tensor->ne[1];
        const int64_t i2 = rem % tensor->ne[2];
        const int64_t i3 = rem / tensor->ne[2];
        for (int64_t i0 = 0; i0 < width; ++i0) {
            const size_t offset = i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0];
            hidden[i0] = tensor_value(data, tensor->type, offset);
        }
    }
};

bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & collector = *static_cast<final_state_collector *>(user_data);
    if (!collector.enabled || std::string(tensor->name) != "result_norm") return false;
    if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] <= 0) return false;
    if (ask) return true;
    collector.collect(tensor);
    return true;
}

double logsumexp(const float * logits, int32_t size) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < size; ++i) maximum = std::max(maximum, logits[i]);
    double total = 0;
    for (int32_t i = 0; i < size; ++i) total += std::exp((double) logits[i] - maximum);
    return maximum + std::log(total);
}

double l2_norm(const std::vector<float> & values) {
    double total = 0;
    for (float value : values) total += (double) value * value;
    return std::sqrt(total);
}

struct state_result {
    std::vector<float> hidden;
    std::vector<float> logits;
    double target_nll = 0;
    double target_probability = 0;
};

state_result evaluate_state(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const std::string & prompt, llama_token target) {
    llama_memory_clear(llama_get_memory(ctx), true);
    collector.hidden.clear();
    collector.enabled = true;
    auto tokens = tokenize(vocab, prompt, true);
    if (tokens.empty() || llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
        throw std::runtime_error("prompt decode failed");
    }
    llama_synchronize(ctx);
    collector.enabled = false;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * raw_logits = llama_get_logits_ith(ctx, -1);
    if (!raw_logits || collector.hidden.empty()) throw std::runtime_error("missing logits or final hidden state");
    state_result result;
    result.hidden = collector.hidden;
    result.logits.assign(raw_logits, raw_logits + n_vocab);
    const double z = logsumexp(raw_logits, n_vocab);
    result.target_nll = z - raw_logits[target];
    result.target_probability = std::exp(raw_logits[target] - z);
    return result;
}

double score_token(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const std::string & prompt, llama_token target) {
    return evaluate_state(ctx, vocab, collector, prompt, target).target_nll;
}

struct output_matrix {
    int64_t width = 0;
    int64_t rows = 0;
    std::vector<float> values;

    const float * row(int64_t index) const { return values.data() + index * width; }
};

output_matrix load_output_matrix(const llama_model & model) {
    const ggml_tensor * tensor = model.output;
    if (!tensor || tensor->ne[2] != 1 || tensor->ne[3] != 1) throw std::runtime_error("invalid output tensor");
    const auto * traits = ggml_get_type_traits(tensor->type);
    if (!traits || !traits->to_float) throw std::runtime_error("output tensor cannot be dequantized");
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    output_matrix result;
    result.width = tensor->ne[0];
    result.rows = tensor->ne[1];
    result.values.resize((size_t) result.width * result.rows);
    for (int64_t row = 0; row < result.rows; ++row) {
        traits->to_float(raw.data() + row * tensor->nb[1], result.values.data() + row * result.width, result.width);
    }
    return result;
}

std::vector<float> probabilities(const std::vector<float> & logits) {
    const double z = logsumexp(logits.data(), logits.size());
    std::vector<float> result(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) result[i] = std::exp(logits[i] - z);
    return result;
}

std::vector<float> exact_target_direction(
        const output_matrix & output, const std::vector<float> & logits,
        llama_token target, int32_t n_threads) {
    if ((int64_t) logits.size() != output.rows || target < 0 || target >= output.rows) {
        throw std::runtime_error("output matrix/logit mismatch");
    }
    const auto probs = probabilities(logits);
    const int32_t threads = std::max(1, std::min<int32_t>(n_threads, output.rows));
    std::vector<std::vector<float>> partial(threads, std::vector<float>(output.width, 0.0f));
    std::vector<std::thread> workers;
    for (int32_t thread = 0; thread < threads; ++thread) {
        workers.emplace_back([&, thread]() {
            const int64_t begin = output.rows * thread / threads;
            const int64_t end = output.rows * (thread + 1) / threads;
            auto & sum = partial[thread];
            for (int64_t row = begin; row < end; ++row) {
                const float scale = probs[row];
                const float * weight = output.row(row);
                for (int64_t column = 0; column < output.width; ++column) sum[column] += scale * weight[column];
            }
        });
    }
    for (auto & worker : workers) worker.join();
    std::vector<float> result(output.width);
    const float * target_row = output.row(target);
    for (int64_t column = 0; column < output.width; ++column) {
        double expected = 0;
        for (int32_t thread = 0; thread < threads; ++thread) expected += partial[thread][column];
        result[column] = target_row[column] - expected;
    }
    return result;
}

struct bank {
    int32_t rank;
    int32_t width;
    std::vector<float> u; // [rank, width] in GGML storage: slot + rank*dimension
    std::vector<float> v; // [width, rank] in GGML storage: dimension + width*slot
    std::vector<std::string> owners;
    double initial_q_norm = 0;

    bank(int32_t rank, int32_t width) : rank(rank), width(width), u((size_t) rank*width),
            v((size_t) rank*width), owners(rank) {}

    void clear() {
        std::fill(u.begin(), u.end(), 0);
        std::fill(v.begin(), v.end(), 0);
        std::fill(owners.begin(), owners.end(), std::string());
        initial_q_norm = 0;
    }

    std::vector<float> apply(const std::vector<float> & key) const {
        std::vector<float> result(width, 0.0f);
        for (int32_t slot = 0; slot < rank; ++slot) {
            double coefficient = 0;
            for (int32_t i = 0; i < width; ++i) coefficient += v[(size_t) width*slot + i] * key[i];
            for (int32_t j = 0; j < width; ++j) result[j] += u[(size_t) rank*j + slot] * coefficient;
        }
        return result;
    }

    std::vector<float> slot_key(int32_t slot) const {
        std::vector<float> result(width);
        for (int32_t i = 0; i < width; ++i) result[i] = v[(size_t) width*slot + i];
        return result;
    }

    std::vector<float> slot_contribution(int32_t slot, const std::vector<float> & key) const {
        double coefficient = 0;
        for (int32_t i = 0; i < width; ++i) coefficient += v[(size_t) width*slot + i]*key[i];
        std::vector<float> result(width);
        for (int32_t j = 0; j < width; ++j) result[j] = u[(size_t) rank*j + slot]*coefficient;
        return result;
    }
};

struct update_result {
    bool feasible = false;
    double alpha = 0;
    double requested_rho = 0;
    double actual_rho = 0;
    double direction_norm = 0;
    double target_probability = 0;
    double target_probability_after = 0;
    double delta_rho_requested = 0;
    double delta_rho_actual = 0;
    double response_rho_requested = 0;
    double response_rho = 0;
    double key_similarity = 0;
    double q_previous_cosine = 0;
    double q_response_cosine = 0;
    bool projected = false;
    bool reference_delta_applied = false;
    double reference_delta_rho = 0;
    double control_direction_cosine = 1;
    std::string matching_semantics = "whole_response";
};

update_result update_bank(
        const options & opts, bank & state, int32_t turn,
        const std::vector<float> & hidden, const std::vector<float> & target_q,
        const std::vector<float> & wrong_q, double target_probability, uint32_t random_salt) {
    const double h_norm = l2_norm(hidden);
    if (!(h_norm > 0)) throw std::runtime_error("zero hidden norm");
    std::vector<float> key(hidden.size());
    for (size_t i = 0; i < hidden.size(); ++i) key[i] = hidden[i] / h_norm;

    for (float & value : state.u) value *= 1.0f - opts.decay;
    const int32_t slot = turn % state.rank;
    for (int32_t j = 0; j < state.width; ++j) state.u[(size_t) state.rank*j + slot] = 0;
    for (int32_t i = 0; i < state.width; ++i) state.v[(size_t) state.width*slot + i] = 0;

    const std::vector<float> residual = state.apply(key);
    std::vector<float> direction(state.width);
    if (opts.rule == "target_delta" || opts.rule == "shuffled_target") {
        const auto & q = opts.rule == "target_delta" ? target_q : wrong_q;
        for (int32_t i = 0; i < state.width; ++i) direction[i] = q[i] - residual[i];
    } else if (opts.rule == "random") {
        std::mt19937 generator(opts.seed + 104729u*turn + 1009u*random_salt);
        std::normal_distribution<float> normal;
        for (float & value : direction) value = normal(generator);
    } else {
        const float gate = opts.rule == "error_gated_hebbian" ? 1.0f - target_probability : 1.0f;
        for (int32_t i = 0; i < state.width; ++i) direction[i] = gate * hidden[i];
    }
    const double direction_norm = l2_norm(direction);
    if (!(direction_norm > 0)) throw std::runtime_error("zero update direction");
    for (float & value : direction) value /= direction_norm;

    const double target_norm = opts.rho * h_norm;
    double r_norm2 = 0;
    double r_dot_u = 0;
    for (int32_t i = 0; i < state.width; ++i) {
        r_norm2 += (double) residual[i] * residual[i];
        r_dot_u += (double) residual[i] * direction[i];
    }
    // v == key, so c = v^T key = 1.
    const double discriminant = 4.0*(r_dot_u*r_dot_u - (r_norm2 - target_norm*target_norm));
    if (discriminant < -1e-10) return {false, 0, opts.rho, 0, direction_norm, target_probability};
    const double root = std::sqrt(std::max(0.0, discriminant));
    const double alpha_1 = (-2.0*r_dot_u + root)/2.0;
    const double alpha_2 = (-2.0*r_dot_u - root)/2.0;
    double alpha = -1;
    if (alpha_1 >= 0) alpha = alpha_1;
    if (alpha_2 >= 0 && (alpha < 0 || alpha_2 < alpha)) alpha = alpha_2;
    if (!(alpha >= 0)) return {false, 0, opts.rho, 0, direction_norm, target_probability};

    for (int32_t i = 0; i < state.width; ++i) state.v[(size_t) state.width*slot + i] = key[i];
    for (int32_t j = 0; j < state.width; ++j) state.u[(size_t) state.rank*j + slot] = alpha * direction[j];
    const auto applied = state.apply(key);
    const double actual_rho = l2_norm(applied)/h_norm;
    return {std::abs(actual_rho - opts.rho) <= 5e-6, alpha, opts.rho, actual_rho, direction_norm, target_probability};
}

bool is_stabilized_rule(const std::string & rule) {
    return rule == "fixed_step_clipped" || rule == "probability_decayed_incremental" ||
           rule == "raw_q_incremental" || rule == "nlms_target_response" ||
           rule == "random_incremental" || rule == "shuffled_incremental";
}

bool needs_exact_q(const std::string & rule) {
    return rule == "target_delta" || rule == "shuffled_target" ||
           rule == "fixed_step_clipped" || rule == "probability_decayed_incremental" ||
           rule == "raw_q_incremental" || rule == "nlms_target_response" ||
           rule == "shuffled_incremental";
}

double dot_product(const std::vector<float> & left, const std::vector<float> & right) {
    double result = 0;
    for (size_t i = 0; i < left.size(); ++i) result += (double) left[i]*right[i];
    return result;
}

double cosine(const std::vector<float> & left, const std::vector<float> & right) {
    const double denominator = l2_norm(left)*l2_norm(right);
    return denominator > 0 ? dot_product(left, right)/denominator : 0;
}

bool make_exact_bounded_control_step(
        const std::vector<float> & residual, const std::vector<float> & requested_direction,
        double step_norm, double bound, std::vector<float> & candidate,
        double & direction_cosine) {
    const double residual_norm = l2_norm(residual);
    if (step_norm < 0 || step_norm > bound + residual_norm + 1e-10) return false;

    candidate.resize(residual.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        candidate[i] = residual[i] + step_norm*requested_direction[i];
    }
    if (l2_norm(candidate) <= bound + 1e-10) {
        direction_cosine = 1;
        return true;
    }
    if (!(residual_norm > 0) || !(step_norm > 0)) return false;

    std::vector<float> radial(residual.size());
    for (size_t i = 0; i < residual.size(); ++i) radial[i] = residual[i]/residual_norm;
    const double radial_limit = (bound*bound - residual_norm*residual_norm - step_norm*step_norm)
        /(2.0*residual_norm*step_norm);
    if (radial_limit < -1.0 - 1e-8) return false;
    const double bounded_radial = std::clamp(radial_limit, -1.0, 1.0);
    const double requested_radial = dot_product(requested_direction, radial);

    std::vector<float> tangent(residual.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        tangent[i] = requested_direction[i] - requested_radial*radial[i];
    }
    double tangent_norm = l2_norm(tangent);
    if (!(tangent_norm > 1e-12)) {
        size_t axis = 0;
        for (size_t i = 1; i < radial.size(); ++i) {
            if (std::abs(radial[i]) < std::abs(radial[axis])) axis = i;
        }
        tangent.assign(residual.size(), 0);
        tangent[axis] = 1;
        for (size_t i = 0; i < residual.size(); ++i) tangent[i] -= radial[axis]*radial[i];
        tangent_norm = l2_norm(tangent);
    }
    for (float & value : tangent) value /= tangent_norm;

    const double tangent_scale = std::sqrt(std::max(0.0, 1.0 - bounded_radial*bounded_radial));
    std::vector<float> adjusted(requested_direction.size());
    for (size_t i = 0; i < adjusted.size(); ++i) {
        adjusted[i] = bounded_radial*radial[i] + tangent_scale*tangent[i];
        candidate[i] = residual[i] + step_norm*adjusted[i];
    }
    direction_cosine = cosine(requested_direction, adjusted);
    return l2_norm(candidate) <= bound + 1e-7;
}

update_result update_bank_stabilized(
        const options & opts, bank & state, int32_t turn,
        const std::vector<float> & hidden, const std::vector<float> & target_q,
        const std::vector<float> & wrong_q, const std::vector<float> & previous_q,
        double target_probability, uint32_t random_salt, double matched_delta_rho) {
    const double h_norm = l2_norm(hidden);
    if (!(h_norm > 0)) throw std::runtime_error("zero hidden norm");
    std::vector<float> key(hidden.size());
    for (size_t i = 0; i < hidden.size(); ++i) key[i] = hidden[i]/h_norm;

    // The isolated acquisition experiment owns slot zero for the association.
    // Repeated exposures update its output vector while its key remains fixed.
    double key_similarity = 1.0;
    if (turn == 0) {
        for (int32_t i = 0; i < state.width; ++i) state.v[i] = key[i];
    } else {
        std::vector<float> stored_key(state.width);
        for (int32_t i = 0; i < state.width; ++i) stored_key[i] = state.v[i];
        key_similarity = cosine(stored_key, key);
        if (key_similarity < 0.999) throw std::runtime_error("same-association key continuity failed");
    }

    for (float & value : state.u) value *= 1.0f - opts.decay;
    const std::vector<float> residual = state.apply(key);
    const double residual_norm = l2_norm(residual);

    std::vector<float> direction(state.width);
    if (opts.rule == "shuffled_incremental") {
        direction = wrong_q;
    } else if (opts.rule == "random_incremental") {
        std::mt19937 generator(opts.seed + 104729u*turn + 1009u*random_salt);
        std::normal_distribution<float> normal;
        for (float & value : direction) value = normal(generator);
    } else {
        direction = target_q;
    }
    const double direction_norm = l2_norm(direction);
    if (!(direction_norm > 0)) throw std::runtime_error("zero stabilized update direction");
    if (state.initial_q_norm == 0 && opts.rule == "raw_q_incremental") state.initial_q_norm = direction_norm;
    for (float & value : direction) value /= direction_norm;

    const double bound = opts.rho*h_norm;
    double delta_target = opts.eta*bound;
    std::string matching_semantics = "incremental_delta";
    const bool reference_delta_applied = opts.rule == "random_incremental" || opts.rule == "shuffled_incremental";
    if (reference_delta_applied) {
        if (!(matched_delta_rho >= 0)) throw std::runtime_error("missing matched control delta");
        delta_target = matched_delta_rho*h_norm;
    } else if (opts.rule == "probability_decayed_incremental") {
        delta_target *= 1.0 - target_probability;
    } else if (opts.rule == "raw_q_incremental") {
        delta_target *= direction_norm/std::max(state.initial_q_norm, 1e-12);
    }

    std::vector<float> candidate(state.width);
    double control_direction_cosine = 1;
    if (opts.rule == "nlms_target_response") {
        matching_semantics = "whole_response";
        const double desired_norm = bound*(1.0 - target_probability);
        for (int32_t i = 0; i < state.width; ++i) {
            const double desired = desired_norm*direction[i];
            candidate[i] = residual[i] + opts.eta*(desired - residual[i]);
        }
        delta_target = l2_norm(candidate);
    } else if (reference_delta_applied) {
        if (!make_exact_bounded_control_step(
                    residual, direction, delta_target, bound, candidate, control_direction_cosine)) {
            throw std::runtime_error("matched control delta is infeasible inside trust region");
        }
    } else {
        for (int32_t i = 0; i < state.width; ++i) candidate[i] = residual[i] + delta_target*direction[i];
    }

    bool projected = false;
    const double candidate_norm = l2_norm(candidate);
    if (candidate_norm > bound) {
        projected = true;
        const double scale = bound/candidate_norm;
        for (float & value : candidate) value *= scale;
    }

    std::vector<float> response_change(state.width);
    for (int32_t i = 0; i < state.width; ++i) response_change[i] = candidate[i] - residual[i];
    const double response_change_norm = l2_norm(response_change);

    // With a fixed normalized key in slot zero, v^T k is one. Update u in
    // output space so the slot response changes by exactly response_change.
    double key_coefficient = 0;
    for (int32_t i = 0; i < state.width; ++i) key_coefficient += state.v[i]*key[i];
    if (std::abs(key_coefficient) < 1e-6) throw std::runtime_error("slot key coefficient is zero");
    for (int32_t j = 0; j < state.width; ++j) {
        state.u[(size_t) state.rank*j] += response_change[j]/key_coefficient;
    }

    const std::vector<float> applied = state.apply(key);
    const double actual_response_rho = l2_norm(applied)/h_norm;
    const double actual_delta_rho = response_change_norm/h_norm;
    const std::vector<float> & diagnostic_q = opts.rule == "shuffled_incremental" ? wrong_q : target_q;
    const double q_previous_cosine = previous_q.empty() || diagnostic_q.empty() ? 1.0 : cosine(diagnostic_q, previous_q);
    const double q_response_cosine = residual_norm > 0 && !diagnostic_q.empty() ? cosine(diagnostic_q, residual) : 0;
    const bool feasible = actual_response_rho <= opts.rho + 5e-6;
    update_result result;
    result.feasible = feasible;
    result.alpha = response_change_norm;
    result.requested_rho = opts.rho;
    result.actual_rho = actual_response_rho;
    result.direction_norm = direction_norm;
    result.target_probability = target_probability;
    result.delta_rho_requested = delta_target/h_norm;
    result.delta_rho_actual = actual_delta_rho;
    result.response_rho_requested = opts.rule == "nlms_target_response"
        ? opts.rho*(1.0 - target_probability) : opts.rho;
    result.response_rho = actual_response_rho;
    result.key_similarity = key_similarity;
    result.q_previous_cosine = q_previous_cosine;
    result.q_response_cosine = q_response_cosine;
    result.projected = projected;
    result.reference_delta_applied = reference_delta_applied;
    result.reference_delta_rho = reference_delta_applied ? matched_delta_rho : 0;
    result.control_direction_cosine = control_direction_cosine;
    result.matching_semantics = matching_semantics;
    return result;
}

void install_bank(llama_context * ctx, const bank & state) {
    if (!ctx->set_fast_weights(state.u.data(), state.u.size(), state.v.data(), state.v.size())) {
        throw std::runtime_error("failed to install fast-weight bank");
    }
}

std::vector<int32_t> checkpoints(int32_t maximum) {
    std::vector<int32_t> values;
    for (int32_t value : {1, 2, 4, 8, 16, 32}) if (value <= maximum) values.push_back(value);
    if (values.empty() || values.back() != maximum) values.push_back(maximum);
    return values;
}

struct query_nll {
    double exact = 0;
    double paraphrase = 0;

    double mean() const { return (exact + 2.0*paraphrase)/3.0; }
};

query_nll measure_query_nll(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const association & item, llama_token target) {
    query_nll result;
    result.exact = score_token(ctx, vocab, collector, item.query, target);
    result.paraphrase = (score_token(ctx, vocab, collector, item.paraphrase_1, target) +
                         score_token(ctx, vocab, collector, item.paraphrase_2, target))/2.0;
    return result;
}

double unrelated_nll(llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector) {
    const std::vector<std::pair<std::string, std::string>> controls = {
        {"The capital of France is", " Paris"},
        {"Two plus two equals", " four"},
        {"The opposite of hot is", " cold"},
    };
    double result = 0;
    for (const auto & item : controls) {
        auto answer = tokenize(vocab, item.second, false);
        if (answer.size() != 1) throw std::runtime_error("unrelated control answer is not one token");
        result += score_token(ctx, vocab, collector, item.first, answer[0]);
    }
    return result/controls.size();
}

void write_row(
        std::ofstream & output, const options & opts, const association & item,
        int32_t exposure, const update_result & update,
        const query_nll & target_before, const query_nll & target_after,
        const query_nll & foil_before, const query_nll & foil_after,
        double unrelated_before, double unrelated_after) {
    output << "{\"status\":\"completed\",\"association\":\"" << json_escape(item.id)
           << "\",\"rule\":\"" << opts.rule << "\",\"seed\":" << opts.seed
           << ",\"rank\":" << opts.rank << ",\"rho\":" << opts.rho
           << ",\"decay\":" << opts.decay << ",\"eta\":" << opts.eta
           << ",\"exposures\":" << exposure
           << ",\"update_feasible\":" << (update.feasible ? "true" : "false")
           << ",\"alpha\":" << update.alpha
           << ",\"actual_rho\":" << update.actual_rho
           << ",\"target_probability_at_update\":" << update.target_probability
           << ",\"target_probability_after_update\":" << update.target_probability_after
           << ",\"delta_rho_requested\":" << update.delta_rho_requested
           << ",\"delta_rho_actual\":" << update.delta_rho_actual
           << ",\"response_rho_requested\":" << update.response_rho_requested
           << ",\"response_rho\":" << update.response_rho
           << ",\"key_similarity\":" << update.key_similarity
           << ",\"q_previous_cosine\":" << update.q_previous_cosine
           << ",\"q_response_cosine\":" << update.q_response_cosine
           << ",\"projected\":" << (update.projected ? "true" : "false")
           << ",\"reference_delta_applied\":" << (update.reference_delta_applied ? "true" : "false")
           << ",\"reference_delta_rho\":" << update.reference_delta_rho
           << ",\"control_direction_cosine\":" << update.control_direction_cosine
           << ",\"matching_semantics\":\"" << update.matching_semantics << "\""
           << ",\"target_nll_before\":" << target_before.mean()
           << ",\"target_nll_after\":" << target_after.mean()
           << ",\"target_nll_change\":" << target_after.mean()-target_before.mean()
           << ",\"target_exact_nll_change\":" << target_after.exact-target_before.exact
           << ",\"target_paraphrase_nll_change\":" << target_after.paraphrase-target_before.paraphrase
           << ",\"foil_nll_before\":" << foil_before.mean()
           << ",\"foil_nll_after\":" << foil_after.mean()
           << ",\"foil_nll_change\":" << foil_after.mean()-foil_before.mean()
           << ",\"foil_exact_nll_change\":" << foil_after.exact-foil_before.exact
           << ",\"foil_paraphrase_nll_change\":" << foil_after.paraphrase-foil_before.paraphrase
           << ",\"selectivity\":" << (foil_after.mean()-foil_before.mean())-(target_after.mean()-target_before.mean())
           << ",\"unrelated_nll_before\":" << unrelated_before
           << ",\"unrelated_nll_after\":" << unrelated_after
           << ",\"unrelated_nll_change\":" << unrelated_after-unrelated_before
           << "}\n";
    output.flush();
}

void write_update_row(
        std::ofstream & output, const options & opts, const association & item,
        int32_t exposure, const update_result & update) {
    output << "{\"status\":\"update\",\"association\":\"" << json_escape(item.id)
           << "\",\"rule\":\"" << opts.rule << "\",\"seed\":" << opts.seed
           << ",\"rank\":" << opts.rank << ",\"rho\":" << opts.rho
           << ",\"decay\":" << opts.decay << ",\"eta\":" << opts.eta
           << ",\"exposures\":" << exposure
           << ",\"update_feasible\":" << (update.feasible ? "true" : "false")
           << ",\"target_probability_before\":" << update.target_probability
           << ",\"target_probability_after\":" << update.target_probability_after
           << ",\"direction_norm\":" << update.direction_norm
           << ",\"delta_rho_requested\":" << update.delta_rho_requested
           << ",\"delta_rho_actual\":" << update.delta_rho_actual
           << ",\"response_rho_requested\":" << update.response_rho_requested
           << ",\"response_rho\":" << update.response_rho
           << ",\"key_similarity\":" << update.key_similarity
           << ",\"q_previous_cosine\":" << update.q_previous_cosine
           << ",\"q_response_cosine\":" << update.q_response_cosine
           << ",\"projected\":" << (update.projected ? "true" : "false")
           << ",\"reference_delta_applied\":" << (update.reference_delta_applied ? "true" : "false")
           << ",\"reference_delta_rho\":" << update.reference_delta_rho
           << ",\"control_direction_cosine\":" << update.control_direction_cosine
           << ",\"matching_semantics\":\"" << update.matching_semantics << "\"}\n";
    output.flush();
}

void finite_difference_check(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const association & item, llama_token target, const std::vector<float> & q,
        std::ofstream & output, int32_t rank) {
    const state_result base = evaluate_state(ctx, vocab, collector, item.learn_prompt, target);
    const double q_norm2 = l2_norm(q)*l2_norm(q);
    std::vector<float> key(base.hidden.size());
    const double h_norm = l2_norm(base.hidden);
    for (size_t i = 0; i < key.size(); ++i) key[i] = base.hidden[i]/h_norm;
    // Quantized LM-head matmuls have a finite activation resolution. Keep the
    // preregistered small steps, then probe larger steps to identify the first
    // numerically resolvable local-descent region.
    for (double epsilon : {1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}) {
        bank probe(rank, base.hidden.size());
        for (size_t i = 0; i < key.size(); ++i) probe.v[i] = key[i];
        for (size_t j = 0; j < q.size(); ++j) probe.u[(size_t) rank*j] = epsilon*q[j]/h_norm;
        install_bank(ctx, probe);
        const double changed = score_token(ctx, vocab, collector, item.learn_prompt, target);
        output << "{\"status\":\"finite_difference\",\"epsilon\":" << epsilon
               << ",\"base_nll\":" << base.target_nll
               << ",\"changed_nll\":" << changed
               << ",\"directional_derivative\":" << (changed-base.target_nll)/epsilon
               << ",\"expected_derivative\":" << -q_norm2
               << ",\"hidden_norm\":" << h_norm
               << ",\"delta_hidden_norm\":" << epsilon*std::sqrt(q_norm2)
               << "}\n";
    }
    ctx->clear_fast_weights();
    output.flush();
}

struct token_metrics {
    double nll = 0;
    double probability = 0;
    int32_t rank = 0;
    double margin = 0;
    llama_token greedy_token = -1;
    bool greedy_recall = false;
    std::vector<float> hidden;
    std::vector<float> logits;
};

token_metrics token_metrics_from_state(const state_result & state, llama_token target) {
    token_metrics result;
    result.nll = state.target_nll;
    result.probability = state.target_probability;
    result.hidden = state.hidden;
    result.logits = state.logits;
    float best_other = -std::numeric_limits<float>::infinity();
    float best = -std::numeric_limits<float>::infinity();
    int32_t better = 0;
    for (int32_t token = 0; token < (int32_t) state.logits.size(); ++token) {
        if (state.logits[token] > best) {
            best = state.logits[token];
            result.greedy_token = token;
        }
        if (token != target) best_other = std::max(best_other, state.logits[token]);
        if (state.logits[token] > state.logits[target]) ++better;
    }
    result.rank = better + 1;
    result.margin = state.logits[target] - best_other;
    result.greedy_recall = result.greedy_token == target;
    return result;
}

token_metrics measure_token(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const std::string & prompt, llama_token target) {
    const state_result state = evaluate_state(ctx, vocab, collector, prompt, target);
    return token_metrics_from_state(state, target);
}

std::vector<float> normalized(const std::vector<float> & values) {
    const double norm = l2_norm(values);
    if (!(norm > 0)) throw std::runtime_error("cannot normalize zero vector");
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = values[i]/norm;
    return result;
}

double projection_angle_degrees(const std::vector<float> & requested, const std::vector<float> & actual) {
    if (!(l2_norm(requested) > 0) || !(l2_norm(actual) > 0)) return 0;
    const double value = std::clamp(cosine(requested, actual), -1.0, 1.0);
    return std::acos(value)*180.0/3.14159265358979323846;
}

struct memory_update_result {
    update_result update;
    int32_t slot = -1;
    std::string assignment_reason;
    double projection_angle_deg = 0;
    std::vector<double> slot_similarities;
    std::vector<float> requested_delta;
    std::vector<float> actual_delta;
};

memory_update_result update_memory_slot(
        const options & opts, bank & state, const std::string & rule,
        int32_t slot, const std::string & association_id, int32_t exposure,
        const std::vector<float> & hidden, const std::vector<float> & target_q,
        const std::vector<float> & shuffled_q, double target_probability,
        uint32_t random_salt, double matched_delta_rho,
        const std::vector<float> & previous_q) {
    const double h_norm = l2_norm(hidden);
    const std::vector<float> key = normalized(hidden);
    memory_update_result result;
    result.slot = slot;
    result.assignment_reason = state.owners[slot].empty() ? "association_owned_slot_created" : "association_owned_slot_reused";
    if (state.owners[slot].empty()) {
        state.owners[slot] = association_id;
        for (int32_t i = 0; i < state.width; ++i) state.v[(size_t) state.width*slot + i] = key[i];
    } else if (state.owners[slot] != association_id) {
        throw std::runtime_error("association attempted to update another association's slot");
    }

    result.slot_similarities.resize(state.rank);
    for (int32_t current = 0; current < state.rank; ++current) {
        result.slot_similarities[current] = state.owners[current].empty() ? 0 : cosine(key, state.slot_key(current));
    }

    const std::vector<float> residual = state.apply(key);
    std::vector<float> requested_direction;
    if (rule == "target") requested_direction = target_q;
    else if (rule == "shuffled") requested_direction = shuffled_q;
    else if (rule == "random") {
        requested_direction.resize(state.width);
        std::mt19937 generator(opts.seed + 104729u*random_salt + 1009u*exposure + 7919u*slot);
        std::normal_distribution<float> normal;
        for (float & value : requested_direction) value = normal(generator);
    } else throw std::runtime_error("invalid memory rule");
    const double direction_norm = l2_norm(requested_direction);
    if (!(direction_norm > 0)) throw std::runtime_error("zero memory update direction");
    for (float & value : requested_direction) value /= direction_norm;

    const double bound = opts.rho*h_norm;
    double requested_step = opts.eta*bound*(1.0 - target_probability);
    const bool matched_control = rule != "target";
    if (matched_control) {
        if (!(matched_delta_rho >= 0)) throw std::runtime_error("missing target delta for matched memory control");
        requested_step = matched_delta_rho*h_norm;
    }

    std::vector<float> candidate;
    double control_direction_cosine = 1;
    if (matched_control) {
        if (!make_exact_bounded_control_step(
                    residual, requested_direction, requested_step, bound, candidate, control_direction_cosine)) {
            throw std::runtime_error("matched memory control step infeasible");
        }
    } else {
        candidate.resize(state.width);
        for (int32_t i = 0; i < state.width; ++i) candidate[i] = residual[i] + requested_step*requested_direction[i];
    }

    bool projected = false;
    if (l2_norm(candidate) > bound) {
        projected = true;
        const double scale = bound/l2_norm(candidate);
        for (float & value : candidate) value *= scale;
    }
    std::vector<float> actual_delta(state.width);
    std::vector<float> requested_delta(state.width);
    for (int32_t i = 0; i < state.width; ++i) {
        actual_delta[i] = candidate[i] - residual[i];
        requested_delta[i] = requested_step*requested_direction[i];
    }

    const std::vector<float> stored_key = state.slot_key(slot);
    const double coefficient = dot_product(stored_key, key);
    if (std::abs(coefficient) < 1e-6) throw std::runtime_error("owned slot key coefficient is zero");
    for (int32_t j = 0; j < state.width; ++j) {
        state.u[(size_t) state.rank*j + slot] += actual_delta[j]/coefficient;
    }

    const std::vector<float> applied = state.apply(key);
    update_result & update = result.update;
    update.feasible = l2_norm(applied)/h_norm <= opts.rho + 5e-6;
    update.alpha = l2_norm(actual_delta);
    update.requested_rho = opts.rho;
    update.actual_rho = l2_norm(applied)/h_norm;
    update.direction_norm = direction_norm;
    update.target_probability = target_probability;
    update.delta_rho_requested = requested_step/h_norm;
    update.delta_rho_actual = l2_norm(actual_delta)/h_norm;
    update.response_rho_requested = opts.rho;
    update.response_rho = update.actual_rho;
    update.key_similarity = cosine(stored_key, key);
    const std::vector<float> & diagnostic_q = rule == "shuffled" ? shuffled_q : target_q;
    update.q_previous_cosine = previous_q.empty() ? 1.0 : cosine(diagnostic_q, previous_q);
    update.q_response_cosine = l2_norm(residual) > 0 ? cosine(diagnostic_q, residual) : 0;
    update.projected = projected || (matched_control && control_direction_cosine < 1.0 - 1e-12);
    update.reference_delta_applied = matched_control;
    update.reference_delta_rho = matched_control ? matched_delta_rho : 0;
    update.control_direction_cosine = control_direction_cosine;
    update.matching_semantics = "incremental";
    result.projection_angle_deg = projection_angle_degrees(requested_delta, actual_delta);
    result.requested_delta = requested_delta;
    result.actual_delta = actual_delta;
    return result;
}

memory_update_result update_task_bank(
        const options & opts, bank & state, const std::string & rule, int32_t turn,
        const std::vector<float> & hidden, const std::vector<float> & target_q,
        const std::vector<float> & control_q, double target_probability,
        uint32_t random_salt, double matched_delta_rho,
        const std::vector<float> & previous_q) {
    const double h_norm = l2_norm(hidden);
    if (!(h_norm > 0)) throw std::runtime_error("zero task hidden norm");
    const std::vector<float> key = normalized(hidden);
    const int32_t slot = turn % state.rank;

    memory_update_result result;
    result.slot = slot;
    result.assignment_reason = state.owners[slot].empty() ? "task_slot_created" : "task_slot_replaced";
    result.slot_similarities.resize(state.rank);
    for (int32_t current = 0; current < state.rank; ++current) {
        result.slot_similarities[current] = state.owners[current].empty() ? 0 : cosine(key, state.slot_key(current));
    }

    // A bounded rank-R task bank retains the other R-1 updates and replaces the
    // oldest cyclic slot. No example receives an association-owned retrieval slot.
    for (int32_t j = 0; j < state.width; ++j) state.u[(size_t) state.rank*j + slot] = 0;
    for (int32_t i = 0; i < state.width; ++i) state.v[(size_t) state.width*slot + i] = 0;
    state.owners[slot] = "task_state";
    const std::vector<float> residual = state.apply(key);

    std::vector<float> direction;
    if (rule == "target") direction = target_q;
    else if (rule == "shuffled" || rule == "label_permuted") direction = control_q;
    else if (rule == "random") {
        direction.resize(state.width);
        std::mt19937 generator(opts.seed + 104729u*random_salt + 1009u*turn);
        std::normal_distribution<float> normal;
        for (float & value : direction) value = normal(generator);
    } else throw std::runtime_error("invalid task rule");
    const double direction_norm = l2_norm(direction);
    if (!(direction_norm > 0)) throw std::runtime_error("zero task update direction");
    for (float & value : direction) value /= direction_norm;

    const double bound = opts.rho*h_norm;
    double requested_step = opts.eta*bound*(1.0 - target_probability);
    const bool matched_control = rule != "target";
    if (matched_control) {
        if (!(matched_delta_rho >= 0)) throw std::runtime_error("missing matched task delta");
        requested_step = matched_delta_rho*h_norm;
    }

    std::vector<float> candidate;
    double control_direction_cosine = 1;
    if (matched_control) {
        if (!make_exact_bounded_control_step(
                    residual, direction, requested_step, bound, candidate, control_direction_cosine)) {
            result.assignment_reason = "matched_control_infeasible";
            result.update.feasible = false;
            result.update.direction_norm = direction_norm;
            result.update.target_probability = target_probability;
            result.update.delta_rho_requested = requested_step/h_norm;
            result.update.response_rho = l2_norm(residual)/h_norm;
            result.update.reference_delta_applied = true;
            result.update.reference_delta_rho = matched_delta_rho;
            result.update.control_direction_cosine = 0;
            result.update.matching_semantics = "incremental";
            result.requested_delta.resize(state.width);
            for (int32_t i = 0; i < state.width; ++i) {
                result.requested_delta[i] = requested_step*direction[i];
            }
            return result;
        }
    } else {
        candidate.resize(state.width);
        for (int32_t i = 0; i < state.width; ++i) candidate[i] = residual[i] + requested_step*direction[i];
    }

    bool projected = false;
    if (l2_norm(candidate) > bound) {
        projected = true;
        const double scale = bound/l2_norm(candidate);
        for (float & value : candidate) value *= scale;
    }
    std::vector<float> actual_delta(state.width);
    std::vector<float> requested_delta(state.width);
    for (int32_t i = 0; i < state.width; ++i) {
        actual_delta[i] = candidate[i] - residual[i];
        requested_delta[i] = requested_step*direction[i];
        state.v[(size_t) state.width*slot + i] = key[i];
        state.u[(size_t) state.rank*i + slot] = actual_delta[i];
    }

    const std::vector<float> applied = state.apply(key);
    update_result & update = result.update;
    update.feasible = l2_norm(applied)/h_norm <= opts.rho + 5e-6;
    update.alpha = l2_norm(actual_delta);
    update.requested_rho = opts.rho;
    update.actual_rho = l2_norm(applied)/h_norm;
    update.direction_norm = direction_norm;
    update.target_probability = target_probability;
    update.delta_rho_requested = requested_step/h_norm;
    update.delta_rho_actual = l2_norm(actual_delta)/h_norm;
    update.response_rho_requested = opts.rho;
    update.response_rho = update.actual_rho;
    update.key_similarity = result.slot_similarities[slot];
    const std::vector<float> & diagnostic_q = rule == "target" ? target_q : control_q;
    update.q_previous_cosine = previous_q.empty() ? 1.0 : cosine(diagnostic_q, previous_q);
    update.q_response_cosine = l2_norm(residual) > 0 ? cosine(diagnostic_q, residual) : 0;
    update.projected = projected || (matched_control && control_direction_cosine < 1.0 - 1e-12);
    update.reference_delta_applied = matched_control;
    update.reference_delta_rho = matched_control ? matched_delta_rho : 0;
    update.control_direction_cosine = control_direction_cosine;
    update.matching_semantics = "incremental";
    result.projection_angle_deg = projection_angle_degrees(requested_delta, actual_delta);
    result.requested_delta = requested_delta;
    result.actual_delta = actual_delta;
    return result;
}

void save_bank(const bank & state, const std::string & path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot save plastic bank");
    const uint64_t magic = 0x314b4e4142574654ULL;
    output.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    output.write(reinterpret_cast<const char *>(&state.rank), sizeof(state.rank));
    output.write(reinterpret_cast<const char *>(&state.width), sizeof(state.width));
    for (const std::string & owner : state.owners) {
        const uint32_t size = owner.size();
        output.write(reinterpret_cast<const char *>(&size), sizeof(size));
        output.write(owner.data(), size);
    }
    output.write(reinterpret_cast<const char *>(state.u.data()), state.u.size()*sizeof(float));
    output.write(reinterpret_cast<const char *>(state.v.data()), state.v.size()*sizeof(float));
    if (!output) throw std::runtime_error("plastic bank serialization failed");
}

bank load_bank(const std::string & path, int32_t expected_rank, int32_t expected_width) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot load plastic bank");
    uint64_t magic = 0;
    int32_t rank = 0;
    int32_t width = 0;
    input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    input.read(reinterpret_cast<char *>(&rank), sizeof(rank));
    input.read(reinterpret_cast<char *>(&width), sizeof(width));
    if (magic != 0x314b4e4142574654ULL || rank != expected_rank || width != expected_width) {
        throw std::runtime_error("plastic bank header mismatch");
    }
    bank result(rank, width);
    for (std::string & owner : result.owners) {
        uint32_t size = 0;
        input.read(reinterpret_cast<char *>(&size), sizeof(size));
        owner.resize(size);
        input.read(owner.data(), size);
    }
    input.read(reinterpret_cast<char *>(result.u.data()), result.u.size()*sizeof(float));
    input.read(reinterpret_cast<char *>(result.v.data()), result.v.size()*sizeof(float));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("plastic bank payload mismatch");
    return result;
}

struct unrelated_reference {
    std::string prompt;
    llama_token target = -1;
    std::vector<float> logits;
    double nll = 0;
};

double distribution_kl(const std::vector<float> & adapted, const std::vector<float> & base) {
    const double za = logsumexp(adapted.data(), adapted.size());
    const double zb = logsumexp(base.data(), base.size());
    double result = 0;
    for (size_t i = 0; i < adapted.size(); ++i) {
        const double pa = std::exp(adapted[i] - za);
        result += pa*((adapted[i] - za) - (base[i] - zb));
    }
    return result;
}

std::vector<unrelated_reference> make_unrelated_reference(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector) {
    const std::vector<std::pair<std::string, std::string>> prompts = {
        {"The capital of France is", " Paris"},
        {"Two plus two equals", " four"},
        {"The opposite of hot is", " cold"},
        {"Water freezes at zero degrees", " Celsius"},
        {"The first month of the year is", " January"},
    };
    std::vector<unrelated_reference> result;
    for (const auto & entry : prompts) {
        const auto target = tokenize(vocab, entry.second, false);
        if (target.size() != 1) throw std::runtime_error("unrelated answer is not one token");
        const auto state = evaluate_state(ctx, vocab, collector, entry.first, target[0]);
        result.push_back({entry.first, target[0], state.logits, state.target_nll});
    }
    return result;
}

std::pair<double, double> measure_unrelated(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const std::vector<unrelated_reference> & references) {
    double nll = 0;
    double kl = 0;
    for (const auto & reference : references) {
        const auto state = evaluate_state(ctx, vocab, collector, reference.prompt, reference.target);
        nll += state.target_nll;
        kl += distribution_kl(state.logits, reference.logits);
    }
    return {nll/references.size(), kl/references.size()};
}

void write_recall(
        std::ofstream & output, const std::string & rule, uint32_t seed,
        const std::string & phase, const std::string & stage, const association & item,
        const std::string & query_type, const std::string & prompt, llama_token target,
        const token_metrics & metrics, bool foil_query = false) {
    output << std::setprecision(17)
           << "{\"rule\":\"" << rule << "\",\"seed\":" << seed
           << ",\"phase\":\"" << phase << "\",\"stage\":\"" << stage
           << "\",\"association_id\":\"" << json_escape(item.id)
           << "\",\"query_type\":\"" << query_type << "\",\"foil_query\":" << (foil_query ? "true" : "false")
           << ",\"prompt\":\"" << json_escape(prompt) << "\",\"target_token_id\":" << target
           << ",\"generated_token_id\":" << metrics.greedy_token
           << ",\"greedy_exact_recall\":" << (metrics.greedy_recall ? "true" : "false")
           << ",\"target_probability\":" << metrics.probability << ",\"target_rank\":" << metrics.rank
           << ",\"target_margin\":" << metrics.margin << ",\"target_nll\":" << metrics.nll << "}\n";
    output.flush();
}

void evaluate_association_queries(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        std::ofstream & output, const std::string & rule, uint32_t seed,
        const std::string & phase, const std::string & stage,
        const association & item, const association & entity_foil, llama_token target) {
    const std::vector<std::pair<std::string, std::string>> queries = {
        {"exact", item.query}, {"near", item.paraphrase_1}, {"distant", item.paraphrase_2},
        {"entity_foil", entity_foil.query},
    };
    for (const auto & query : queries) {
        write_recall(output, rule, seed, phase, stage, item, query.first, query.second, target,
            measure_token(ctx, vocab, collector, query.second, target), query.first == "entity_foil");
    }
}

void write_slot_matrix(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        std::ofstream & output, const bank & state, const std::vector<association> & items,
        size_t learned, const std::string & rule, uint32_t seed, const std::string & stage) {
    for (size_t item_index = 0; item_index < learned; ++item_index) {
        const auto target = tokenize(vocab, items[item_index].target, false);
        const auto current = evaluate_state(ctx, vocab, collector, items[item_index].learn_prompt, target[0]);
        const auto key = normalized(current.hidden);
        for (int32_t slot = 0; slot < state.rank; ++slot) {
            const double similarity = state.owners[slot].empty() ? 0 : cosine(key, state.slot_key(slot));
            const double contribution_rho = l2_norm(state.slot_contribution(slot, key))/l2_norm(current.hidden);
            output << std::setprecision(17)
                   << "{\"rule\":\"" << rule << "\",\"seed\":" << seed
                   << ",\"stage\":\"" << stage << "\",\"association_id\":\"" << items[item_index].id
                   << "\",\"slot_id\":" << slot << ",\"slot_owner\":\"" << json_escape(state.owners[slot])
                   << "\",\"key_similarity\":" << similarity
                   << ",\"slot_contribution_rho\":" << contribution_rho << "}\n";
        }
    }
    output.flush();
}

std::string schedule_key(const std::string & phase, const std::string & association_id, int32_t exposure) {
    return phase + "|" + association_id + "|" + std::to_string(exposure);
}

void learn_memory_association(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const options & opts, bank & state,
        const std::string & rule, const std::string & phase, const association & item,
        llama_token target, llama_token shuffled_target, int32_t slot, int32_t exposures,
        std::map<std::string, double> & target_schedule,
        const std::vector<unrelated_reference> & unrelated, std::ofstream & updates) {
    std::vector<float> previous_q;
    for (int32_t exposure_index = 1; exposure_index <= exposures; ++exposure_index) {
        install_bank(ctx, state);
        const auto before = evaluate_state(ctx, vocab, collector, item.learn_prompt, target);
        const auto target_q = exact_target_direction(output_head, before.logits, target, opts.q_threads);
        const auto shuffled_q = exact_target_direction(output_head, before.logits, shuffled_target, opts.q_threads);
        const std::string key = schedule_key(phase, item.id, exposure_index);
        double matched = -1;
        if (rule != "target") {
            const auto found = target_schedule.find(key);
            if (found == target_schedule.end()) throw std::runtime_error("missing target schedule for " + key);
            matched = found->second;
        }
        auto result = update_memory_slot(opts, state, rule, slot, item.id, exposure_index,
            before.hidden, target_q, shuffled_q, before.target_probability,
            (uint32_t) std::hash<std::string>{}(phase + item.id), matched, previous_q);
        if (!result.update.feasible || result.update.response_rho > opts.rho + 5e-6) {
            throw std::runtime_error("memory trust-region violation");
        }
        install_bank(ctx, state);
        const auto after = evaluate_state(ctx, vocab, collector, item.learn_prompt, target);
        const auto after_metrics = token_metrics_from_state(after, target);
        const auto exact_metrics = measure_token(ctx, vocab, collector, item.query, target);
        const auto near_metrics = measure_token(ctx, vocab, collector, item.paraphrase_1, target);
        const auto distant_metrics = measure_token(ctx, vocab, collector, item.paraphrase_2, target);
        const auto foil_tokens = tokenize(vocab, item.foil, false);
        if (foil_tokens.size() != 1) throw std::runtime_error("association foil is not one token");
        const auto foil_metrics = measure_token(ctx, vocab, collector, item.query, foil_tokens[0]);
        const auto unrelated_metrics = measure_unrelated(ctx, vocab, collector, unrelated);
        result.update.target_probability_after = after.target_probability;
        if (rule == "target") target_schedule[key] = result.update.delta_rho_actual;
        updates << std::setprecision(17)
                << "{\"rule\":\"" << rule << "\",\"seed\":" << opts.seed
                << ",\"phase\":\"" << phase << "\",\"association_id\":\"" << item.id
                << "\",\"exposure_number\":" << exposure_index << ",\"slot_id\":" << slot
                << ",\"slot_assignment_reason\":\"" << result.assignment_reason
                << "\",\"target_token_id\":" << target
                << ",\"target_probability\":" << before.target_probability
                << ",\"target_probability_after\":" << after.target_probability
                << ",\"target_rank\":" << after_metrics.rank
                << ",\"target_logit_margin\":" << after_metrics.margin
                << ",\"exact_nll\":" << exact_metrics.nll
                << ",\"near_paraphrase_nll\":" << near_metrics.nll
                << ",\"distant_paraphrase_nll\":" << distant_metrics.nll
                << ",\"foil_nll\":" << foil_metrics.nll
                << ",\"generated_exact_recall\":" << (exact_metrics.greedy_recall ? "true" : "false")
                << ",\"generated_near_paraphrase_recall\":" << (near_metrics.greedy_recall ? "true" : "false")
                << ",\"generated_distant_paraphrase_recall\":" << (distant_metrics.greedy_recall ? "true" : "false")
                << ",\"unrelated_nll\":" << unrelated_metrics.first
                << ",\"unrelated_kl\":" << unrelated_metrics.second
                << ",\"rho_h\":" << result.update.response_rho
                << ",\"rho_delta\":" << result.update.delta_rho_actual
                << ",\"requested_update_rho\":" << result.update.delta_rho_requested
                << ",\"actual_update_rho\":" << result.update.delta_rho_actual
                << ",\"final_state_rho\":" << result.update.response_rho
                << ",\"angle_q_pk\":" << std::acos(std::clamp(result.update.q_response_cosine, -1.0, 1.0))*180.0/3.14159265358979323846
                << ",\"angle_q_previous\":" << std::acos(std::clamp(result.update.q_previous_cosine, -1.0, 1.0))*180.0/3.14159265358979323846
                << ",\"projection_applied\":" << (result.update.projected ? "true" : "false")
                << ",\"projection_direction_change_deg\":" << result.projection_angle_deg
                << ",\"requested_direction\":\"" << rule << "\""
                << ",\"applied_direction\":\"" << (result.update.projected ? "trust_region_adjusted" : rule) << "\""
                << ",\"matching_semantics\":\"incremental\",\"key_similarities\":[";
        for (size_t i = 0; i < result.slot_similarities.size(); ++i) {
            if (i) updates << ',';
            updates << result.slot_similarities[i];
        }
        updates << "]}\n";
        updates.flush();
        previous_q = rule == "shuffled" ? shuffled_q : target_q;
    }
}

std::string make_context_prompt(
        const llama_vocab * vocab, const std::string & query, int32_t target_tokens) {
    const std::string filler = "A neutral ledger records ordinary weather, calendar, and inventory observations. ";
    const int32_t filler_tokens = std::max<int32_t>(1, tokenize(vocab, filler, false).size());
    const int32_t query_tokens = tokenize(vocab, query, true).size();
    const int32_t repetitions = std::max(0, (target_tokens - query_tokens + filler_tokens - 1)/filler_tokens);
    std::string result;
    result.reserve((size_t) repetitions*filler.size() + query.size());
    for (int32_t i = 0; i < repetitions; ++i) result += filler;
    return result + query;
}

void run_memory_condition(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const options & opts,
        const std::vector<association> & benchmark, const std::string & rule,
        std::map<std::string, double> & target_schedule,
        const std::vector<unrelated_reference> & unrelated,
        const std::filesystem::path & root) {
    const std::filesystem::path condition = root/rule;
    std::filesystem::create_directories(condition);
    std::ofstream updates(condition/"updates.jsonl", std::ios::trunc);
    std::ofstream recalls(condition/"generated_recall.jsonl", std::ios::trunc);
    std::ofstream traces(condition/"slot_interference.jsonl", std::ios::trunc);
    std::ofstream stages(condition/"stage_metrics.jsonl", std::ios::trunc);
    if (!updates || !recalls || !traces || !stages) throw std::runtime_error("cannot create memory condition outputs");

    auto target_for = [&](size_t index) {
        auto result = tokenize(vocab, benchmark[index].target, false);
        if (result.size() != 1) throw std::runtime_error("association target is not one token: " + benchmark[index].id);
        return result[0];
    };
    auto write_stage = [&](const std::string & phase, const std::string & stage) {
        const auto metrics = measure_unrelated(ctx, vocab, collector, unrelated);
        stages << std::setprecision(17) << "{\"rule\":\"" << rule << "\",\"seed\":" << opts.seed
               << ",\"phase\":\"" << phase << "\",\"stage\":\"" << stage
               << "\",\"unrelated_nll\":" << metrics.first << ",\"unrelated_kl\":" << metrics.second << "}\n";
        stages.flush();
    };

    // Phase 1: one association, generated one-token recall and persistence.
    bank single(opts.rank, output_head.width);
    ctx->clear_fast_weights();
    learn_memory_association(ctx, vocab, collector, output_head, opts, single, rule, "phase1",
        benchmark[0], target_for(0), target_for(1), 0, opts.exposures, target_schedule, unrelated, updates);
    install_bank(ctx, single);
    evaluate_association_queries(ctx, vocab, collector, recalls, rule, opts.seed,
        "phase1", "after_A", benchmark[0], benchmark[8], target_for(0));
    write_stage("phase1", "after_A");

    const auto pre_serial = measure_token(ctx, vocab, collector, benchmark[0].query, target_for(0));
    const std::string bank_path = (condition/"plastic_bank.bin").string();
    save_bank(single, bank_path);
    ctx->clear_fast_weights();
    bank reloaded = load_bank(bank_path, opts.rank, output_head.width);
    install_bank(ctx, reloaded);
    const auto post_serial = measure_token(ctx, vocab, collector, benchmark[0].query, target_for(0));
    double serial_max_logit_delta = 0;
    for (size_t i = 0; i < pre_serial.logits.size(); ++i) {
        serial_max_logit_delta = std::max(serial_max_logit_delta,
            std::abs((double) pre_serial.logits[i] - post_serial.logits[i]));
    }
    std::ofstream serialization(condition/"serialization_validation.json", std::ios::trunc);
    serialization << std::setprecision(17)
                  << "{\"rule\":\"" << rule << "\",\"max_logit_abs_diff\":" << serial_max_logit_delta
                  << ",\"pre_probability\":" << pre_serial.probability
                  << ",\"post_probability\":" << post_serial.probability
                  << ",\"pre_greedy_recall\":" << (pre_serial.greedy_recall ? "true" : "false")
                  << ",\"post_greedy_recall\":" << (post_serial.greedy_recall ? "true" : "false") << "}\n";

    // Phase 5: read-only context growth. No plastic update occurs here.
    for (int32_t context_tokens : {128, 512, 2048, 8192}) {
        const std::string prompt = make_context_prompt(vocab, benchmark[0].query, context_tokens);
        const auto metrics = measure_token(ctx, vocab, collector, prompt, target_for(0));
        const auto context_key = normalized(metrics.hidden);
        const double key_cosine = cosine(single.slot_key(0), context_key);
        write_recall(recalls, rule, opts.seed, "phase5", "context_" + std::to_string(context_tokens),
            benchmark[0], "context", prompt, target_for(0), metrics);
        traces << std::setprecision(17) << "{\"rule\":\"" << rule << "\",\"seed\":" << opts.seed
               << ",\"stage\":\"context_" << context_tokens << "\",\"association_id\":\"" << benchmark[0].id
               << "\",\"slot_id\":0,\"key_similarity\":" << key_cosine
               << ",\"slot_contribution_rho\":" << l2_norm(single.slot_contribution(0, context_key))/l2_norm(metrics.hidden) << "}\n";
        traces.flush();
    }

    // Phases 2 and 6: one shared rank-4 bank, sequential A/B/C/D insertion.
    bank multi(opts.rank, output_head.width);
    ctx->clear_fast_weights();
    for (size_t learned = 0; learned < 4; ++learned) {
        learn_memory_association(ctx, vocab, collector, output_head, opts, multi, rule, "phase2",
            benchmark[learned], target_for(learned), target_for((learned + 1)%4), learned,
            opts.exposures, target_schedule, unrelated, updates);
        install_bank(ctx, multi);
        const std::string stage = "after_" + std::string(1, 'A' + learned);
        for (size_t item_index = 0; item_index <= learned; ++item_index) {
            evaluate_association_queries(ctx, vocab, collector, recalls, rule, opts.seed,
                "phase2", stage, benchmark[item_index], benchmark[8 + item_index], target_for(item_index));
        }
        write_slot_matrix(ctx, vocab, collector, traces, multi, benchmark, learned + 1, rule, opts.seed, stage);
        write_stage("phase2", stage);
    }

    // Phase 3: overwrite A's value in slot zero with a verified alternate token.
    const llama_token old_target = target_for(0);
    const llama_token new_target = target_for(4);
    const auto old_before = measure_token(ctx, vocab, collector, benchmark[0].query, old_target);
    const auto new_before = measure_token(ctx, vocab, collector, benchmark[0].query, new_target);
    association overwritten = benchmark[0];
    overwritten.target = benchmark[4].target;
    learn_memory_association(ctx, vocab, collector, output_head, opts, multi, rule, "phase3_overwrite",
        overwritten, new_target, target_for(1), 0, opts.exposures, target_schedule, unrelated, updates);
    install_bank(ctx, multi);
    const auto old_after = measure_token(ctx, vocab, collector, benchmark[0].query, old_target);
    const auto new_after = measure_token(ctx, vocab, collector, benchmark[0].query, new_target);
    stages << std::setprecision(17)
           << "{\"rule\":\"" << rule << "\",\"seed\":" << opts.seed
           << ",\"phase\":\"phase3_overwrite\",\"stage\":\"after_overwrite\""
           << ",\"old_probability_before\":" << old_before.probability
           << ",\"old_probability_after\":" << old_after.probability
           << ",\"new_probability_before\":" << new_before.probability
           << ",\"new_probability_after\":" << new_after.probability
           << ",\"old_greedy_before\":" << (old_before.greedy_recall ? "true" : "false")
           << ",\"old_greedy_after\":" << (old_after.greedy_recall ? "true" : "false")
           << ",\"new_greedy_before\":" << (new_before.greedy_recall ? "true" : "false")
           << ",\"new_greedy_after\":" << (new_after.greedy_recall ? "true" : "false")
           << ",\"overwrite_score\":" << ((std::log(std::max(new_after.probability, 1e-30)) - std::log(std::max(new_before.probability, 1e-30)))
                - (std::log(std::max(old_after.probability, 1e-30)) - std::log(std::max(old_before.probability, 1e-30)))) << "}\n";
    for (size_t item_index = 0; item_index < 4; ++item_index) {
        const llama_token evaluation_target = item_index == 0 ? new_target : target_for(item_index);
        evaluate_association_queries(ctx, vocab, collector, recalls, rule, opts.seed,
            "phase3_overwrite", "after_overwrite", benchmark[item_index], benchmark[8 + item_index], evaluation_target);
    }
    write_slot_matrix(ctx, vocab, collector, traces, multi, benchmark, 4, rule, opts.seed, "after_overwrite");
    write_stage("phase3_overwrite", "after_overwrite");
}

#include "task-adaptation.inc"
#include "sleep-consolidation.inc"

void run_memory_benchmark(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const options & opts,
        const std::vector<association> & benchmark) {
    if (opts.rank != 4 || std::abs(opts.rho - 0.001f) > 1e-9 || opts.decay != 0 || opts.eta != 1) {
        throw std::runtime_error("memory benchmark requires rank=4 rho=0.001 decay=0 eta=1");
    }
    if (benchmark.size() < 12) throw std::runtime_error("memory benchmark requires at least 12 associations");
    const std::filesystem::path root(opts.memory_output_dir);
    std::filesystem::create_directories(root);

    ctx->clear_fast_weights();
    const auto first_target = tokenize(vocab, benchmark[0].target, false);
    if (first_target.size() != 1) throw std::runtime_error("phase-zero target is not one token");
    const auto baseline = evaluate_state(ctx, vocab, collector, benchmark[0].query, first_target[0]);
    bank zero(opts.rank, output_head.width);
    install_bank(ctx, zero);
    const auto zero_state = evaluate_state(ctx, vocab, collector, benchmark[0].query, first_target[0]);
    double zero_diff = 0;
    for (size_t i = 0; i < baseline.logits.size(); ++i) zero_diff = std::max(zero_diff,
        std::abs((double) baseline.logits[i] - zero_state.logits[i]));
    const auto probe_key = normalized(baseline.hidden);
    zero.owners[0] = "phase0_probe";
    for (int32_t i = 0; i < zero.width; ++i) zero.v[i] = probe_key[i];
    zero.u[0] = 0.01f;
    install_bank(ctx, zero);
    (void) evaluate_state(ctx, vocab, collector, benchmark[0].query, first_target[0]);
    ctx->clear_fast_weights();
    const auto reset_state = evaluate_state(ctx, vocab, collector, benchmark[0].query, first_target[0]);
    double reset_diff = 0;
    for (size_t i = 0; i < baseline.logits.size(); ++i) reset_diff = std::max(reset_diff,
        std::abs((double) baseline.logits[i] - reset_state.logits[i]));

    std::ofstream correctness(root/"correctness.jsonl", std::ios::trunc);
    correctness << std::setprecision(17)
                << "{\"check\":\"zero_adapter_equivalence\",\"max_logit_abs_diff\":" << zero_diff << "}\n"
                << "{\"check\":\"reset_equivalence\",\"max_logit_abs_diff\":" << reset_diff << "}\n";
    const auto q = exact_target_direction(output_head, baseline.logits, first_target[0], opts.q_threads);
    finite_difference_check(ctx, vocab, collector, benchmark[0], first_target[0], q, correctness, opts.rank);
    correctness.flush();

    ctx->clear_fast_weights();
    const auto unrelated = make_unrelated_reference(ctx, vocab, collector);
    const std::filesystem::path frozen_dir = root/"frozen";
    std::filesystem::create_directories(frozen_dir);
    std::ofstream frozen_recall(frozen_dir/"generated_recall.jsonl", std::ios::trunc);
    for (size_t item_index = 0; item_index < 4; ++item_index) {
        const auto target = tokenize(vocab, benchmark[item_index].target, false);
        evaluate_association_queries(ctx, vocab, collector, frozen_recall, "frozen", opts.seed,
            "baseline", "baseline", benchmark[item_index], benchmark[8 + item_index], target[0]);
    }
    const auto frozen_unrelated = measure_unrelated(ctx, vocab, collector, unrelated);
    std::ofstream frozen_stage(frozen_dir/"stage_metrics.jsonl", std::ios::trunc);
    frozen_stage << std::setprecision(17) << "{\"rule\":\"frozen\",\"seed\":" << opts.seed
                 << ",\"phase\":\"baseline\",\"stage\":\"baseline\",\"unrelated_nll\":"
                 << frozen_unrelated.first << ",\"unrelated_kl\":" << frozen_unrelated.second << "}\n";
    std::map<std::string, double> target_schedule;
    for (const std::string rule : {"target", "random", "shuffled"}) {
        std::fprintf(stderr, "memory benchmark start rule=%s seed=%u\n", rule.c_str(), opts.seed);
        run_memory_condition(ctx, vocab, collector, output_head, opts, benchmark, rule,
            target_schedule, unrelated, root);
        std::fprintf(stderr, "memory benchmark complete rule=%s seed=%u\n", rule.c_str(), opts.seed);
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse_args(argc, argv);
        const auto benchmark = opts.task_output_dir.empty()
            ? load_benchmark(opts.benchmark) : std::vector<association>();
        const auto task_benchmark = opts.task_output_dir.empty()
            ? std::vector<task_example>() : load_task_benchmark(opts.task_benchmark);
        const auto control_delta_schedule = load_delta_schedule(opts.delta_schedule);
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = opts.gpu_layers;
        llama_model * public_model = llama_model_load_from_file(opts.model.c_str(), model_params);
        if (!public_model) throw std::runtime_error("model load failed");
        auto & model = *public_model;

        final_state_collector collector;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.ctx_size;
        context_params.n_batch = opts.ctx_size;
        context_params.n_ubatch = std::min(opts.ctx_size, 512);
        context_params.n_seq_max = 1;
        context_params.cb_eval = eval_callback;
        context_params.cb_eval_user_data = &collector;
        llama_context * public_ctx = llama_init_from_model(public_model, context_params);
        if (!public_ctx) throw std::runtime_error("context creation failed");
        auto * ctx = public_ctx;
        const int32_t runtime_rank = opts.sleep_output_dir.empty() ? opts.rank : opts.rank + 8;
        if (!ctx->init_fast_weights(runtime_rank)) throw std::runtime_error("fast-weight initialization failed");
        const llama_vocab * vocab = llama_model_get_vocab(public_model);

        output_matrix output_matrix_f32;
        if (needs_exact_q(opts.rule) || !opts.memory_output_dir.empty() || !opts.task_output_dir.empty() ||
                !opts.sleep_output_dir.empty()) {
            std::fprintf(stderr, "loading exact dequantized LM head rows=%lld width=%lld\n",
                (long long) model.output->ne[1], (long long) model.output->ne[0]);
            output_matrix_f32 = load_output_matrix(model);
            std::fprintf(stderr, "exact LM head ready: %.2f GiB\n",
                output_matrix_f32.values.size()*sizeof(float)/(1024.0*1024.0*1024.0));
        }

        if (!opts.sleep_output_dir.empty()) {
            run_sleep_consolidation(ctx, vocab, collector, output_matrix_f32, opts, benchmark);
            llama_free(public_ctx);
            llama_model_free(public_model);
            return 0;
        }
        if (!opts.memory_output_dir.empty()) {
            run_memory_benchmark(ctx, vocab, collector, output_matrix_f32, opts, benchmark);
            llama_free(public_ctx);
            llama_model_free(public_model);
            return 0;
        }
        if (!opts.task_output_dir.empty()) {
            run_task_benchmark(ctx, vocab, collector, output_matrix_f32, opts, task_benchmark);
            llama_free(public_ctx);
            llama_model_free(public_model);
            return 0;
        }

        std::ofstream output(opts.output, std::ios::app);
        if (!output) throw std::runtime_error("cannot open output JSONL");
        bool finite_difference_done = false;
        const auto exposure_checkpoints = checkpoints(opts.exposures);

        for (size_t association_index = 0; association_index < benchmark.size(); ++association_index) {
            const association & item = benchmark[association_index];
            auto target_tokens = tokenize(vocab, item.target, false);
            auto foil_tokens = tokenize(vocab, item.foil, false);
            if (target_tokens.size() != 1 || foil_tokens.size() != 1) {
                throw std::runtime_error("target and foil must each tokenize to one token: " + item.id);
            }
            const llama_token target = target_tokens[0];
            const llama_token foil = foil_tokens[0];
            const association & wrong_item = benchmark[(association_index + 1 + opts.seed % (benchmark.size()-1)) % benchmark.size()];
            auto wrong_tokens = tokenize(vocab, wrong_item.target, false);
            if (wrong_tokens.size() != 1) throw std::runtime_error("wrong target is not one token");

            bank state(opts.rank, model.hparams.n_embd);
            ctx->clear_fast_weights();
            const query_nll target_before = measure_query_nll(ctx, vocab, collector, item, target);
            const query_nll foil_before = measure_query_nll(ctx, vocab, collector, item, foil);
            const double unrelated_before = unrelated_nll(ctx, vocab, collector);
            update_result last_update;
            std::vector<float> previous_q;

            for (int32_t turn = 0; turn < opts.exposures; ++turn) {
                const state_result exposure = evaluate_state(ctx, vocab, collector, item.learn_prompt, target);
                std::vector<float> target_q;
                std::vector<float> wrong_q;
                if (needs_exact_q(opts.rule)) {
                    target_q = exact_target_direction(output_matrix_f32, exposure.logits, target, opts.q_threads);
                    wrong_q = target_q;
                    const float * target_row = output_matrix_f32.row(target);
                    const float * wrong_row = output_matrix_f32.row(wrong_tokens[0]);
                    for (int32_t i = 0; i < state.width; ++i) wrong_q[i] += wrong_row[i] - target_row[i];
                }

                if (!finite_difference_done && (opts.rule == "target_delta" || opts.rule == "fixed_step_clipped")) {
                    finite_difference_check(ctx, vocab, collector, item, target, target_q, output, opts.rank);
                    finite_difference_done = true;
                    install_bank(ctx, state);
                }

                if (is_stabilized_rule(opts.rule)) {
                    double matched_delta_rho = -1;
                    if (opts.rule == "random_incremental" || opts.rule == "shuffled_incremental") {
                        const auto found = control_delta_schedule.find({item.id, turn+1});
                        if (found == control_delta_schedule.end()) {
                            throw std::runtime_error("missing association/turn in delta schedule");
                        }
                        matched_delta_rho = found->second;
                    }
                    last_update = update_bank_stabilized(
                        opts, state, turn, exposure.hidden, target_q, wrong_q, previous_q,
                        exposure.target_probability, association_index, matched_delta_rho);
                } else {
                    last_update = update_bank(
                        opts, state, turn, exposure.hidden, target_q, wrong_q,
                        exposure.target_probability, association_index);
                }
                if (!last_update.feasible) {
                    output << "{\"status\":\"update_infeasible\",\"association\":\"" << json_escape(item.id)
                           << "\",\"rule\":\"" << opts.rule << "\",\"turn\":" << turn+1
                           << ",\"rho\":" << opts.rho << "}\n";
                    output.flush();
                    break;
                }
                install_bank(ctx, state);
                const state_result after_update = evaluate_state(ctx, vocab, collector, item.learn_prompt, target);
                last_update.target_probability_after = after_update.target_probability;
                write_update_row(output, opts, item, turn+1, last_update);
                if (opts.rule == "shuffled_incremental") previous_q = wrong_q;
                else if (!target_q.empty()) previous_q = target_q;

                if (std::find(exposure_checkpoints.begin(), exposure_checkpoints.end(), turn+1) != exposure_checkpoints.end()) {
                    const query_nll target_after = measure_query_nll(ctx, vocab, collector, item, target);
                    const query_nll foil_after = measure_query_nll(ctx, vocab, collector, item, foil);
                    const double unrelated_after = unrelated_nll(ctx, vocab, collector);
                    write_row(output, opts, item, turn+1, last_update,
                        target_before, target_after, foil_before, foil_after, unrelated_before, unrelated_after);
                }
            }
            std::fprintf(stderr, "completed association %zu/%zu id=%s rule=%s rho=%.4f\n",
                association_index+1, benchmark.size(), item.id.c_str(), opts.rule.c_str(), opts.rho);
        }

        llama_free(public_ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "forward-target-plasticity error: %s\n", error.what());
        usage(argv[0]);
        return 1;
    }
}
