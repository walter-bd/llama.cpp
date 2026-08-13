#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int N_TAPS = llama_adapter_attnres_sidecar::N_TAPS;
constexpr int RANK = 4;
constexpr double RHO_MAX = 0.001;

struct options {
    std::string model;
    std::string output_dir;
    int32_t gpu_layers = 99;
    int32_t ctx_size = 512;
    int32_t q_threads = 16;
    uint32_t seed = 15031;
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
        else if (arg == "--q-threads") out.q_threads = std::stoi(value(i));
        else if (arg == "--seed") out.seed = std::stoul(value(i));
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
    std::vector<float> hidden;
    std::vector<float> routed;
    std::vector<float> scores;
    std::vector<float> weights;
    std::array<std::vector<float>, N_TAPS> taps;

    void clear() {
        hidden.clear();
        routed.clear();
        scores.clear();
        weights.clear();
        for (auto & tap : taps) tap.clear();
    }
};

bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & collector = *static_cast<trace_collector *>(user_data);
    if (!collector.enabled) return false;
    const std::string name(tensor->name);
    bool wanted = name == "result_norm" || name == "attnres_routed" ||
        name == "attnres_scores" || name == "attnres_weights";
    int tap = -1;
    if (name.rfind("attnres_tap_", 0) == 0) {
        tap = std::stoi(name.substr(std::string("attnres_tap_").size()));
        wanted = tap >= 0 && tap < N_TAPS;
    }
    if (!wanted) return false;
    if (ask) return true;
    const auto values = copy_last_vector(tensor);
    if (name == "result_norm") collector.hidden = values;
    else if (name == "attnres_routed") collector.routed = values;
    else if (name == "attnres_scores") collector.scores = values;
    else if (name == "attnres_weights") collector.weights = values;
    else collector.taps[tap] = values;
    return true;
}

double l2(const std::vector<float> & values) {
    double total = 0;
    for (float value : values) total += (double) value*value;
    return std::sqrt(total);
}

double dot(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("dot size mismatch");
    double total = 0;
    for (size_t i = 0; i < a.size(); ++i) total += (double) a[i]*b[i];
    return total;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    const double denom = l2(a)*l2(b);
    return denom > 0 ? dot(a, b)/denom : 0;
}

std::vector<float> normalized(const std::vector<float> & values) {
    std::vector<float> out(values);
    const double norm = l2(out);
    if (norm <= 0) return out;
    for (float & value : out) value /= norm;
    return out;
}

double logsumexp(const float * logits, int32_t size) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < size; ++i) maximum = std::max(maximum, logits[i]);
    double total = 0;
    for (int32_t i = 0; i < size; ++i) total += std::exp((double) logits[i] - maximum);
    return maximum + std::log(total);
}

struct eval_result {
    std::vector<float> hidden;
    std::vector<float> routed;
    std::vector<float> logits;
    std::vector<float> scores;
    std::vector<float> weights;
    std::array<std::vector<float>, N_TAPS> taps;
    double probability = 0;
    double nll = 0;
    double margin = 0;
    int32_t rank = 0;
    llama_token greedy = -1;
};

eval_result evaluate(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const std::string & prompt,
        llama_token target) {
    llama_memory_clear(llama_get_memory(ctx), true);
    collector.clear();
    collector.enabled = true;
    auto tokens = tokenize(vocab, prompt, true);
    if (tokens.empty() || llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
        throw std::runtime_error("prompt decode failed");
    }
    llama_synchronize(ctx);
    collector.enabled = false;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * raw = llama_get_logits_ith(ctx, -1);
    if (!raw || collector.hidden.empty()) throw std::runtime_error("missing logits or final hidden");
    eval_result out;
    out.hidden = collector.hidden;
    out.routed = collector.routed;
    out.scores = collector.scores;
    out.weights = collector.weights;
    out.taps = collector.taps;
    out.logits.assign(raw, raw + n_vocab);
    const double z = logsumexp(raw, n_vocab);
    out.nll = z - raw[target];
    out.probability = std::exp(raw[target] - z);
    out.greedy = 0;
    float best_other = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        if (raw[token] > raw[out.greedy]) out.greedy = token;
        if (token != target) best_other = std::max(best_other, raw[token]);
        if (raw[token] > raw[target]) ++out.rank;
    }
    out.rank += 1;
    out.margin = raw[target] - best_other;
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
    if (!traits || !traits->to_float) throw std::runtime_error("output tensor cannot be dequantized");
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

std::vector<float> exact_target_direction(
        const output_matrix & output,
        const std::vector<float> & logits,
        llama_token target,
        int32_t requested_threads) {
    const double z = logsumexp(logits.data(), logits.size());
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) probs[i] = std::exp(logits[i] - z);
    const int32_t n_threads = std::max(1, std::min<int32_t>(requested_threads, output.rows));
    std::vector<std::vector<float>> partial(n_threads, std::vector<float>(output.width));
    std::vector<std::thread> workers;
    for (int32_t thread = 0; thread < n_threads; ++thread) {
        workers.emplace_back([&, thread]() {
            const int64_t begin = output.rows*thread/n_threads;
            const int64_t end = output.rows*(thread + 1)/n_threads;
            for (int64_t row = begin; row < end; ++row) {
                const float * weight = output.row(row);
                for (int64_t column = 0; column < output.width; ++column) {
                    partial[thread][column] += probs[row]*weight[column];
                }
            }
        });
    }
    for (auto & worker : workers) worker.join();
    std::vector<float> out(output.width);
    const float * target_row = output.row(target);
    for (int64_t column = 0; column < output.width; ++column) {
        double expected = 0;
        for (int32_t thread = 0; thread < n_threads; ++thread) expected += partial[thread][column];
        out[column] = target_row[column] - expected;
    }
    return out;
}

struct bank {
    int32_t width;
    std::vector<float> u;
    std::vector<float> v;
    int32_t used = 0;

    explicit bank(int32_t width) : width(width), u((size_t) RANK*width), v((size_t) RANK*width) {}

    std::vector<float> apply(const std::vector<float> & key) const {
        std::vector<float> out(width);
        for (int32_t slot = 0; slot < RANK; ++slot) {
            double coefficient = 0;
            for (int32_t i = 0; i < width; ++i) coefficient += v[(size_t) width*slot + i]*key[i];
            for (int32_t j = 0; j < width; ++j) out[j] += u[(size_t) RANK*j + slot]*coefficient;
        }
        return out;
    }

    std::vector<float> slot_key(int32_t slot) const {
        std::vector<float> out(width);
        for (int32_t i = 0; i < width; ++i) out[i] = v[(size_t) width*slot + i];
        return out;
    }
};

struct route {
    std::string name;
    llama_attnres_route_mode mode = LLAMA_ATTNRES_ROUTE_QUERY;
    int32_t tap = -1;
    std::vector<float> query;
};

void install_route(llama_context * ctx, const route & selected, const bank & state) {
    if (!ctx->init_attnres_sidecar(RANK, selected.mode, selected.tap, true)) {
        throw std::runtime_error("sidecar initialization failed");
    }
    if (selected.mode == LLAMA_ATTNRES_ROUTE_QUERY) {
        if ((int32_t) selected.query.size() != state.width ||
                !ctx->set_attnres_sidecar_query(selected.query.data(), selected.query.size())) {
            throw std::runtime_error("sidecar query installation failed");
        }
    }
    if (!ctx->set_attnres_sidecar(state.u.data(), state.u.size(), state.v.data(), state.v.size())) {
        throw std::runtime_error("sidecar bank installation failed");
    }
}

void install_final_bank(llama_context * ctx, const bank & state) {
    if (!ctx->set_fast_weights(state.u.data(), state.u.size(), state.v.data(), state.v.size())) {
        throw std::runtime_error("final-only bank installation failed");
    }
}

struct update_result {
    bank state;
    double requested_rho = 0;
    double actual_rho = 0;
    double final_rho = 0;
    bool projected = false;

    explicit update_result(int32_t width) : state(width) {}
};

update_result make_update(
        const bank & original,
        const std::vector<float> & key,
        const std::vector<float> & hidden,
        const std::vector<float> & direction,
        double target_probability) {
    if (original.used >= RANK) throw std::runtime_error("fast rank exhausted");
    update_result out(original.width);
    out.state = original;
    const auto key_hat = normalized(key);
    const auto q_hat = normalized(direction);
    const auto current = original.apply(key);
    const double h_norm = l2(hidden);
    const double requested_norm = RHO_MAX*h_norm*(1.0 - target_probability);
    std::vector<float> delta(original.width);
    std::vector<float> candidate(original.width);
    for (int32_t i = 0; i < original.width; ++i) {
        delta[i] = requested_norm*q_hat[i];
        candidate[i] = current[i] + delta[i];
    }
    const double bound = RHO_MAX*h_norm;
    if (l2(candidate) > bound) {
        out.projected = true;
        const double scale = bound/l2(candidate);
        for (int32_t i = 0; i < original.width; ++i) {
            const double projected = candidate[i]*scale;
            delta[i] = projected - current[i];
            candidate[i] = projected;
        }
    }
    const double coefficient = dot(key_hat, key);
    const int32_t slot = original.used;
    for (int32_t i = 0; i < original.width; ++i) {
        out.state.v[(size_t) original.width*slot + i] = key_hat[i];
        out.state.u[(size_t) RANK*i + slot] = coefficient > 0 ? delta[i]/coefficient : 0;
    }
    out.state.used = original.used + 1;
    out.requested_rho = requested_norm/h_norm;
    out.actual_rho = l2(delta)/h_norm;
    out.final_rho = l2(candidate)/h_norm;
    return out;
}

std::vector<float> deterministic_query(int32_t width, uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal;
    std::vector<float> out(width);
    for (float & value : out) value = normal(generator);
    return normalized(out);
}

std::vector<route> candidate_routes(int32_t width, uint32_t seed, const std::vector<float> & correction_q) {
    std::vector<route> out;
    out.push_back({"uniform", LLAMA_ATTNRES_ROUTE_QUERY, -1, std::vector<float>(width)});
    for (int32_t tap = 0; tap < N_TAPS; ++tap) {
        out.push_back({"tap_" + std::to_string(tap), LLAMA_ATTNRES_ROUTE_ONE_HOT, tap, {}});
    }
    out.push_back({"final_representation_query", LLAMA_ATTNRES_ROUTE_FINAL_QUERY, -1, {}});
    out.push_back({"fixed_seed_query", LLAMA_ATTNRES_ROUTE_QUERY, -1, deterministic_query(width, seed)});
    out.push_back({"correction_direction_query", LLAMA_ATTNRES_ROUTE_QUERY, -1, normalized(correction_q)});
    return out;
}

double attention_entropy(const std::vector<float> & weights) {
    double out = 0;
    for (float value : weights) if (value > 0) out -= value*std::log(value);
    return out;
}

void write_route_diagnostic(
        std::ofstream & output,
        const std::string & phase,
        const std::string & item,
        const route & selected,
        const eval_result & state) {
    output << std::setprecision(17) << "{\"phase\":\"" << phase << "\",\"item\":\""
           << json_escape(item) << "\",\"route\":\"" << selected.name << "\",\"scores\":[";
    for (size_t i = 0; i < state.scores.size(); ++i) output << (i ? "," : "") << state.scores[i];
    output << "],\"alpha\":[";
    for (size_t i = 0; i < state.weights.size(); ++i) output << (i ? "," : "") << state.weights[i];
    size_t winner = state.weights.empty() ? 0 : std::distance(state.weights.begin(),
        std::max_element(state.weights.begin(), state.weights.end()));
    output << "],\"entropy\":" << attention_entropy(state.weights)
           << ",\"max_alpha\":" << (state.weights.empty() ? 1.0 : state.weights[winner])
           << ",\"winning_tap\":" << winner << ",\"z_norm\":" << l2(state.routed)
           << ",\"pairwise_key_cosine\":[";
    bool first = true;
    for (int i = 0; i < N_TAPS; ++i) for (int j = i + 1; j < N_TAPS; ++j) {
        output << (first ? "" : ",") << cosine(state.taps[i], state.taps[j]);
        first = false;
    }
    output << "]}\n";
}

struct item {
    std::string id;
    std::string prompt;
    llama_token target;
    std::string target_text;
};

std::vector<std::pair<llama_token, std::string>> choose_single_tokens(const llama_vocab * vocab, size_t count) {
    const std::vector<std::string> candidates = {
        " Q", " X", " J", " V", " K", " Z", " R", " T", " P", " G", " W", " F",
        " A", " B", " C", " D", " E", " H", " L", " M", " N", " S", " U", " Y",
    };
    std::vector<std::pair<llama_token, std::string>> out;
    for (const auto & text : candidates) {
        const auto tokens = tokenize(vocab, text, false);
        if (tokens.size() == 1 && std::none_of(out.begin(), out.end(), [&](const auto & p) { return p.first == tokens[0]; })) {
            out.emplace_back(tokens[0], text);
        }
        if (out.size() == count) break;
    }
    if (out.size() < count) throw std::runtime_error("insufficient verified single-token labels");
    return out;
}

std::vector<item> association_items(const std::vector<std::pair<llama_token, std::string>> & labels) {
    const std::array<const char *, 4> names = {"Velkor", "Narev", "Soltik", "Brevan"};
    std::vector<item> out;
    for (size_t i = 0; i < names.size(); ++i) {
        out.push_back({"association_" + std::to_string(i),
            "An arbitrary marker was assigned independently. The marker for " + std::string(names[i]) + " is", labels[i].first, labels[i].second});
    }
    return out;
}

std::vector<item> reflect_items(
        const std::vector<std::pair<llama_token, std::string>> & labels,
        uint32_t seed) {
    const std::array<const char *, 8> relations = {"zor", "mek", "pul", "dax", "niv", "suv", "rik", "tav"};
    const std::array<const char *, 16> entities = {
        "Veln", "Koru", "Mira", "Dast", "Pelu", "Rin", "Sova", "Tern",
        "Bex", "Luma", "Noki", "Faro", "Grev", "Hali", "Jora", "Cint",
    };
    std::vector<item> out;
    for (int index = 0; index < 28; ++index) {
        // Cover varied relation pairs without making the affine target rule
        // collapse to one label along the generated sequence.
        const int left = (index + seed)%8;
        const int right = (2*(index%8) + index/8 + seed/8)%8;
        const int target = (3*left + 5*right + seed)%8;
        const std::string prompt =
            "Relations are arbitrary symbols. " + std::string(entities[index%16]) + " has relation " + relations[left] +
            " to " + entities[(index + 3)%16] + ". " + entities[(index + 3)%16] + " has relation " + relations[right] +
            " to " + entities[(index + 7)%16] + ". Return only the arbitrary relation label for " +
            entities[index%16] + " to " + entities[(index + 7)%16] + ":";
        out.push_back({(index < 4 ? "train_" : "heldout_") + std::to_string(index), prompt,
            labels[target].first, labels[target].second});
    }
    return out;
}

struct protective_reference {
    std::string prompt;
    llama_token target;
    std::vector<float> logits;
};

double kl_divergence(const std::vector<float> & current, const std::vector<float> & baseline) {
    const double z_current = logsumexp(current.data(), current.size());
    const double z_base = logsumexp(baseline.data(), baseline.size());
    double kl = 0;
    for (size_t i = 0; i < current.size(); ++i) {
        const double p = std::exp(current[i] - z_current);
        if (p > 0) kl += p*((current[i] - z_current) - (baseline[i] - z_base));
    }
    return kl;
}

struct route_probe {
    route selected;
    double improvement = -std::numeric_limits<double>::infinity();
};

route_probe select_route(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const item & training,
        const bank & current,
        uint32_t seed,
        int32_t q_threads,
        std::ofstream & probes) {
    route uniform{"uniform", LLAMA_ATTNRES_ROUTE_QUERY, -1, std::vector<float>(current.width)};
    install_route(ctx, uniform, current);
    ctx->clear_fast_weights();
    const auto neutral = evaluate(ctx, vocab, collector, training.prompt, training.target);
    const auto q = exact_target_direction(output, neutral.logits, training.target, q_threads);
    route_probe best;
    for (const auto & candidate : candidate_routes(current.width, seed, q)) {
        install_route(ctx, candidate, current);
        const auto before = evaluate(ctx, vocab, collector, training.prompt, training.target);
        const auto direction = exact_target_direction(output, before.logits, training.target, q_threads);
        const auto updated = make_update(current, before.routed, before.hidden, direction, before.probability);
        install_route(ctx, candidate, updated.state);
        const auto after = evaluate(ctx, vocab, collector, training.prompt, training.target);
        const double improvement = before.nll - after.nll;
        probes << std::setprecision(17) << "{\"item\":\"" << training.id << "\",\"route\":\""
               << candidate.name << "\",\"nll_before\":" << before.nll << ",\"nll_after\":" << after.nll
               << ",\"improvement\":" << improvement << ",\"requested_rho\":" << updated.requested_rho
               << ",\"actual_rho\":" << updated.actual_rho << ",\"final_rho\":" << updated.final_rho
               << ",\"projected\":" << (updated.projected ? "true" : "false") << "}\n";
        if (improvement > best.improvement) best = {candidate, improvement};
    }
    install_route(ctx, best.selected, current);
    return best;
}

void write_measurement(
        std::ofstream & output,
        const std::string & phase,
        const std::string & condition,
        int checkpoint,
        const item & current,
        const eval_result & state) {
    output << std::setprecision(17) << "{\"phase\":\"" << phase << "\",\"condition\":\""
           << condition << "\",\"checkpoint\":" << checkpoint << ",\"item\":\"" << current.id
           << "\",\"target_token\":" << current.target << ",\"target_text\":\"" << json_escape(current.target_text)
           << "\",\"probability\":" << state.probability << ",\"nll\":" << state.nll
           << ",\"rank\":" << state.rank << ",\"margin\":" << state.margin
           << ",\"greedy_token\":" << state.greedy << ",\"greedy_recall\":"
           << (state.greedy == current.target ? "true" : "false") << "}\n";
}

enum class condition_kind { final_only, sidecar };

struct condition {
    std::string name;
    condition_kind kind = condition_kind::sidecar;
    route routing;
    bool validator_select = false;
    bool shuffled = false;
};

void run_phase1(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const options & opts,
        const std::vector<item> & associations,
        const std::filesystem::path & root) {
    std::ofstream results(root/"phase1_association_results.jsonl", std::ios::trunc);
    std::ofstream probes(root/"routing_probes.jsonl", std::ios::trunc);
    std::ofstream geometry(root/"sidecar_geometry.jsonl", std::ios::trunc);
    for (const auto & association : associations) {
        bank empty(output.width);
        const auto winner = select_route(ctx, vocab, collector, output, association, empty,
            opts.seed, opts.q_threads, probes);
        install_route(ctx, winner.selected, empty);
        const auto before = evaluate(ctx, vocab, collector, association.prompt, association.target);
        const auto q = exact_target_direction(output, before.logits, association.target, opts.q_threads);
        const auto update = make_update(empty, before.routed, before.hidden, q, before.probability);
        install_route(ctx, winner.selected, update.state);
        const auto after = evaluate(ctx, vocab, collector, association.prompt, association.target);
        write_measurement(results, "phase1", "validator_selected", 0, association, before);
        write_measurement(results, "phase1", "validator_selected", 1, association, after);
        write_route_diagnostic(geometry, "phase1", association.id, winner.selected, after);
    }
}

void run_condition(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const options & opts,
        const std::vector<item> & tasks,
        const std::vector<protective_reference> & protective,
        condition spec,
        const std::filesystem::path & root,
        std::ofstream & probes,
        std::ofstream & geometry) {
    std::ofstream heldout(root/"phase2_reflect_results.jsonl", std::ios::app);
    std::ofstream protection(root/"protective_results.jsonl", std::ios::app);
    bank state(output.width);
    ctx->clear_fast_weights();
    if (spec.validator_select) {
        const auto winner = select_route(ctx, vocab, collector, output, tasks[0], state,
            opts.seed + 1, opts.q_threads, probes);
        spec.routing = winner.selected;
    }
    if (spec.kind == condition_kind::sidecar) install_route(ctx, spec.routing, state);
    else {
        route uniform{"uniform", LLAMA_ATTNRES_ROUTE_QUERY, -1, std::vector<float>(output.width)};
        install_route(ctx, uniform, bank(output.width));
        install_final_bank(ctx, state);
    }

    auto eval_checkpoint = [&](int checkpoint) {
        int correct = 0;
        for (size_t index = 4; index < tasks.size(); ++index) {
            const auto current = evaluate(ctx, vocab, collector, tasks[index].prompt, tasks[index].target);
            write_measurement(heldout, "phase2", spec.name, checkpoint, tasks[index], current);
            correct += current.greedy == tasks[index].target;
        }
        double mean_kl = 0;
        double mean_nll = 0;
        for (const auto & reference : protective) {
            const auto current = evaluate(ctx, vocab, collector, reference.prompt, reference.target);
            mean_kl += kl_divergence(current.logits, reference.logits);
            mean_nll += current.nll;
        }
        mean_kl /= protective.size();
        mean_nll /= protective.size();
        protection << std::setprecision(17) << "{\"condition\":\"" << spec.name
                   << "\",\"checkpoint\":" << checkpoint << ",\"heldout_correct\":" << correct
                   << ",\"heldout_total\":" << tasks.size() - 4 << ",\"mean_kl\":" << mean_kl
                   << ",\"mean_nll\":" << mean_nll << "}\n";
    };

    eval_checkpoint(0);
    for (int correction = 0; correction < 4; ++correction) {
        if (spec.kind == condition_kind::sidecar) install_route(ctx, spec.routing, state);
        else install_final_bank(ctx, state);
        const auto before = evaluate(ctx, vocab, collector, tasks[correction].prompt, tasks[correction].target);
        const llama_token direction_target = spec.shuffled ? tasks[(correction + 1)%4].target : tasks[correction].target;
        const auto direction = exact_target_direction(output, before.logits, direction_target, opts.q_threads);
        const auto & key = spec.kind == condition_kind::sidecar ? before.routed : before.hidden;
        const auto update = make_update(state, key, before.hidden, direction, before.probability);
        bank candidate = update.state;
        if (spec.kind == condition_kind::sidecar) install_route(ctx, spec.routing, candidate);
        else install_final_bank(ctx, candidate);
        const auto after = evaluate(ctx, vocab, collector, tasks[correction].prompt, tasks[correction].target);
        const bool accepted = after.nll < before.nll;
        if (accepted) state = candidate;
        else {
            if (spec.kind == condition_kind::sidecar) install_route(ctx, spec.routing, state);
            else install_final_bank(ctx, state);
        }
        probes << std::setprecision(17) << "{\"item\":\"" << tasks[correction].id
               << "\",\"condition\":\"" << spec.name << "\",\"event\":" << correction + 1
               << ",\"route\":\"" << spec.routing.name << "\",\"accepted\":" << (accepted ? "true" : "false")
               << ",\"nll_before\":" << before.nll << ",\"nll_after\":" << after.nll
               << ",\"requested_rho\":" << update.requested_rho << ",\"actual_rho\":" << update.actual_rho
               << ",\"final_rho\":" << update.final_rho << ",\"projected\":"
               << (update.projected ? "true" : "false") << "}\n";
        if (spec.kind == condition_kind::sidecar) {
            const auto installed = evaluate(ctx, vocab, collector, tasks[correction].prompt, tasks[correction].target);
            write_route_diagnostic(geometry, "phase2", spec.name + ":" + tasks[correction].id, spec.routing, installed);
        }
        eval_checkpoint(correction + 1);
    }
}

void write_serialization(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const item & probe,
        const route & selected,
        const bank & state,
        const std::filesystem::path & root) {
    install_route(ctx, selected, state);
    const auto before = evaluate(ctx, vocab, collector, probe.prompt, probe.target);
    const auto path = root/"sidecar_state.bin";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        const int32_t name_size = selected.name.size();
        output.write(reinterpret_cast<const char *>(&name_size), sizeof(name_size));
        output.write(selected.name.data(), name_size);
        output.write(reinterpret_cast<const char *>(&selected.mode), sizeof(selected.mode));
        output.write(reinterpret_cast<const char *>(&selected.tap), sizeof(selected.tap));
        const int32_t query_size = selected.query.size();
        output.write(reinterpret_cast<const char *>(&query_size), sizeof(query_size));
        output.write(reinterpret_cast<const char *>(selected.query.data()), query_size*sizeof(float));
        output.write(reinterpret_cast<const char *>(&state.used), sizeof(state.used));
        output.write(reinterpret_cast<const char *>(state.u.data()), state.u.size()*sizeof(float));
        output.write(reinterpret_cast<const char *>(state.v.data()), state.v.size()*sizeof(float));
    }
    bank loaded(state.width);
    route loaded_route;
    {
        std::ifstream input(path, std::ios::binary);
        int32_t name_size = 0;
        input.read(reinterpret_cast<char *>(&name_size), sizeof(name_size));
        loaded_route.name.resize(name_size);
        input.read(loaded_route.name.data(), name_size);
        input.read(reinterpret_cast<char *>(&loaded_route.mode), sizeof(loaded_route.mode));
        input.read(reinterpret_cast<char *>(&loaded_route.tap), sizeof(loaded_route.tap));
        int32_t query_size = 0;
        input.read(reinterpret_cast<char *>(&query_size), sizeof(query_size));
        loaded_route.query.resize(query_size);
        input.read(reinterpret_cast<char *>(loaded_route.query.data()), query_size*sizeof(float));
        input.read(reinterpret_cast<char *>(&loaded.used), sizeof(loaded.used));
        input.read(reinterpret_cast<char *>(loaded.u.data()), loaded.u.size()*sizeof(float));
        input.read(reinterpret_cast<char *>(loaded.v.data()), loaded.v.size()*sizeof(float));
        if (!input) throw std::runtime_error("sidecar state reload failed");
    }
    install_route(ctx, loaded_route, loaded);
    const auto after = evaluate(ctx, vocab, collector, probe.prompt, probe.target);
    double max_diff = 0;
    for (size_t i = 0; i < before.logits.size(); ++i) max_diff = std::max(max_diff,
        std::abs((double) before.logits[i] - after.logits[i]));
    std::ofstream output(root/"serialization_validation.json", std::ios::trunc);
    output << std::setprecision(17) << "{\"max_logit_abs_diff\":" << max_diff
           << ",\"greedy_equal\":" << (before.greedy == after.greedy ? "true" : "false")
           << ",\"rank_equal\":" << (state.used == loaded.used ? "true" : "false") << "}\n";
}

} // namespace

int legacy_main_unused(int argc, char ** argv) {
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
        if (model.hparams.n_embd != 2560 || model.hparams.n_layer() != 32) {
            throw std::runtime_error("PoC requires Qwen3.5-4B dimensions 32x2560");
        }

        trace_collector collector;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.ctx_size;
        context_params.n_batch = opts.ctx_size;
        context_params.n_ubatch = std::min(opts.ctx_size, 512);
        context_params.n_seq_max = 1;
        context_params.cb_eval = eval_callback;
        context_params.cb_eval_user_data = &collector;
        llama_context * ctx = llama_init_from_model(public_model, context_params);
        if (!ctx) throw std::runtime_error("context creation failed");
        if (!ctx->init_fast_weights(RANK)) throw std::runtime_error("final-only adapter initialization failed");
        const llama_vocab * vocab = llama_model_get_vocab(public_model);
        const auto labels = choose_single_tokens(vocab, 8);
        const auto associations = association_items(labels);
        const auto tasks = reflect_items(labels, opts.seed%8);

        const llama_token neutral_target = associations[0].target;
        ctx->clear_fast_weights();
        const auto baseline = evaluate(ctx, vocab, collector, associations[0].prompt, neutral_target);
        bank zero(model.hparams.n_embd);
        route uniform{"uniform", LLAMA_ATTNRES_ROUTE_QUERY, -1, std::vector<float>(model.hparams.n_embd)};
        install_route(ctx, uniform, zero);
        const auto enabled = evaluate(ctx, vocab, collector, associations[0].prompt, neutral_target);
        double neutrality_diff = 0;
        for (size_t i = 0; i < baseline.logits.size(); ++i) neutrality_diff = std::max(neutrality_diff,
            std::abs((double) baseline.logits[i] - enabled.logits[i]));
        {
            std::ofstream architecture(root/"architecture.json", std::ios::trunc);
            architecture << std::setprecision(17)
                << "{\"model_width\":2560,\"blocks\":32,\"depth_taps\":[4,8,12,16,20,24,28,32]"
                << ",\"rank\":4,\"rho_max\":0.001,\"score_scaling\":\"1/sqrt(2560)\""
                << ",\"persistent_bytes\":92160,\"transient_depth_bytes\":81920"
                << ",\"zero_state_max_logit_abs_diff\":" << neutrality_diff << "}\n";
        }

        std::fprintf(stderr, "loading exact LM head rows=%lld width=%lld\n",
            (long long) model.output->ne[1], (long long) model.output->ne[0]);
        const auto output = load_output_matrix(model);
        run_phase1(ctx, vocab, collector, output, opts, associations, root);

        std::vector<protective_reference> protective;
        const std::array<std::string, 8> protective_prompts = {
            "The capital of France is", "Two plus two equals", "Water freezes at",
            "The opposite of hot is", "Complete the sequence 1, 2, 3,", "A triangle has",
            "The first month of the year is", "The chemical symbol for oxygen is",
        };
        install_route(ctx, uniform, zero);
        for (const auto & prompt : protective_prompts) {
            const auto first = evaluate(ctx, vocab, collector, prompt, 0);
            const auto base = evaluate(ctx, vocab, collector, prompt, first.greedy);
            protective.push_back({prompt, first.greedy, base.logits});
        }

        std::ofstream probes(root/"routing_probes.jsonl", std::ios::app);
        std::ofstream geometry(root/"sidecar_geometry.jsonl", std::ios::app);
        const auto fixed = deterministic_query(output.width, opts.seed);
        const std::vector<condition> conditions = {
            {"final_only", condition_kind::final_only, {}, false, false},
            {"uniform", condition_kind::sidecar, uniform, false, false},
            {"final_representation_query", condition_kind::sidecar,
                {"final_representation_query", LLAMA_ATTNRES_ROUTE_FINAL_QUERY, -1, {}}, false, false},
            {"fixed_seed_query", condition_kind::sidecar,
                {"fixed_seed_query", LLAMA_ATTNRES_ROUTE_QUERY, -1, fixed}, false, false},
            {"validator_selected", condition_kind::sidecar, uniform, true, false},
            {"random_depth", condition_kind::sidecar,
                {"random_depth", LLAMA_ATTNRES_ROUTE_ONE_HOT, (int32_t) (opts.seed%N_TAPS), {}}, false, false},
            {"shuffled_correction", condition_kind::sidecar, uniform, true, true},
        };
        for (const auto & current : conditions) {
            std::fprintf(stderr, "phase2 condition=%s\n", current.name.c_str());
            run_condition(ctx, vocab, collector, output, opts, tasks, protective, current, root, probes, geometry);
        }

        // Serialization uses a deterministic non-empty rank-one sidecar state.
        install_route(ctx, uniform, zero);
        const auto serial_before = evaluate(ctx, vocab, collector, associations[0].prompt, associations[0].target);
        const auto serial_q = exact_target_direction(output, serial_before.logits, associations[0].target, opts.q_threads);
        const auto serial_update = make_update(zero, serial_before.routed, serial_before.hidden, serial_q, serial_before.probability);
        write_serialization(ctx, vocab, collector, associations[0], uniform, serial_update.state, root);

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17) << "{\"wall_seconds\":" << elapsed
                << ",\"phase1_associations\":4,\"phase2_training\":4,\"phase2_heldout\":24"
                << ",\"route_candidates\":12,\"conditions\":7}\n";
        std::ofstream manifest(root/"manifest.json", std::ios::trunc);
        manifest << "{\"experiment\":\"qwen35_attnres_sidecar_poc\",\"seed\":" << opts.seed
                 << ",\"base_model_frozen\":true,\"query_plastic\":false,\"backpropagation\":false}\n";
        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}

namespace {

struct diagnostic_item {
    std::string id;
    std::string family;
    int32_t family_index = 0;
    std::string prompt;
    llama_token target = -1;
    std::string target_text;
};

struct token_metrics {
    double probability = 0;
    double nll = 0;
    double margin = 0;
    int32_t rank = 0;
    llama_token greedy = -1;
};

struct feature_bundle {
    std::map<std::string, std::vector<float>> values;
    std::vector<float> attn_scores;
    std::vector<float> attn_alpha;
};

struct geometry_stats {
    std::vector<double> same;
    std::vector<double> different;
    std::vector<double> protective;
};

uint64_t diagnostic_eval_calls = 0;

eval_result diagnostic_evaluate(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const std::string & prompt,
        llama_token target) {
    ++diagnostic_eval_calls;
    return evaluate(ctx, vocab, collector, prompt, target);
}

token_metrics metrics_from_logits(const std::vector<float> & logits, llama_token target) {
    token_metrics out;
    const double z = logsumexp(logits.data(), logits.size());
    out.nll = z - logits[target];
    out.probability = std::exp(logits[target] - z);
    out.greedy = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
    float best_other = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < (int32_t) logits.size(); ++token) {
        if (token != target) best_other = std::max(best_other, logits[token]);
        if (logits[token] > logits[target]) ++out.rank;
    }
    ++out.rank;
    out.margin = logits[target] - best_other;
    return out;
}

std::vector<float> residual_logits(
        const output_matrix & output,
        const std::vector<float> & baseline,
        const std::vector<float> & residual) {
    if ((int64_t) residual.size() != output.width) throw std::runtime_error("residual width mismatch");
    std::vector<float> logits(baseline);
    for (int64_t row = 0; row < output.rows; ++row) {
        const float * weights = output.row(row);
        double delta = 0.0;
        for (int64_t col = 0; col < output.width; ++col) {
            delta += (double) weights[col] * residual[col];
        }
        logits[row] += delta;
    }
    return logits;
}

std::vector<float> output_projection(
        const output_matrix & output,
        const std::vector<float> & direction,
        int32_t requested_threads = 16) {
    if ((int64_t) direction.size() != output.width) throw std::runtime_error("projection width mismatch");
    std::vector<float> projected(output.rows);
    const int32_t n_threads = std::max(1, std::min<int32_t>(requested_threads, output.rows));
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int32_t thread = 0; thread < n_threads; ++thread) {
        workers.emplace_back([&, thread]() {
            const int64_t begin = output.rows*thread/n_threads;
            const int64_t end = output.rows*(thread + 1)/n_threads;
            for (int64_t row = begin; row < end; ++row) {
                const float * weights = output.row(row);
                double value = 0.0;
                for (int64_t col = 0; col < output.width; ++col) {
                    value += (double) weights[col] * direction[col];
                }
                projected[row] = value;
            }
        });
    }
    for (auto & worker : workers) worker.join();
    return projected;
}

std::vector<float> output_row_difference(
        const output_matrix & output,
        llama_token positive,
        llama_token negative) {
    std::vector<float> out(output.width);
    const float * pos = output.row(positive);
    const float * neg = output.row(negative);
    for (int64_t i = 0; i < output.width; ++i) out[i] = pos[i] - neg[i];
    return out;
}

std::vector<float> scaled_vector(const std::vector<float> & direction, double norm) {
    auto out = normalized(direction);
    for (float & value : out) value *= norm;
    return out;
}

double mean(const std::vector<double> & values) {
    if (values.empty()) return 0;
    return std::accumulate(values.begin(), values.end(), 0.0)/values.size();
}

double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size()/2;
    return values.size()%2 ? values[middle] : 0.5*(values[middle - 1] + values[middle]);
}

double quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const double position = q*(values.size() - 1);
    const size_t lower = std::floor(position);
    const size_t upper = std::ceil(position);
    const double weight = position - lower;
    return values[lower]*(1.0 - weight) + values[upper]*weight;
}

double standard_deviation(const std::vector<double> & values) {
    if (values.size() < 2) return 0;
    const double center = mean(values);
    double total = 0;
    for (double value : values) total += (value - center)*(value - center);
    return std::sqrt(total/(values.size() - 1));
}

double roc_auc(const std::vector<double> & positive, const std::vector<double> & negative) {
    if (positive.empty() || negative.empty()) return 0.5;
    double wins = 0;
    for (double p : positive) for (double n : negative) {
        if (p > n) wins += 1;
        else if (p == n) wins += 0.5;
    }
    return wins/(positive.size()*negative.size());
}

std::vector<float> rms_normalized(const std::vector<float> & values) {
    if (values.empty()) return {};
    double square = 0;
    for (float value : values) square += (double) value*value;
    const double scale = 1.0/std::sqrt(square/values.size() + 1e-6);
    std::vector<float> out(values);
    for (float & value : out) value *= scale;
    return out;
}

feature_bundle make_features(const eval_result & state, const std::vector<float> & query) {
    feature_bundle out;
    if ((int32_t) query.size() != (int32_t) state.taps[0].size()) {
        throw std::runtime_error("query/tap width mismatch");
    }
    const int32_t width = query.size();
    std::vector<float> uniform(width);
    for (int32_t tap = 0; tap < N_TAPS; ++tap) {
        if ((int32_t) state.taps[tap].size() != width) throw std::runtime_error("missing depth tap");
        out.values["tap_" + std::to_string((tap + 1)*4)] = normalized(state.taps[tap]);
        for (int32_t i = 0; i < width; ++i) uniform[i] += state.taps[tap][i]/N_TAPS;
        const auto key = rms_normalized(state.taps[tap]);
        out.attn_scores.push_back(dot(query, key)/std::sqrt((double) width));
    }
    out.values["final_only"] = normalized(state.taps[N_TAPS - 1]);
    out.values["uniform"] = normalized(uniform);
    const double maximum = *std::max_element(out.attn_scores.begin(), out.attn_scores.end());
    double denominator = 0;
    for (double score : out.attn_scores) denominator += std::exp(score - maximum);
    std::vector<float> routed(width);
    for (int32_t tap = 0; tap < N_TAPS; ++tap) {
        const double alpha = std::exp(out.attn_scores[tap] - maximum)/denominator;
        out.attn_alpha.push_back(alpha);
        for (int32_t i = 0; i < width; ++i) routed[i] += alpha*state.taps[tap][i];
    }
    out.values["fixed_attnres"] = normalized(routed);
    return out;
}

std::vector<diagnostic_item> make_rule_items(
        const std::vector<std::pair<llama_token, std::string>> & labels) {
    const std::array<std::array<std::pair<const char *, const char *>, 4>, 3> pairs = {{
        {{{"zor", "mek"}, {"pul", "dax"}, {"niv", "suv"}, {"rik", "tav"}}},
        {{{"mek", "pul"}, {"dax", "niv"}, {"suv", "rik"}, {"tav", "zor"}}},
        {{{"zor", "dax"}, {"mek", "niv"}, {"pul", "suv"}, {"rik", "zor"}}},
    }};
    const std::array<const char *, 24> entities = {
        "Veln", "Koru", "Mira", "Dast", "Pelu", "Rin", "Sova", "Tern",
        "Bex", "Luma", "Noki", "Faro", "Grev", "Hali", "Jora", "Cint",
        "Aven", "Boru", "Cira", "Demu", "Eris", "Fenn", "Galo", "Hira",
    };
    std::vector<diagnostic_item> out;
    for (int family = 0; family < 3; ++family) {
        for (int index = 0; index < 8; ++index) {
            const auto relation = pairs[family][index%4];
            const int entity = family*8 + index;
            const std::string prompt =
                "Synthetic relations have no ordinary meaning. " + std::string(entities[entity]) +
                " has relation " + relation.first + " to " + entities[(entity + 5)%entities.size()] +
                ". " + entities[(entity + 5)%entities.size()] + " has relation " + relation.second +
                " to " + entities[(entity + 11)%entities.size()] +
                ". Under this episode's hidden composition rule, return only the resulting marker:";
            out.push_back({
                std::string(1, 'A' + family) + std::to_string(index + 1),
                std::string(1, 'A' + family),
                index,
                prompt,
                labels[family].first,
                labels[family].second,
            });
        }
    }
    return out;
}

std::vector<std::string> protective_prompts() {
    return {
        "The capital of France is",
        "Two plus two equals",
        "Water freezes at",
        "The opposite of hot is",
        "Complete the sequence 1, 2, 3,",
        "A triangle has",
        "The first month of the year is",
        "The chemical symbol for oxygen is",
    };
}

void write_float_array(std::ostream & out, const std::vector<float> & values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << "]";
}

std::pair<double, double> stage_a(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const std::vector<std::pair<llama_token, std::string>> & labels,
        const std::filesystem::path & root) {
    const std::array<const char *, 4> prompts = {
        "A synthetic probe requests one arbitrary marker. The marker is",
        "Return one token for this deliberately unknown code:",
        "The answer to this arbitrary token-control probe is",
        "Emit the assigned synthetic marker only:",
    };
    const std::array<double, 5> ratios = {0.001, 0.003, 0.01, 0.03, 0.10};
    std::ofstream rows(root/"stage_a_control_thresholds.jsonl", std::ios::trunc);
    std::ofstream generated(root/"generated_outputs.jsonl", std::ios::trunc);
    bank zero(output.width);
    route uniform{"uniform", LLAMA_ATTNRES_ROUTE_QUERY, -1, std::vector<float>(output.width)};
    install_route(ctx, uniform, zero);
    std::vector<double> thresholds;
    int fixed_hits[ratios.size()] = {};
    for (int index = 0; index < 16; ++index) {
        const auto & label = labels[index];
        auto base = diagnostic_evaluate(ctx, vocab, collector, prompts[index%prompts.size()], label.first);
        if (base.greedy == label.first) throw std::runtime_error("Stage A target unexpectedly already top-1");
        const auto direction = output_row_difference(output, label.first, base.greedy);
        const auto logit_slope = output_projection(output, direction);
        const double direction_sq = dot(direction, direction);
        const double pair_alpha = (base.logits[base.greedy] - base.logits[label.first])/direction_sq;
        auto top1_for_alpha = [&](double alpha) {
            std::vector<float> logits(base.logits);
            for (size_t token = 0; token < logits.size(); ++token) logits[token] += alpha*logit_slope[token];
            return metrics_from_logits(logits, label.first);
        };
        double lower = 0;
        double upper = std::max(1e-12, pair_alpha);
        token_metrics upper_metrics = top1_for_alpha(upper);
        int expansion = 0;
        while (upper_metrics.greedy != label.first && expansion++ < 80) {
            lower = upper;
            upper *= 2;
            upper_metrics = top1_for_alpha(upper);
        }
        if (upper_metrics.greedy != label.first) throw std::runtime_error("Stage A top-1 threshold not bracketed");
        for (int iteration = 0; iteration < 64; ++iteration) {
            const double middle = 0.5*(lower + upper);
            if (top1_for_alpha(middle).greedy == label.first) upper = middle;
            else lower = middle;
        }
        const double threshold_rho = upper*l2(direction)/l2(base.hidden);
        thresholds.push_back(threshold_rho);
        rows << std::setprecision(17)
             << "{\"record_type\":\"threshold\",\"case_id\":" << index
             << ",\"target_token\":" << label.first << ",\"target_text\":\"" << json_escape(label.second)
             << "\",\"baseline_greedy\":" << base.greedy << ",\"pair_alpha\":" << pair_alpha
             << ",\"global_alpha\":" << upper << ",\"rho_top1\":" << threshold_rho
             << ",\"baseline_rank\":" << base.rank << ",\"baseline_nll\":" << base.nll << "}\n";
        generated << "{\"stage\":\"A\",\"case_id\":" << index << ",\"condition\":\"baseline\",\"greedy_token\":"
                  << base.greedy << ",\"target_token\":" << label.first << "}\n";
        for (size_t ratio_index = 0; ratio_index < ratios.size(); ++ratio_index) {
            const double alpha = ratios[ratio_index]*l2(base.hidden)/l2(direction);
            const auto metric = top1_for_alpha(alpha);
            fixed_hits[ratio_index] += metric.greedy == label.first;
            rows << std::setprecision(17)
                 << "{\"record_type\":\"fixed_ratio\",\"case_id\":" << index
                 << ",\"target_token\":" << label.first << ",\"rho\":" << ratios[ratio_index]
                 << ",\"target_top1\":" << (metric.greedy == label.first ? "true" : "false")
                 << ",\"target_rank\":" << metric.rank << ",\"target_margin\":" << metric.margin
                 << ",\"target_probability\":" << metric.probability << ",\"target_nll\":" << metric.nll << "}\n";
        }
    }
    const double threshold_median = median(thresholds);
    const double behavioral_scale = std::min(threshold_median, 0.03);
    std::ofstream summary(root/"stage_a_summary.json", std::ios::trunc);
    summary << std::setprecision(17)
            << "{\"cases\":16,\"min_rho_top1\":" << *std::min_element(thresholds.begin(), thresholds.end())
            << ",\"median_rho_top1\":" << threshold_median
            << ",\"mean_rho_top1\":" << mean(thresholds)
            << ",\"p75_rho_top1\":" << quantile(thresholds, 0.75)
            << ",\"max_rho_top1\":" << *std::max_element(thresholds.begin(), thresholds.end())
            << ",\"rho_0_001_position\":\""
            << (0.001 < 0.5*threshold_median ? "below" : (0.001 > 2*threshold_median ? "above" : "near"))
            << "\",\"behavioral_test_rho\":" << behavioral_scale
            << ",\"behavioral_scale_capped\":" << (threshold_median > behavioral_scale ? "true" : "false")
            << ",\"fixed_ratio_top1_fraction\":{";
    for (size_t i = 0; i < ratios.size(); ++i) {
        summary << (i ? "," : "") << "\"" << ratios[i] << "\":" << fixed_hits[i]/16.0;
    }
    summary << "}}\n";
    return {threshold_median, behavioral_scale};
}

std::string stage_b(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const std::vector<diagnostic_item> & tasks,
        const std::vector<float> & query,
        const std::filesystem::path & root) {
    bank zero(query.size());
    route fixed{"fixed_attnres", LLAMA_ATTNRES_ROUTE_QUERY, -1, query};
    install_route(ctx, fixed, zero);
    std::map<std::string, feature_bundle> task_features;
    std::ofstream attention(root/"stage_b_attention.jsonl", std::ios::trunc);
    for (const auto & task : tasks) {
        const auto state = diagnostic_evaluate(ctx, vocab, collector, task.prompt, task.target);
        auto features = make_features(state, query);
        attention << std::setprecision(17)
                  << "{\"item_id\":\"" << task.id << "\",\"family\":\"" << task.family
                  << "\",\"scores\":";
        write_float_array(attention, features.attn_scores);
        attention << ",\"alpha\":";
        write_float_array(attention, features.attn_alpha);
        const auto winner = std::distance(features.attn_alpha.begin(),
            std::max_element(features.attn_alpha.begin(), features.attn_alpha.end()));
        attention << ",\"entropy\":" << attention_entropy(features.attn_alpha)
                  << ",\"max_alpha\":" << features.attn_alpha[winner]
                  << ",\"winning_depth\":" << (winner + 1)*4
                  << ",\"z_norm\":" << l2(state.routed) << "}\n";
        task_features.emplace(task.id, std::move(features));
    }
    std::vector<std::pair<std::string, feature_bundle>> protective_features;
    for (const auto & prompt : protective_prompts()) {
        const auto first = diagnostic_evaluate(ctx, vocab, collector, prompt, 0);
        protective_features.push_back({prompt, make_features(first, query)});
    }
    const std::vector<std::string> methods = {
        "final_only", "uniform", "tap_4", "tap_8", "tap_12", "tap_16",
        "tap_20", "tap_24", "tap_28", "tap_32", "fixed_attnres",
    };
    std::map<std::string, geometry_stats> stats;
    std::ofstream rows(root/"stage_b_key_geometry.jsonl", std::ios::trunc);
    for (const auto & method : methods) {
        for (size_t i = 0; i < tasks.size(); ++i) {
            for (size_t j = i + 1; j < tasks.size(); ++j) {
                const bool same = tasks[i].family == tasks[j].family;
                const double similarity = cosine(task_features.at(tasks[i].id).values.at(method),
                                                 task_features.at(tasks[j].id).values.at(method));
                (same ? stats[method].same : stats[method].different).push_back(similarity);
                rows << std::setprecision(17)
                     << "{\"method\":\"" << method << "\",\"pair_type\":\"" << (same ? "same_rule" : "different_rule")
                     << "\",\"left\":\"" << tasks[i].id << "\",\"right\":\"" << tasks[j].id
                     << "\",\"cosine\":" << similarity << "}\n";
            }
            for (size_t p = 0; p < protective_features.size(); ++p) {
                const double similarity = cosine(task_features.at(tasks[i].id).values.at(method),
                                                 protective_features[p].second.values.at(method));
                stats[method].protective.push_back(similarity);
                rows << std::setprecision(17)
                     << "{\"method\":\"" << method << "\",\"pair_type\":\"protective\",\"left\":\""
                     << tasks[i].id << "\",\"right\":\"protective_" << p << "\",\"cosine\":" << similarity << "}\n";
            }
        }
    }
    std::string best_tap = "tap_4";
    double best_tap_gap = -std::numeric_limits<double>::infinity();
    std::ofstream summary(root/"stage_b_summary.json", std::ios::trunc);
    summary << std::setprecision(17) << "{\"methods\":{";
    for (size_t method_index = 0; method_index < methods.size(); ++method_index) {
        const auto & method = methods[method_index];
        const auto & current = stats.at(method);
        const double same_mean = mean(current.same);
        const double different_mean = mean(current.different);
        const double gap = same_mean - different_mean;
        if (method.rfind("tap_", 0) == 0 && gap > best_tap_gap) {
            best_tap_gap = gap;
            best_tap = method;
        }
        int nearest_correct = 0;
        for (const auto & item : tasks) {
            double nearest = -2;
            std::string family;
            for (const auto & candidate : tasks) if (candidate.id != item.id) {
                const double similarity = cosine(task_features.at(item.id).values.at(method),
                                                 task_features.at(candidate.id).values.at(method));
                if (similarity > nearest) {
                    nearest = similarity;
                    family = candidate.family;
                }
            }
            nearest_correct += family == item.family;
        }
        summary << (method_index ? "," : "") << "\"" << method << "\":{"
                << "\"mean_same\":" << same_mean << ",\"mean_different\":" << different_mean
                << ",\"gap\":" << gap << ",\"median_same\":" << median(current.same)
                << ",\"median_different\":" << median(current.different)
                << ",\"std_same\":" << standard_deviation(current.same)
                << ",\"std_different\":" << standard_deviation(current.different)
                << ",\"roc_auc\":" << roc_auc(current.same, current.different)
                << ",\"nearest_neighbor_same_rule_accuracy\":" << nearest_correct/(double) tasks.size()
                << ",\"mean_protective_cosine\":" << mean(current.protective) << "}";
    }
    const double attn_gap = mean(stats["fixed_attnres"].same) - mean(stats["fixed_attnres"].different);
    const double final_gap = mean(stats["final_only"].same) - mean(stats["final_only"].different);
    const double uniform_gap = mean(stats["uniform"].same) - mean(stats["uniform"].different);
    summary << "},\"best_individual_tap\":\"" << best_tap << "\",\"attnres_gap\":" << attn_gap
            << ",\"final_gap\":" << final_gap << ",\"uniform_gap\":" << uniform_gap
            << ",\"attnres_better_than_final\":" << (attn_gap > final_gap ? "true" : "false")
            << ",\"attnres_better_than_uniform\":" << (attn_gap > uniform_gap ? "true" : "false") << "}\n";
    return best_tap;
}

route route_for_method(
        const std::string & method,
        const std::vector<float> & query,
        int32_t width) {
    if (method == "uniform") return {"uniform", LLAMA_ATTNRES_ROUTE_QUERY, -1, std::vector<float>(width)};
    if (method == "fixed_attnres") return {"fixed_attnres", LLAMA_ATTNRES_ROUTE_QUERY, -1, query};
    if (method == "final_only") return {"final_only", LLAMA_ATTNRES_ROUTE_ONE_HOT, 7, {}};
    if (method.rfind("tap_", 0) == 0) {
        const int depth = std::stoi(method.substr(4));
        return {method, LLAMA_ATTNRES_ROUTE_ONE_HOT, depth/4 - 1, {}};
    }
    throw std::runtime_error("unknown Stage C method: " + method);
}

bank make_one_memory(
        const std::vector<float> & key,
        const std::vector<float> & hidden,
        const std::vector<float> & direction,
        double rho) {
    bank out(key.size());
    const auto key_hat = normalized(key);
    const auto q_hat = normalized(direction);
    const double source_coefficient = dot(key_hat, key);
    if (source_coefficient <= 0) throw std::runtime_error("invalid source key norm");
    const double response_norm = rho*l2(hidden);
    for (int32_t i = 0; i < out.width; ++i) {
        out.v[i] = key_hat[i]/source_coefficient;
        out.u[(size_t) RANK*i] = response_norm*q_hat[i];
    }
    out.used = 1;
    return out;
}

void stage_c(
        llama_context * ctx,
        const llama_vocab * vocab,
        trace_collector & collector,
        const output_matrix & output,
        const std::vector<diagnostic_item> & tasks,
        const std::vector<float> & query,
        const std::string & best_tap,
        double behavioral_rho,
        const std::filesystem::path & root) {
    const diagnostic_item & source = tasks.at(0);
    const diagnostic_item & shuffled_source = tasks.at(8);
    const std::vector<std::string> methods = {"final_only", "uniform", best_tap, "fixed_attnres"};
    const std::array<double, 2> scales = {0.001, behavioral_rho};
    std::ofstream rows(root/"stage_c_memory_results.jsonl", std::ios::trunc);
    std::ofstream protective(root/"protective_results.jsonl", std::ios::trunc);
    std::ofstream generated(root/"generated_outputs.jsonl", std::ios::app);
    const auto protective_cases = protective_prompts();
    std::map<std::string, std::pair<double, double>> aggregate;
    std::map<std::string, std::pair<int, int>> generated_aggregate;
    bank zero(output.width);
    for (const auto & method : methods) {
        const auto selected_route = route_for_method(method, query, output.width);
        install_route(ctx, selected_route, zero);
        const auto source_base = diagnostic_evaluate(ctx, vocab, collector, source.prompt, source.target);
        const auto direction = exact_target_direction(output, source_base.logits, source.target, 16);
        const auto finite_residual = scaled_vector(direction, 1e-5);
        const auto finite_metric = metrics_from_logits(residual_logits(output, source_base.logits, finite_residual), source.target);
        if (!(finite_metric.nll < source_base.nll)) throw std::runtime_error("target direction finite-difference validation failed");
        install_route(ctx, selected_route, zero);
        const auto shuffled_base = diagnostic_evaluate(ctx, vocab, collector, shuffled_source.prompt, shuffled_source.target);
        const auto shuffled_direction = exact_target_direction(output, shuffled_base.logits, shuffled_source.target, 16);
        for (double rho : scales) {
            const auto memory = make_one_memory(source_base.routed, source_base.hidden, direction, rho);
            const std::array<std::pair<std::string, const std::vector<float> *>, 2> directions = {{
                {"target", &direction}, {"shuffled_target", &shuffled_direction},
            }};
            for (const auto & direction_spec : directions) {
                if (direction_spec.first == "shuffled_target" && method != "fixed_attnres") continue;
                const auto installed_memory = direction_spec.first == "target"
                    ? memory
                    : make_one_memory(source_base.routed, source_base.hidden, *direction_spec.second, rho);
                for (const auto & test : tasks) {
                    const bool self = test.id == source.id;
                    const bool same = test.family == source.family && !self;
                    const bool different = test.family != source.family;
                    if (!self && !(same && test.family_index <= 4) &&
                            !(different && test.family == "B" && test.family_index <= 3)) continue;
                    install_route(ctx, selected_route, zero);
                    const auto baseline = diagnostic_evaluate(ctx, vocab, collector, test.prompt, test.target);
                    install_route(ctx, selected_route, installed_memory);
                    const auto plastic = diagnostic_evaluate(ctx, vocab, collector, test.prompt, test.target);
                    const double activation = dot(normalized(source_base.routed), plastic.routed)/l2(source_base.routed);
                    const auto response = installed_memory.apply(plastic.routed);
                    const double actual_rho = l2(response)/l2(plastic.hidden);
                    const double improvement = baseline.nll - plastic.nll;
                    const std::string group = self ? "self_key" : (same ? "same_rule" : "different_rule");
                    rows << std::setprecision(17)
                         << "{\"method\":\"" << method << "\",\"direction\":\"" << direction_spec.first
                         << "\",\"rho_source\":" << rho << ",\"item_id\":\"" << test.id
                         << "\",\"group\":\"" << group << "\",\"baseline_probability\":" << baseline.probability
                         << ",\"plastic_probability\":" << plastic.probability
                         << ",\"baseline_nll\":" << baseline.nll << ",\"plastic_nll\":" << plastic.nll
                         << ",\"nll_improvement\":" << improvement << ",\"baseline_rank\":" << baseline.rank
                         << ",\"plastic_rank\":" << plastic.rank << ",\"baseline_margin\":" << baseline.margin
                         << ",\"plastic_margin\":" << plastic.margin << ",\"baseline_greedy\":" << baseline.greedy
                         << ",\"plastic_greedy\":" << plastic.greedy << ",\"greedy_correct\":"
                         << (plastic.greedy == test.target ? "true" : "false")
                         << ",\"sidecar_residual_norm\":" << l2(response) << ",\"rho_actual\":" << actual_rho
                         << ",\"memory_activation_scalar\":" << activation
                         << ",\"source_test_key_cosine\":" << cosine(source_base.routed, plastic.routed) << "}\n";
                    generated << "{\"stage\":\"C\",\"method\":\"" << method << "\",\"direction\":\""
                              << direction_spec.first << "\",\"rho\":" << rho << ",\"item_id\":\"" << test.id
                              << "\",\"target_token\":" << test.target << ",\"baseline_greedy\":" << baseline.greedy
                              << ",\"plastic_greedy\":" << plastic.greedy << "}\n";
                    if (direction_spec.first == "target" && !self) {
                        auto & sums = aggregate[method + ":" + std::to_string(rho)];
                        auto & counts = generated_aggregate[method + ":" + std::to_string(rho)];
                        if (same) {
                            sums.first += improvement;
                            counts.first += plastic.greedy == test.target;
                        } else {
                            sums.second += improvement;
                            counts.second += plastic.greedy == test.target;
                        }
                    }
                }
                for (size_t protective_index = 0; protective_index < protective_cases.size(); ++protective_index) {
                    const auto & prompt = protective_cases[protective_index];
                    install_route(ctx, selected_route, zero);
                    const auto first = diagnostic_evaluate(ctx, vocab, collector, prompt, 0);
                    const auto baseline = diagnostic_evaluate(ctx, vocab, collector, prompt, first.greedy);
                    install_route(ctx, selected_route, installed_memory);
                    const auto plastic = diagnostic_evaluate(ctx, vocab, collector, prompt, first.greedy);
                    protective << std::setprecision(17)
                               << "{\"method\":\"" << method << "\",\"direction\":\"" << direction_spec.first
                               << "\",\"rho_source\":" << rho << ",\"prompt_id\":" << protective_index
                               << ",\"kl\":" << kl_divergence(plastic.logits, baseline.logits)
                               << ",\"nll_change\":" << plastic.nll - baseline.nll
                               << ",\"greedy_changed\":" << (plastic.greedy != baseline.greedy ? "true" : "false") << "}\n";
                }
            }
        }
    }
    std::ofstream summary(root/"stage_c_summary.json", std::ios::trunc);
    summary << std::setprecision(17) << "{\"source_item\":\"" << source.id << "\",\"best_tap\":\""
            << best_tap << "\",\"behavioral_rho\":" << behavioral_rho << ",\"conditions\":{";
    bool first = true;
    for (const auto & entry : aggregate) {
        const auto counts = generated_aggregate[entry.first];
        summary << (first ? "" : ",") << "\"" << entry.first << "\":{\"same_rule_mean_nll_improvement\":"
                << entry.second.first/4.0 << ",\"different_rule_mean_nll_improvement\":" << entry.second.second/4.0
                << ",\"selectivity\":" << (entry.second.first - entry.second.second)/4.0
                << ",\"same_rule_greedy_correct\":" << counts.first
                << ",\"different_rule_greedy_correct\":" << counts.second << "}";
        first = false;
    }
    summary << "}}\n";

    // Serialize one non-empty AttnRes memory and verify exact reload behavior.
    const auto serial_route = route_for_method("fixed_attnres", query, output.width);
    install_route(ctx, serial_route, zero);
    const auto serial_base = diagnostic_evaluate(ctx, vocab, collector, source.prompt, source.target);
    const auto serial_direction = exact_target_direction(output, serial_base.logits, source.target, 16);
    const auto serial_bank = make_one_memory(serial_base.routed, serial_base.hidden, serial_direction, behavioral_rho);
    item serial_item{source.id, source.prompt, source.target, source.target_text};
    write_serialization(ctx, vocab, collector, serial_item, serial_route, serial_bank, root);
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
        if (model.hparams.n_embd != 2560 || model.hparams.n_layer() != 32) {
            throw std::runtime_error("diagnostic requires Qwen3.5-4B dimensions 32x2560");
        }

        trace_collector collector;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.ctx_size;
        context_params.n_batch = opts.ctx_size;
        context_params.n_ubatch = std::min(opts.ctx_size, 512);
        context_params.n_seq_max = 1;
        context_params.cb_eval = eval_callback;
        context_params.cb_eval_user_data = &collector;
        llama_context * ctx = llama_init_from_model(public_model, context_params);
        if (!ctx) throw std::runtime_error("context creation failed");
        if (!ctx->init_fast_weights(RANK)) throw std::runtime_error("adapter initialization failed");
        const llama_vocab * vocab = llama_model_get_vocab(public_model);
        const auto labels = choose_single_tokens(vocab, 16);
        const auto tasks = make_rule_items(labels);
        const auto query = deterministic_query(model.hparams.n_embd, opts.seed);
        const auto output = load_output_matrix(model);

        bank zero(model.hparams.n_embd);
        route uniform{"uniform", LLAMA_ATTNRES_ROUTE_QUERY, -1, std::vector<float>(model.hparams.n_embd)};
        ctx->clear_fast_weights();
        const auto neutral_base = diagnostic_evaluate(ctx, vocab, collector, tasks[0].prompt, tasks[0].target);
        install_route(ctx, uniform, zero);
        const auto neutral_enabled = diagnostic_evaluate(ctx, vocab, collector, tasks[0].prompt, tasks[0].target);
        double zero_diff = 0;
        for (size_t i = 0; i < neutral_base.logits.size(); ++i) {
            zero_diff = std::max(zero_diff, std::abs((double) neutral_base.logits[i] - neutral_enabled.logits[i]));
        }
        if (zero_diff > 1e-6) throw std::runtime_error("zero-state neutrality failed");

        const auto threshold = stage_a(ctx, vocab, collector, output, labels, root);
        const auto best_tap = stage_b(ctx, vocab, collector, tasks, query, root);
        stage_c(ctx, vocab, collector, output, tasks, query, best_tap, threshold.second, root);

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << std::setprecision(17)
                << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":" << diagnostic_eval_calls
                << ",\"prompt_scoring_forwards\":" << diagnostic_eval_calls
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size << "}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"persistent_sidecar_bytes\":92160,\"transient_depth_bytes\":81920,"
               << "\"rank\":4,\"width\":2560,\"depth_taps\":8,\"peak_sidecar_state_bytes\":174080}\n";
        std::ofstream manifest(root/"manifest.json", std::ios::trunc);
        manifest << "{\"experiment\":\"qwen35_attnres_address_space_diagnostic\","
                 << "\"seed\":" << opts.seed << ",\"stages\":[\"A\",\"B\",\"C\"],"
                 << "\"base_model_frozen\":true,\"query_plastic\":false,\"backpropagation\":false,"
                 << "\"depth_taps\":[4,8,12,16,20,24,28,32],"
                 << "\"same_key_primary_evidence\":false,\"zero_state_max_logit_abs_diff\":" << zero_diff << "}\n";
        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
