#include "../injection-sensitivity-allocation/multi-injection-support.inc"
#include "chat.h"

#include <cctype>
#include <random>

namespace {

constexpr int32_t FAST_RANK = 8;
constexpr double ETA_POSITIVE = 0.05;
constexpr double ETA_NEGATIVE = 0.05;
constexpr double PRIMARY_BUDGET = 0.03;
constexpr double SECONDARY_BUDGET = 0.05;
constexpr uint64_t BENCHMARK_SEED = 20260812ULL;
constexpr int32_t PROCEDURAL_SEEDS = 3;
constexpr int32_t REASONING_SEEDS = 1;
constexpr int32_t REASONING_THINK_BUDGET = 48;
constexpr int32_t REASONING_TOKEN_BUDGET = 64;

const std::array<int32_t, 4> CHECKPOINTS = {0, 1, 2, 4};
const std::array<const char *, 6> PROCEDURAL_CONDITIONS = {
    "frozen", "blind_repetition", "reward_only", "teacher_validated",
    "shuffled_teacher", "random_direction",
};
const std::array<const char *, 5> REASONING_CONDITIONS = {
    "frozen_reasoning_on", "teacher_reasoning_off", "teacher_reasoning_on",
    "reasoning_on_shuffled_teacher", "reasoning_on_random_direction",
};
const std::array<int32_t, 3> REASONING_OFFSETS = {1, 3, 6};

struct component {
    std::vector<float> q;
    std::vector<float> z;
    double coefficient = 0;
    int32_t age = 0;
    std::string source;
    std::string origin;
};

struct fast_bank {
    std::vector<component> components;
    int32_t compression_events = 0;

    void clear() {
        components.clear();
        compression_events = 0;
    }

    bool add(component value) {
        bool compressed = false;
        if ((int32_t) components.size() >= FAST_RANK) {
            auto victim = std::min_element(components.begin(), components.end(), [](const auto & a, const auto & b) {
                if (std::abs(a.coefficient) != std::abs(b.coefficient)) {
                    return std::abs(a.coefficient) < std::abs(b.coefficient);
                }
                return a.age < b.age;
            });
            components.erase(victim);
            ++compression_events;
            compressed = true;
        }
        components.push_back(std::move(value));
        return compressed;
    }

    std::vector<float> apply(const std::vector<float> & key) const {
        std::vector<float> out(WIDTH, 0.0f);
        for (const auto & term : components) {
            double activation = 0;
            for (int32_t i = 0; i < WIDTH; ++i) activation += (double) term.z[i]*key[i];
            const double scale = term.coefficient*activation;
            for (int32_t i = 0; i < WIDTH; ++i) out[i] += scale*term.q[i];
        }
        return out;
    }

    double frobenius_norm() const {
        double squared = 0;
        for (const auto & a : components) {
            for (const auto & b : components) {
                double q_dot = 0;
                double z_dot = 0;
                for (int32_t i = 0; i < WIDTH; ++i) {
                    q_dot += (double) a.q[i]*b.q[i];
                    z_dot += (double) a.z[i]*b.z[i];
                }
                squared += a.coefficient*b.coefficient*q_dot*z_dot;
            }
        }
        return std::sqrt(std::max(0.0, squared));
    }
};

struct plastic_eval {
    eval_result frozen;
    token_metrics metrics;
    std::vector<float> logits;
    std::vector<float> key;
    double raw_response_norm = 0;
    double budget = 0;
    std::map<int32_t, double> rhos;
};

struct generated_result {
    std::string text;
    std::string canonical;
    std::vector<llama_token> tokens;
    plastic_eval trace;
    llama_token answer_token = -1;
    int32_t reasoning_tokens = 0;
    int32_t answer_tokens = 0;
    bool reasoning_occurred = false;
    bool reasoning_completed = false;
    bool reasoning_forced_close = false;
    bool correct_hypothesis = false;
    bool correct = false;
};

struct procedure_example {
    std::array<std::string, 3> symbols;
    std::string expected;
    llama_token first_target = -1;
    std::string id;
};

struct procedure_episode {
    int32_t seed = 0;
    std::vector<procedure_example> training;
    std::vector<procedure_example> heldout;
};

struct reasoning_example {
    int32_t input = 0;
    int32_t target_index = 0;
    llama_token target = -1;
    std::string id;
};

struct reasoning_episode {
    int32_t offset = 0;
    int32_t seed = 0;
    int32_t demo_a = 0;
    int32_t demo_b = 2;
    std::vector<reasoning_example> training;
    std::vector<reasoning_example> heldout;
};

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize(-count);
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    return count > 0 ? std::string(buffer.data(), count) : std::string();
}

std::string trim(std::string value) {
    while (!value.empty() && std::isspace((unsigned char) value.front())) value.erase(value.begin());
    while (!value.empty() && std::isspace((unsigned char) value.back())) value.pop_back();
    return value;
}

std::string canonical_procedure(std::string value) {
    const auto newline = value.find_first_of("\r\n");
    if (newline != std::string::npos) value.resize(newline);
    value = trim(value);
    if (!value.empty() && value.front() == '`') value.erase(value.begin());
    if (!value.empty() && value.back() == '`') value.pop_back();
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); }), value.end());
    return value;
}

llama_token answer_symbol_token(const llama_vocab * vocab, const std::string & text,
                                const std::vector<llama_token> & symbol_tokens) {
    for (auto it = text.rbegin(); it != text.rend(); ++it) {
        if (*it < 'A' || *it > 'Z') continue;
        const auto tokens = tokenize(vocab, std::string(" ") + *it, false);
        if (tokens.size() == 1 && std::find(symbol_tokens.begin(), symbol_tokens.end(), tokens[0]) != symbol_tokens.end()) {
            return tokens[0];
        }
    }
    return -1;
}

std::vector<float> normalized_output_row(const output_matrix & output, llama_token token) {
    std::vector<float> row(output.width);
    std::copy(output.row(token), output.row(token) + output.width, row.begin());
    return normalized(row);
}

std::vector<float> normalized_token_direction(
        const output_matrix & output,
        llama_token positive,
        llama_token negative) {
    // Full-string validation can fail even when a multi-token answer starts with
    // the target token. That makes the token-row difference exactly zero.
    if (positive == negative) {
        return normalized_output_row(output, positive);
    }
    return normalized(output_row_difference(output, positive, negative));
}

std::vector<float> deterministic_random_direction(uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::normal_distribution<float> normal;
    std::vector<float> values(WIDTH);
    for (float & value : values) value = normal(generator);
    return normalized(values);
}

plastic_eval evaluate_bank(
        llama_context * ctx, const llama_vocab * vocab, trace_collector & collector,
        const output_matrix & output, const fast_bank & bank, const std::string & prompt,
        llama_token target, double budget) {
    clear_internal_injection(ctx);
    plastic_eval out;
    out.frozen = evaluate(ctx, vocab, collector, prompt, target, true);
    out.key = normalized(out.frozen.final_hidden);
    out.logits = out.frozen.logits;
    out.metrics = out.frozen.metrics;
    if (bank.components.empty()) return out;
    const auto response = bank.apply(out.key);
    out.raw_response_norm = l2(response);
    if (out.raw_response_norm <= 1e-12) return out;
    const auto direction = normalized(response);
    item definition{0, prompt, target, ""};
    case_state state{definition, out.frozen, direction, output_projection(output, direction)};
    const multi_configuration config{"plastic", "block24_block28_final", {23, 27, -1}};
    const auto weights = normalized_weights({0.23072, 0.38322, 0.89438});
    const double bounded_budget = std::min(SECONDARY_BUDGET, std::max(0.0, budget));
    const auto probe = probe_multi(ctx, vocab, collector, state, config, weights, bounded_budget);
    out.metrics = probe.metrics;
    out.logits = probe.logits;
    out.budget = probe.actual_budget;
    out.rhos = probe.actual_rhos;
    clear_internal_injection(ctx);
    return out;
}

generated_result generate_bank(
        llama_context * ctx, const llama_vocab * vocab, trace_collector & collector,
        const output_matrix & output, const fast_bank & bank, const std::string & prompt,
        llama_token target, double budget, int32_t max_tokens, bool procedure_mode,
        int32_t reasoning_offset, const std::vector<llama_token> & reasoning_symbols) {
    generated_result result;
    std::string prefix;
    const bool thinking_mode = !procedure_mode && max_tokens == REASONING_TOKEN_BUDGET;
    bool thinking_complete = false;
    for (int32_t i = 0; i < max_tokens; ++i) {
        if (thinking_mode && !thinking_complete && i >= REASONING_THINK_BUDGET) {
            // The custom greedy loop does not use llama.cpp's sampler-side
            // reasoning budget, so reproduce its deterministic close behavior.
            prefix += "</think>\n";
            result.text += "</think>\n";
            thinking_complete = true;
            result.reasoning_forced_close = true;
        }
        auto value = evaluate_bank(ctx, vocab, collector, output, bank, prompt + prefix, target, budget);
        if (i == 0) result.trace = value;
        const llama_token next = value.metrics.greedy;
        if (llama_vocab_is_eog(vocab, next)) break;
        const std::string piece = token_piece(vocab, next);
        const bool token_is_reasoning = thinking_mode && !thinking_complete;
        result.tokens.push_back(next);
        result.text += piece;
        prefix += piece;
        if (token_is_reasoning) ++result.reasoning_tokens;
        if (procedure_mode && (piece.find('\n') != std::string::npos || result.tokens.size() >= 8)) break;
        if (!procedure_mode) {
            thinking_complete = thinking_complete || result.text.find("</think>") != std::string::npos;
            const bool answer_position = !thinking_mode || thinking_complete;
            if (answer_position && std::find(reasoning_symbols.begin(), reasoning_symbols.end(), next) != reasoning_symbols.end()) {
                result.trace = std::move(value);
                result.answer_token = next;
                break;
            }
        }
    }
    result.canonical = procedure_mode ? canonical_procedure(result.text) : trim(result.text);
    if (procedure_mode) result.answer_token = result.tokens.empty() ? -1 : result.tokens.front();
    if (!procedure_mode) {
        // Qwen's chat template prefills the opening thinking marker in the prompt,
        // so generated text normally contains only the closing marker.
        result.reasoning_occurred = thinking_mode && !result.tokens.empty();
        result.reasoning_completed = thinking_complete;
        if (!thinking_complete && thinking_mode) {
            // An unfinished thinking trajectory has no validated final answer.
            result.answer_token = -1;
        }
        result.answer_tokens = (int32_t) result.tokens.size() - result.reasoning_tokens;
        const std::string plus = "+" + std::to_string(reasoning_offset);
        const std::string offset = "offset " + std::to_string(reasoning_offset);
        result.correct_hypothesis = result.text.find(plus) != std::string::npos || result.text.find(offset) != std::string::npos;
    }
    result.correct = result.answer_token == target;
    return result;
}

void write_metrics(std::ostream & out, const token_metrics & m) {
    out << std::setprecision(17)
        << "\"target_probability\":" << m.probability
        << ",\"target_nll\":" << m.nll
        << ",\"target_rank\":" << m.rank
        << ",\"target_margin\":" << m.margin
        << ",\"greedy_token\":" << m.greedy;
}

void write_rhos(std::ostream & out, const plastic_eval & value) {
    auto get = [&](int32_t site) { auto it = value.rhos.find(site); return it == value.rhos.end() ? 0.0 : it->second; };
    out << std::setprecision(17) << "\"B\":" << value.budget
        << ",\"rho24\":" << get(23) << ",\"rho28\":" << get(27)
        << ",\"rho_final\":" << get(-1) << ",\"authority_clipped\":"
        << (value.budget > SECONDARY_BUDGET + 1e-9 ? "true" : "false")
        << ",\"raw_sidecar_response_norm\":" << value.raw_response_norm;
}

bool add_component(fast_bank & bank, std::vector<float> q, const std::vector<float> & z,
                   double coefficient, int32_t age, const std::string & source, const std::string & origin) {
    return bank.add({normalized(q), normalized(z), coefficient, age, source, origin});
}

double pending_norm(const std::vector<component> & pending, bool negative) {
    double sum = 0;
    for (const auto & term : pending) {
        const bool neg = term.origin.find("negative") != std::string::npos;
        if (neg == negative) sum += term.coefficient*term.coefficient;
    }
    return std::sqrt(sum);
}

std::vector<llama_token> validate_symbols(const llama_vocab * vocab, int count) {
    std::vector<llama_token> result;
    for (int i = 0; i < count; ++i) {
        const std::string text = std::string(" ") + (char) ('A' + i);
        const auto tokens = tokenize(vocab, text, false);
        if (tokens.size() != 1) throw std::runtime_error("symbol is not one token: " + text);
        if (std::find(result.begin(), result.end(), tokens[0]) != result.end()) throw std::runtime_error("symbol token collision");
        result.push_back(tokens[0]);
    }
    return result;
}

std::string procedure_expected(const std::array<std::string, 3> & s) {
    return s[2] + "-" + s[0] + "-" + s[1];
}

std::string procedure_prompt(const procedure_example & ex, bool rule_visible) {
    std::ostringstream out;
    if (rule_visible) {
        out << "You are executing a simple procedure.\nRULE: Given three symbols X Y Z, answer THIRD-FIRST-SECOND.\n"
            << "Example: Input: A B C\nOutput: C-A-B\n\n";
    }
    out << "Input: " << ex.symbols[0] << " " << ex.symbols[1] << " " << ex.symbols[2] << "\nOutput:";
    return out.str();
}

std::string procedure_corrected_prompt(const procedure_example & ex, const std::string & expected) {
    return "Incorrect.\nThe procedure is THIRD-FIRST-SECOND.\nFor input " + ex.symbols[0] + " " + ex.symbols[1] + " " + ex.symbols[2] +
           ": third = " + ex.symbols[2] + ", first = " + ex.symbols[0] + ", second = " + ex.symbols[1] +
           ".\nTherefore the correct answer is " + expected + ".\nNow solve the original input again.\nInput: " +
           ex.symbols[0] + " " + ex.symbols[1] + " " + ex.symbols[2] + "\nOutput:";
}

std::vector<procedure_episode> make_procedure_episodes(const std::vector<llama_token> & symbols) {
    const std::array<std::array<int, 3>, 12> base = {{
        {{0,1,2}}, {{3,4,5}}, {{6,7,8}}, {{9,10,11}},
        {{0,3,6}}, {{1,4,7}}, {{2,5,8}}, {{3,6,9}},
        {{4,7,10}}, {{5,8,11}}, {{0,4,8}}, {{1,5,9}},
    }};
    std::vector<procedure_episode> episodes;
    for (int seed = 0; seed < PROCEDURAL_SEEDS; ++seed) {
        procedure_episode ep;
        ep.seed = seed;
        for (int i = 0; i < 12; ++i) {
            std::array<std::string, 3> labels;
            for (int j = 0; j < 3; ++j) labels[j] = std::string(1, (char) ('A' + (base[i][j] + seed) % 12));
            procedure_example ex{labels, procedure_expected(labels), symbols[(base[i][2] + seed) % 12], "P" + std::to_string(seed) + "_" + std::to_string(i)};
            (i < 4 ? ep.training : ep.heldout).push_back(std::move(ex));
        }
        episodes.push_back(std::move(ep));
    }
    return episodes;
}

std::vector<reasoning_episode> make_reasoning_episodes(const std::vector<llama_token> & symbols) {
    std::vector<reasoning_episode> result;
    for (int offset : REASONING_OFFSETS) {
        for (int seed = 0; seed < REASONING_SEEDS; ++seed) {
            reasoning_episode ep;
            ep.offset = offset;
            ep.seed = seed;
            ep.demo_a = (seed + 0) % 8;
            ep.demo_b = (seed + 2) % 8;
            const std::array<int, 4> base_train = {1, 3, 5, 7};
            std::set<int> training;
            for (int input : base_train) training.insert((input + seed) % 8);
            for (int input = 0; input < 8; ++input) {
                reasoning_example ex{input, (input + offset) % 8, symbols[(input + offset) % 8],
                                     "R" + std::to_string(offset) + "_" + std::to_string(seed) + "_" + std::to_string(input)};
                (training.count(input) ? ep.training : ep.heldout).push_back(ex);
            }
            result.push_back(std::move(ep));
        }
    }
    return result;
}

std::string reasoning_user_prompt(const reasoning_episode & ep, const reasoning_example & ex) {
    auto symbol = [](int value) { return std::string(1, (char) ('A' + value)); };
    return "The symbols form a cycle: A B C D E F G H.\nEvery example uses one constant forward offset. "
           "Use the examples to determine the offset. Think through the offset before answering.\n\nExample 1: " +
           symbol(ep.demo_a) + " -> " + symbol((ep.demo_a + ep.offset) % 8) + "\nExample 2: " +
           symbol(ep.demo_b) + " -> " + symbol((ep.demo_b + ep.offset) % 8) + "\nQuestion: " +
           symbol(ex.input) + " -> ?\nAfter reasoning, give the final answer as one symbol.";
}

std::string reasoning_corrected_user_prompt(const reasoning_episode & ep, const reasoning_example & ex, int teacher_target) {
    auto symbol = [](int value) { return std::string(1, (char) ('A' + value)); };
    return "The previous answer was incorrect. Reason through the shared transformation.\n" +
           symbol(ep.demo_a) + " -> " + symbol((ep.demo_a + ep.offset) % 8) + " means move forward " + std::to_string(ep.offset) +
           " positions. " + symbol(ep.demo_b) + " -> " + symbol((ep.demo_b + ep.offset) % 8) +
           " confirms the same offset. Therefore the common rule is +" + std::to_string(ep.offset) +
           " modulo 8. For " + symbol(ex.input) + ", moving forward " + std::to_string(ep.offset) + " gives " +
           symbol(teacher_target) + ". Now reason through the problem again and give the final answer as one symbol.";
}

std::string render_chat(const common_chat_templates * templates, const std::string & user, bool enable_thinking,
                        bool & supports_thinking) {
    common_chat_msg message;
    message.role = "user";
    message.content = user;
    common_chat_templates_inputs inputs;
    inputs.messages.push_back(message);
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = enable_thinking;
    const auto params = common_chat_templates_apply(templates, inputs);
    supports_thinking = params.supports_thinking;
    return params.prompt;
}

void write_json_string_array(std::ostream & out, const std::vector<std::string> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << '"' << json_escape(values[i]) << '"';
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
            throw std::runtime_error("experiment requires Qwen3.5-4B dimensions 32x2560");
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
        const auto procedure_symbols = validate_symbols(vocab, 12);
        const auto reasoning_symbols = std::vector<llama_token>(procedure_symbols.begin(), procedure_symbols.begin() + 8);
        const auto procedure_episodes = make_procedure_episodes(procedure_symbols);
        const auto reasoning_episodes = make_reasoning_episodes(reasoning_symbols);
        auto chat_templates = common_chat_templates_init(public_model, "");
        if (!chat_templates) throw std::runtime_error("chat template initialization failed");
        const bool template_supports_thinking = common_chat_templates_support_enable_thinking(chat_templates.get());
        const bool skip_procedural = std::getenv("TEACHER_CURRICULUM_REASONING_ONLY") != nullptr;
        const bool skip_reasoning = std::getenv("TEACHER_CURRICULUM_PROCEDURAL_ONLY") != nullptr;
        const char * procedural_condition_filter = std::getenv("TEACHER_CURRICULUM_PROCEDURAL_CONDITION");
        const char * reasoning_offset_filter = std::getenv("TEACHER_CURRICULUM_REASONING_OFFSET");

        std::ofstream procedural_events(root/"procedural_events.jsonl", std::ios::trunc);
        std::ofstream procedural_results(root/"procedural_results.jsonl", std::ios::trunc);
        std::ofstream procedural_corrected(root/"procedural_corrected_attempts.jsonl", std::ios::trunc);
        std::ofstream reasoning_events(root/"reasoning_events.jsonl", std::ios::trunc);
        std::ofstream reasoning_results(root/"reasoning_results.jsonl", std::ios::trunc);
        std::ofstream reasoning_corrected(root/"reasoning_corrected_attempts.jsonl", std::ios::trunc);
        std::ofstream history(root/"plastic_state_history.jsonl", std::ios::trunc);
        std::ofstream protective(root/"protective_results.jsonl", std::ios::trunc);
        for (std::ofstream * stream : {&procedural_events, &procedural_results, &procedural_corrected,
                                      &reasoning_events, &reasoning_results, &reasoning_corrected,
                                      &history, &protective}) {
            stream->setf(std::ios::unitbuf);
        }

        uint64_t procedural_teaching_calls_start = decode_calls;
        uint64_t procedural_evaluation_calls = 0;
        uint64_t reasoning_teaching_calls_start = 0;
        uint64_t reasoning_evaluation_calls = 0;
        uint64_t protective_calls = 0;
        uint64_t update_rows = 0;
        double zero_state_max = 0;
        double reset_max = 0;
        double max_budget_observed = 0;
        bool no_update_before_feedback = true;
        const std::array<std::string, 8> protective_prompts = {
            "A common week has this many days:", "Five plus six equals", "The season after spring is",
            "A triangle has this many sides:", "The opposite of narrow is", "Water freezes at zero degrees",
            "Monday, Tuesday,", "The clear daytime sky is usually",
        };

        auto log_history = [&](const char * part, const char * condition, int family, int seed, int checkpoint,
                               const fast_bank & bank) {
            history << std::setprecision(17) << "{\"part\":\"" << part << "\",\"condition\":\"" << condition
                    << "\",\"family\":" << family << ",\"seed\":" << seed << ",\"checkpoint\":" << checkpoint
                    << ",\"rank\":" << bank.components.size() << ",\"plastic_frobenius_norm\":" << bank.frobenius_norm()
                    << ",\"compression_events\":" << bank.compression_events << "}\n";
        };

        // PART I: an explicit, reasoning-free THIRD-FIRST-SECOND procedure.
        if (!skip_procedural) for (int condition = 0; condition < (int) PROCEDURAL_CONDITIONS.size(); ++condition) {
            if (procedural_condition_filter != nullptr &&
                std::string(PROCEDURAL_CONDITIONS[condition]) != procedural_condition_filter) {
                continue;
            }
            for (const auto & ep : procedure_episodes) {
                fast_bank bank;
                auto evaluate_checkpoint = [&](int checkpoint) {
                    log_history("procedural", PROCEDURAL_CONDITIONS[condition], 0, ep.seed, checkpoint, bank);
                    const uint64_t before = decode_calls;
                    for (const auto & ex : ep.heldout) {
                        for (bool rule_visible : {true, false}) {
                            const auto generated = generate_bank(ctx, vocab, collector, output, bank,
                                procedure_prompt(ex, rule_visible), ex.first_target, PRIMARY_BUDGET, 8, true, 0, {});
                            max_budget_observed = std::max(max_budget_observed, generated.trace.budget);
                            procedural_results << "{\"split\":\"heldout\",\"condition\":\""
                                << PROCEDURAL_CONDITIONS[condition] << "\",\"seed\":" << ep.seed
                                << ",\"checkpoint\":" << checkpoint << ",\"rule_visible\":"
                                << (rule_visible ? "true" : "false") << ",\"example_id\":\"" << ex.id
                                << "\",\"same_exact_prompt_seen\":false,\"same_exact_pair_seen\":false,\"same_latent_rule\":true,"
                                << "\"expected\":\"" << ex.expected << "\",\"generated\":\""
                                << json_escape(generated.canonical) << "\",\"correct\":"
                                << (generated.canonical == ex.expected ? "true" : "false") << ',';
                            write_metrics(procedural_results, generated.trace.metrics);
                            procedural_results << ','; write_rhos(procedural_results, generated.trace); procedural_results << "}\n";
                        }
                    }
                    for (const auto & ex : ep.training) {
                        const auto generated = generate_bank(ctx, vocab, collector, output, bank,
                            procedure_prompt(ex, true), ex.first_target, PRIMARY_BUDGET, 8, true, 0, {});
                        procedural_results << "{\"split\":\"training\",\"condition\":\""
                            << PROCEDURAL_CONDITIONS[condition] << "\",\"seed\":" << ep.seed
                            << ",\"checkpoint\":" << checkpoint << ",\"rule_visible\":true,\"example_id\":\""
                            << ex.id << "\",\"expected\":\"" << ex.expected << "\",\"generated\":\""
                            << json_escape(generated.canonical) << "\",\"correct\":"
                            << (generated.canonical == ex.expected ? "true" : "false") << ',';
                        write_metrics(procedural_results, generated.trace.metrics);
                        procedural_results << ','; write_rhos(procedural_results, generated.trace); procedural_results << "}\n";
                    }
                    if (condition == 1 || condition == 3) {
                        for (size_t i = 0; i < protective_prompts.size(); ++i) {
                            const auto baseline = evaluate(ctx, vocab, collector, protective_prompts[i], 0, true);
                            const auto value = evaluate_bank(ctx, vocab, collector, output, bank, protective_prompts[i],
                                                             baseline.metrics.greedy, PRIMARY_BUDGET);
                            protective << "{\"part\":\"procedural\",\"condition\":\""
                                       << PROCEDURAL_CONDITIONS[condition] << "\",\"seed\":" << ep.seed
                                       << ",\"checkpoint\":" << checkpoint << ",\"prompt_id\":" << i
                                       << ",\"kl_adapted_base\":" << kl_divergence(value.logits, baseline.logits)
                                       << ",\"greedy_changed\":"
                                       << (value.metrics.greedy != baseline.metrics.greedy ? "true" : "false") << "}\n";
                            protective_calls += 2;
                        }
                    }
                    procedural_evaluation_calls += decode_calls - before;
                };

                evaluate_checkpoint(0);
                for (int event = 0; event < 4; ++event) {
                    const auto & ex = ep.training[event];
                    const auto before_norm = bank.frobenius_norm();
                    const auto action = generate_bank(ctx, vocab, collector, output, bank,
                        procedure_prompt(ex, true), ex.first_target, PRIMARY_BUDGET, 8, true, 0, {});
                    const bool action_correct = action.canonical == ex.expected;
                    const int rank_at_trace = bank.components.size();
                    const double norm_at_trace = bank.frobenius_norm();
                    no_update_before_feedback = no_update_before_feedback && rank_at_trace == (int) bank.components.size()
                        && norm_at_trace == bank.frobenius_norm();
                    const llama_token produced = action.answer_token >= 0 ? action.answer_token : action.trace.metrics.greedy;
                    llama_token feedback_target = ex.first_target;
                    std::string feedback_expected = ex.expected;
                    if (condition == 4) {
                        const auto & other = ep.training[(event + 1) % ep.training.size()];
                        feedback_target = other.first_target;
                        feedback_expected = other.expected;
                    }

                    std::vector<component> pending;
                    bool corrected_attempted = false;
                    bool corrected_success = false;
                    generated_result corrected;
                    if (condition == 1) {
                        pending.push_back({normalized_token_direction(output, ex.first_target, produced), action.trace.key,
                                           ETA_POSITIVE, event + 1, ex.id, "blind_positive"});
                    } else if (condition == 2) {
                        if (action_correct) {
                            pending.push_back({normalized_output_row(output, ex.first_target), action.trace.key,
                                               ETA_POSITIVE, event + 1, ex.id, "reward_positive"});
                        } else {
                            pending.push_back({normalized_token_direction(output, produced, ex.first_target), action.trace.key,
                                               -ETA_NEGATIVE, event + 1, ex.id, "reward_negative"});
                        }
                    } else if (condition >= 3) {
                        if (action_correct) {
                            auto q = condition == 5 ? deterministic_random_direction(BENCHMARK_SEED + 10000*ep.seed + event)
                                                    : normalized_output_row(output, ex.first_target);
                            pending.push_back({std::move(q), action.trace.key, ETA_POSITIVE, event + 1, ex.id, "positive_success"});
                        } else {
                            corrected_attempted = true;
                            corrected = generate_bank(ctx, vocab, collector, output, bank,
                                procedure_corrected_prompt(ex, feedback_expected), feedback_target,
                                PRIMARY_BUDGET, 8, true, 0, {});
                            corrected_success = corrected.canonical == feedback_expected;
                            auto negative_q = condition == 5
                                ? deterministic_random_direction(BENCHMARK_SEED + 20000*ep.seed + 2*event)
                                : normalized_token_direction(output, produced, feedback_target);
                            pending.push_back({std::move(negative_q), action.trace.key, -ETA_NEGATIVE,
                                               event + 1, ex.id, "negative_wrong"});
                            if (corrected_success) {
                                auto positive_q = condition == 5
                                    ? deterministic_random_direction(BENCHMARK_SEED + 20000*ep.seed + 2*event + 1)
                                    : normalized_token_direction(output, feedback_target, produced);
                                pending.push_back({std::move(positive_q), corrected.trace.key, ETA_POSITIVE,
                                                   event + 1, ex.id, "positive_validated_retry"});
                            }
                        }
                    }

                    const double positive_update_norm = pending_norm(pending, false);
                    const double negative_update_norm = pending_norm(pending, true);
                    bool compressed = false;
                    if (condition != 0) for (auto & term : pending) compressed = bank.add(std::move(term)) || compressed;
                    procedural_events << std::setprecision(17) << "{\"phase_order\":[\"ACT\",\"TRACE\",\"FEEDBACK\",\"CORRECTED_ATTEMPT\",\"UPDATE\"],"
                        << "\"condition\":\"" << PROCEDURAL_CONDITIONS[condition] << "\",\"seed\":" << ep.seed
                        << ",\"event\":" << event + 1 << ",\"example_id\":\"" << ex.id << "\",\"action_correct\":"
                        << (action_correct ? "true" : "false") << ",\"reward\":" << (action_correct ? 1 : -1)
                        << ",\"update_before_feedback\":false,\"corrected_attempted\":" << (corrected_attempted ? "true" : "false")
                        << ",\"corrected_success\":" << (corrected_success ? "true" : "false")
                        << ",\"positive_update_norm\":" << positive_update_norm
                        << ",\"negative_update_norm\":" << negative_update_norm
                        << ",\"state_norm_before\":" << before_norm << ",\"state_norm_after\":" << bank.frobenius_norm()
                        << ",\"rank_before\":" << rank_at_trace << ",\"rank_after\":" << bank.components.size()
                        << ",\"compression_event\":" << (compressed ? "true" : "false") << "}\n";
                    if (corrected_attempted) {
                        procedural_corrected << "{\"condition\":\"" << PROCEDURAL_CONDITIONS[condition]
                            << "\",\"seed\":" << ep.seed << ",\"event\":" << event + 1 << ",\"example_id\":\""
                            << ex.id << "\",\"expected\":\"" << feedback_expected << "\",\"generated\":\""
                            << json_escape(corrected.canonical) << "\",\"corrected_success\":"
                            << (corrected_success ? "true" : "false") << ",\"positive_eligibility_created\":"
                            << (corrected_success ? "true" : "false") << "}\n";
                    }
                    ++update_rows;
                    if (event == 0 || event == 1 || event == 3) evaluate_checkpoint(event + 1);
                }

                clear_internal_injection(ctx);
                const auto direct = evaluate(ctx, vocab, collector, procedure_prompt(ep.heldout.front(), true), ep.heldout.front().first_target, false);
                bank.clear();
                clear_internal_injection(ctx);
                const auto reset = evaluate(ctx, vocab, collector, procedure_prompt(ep.heldout.front(), true), ep.heldout.front().first_target, false);
                reset_max = std::max(reset_max, max_logit_difference(direct.logits, reset.logits));
            }
        }

        reasoning_teaching_calls_start = decode_calls;
        // PART II: reasoning-enabled versus direct teacher correction on a hidden offset.
        if (!skip_reasoning) for (int condition = 0; condition < (int) REASONING_CONDITIONS.size(); ++condition) {
            const bool reasoning_enabled = condition != 1;
            for (const auto & ep : reasoning_episodes) {
                if (reasoning_offset_filter != nullptr &&
                    ep.offset != std::stoi(reasoning_offset_filter)) {
                    continue;
                }
                fast_bank bank;
                auto evaluate_checkpoint = [&](int checkpoint) {
                    log_history("reasoning", REASONING_CONDITIONS[condition], ep.offset, ep.seed, checkpoint, bank);
                    const uint64_t before = decode_calls;
                    for (const auto & ex : ep.heldout) {
                        bool supports = false;
                        const auto rendered = render_chat(chat_templates.get(), reasoning_user_prompt(ep, ex), reasoning_enabled, supports);
                        const auto generated = generate_bank(ctx, vocab, collector, output, bank, rendered,
                            ex.target, PRIMARY_BUDGET, reasoning_enabled ? REASONING_TOKEN_BUDGET : 12,
                            false, ep.offset, reasoning_symbols);
                        max_budget_observed = std::max(max_budget_observed, generated.trace.budget);
                        reasoning_results << "{\"split\":\"heldout\",\"condition\":\"" << REASONING_CONDITIONS[condition]
                            << "\",\"offset\":" << ep.offset << ",\"seed\":" << ep.seed << ",\"checkpoint\":" << checkpoint
                            << ",\"example_id\":\"" << ex.id << "\",\"reasoning_enabled\":" << (reasoning_enabled ? "true" : "false")
                            << ",\"template_supports_thinking\":" << (supports ? "true" : "false")
                            << ",\"reasoning_occurred\":" << (generated.reasoning_occurred ? "true" : "false")
                            << ",\"reasoning_completed\":" << (generated.reasoning_completed ? "true" : "false")
                            << ",\"reasoning_forced_close\":" << (generated.reasoning_forced_close ? "true" : "false")
                            << ",\"reasoning_tokens\":" << generated.reasoning_tokens << ",\"answer_tokens\":" << generated.answer_tokens
                            << ",\"correct_hypothesis\":" << (generated.correct_hypothesis ? "true" : "false")
                            << ",\"correct\":" << (generated.correct ? "true" : "false") << ',';
                        write_metrics(reasoning_results, generated.trace.metrics);
                        reasoning_results << ','; write_rhos(reasoning_results, generated.trace); reasoning_results << "}\n";
                    }
                    if (condition == 2) {
                        for (size_t i = 0; i < protective_prompts.size(); ++i) {
                            const auto baseline = evaluate(ctx, vocab, collector, protective_prompts[i], 0, true);
                            const auto value = evaluate_bank(ctx, vocab, collector, output, bank, protective_prompts[i],
                                                             baseline.metrics.greedy, PRIMARY_BUDGET);
                            protective << "{\"part\":\"reasoning\",\"condition\":\"teacher_reasoning_on\",\"offset\":"
                                       << ep.offset << ",\"seed\":" << ep.seed << ",\"checkpoint\":" << checkpoint
                                       << ",\"prompt_id\":" << i << ",\"kl_adapted_base\":"
                                       << kl_divergence(value.logits, baseline.logits) << ",\"greedy_changed\":"
                                       << (value.metrics.greedy != baseline.metrics.greedy ? "true" : "false") << "}\n";
                            protective_calls += 2;
                        }
                    }
                    reasoning_evaluation_calls += decode_calls - before;
                };

                evaluate_checkpoint(0);
                for (int event = 0; event < 4; ++event) {
                    const auto & ex = ep.training[event];
                    bool supports = false;
                    const auto prompt = render_chat(chat_templates.get(), reasoning_user_prompt(ep, ex), reasoning_enabled, supports);
                    const auto action = generate_bank(ctx, vocab, collector, output, bank, prompt, ex.target,
                        PRIMARY_BUDGET, reasoning_enabled ? REASONING_TOKEN_BUDGET : 12,
                        false, ep.offset, reasoning_symbols);
                    // Behavioral success is the exact final answer. Whether the
                    // generated rationale explicitly names the offset is logged
                    // separately and must not veto an otherwise correct retry.
                    const bool action_validated = action.correct;
                    const int rank_at_trace = bank.components.size();
                    const double norm_at_trace = bank.frobenius_norm();
                    no_update_before_feedback = no_update_before_feedback && rank_at_trace == (int) bank.components.size()
                        && norm_at_trace == bank.frobenius_norm();
                    const llama_token produced = action.answer_token >= 0 ? action.answer_token : action.trace.metrics.greedy;
                    int teacher_target_index = ex.target_index;
                    llama_token teacher_target = ex.target;
                    if (condition == 3) {
                        teacher_target_index = (ex.target_index + 1) % 8;
                        teacher_target = reasoning_symbols[teacher_target_index];
                    }
                    std::vector<component> pending;
                    bool corrected_attempted = false;
                    bool corrected_success = false;
                    generated_result corrected;
                    if (condition != 0) {
                        if (action_validated) {
                            auto q = condition == 4 ? deterministic_random_direction(BENCHMARK_SEED + 30000*ep.offset + event)
                                                    : normalized_output_row(output, ex.target);
                            pending.push_back({std::move(q), action.trace.key, ETA_POSITIVE, event + 1, ex.id, "positive_success"});
                        } else {
                            corrected_attempted = true;
                            bool corrected_supports = false;
                            const auto corrected_prompt = render_chat(chat_templates.get(),
                                reasoning_corrected_user_prompt(ep, ex, teacher_target_index), reasoning_enabled, corrected_supports);
                            corrected = generate_bank(ctx, vocab, collector, output, bank, corrected_prompt, teacher_target,
                                PRIMARY_BUDGET, reasoning_enabled ? REASONING_TOKEN_BUDGET : 12,
                                false, ep.offset, reasoning_symbols);
                            corrected_success = corrected.answer_token == teacher_target;
                            auto negative_q = condition == 4
                                ? deterministic_random_direction(BENCHMARK_SEED + 40000*ep.offset + 2*event)
                                : normalized_token_direction(output, produced, teacher_target);
                            pending.push_back({std::move(negative_q), action.trace.key, -ETA_NEGATIVE,
                                               event + 1, ex.id, "negative_wrong"});
                            if (corrected_success) {
                                auto positive_q = condition == 4
                                    ? deterministic_random_direction(BENCHMARK_SEED + 40000*ep.offset + 2*event + 1)
                                    : normalized_token_direction(output, teacher_target, produced);
                                pending.push_back({std::move(positive_q), corrected.trace.key, ETA_POSITIVE,
                                                   event + 1, ex.id, "positive_validated_reasoning"});
                            }
                        }
                    }
                    const double positive_update_norm = pending_norm(pending, false);
                    const double negative_update_norm = pending_norm(pending, true);
                    bool compressed = false;
                    for (auto & term : pending) compressed = bank.add(std::move(term)) || compressed;
                    reasoning_events << std::setprecision(17) << "{\"phase_order\":[\"ACT\",\"TRACE\",\"FEEDBACK\",\"CORRECTED_ATTEMPT\",\"UPDATE\"],"
                        << "\"condition\":\"" << REASONING_CONDITIONS[condition] << "\",\"offset\":" << ep.offset
                        << ",\"seed\":" << ep.seed << ",\"event\":" << event + 1 << ",\"example_id\":\"" << ex.id
                        << "\",\"reasoning_enabled\":" << (reasoning_enabled ? "true" : "false")
                        << ",\"action_correct\":" << (action.correct ? "true" : "false")
                        << ",\"action_validated\":" << (action_validated ? "true" : "false")
                        << ",\"reasoning_occurred\":" << (action.reasoning_occurred ? "true" : "false")
                        << ",\"reasoning_completed\":" << (action.reasoning_completed ? "true" : "false")
                        << ",\"reasoning_forced_close\":" << (action.reasoning_forced_close ? "true" : "false")
                        << ",\"correct_hypothesis\":" << (action.correct_hypothesis ? "true" : "false")
                        << ",\"update_before_feedback\":false,\"corrected_attempted\":" << (corrected_attempted ? "true" : "false")
                        << ",\"corrected_success\":" << (corrected_success ? "true" : "false")
                        << ",\"positive_update_norm\":" << positive_update_norm
                        << ",\"negative_update_norm\":" << negative_update_norm
                        << ",\"rank_before\":" << rank_at_trace << ",\"rank_after\":" << bank.components.size()
                        << ",\"compression_event\":" << (compressed ? "true" : "false") << "}\n";
                    if (corrected_attempted) {
                        reasoning_corrected << "{\"condition\":\"" << REASONING_CONDITIONS[condition]
                            << "\",\"offset\":" << ep.offset << ",\"seed\":" << ep.seed << ",\"event\":" << event + 1
                            << ",\"example_id\":\"" << ex.id << "\",\"reasoning_enabled\":" << (reasoning_enabled ? "true" : "false")
                            << ",\"reasoning_occurred\":" << (corrected.reasoning_occurred ? "true" : "false")
                            << ",\"reasoning_completed\":" << (corrected.reasoning_completed ? "true" : "false")
                            << ",\"reasoning_forced_close\":" << (corrected.reasoning_forced_close ? "true" : "false")
                            << ",\"reasoning_tokens\":" << corrected.reasoning_tokens
                            << ",\"correct_hypothesis\":" << (corrected.correct_hypothesis ? "true" : "false")
                            << ",\"corrected_success\":" << (corrected_success ? "true" : "false")
                            << ",\"positive_eligibility_created\":" << (corrected_success ? "true" : "false") << "}\n";
                    }
                    ++update_rows;
                    if (event == 0 || event == 1 || event == 3) evaluate_checkpoint(event + 1);
                }
            }
        }

        fast_bank empty;
        const auto & validation_ex = procedure_episodes.front().heldout.front();
        const auto validation_prompt = procedure_prompt(validation_ex, true);
        clear_internal_injection(ctx);
        const auto direct = evaluate(ctx, vocab, collector, validation_prompt, validation_ex.first_target, true);
        const auto zero = evaluate_bank(ctx, vocab, collector, output, empty, validation_prompt, validation_ex.first_target, PRIMARY_BUDGET);
        zero_state_max = max_logit_difference(direct.logits, zero.logits);

        std::ofstream proc_benchmark(root/"procedural_benchmark.json", std::ios::trunc);
        proc_benchmark << "{\"seed\":" << BENCHMARK_SEED << ",\"procedure\":\"THIRD-FIRST-SECOND\",\"rule_explicit_during_teaching\":true,"
                       << "\"seeds\":" << PROCEDURAL_SEEDS << ",\"training_per_episode\":4,\"heldout_per_episode\":8,"
                       << "\"heldout_exact_prompt_seen\":false,\"heldout_exact_pair_seen\":false,\"same_latent_rule\":true}\n";
        std::ofstream reasoning_benchmark(root/"reasoning_benchmark.json", std::ios::trunc);
        reasoning_benchmark << "{\"seed\":" << BENCHMARK_SEED << ",\"rule\":\"y=(x+k) mod 8\",\"offsets\":[1,3,6],"
                            << "\"seeds_per_offset\":" << REASONING_SEEDS << ",\"training_per_episode\":4,\"heldout_per_episode\":4,"
                            << "\"runtime_offset_filter\":" << (reasoning_offset_filter ? reasoning_offset_filter : "null") << ','
                            << "\"thinking_token_budget\":" << REASONING_TOKEN_BUDGET
                            << ",\"forced_think_close_after_tokens\":" << REASONING_THINK_BUDGET << ','
                            << "\"reasoning_template_supports_enable_thinking\":" << (template_supports_thinking ? "true" : "false")
                            << ",\"teacher_explains_offset\":true}\n";
        std::ofstream tokens(root/"token_validation.json", std::ios::trunc);
        tokens << "{\"all_symbols_single_token\":true,\"symbols\":[";
        for (int i = 0; i < 12; ++i) tokens << (i ? "," : "") << "{\"symbol\":\"" << (char) ('A' + i)
            << "\",\"token_id\":" << procedure_symbols[i] << "}";
        tokens << "],\"procedure_answers_multitoken\":true,\"procedure_scoring\":\"canonical_exact_generated_string\"}\n";

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ofstream validation(root/"validation.json", std::ios::trunc);
        validation << std::setprecision(17) << "{\"zero_state_max_logit_difference\":" << zero_state_max
                   << ",\"reset_max_logit_difference\":" << reset_max
                   << ",\"no_update_before_feedback\":" << (no_update_before_feedback ? "true" : "false")
                   << ",\"positive_eligibility_requires_validated_success\":true,\"max_global_authority\":" << max_budget_observed
                   << ",\"authority_limit\":0.05,\"base_model_mutated\":false,\"backpropagation\":false,\"optimizer\":false}\n";
        std::ofstream runtime(root/"runtime_metrics.json", std::ios::trunc);
        runtime << "{\"wall_seconds\":" << elapsed << ",\"model_decode_calls\":" << decode_calls
                << ",\"procedural_teaching_and_total_calls\":" << reasoning_teaching_calls_start - procedural_teaching_calls_start
                << ",\"procedural_evaluation_calls\":" << procedural_evaluation_calls
                << ",\"reasoning_teaching_and_total_calls\":" << decode_calls - reasoning_teaching_calls_start
                << ",\"reasoning_evaluation_calls\":" << reasoning_evaluation_calls
                << ",\"protective_calls\":" << protective_calls << ",\"update_rows\":" << update_rows
                << ",\"gpu_layers\":" << opts.gpu_layers << ",\"ctx_size\":" << opts.ctx_size << "}\n";
        std::ofstream memory(root/"memory_metrics.json", std::ios::trunc);
        memory << "{\"fast_rank\":8,\"persistent_sidecar_bytes_fp32\":163840,\"eligibility_trace_bytes_fp32\":20480,"
               << "\"temporary_correction_vector_bytes_fp32\":10240,\"reward_predictor_bytes\":0,\"full_dense_matrix_bytes\":0,"
               << "\"model_and_kv_memory_excluded\":true}\n";
        std::ofstream raw_manifest(root/"raw_manifest.json", std::ios::trunc);
        const char * sparse = std::getenv("LLAMA_SPARSE_RESIDUAL");
        raw_manifest << "{\"experiment\":\"qwen35_teacher_curriculum_plasticity\",\"model\":\"" << json_escape(opts.model)
                     << "\",\"sparse_residual_enabled\":" << (sparse ? "true" : "false")
                     << ",\"primary_budget\":0.03,\"secondary_budget\":0.05,\"fast_rank\":8,\"eta_positive\":0.05,"
                     << "\"eta_negative\":0.05,\"injection_sites\":[\"block_24\",\"block_28\",\"final\"],"
                     << "\"weights\":[0.23072,0.38322,0.89438],\"base_model_mutated\":false,\"backpropagation\":false,"
                     << "\"autograd\":false,\"optimizer\":false,\"reasoning_template_supports_thinking\":"
                     << (template_supports_thinking ? "true" : "false") << "}\n";

        clear_internal_injection(ctx);
        llama_free(ctx);
        llama_model_free(public_model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
