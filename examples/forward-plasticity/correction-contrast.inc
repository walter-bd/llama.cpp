// Isolated correction-triggered contrast benchmark. Included by forward-plasticity.cpp
// so it can reuse the validated activation collector and mutable LoRA tensor path.

struct contrast_case {
    std::string id;
    std::string role;
    int32_t first = 0;
    int32_t second = 0;
    std::string target;
};

struct contrast_trace {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> logits;
    llama_token greedy = -1;
    llama_token response = -1;
    llama_token target = -1;
    double probability = 0;
    double nll = 0;
    int32_t target_rank = 0;
    double margin = 0;
};

struct contrast_term {
    std::vector<float> x;
    std::vector<float> y;
    double coefficient = 0;
};

struct contrast_bank {
    int64_t d_in = 0;
    int64_t d_out = 0;
    int64_t rank = 0;
    std::vector<float> a;
    std::vector<float> b;

    contrast_bank(int64_t input, int64_t output, int64_t slots) :
        d_in(input), d_out(output), rank(slots), a(input*slots, 0.0f), b(slots*output, 0.0f) {}
};

struct contrast_update_result {
    double raw_positive_norm = 0;
    double raw_negative_norm = 0;
    double raw_contrast_norm = 0;
    double positive_negative_cosine = 0;
    double normalized_contrast_norm = 0;
    double control_match_scale = 1;
    double requested_update_rho = 0;
    double actual_update_rho = 0;
    double final_state_rho = 0;
    double state_frobenius_norm = 0;
    bool clipped = false;
    double clip_scale = 1;
    double preexisting_state_scale = 1;
    double final_state_scale = 1;
};

std::string json_escape_contrast(const std::string & value) {
    std::ostringstream result;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': result << "\\\\"; break;
            case '"': result << "\\\""; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (ch < 0x20) {
                    result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int) ch << std::dec;
                } else {
                    result << ch;
                }
        }
    }
    return result.str();
}

std::vector<std::string> split_tabs_contrast(const std::string & line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream input(line);
    while (std::getline(input, field, '\t')) fields.push_back(field);
    return fields;
}

std::vector<contrast_case> load_contrast_cases(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open correction task TSV: " + path);
    std::vector<contrast_case> result;
    std::string line;
    bool header = true;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (header) {
            header = false;
            if (line == "id\trole\tfirst\tsecond\ttarget") continue;
        }
        const auto fields = split_tabs_contrast(line);
        if (fields.size() != 5) throw std::runtime_error("invalid correction task TSV row");
        result.push_back({ fields[0], fields[1], std::stoi(fields[2]), std::stoi(fields[3]), fields[4] });
    }
    return result;
}

std::string negative_prompt_contrast(const contrast_case & item) {
    std::ostringstream prompt;
    prompt << "VEXA protocol trial. Inputs: " << item.first << " and " << item.second
           << ". Return exactly one uppercase protocol symbol, with no explanation. Answer:";
    return prompt.str();
}

std::string positive_prompt_contrast(const contrast_case & item) {
    const int32_t residue = (item.first + 2*item.second + 1) % 5;
    std::ostringstream prompt;
    prompt << "VEXA protocol trial. Inputs: " << item.first << " and " << item.second << ".\n"
           << "Correction: VEXA computes (first + twice second + 1) modulo 5. "
           << "Residues 0, 1, 2, 3, 4 map to A, B, C, D, E. "
           << "For this trial the residue is " << residue << " and the corrected symbol is " << item.target << ".\n"
           << "Now return exactly that one uppercase protocol symbol, with no explanation. Answer:";
    return prompt.str();
}

llama_token one_token_contrast(const llama_vocab * vocab, const std::string & target) {
    const auto tokens = tokenize(vocab, " " + target, false);
    if (tokens.size() != 1) throw std::runtime_error("contrast target is not one token: " + target);
    return tokens[0];
}

double dot_contrast(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) throw std::runtime_error("contrast vector dimension mismatch");
    double result = 0;
    for (size_t i = 0; i < lhs.size(); ++i) result += (double) lhs[i]*rhs[i];
    return result;
}

std::vector<float> normalized_contrast(const std::vector<float> & values) {
    const double norm = l2_norm(values);
    if (!(norm > 0)) throw std::runtime_error("cannot normalize zero contrast vector");
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result[i] = values[i]/norm;
    return result;
}

std::string token_piece_contrast(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(64);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        buffer.resize(-count);
        count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    if (count < 0) return "";
    return std::string(buffer.data(), count);
}

contrast_trace capture_response_contrast(
        llama_context * ctx,
        const llama_vocab * vocab,
        activation_collector & collector,
        const std::string & prompt,
        llama_token target,
        bool force_target_response) {
    llama_memory_clear(llama_get_memory(ctx), true);
    collector.enabled = false;
    collector.reset();
    auto prompt_tokens = tokenize(vocab, prompt, true);
    if (prompt_tokens.empty()) throw std::runtime_error("empty contrast prompt");
    if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size())) != 0) {
        throw std::runtime_error("contrast prompt decode failed");
    }
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    const float * logits_ptr = llama_get_logits_ith(ctx, -1);
    if (!logits_ptr) throw std::runtime_error("missing contrast logits");
    contrast_trace result;
    result.logits.assign(logits_ptr, logits_ptr + vocab_size);
    result.target = target;
    result.greedy = std::max_element(result.logits.begin(), result.logits.end()) - result.logits.begin();
    result.response = force_target_response ? target : result.greedy;
    const double z = logsumexp(result.logits.data(), vocab_size);
    result.nll = z - result.logits[target];
    result.probability = std::exp(-result.nll);
    result.target_rank = 1;
    float best_other = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < vocab_size; ++i) {
        if (result.logits[i] > result.logits[target]) ++result.target_rank;
        if (i != target) best_other = std::max(best_other, result.logits[i]);
    }
    result.margin = result.logits[target] - best_other;

    // Capture only the response token. The causal state now sees the correction,
    // while the shared original problem prefix is excluded from both traces.
    collector.reset();
    collector.enabled = true;
    if (llama_decode(ctx, llama_batch_get_one(&result.response, 1)) != 0) {
        collector.enabled = false;
        throw std::runtime_error("contrast response decode failed");
    }
    llama_synchronize(ctx);
    collector.enabled = false;
    if (collector.n_x != 1 || collector.n_y != 1) {
        throw std::runtime_error("contrast capture expected exactly one response-token vector");
    }
    result.x.resize(collector.sum_x.size());
    result.y.resize(collector.sum_y.size());
    for (size_t i = 0; i < result.x.size(); ++i) result.x[i] = collector.sum_x[i];
    for (size_t i = 0; i < result.y.size(); ++i) result.y[i] = collector.sum_y[i];
    return result;
}

double outer_inner_contrast(
        const std::vector<float> & y1, const std::vector<float> & x1,
        const std::vector<float> & y2, const std::vector<float> & x2) {
    return dot_contrast(y1, y2)*dot_contrast(x1, x2);
}

double raw_update_inner_contrast(
        const contrast_trace & p1, const contrast_trace & n1,
        const contrast_trace & p2, const contrast_trace & n2) {
    return outer_inner_contrast(p1.y, p1.x, p2.y, p2.x)
         - outer_inner_contrast(p1.y, p1.x, n2.y, n2.x)
         - outer_inner_contrast(n1.y, n1.x, p2.y, p2.x)
         + outer_inner_contrast(n1.y, n1.x, n2.y, n2.x);
}

std::vector<float> response_contrast(const contrast_bank & bank, const std::vector<float> & x) {
    std::vector<float> result(bank.d_out, 0.0f);
    for (int64_t slot = 0; slot < bank.rank; ++slot) {
        double coefficient = 0;
        for (int64_t i = 0; i < bank.d_in; ++i) coefficient += bank.a[i + bank.d_in*slot]*x[i];
        for (int64_t j = 0; j < bank.d_out; ++j) result[j] += bank.b[bank.rank*j + slot]*coefficient;
    }
    return result;
}

std::vector<float> response_terms_contrast(const std::vector<contrast_term> & terms, const std::vector<float> & x) {
    if (terms.empty()) return {};
    std::vector<float> result(terms[0].y.size(), 0.0f);
    for (const auto & term : terms) {
        const double projection = dot_contrast(term.x, x)*term.coefficient;
        for (size_t j = 0; j < result.size(); ++j) result[j] += term.y[j]*projection;
    }
    return result;
}

double bank_frobenius_contrast(const contrast_bank & bank) {
    double norm2 = 0;
    for (int64_t s = 0; s < bank.rank; ++s) {
        for (int64_t t = 0; t < bank.rank; ++t) {
            double aa = 0;
            double bb = 0;
            for (int64_t i = 0; i < bank.d_in; ++i) aa += bank.a[i + bank.d_in*s]*bank.a[i + bank.d_in*t];
            for (int64_t j = 0; j < bank.d_out; ++j) bb += bank.b[bank.rank*j + s]*bank.b[bank.rank*j + t];
            norm2 += aa*bb;
        }
    }
    return std::sqrt(std::max(0.0, norm2));
}

double terms_frobenius_contrast(const std::vector<contrast_term> & terms) {
    double norm2 = 0;
    for (const auto & lhs : terms) {
        for (const auto & rhs : terms) {
            norm2 += lhs.coefficient*rhs.coefficient*outer_inner_contrast(lhs.y, lhs.x, rhs.y, rhs.x);
        }
    }
    return std::sqrt(std::max(0.0, norm2));
}

void scale_terms_contrast(std::vector<contrast_term> & terms, double scale) {
    for (auto & term : terms) term.coefficient *= scale;
}

double trust_scale_contrast(
        const std::vector<float> & old_response,
        const std::vector<float> & delta,
        double maximum) {
    std::vector<float> candidate(old_response.size());
    for (size_t i = 0; i < candidate.size(); ++i) candidate[i] = old_response[i] + delta[i];
    if (l2_norm(candidate) <= maximum) return 1.0;
    if (l2_norm(old_response) > maximum*(1.0 + 1e-6)) return 0.0;
    double low = 0;
    double high = 1;
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double middle = (low + high)/2;
        for (size_t i = 0; i < candidate.size(); ++i) candidate[i] = old_response[i] + middle*delta[i];
        if (l2_norm(candidate) <= maximum) low = middle;
        else high = middle;
    }
    return low;
}

void install_terms_contrast(
        contrast_bank & bank,
        int32_t event,
        const std::vector<contrast_term> & terms) {
    const int64_t first_slot = 2*event;
    for (int64_t local = 0; local < 2; ++local) {
        const int64_t slot = first_slot + local;
        if (slot >= bank.rank) throw std::runtime_error("contrast bank exhausted");
        for (int64_t i = 0; i < bank.d_in; ++i) bank.a[i + bank.d_in*slot] = 0;
        for (int64_t j = 0; j < bank.d_out; ++j) bank.b[bank.rank*j + slot] = 0;
        if (local >= (int64_t) terms.size()) continue;
        const auto & term = terms[local];
        for (int64_t i = 0; i < bank.d_in; ++i) bank.a[i + bank.d_in*slot] = term.x[i];
        for (int64_t j = 0; j < bank.d_out; ++j) bank.b[bank.rank*j + slot] = term.coefficient*term.y[j];
    }
}

std::vector<contrast_term> formula_terms_contrast(
        const std::string & condition,
        const contrast_trace & positive,
        const contrast_trace & negative,
        const contrast_trace & shuffled_positive,
        double epsilon) {
    auto term = [&](const contrast_trace & trace, double coefficient) {
        return contrast_term { normalized_contrast(trace.x), normalized_contrast(trace.y), epsilon*coefficient };
    };
    if (condition == "contrastive") return { term(positive, 1), term(negative, -1) };
    if (condition == "positive_only") return { term(positive, 1) };
    if (condition == "shuffled_correction") return { term(shuffled_positive, 1), term(negative, -1) };
    if (condition == "sign_reversed") return { term(positive, -1), term(negative, 1) };
    throw std::runtime_error("unknown contrast condition: " + condition);
}

contrast_update_result apply_correction_contrast(
        const std::string & condition,
        double epsilon,
        double ratio_cap,
        int32_t event,
        const contrast_trace & positive,
        const contrast_trace & negative,
        const contrast_trace & shuffled_positive,
        llama_adapter_lora_weight & weight,
        contrast_bank & bank) {
    contrast_update_result result;
    result.raw_positive_norm = l2_norm(positive.y)*l2_norm(positive.x);
    result.raw_negative_norm = l2_norm(negative.y)*l2_norm(negative.x);
    const double raw_inner = outer_inner_contrast(positive.y, positive.x, negative.y, negative.x);
    result.raw_contrast_norm = std::sqrt(std::max(0.0,
        result.raw_positive_norm*result.raw_positive_norm +
        result.raw_negative_norm*result.raw_negative_norm - 2*raw_inner));
    result.positive_negative_cosine = raw_inner/(result.raw_positive_norm*result.raw_negative_norm + 1e-30);

    auto proper = formula_terms_contrast("contrastive", positive, negative, shuffled_positive, epsilon);
    auto terms = formula_terms_contrast(condition, positive, negative, shuffled_positive, epsilon);
    result.normalized_contrast_norm = terms_frobenius_contrast(proper);
    const double target_response = l2_norm(response_terms_contrast(proper, negative.x));
    const double candidate_response = l2_norm(response_terms_contrast(terms, negative.x));
    if (condition != "contrastive" && candidate_response > 0) {
        result.control_match_scale = target_response/candidate_response;
        scale_terms_contrast(terms, result.control_match_scale);
    }

    const auto original_response = response_contrast(bank, negative.x);
    auto old_response = original_response;
    auto delta = response_terms_contrast(terms, negative.x);
    const double base_norm = l2_norm(negative.y);
    const double maximum = ratio_cap*base_norm;
    const double old_norm = l2_norm(old_response);
    if (old_norm > maximum) {
        result.preexisting_state_scale = maximum/(old_norm + 1e-30)*0.99999;
        for (float & value : bank.b) value *= result.preexisting_state_scale;
        set_bank(weight, bank.a, bank.b);
        old_response = response_contrast(bank, negative.x);
        result.clipped = true;
    }
    result.requested_update_rho = l2_norm(delta)/(base_norm + 1e-30);
    result.clip_scale = trust_scale_contrast(old_response, delta, maximum);
    if (result.clip_scale < 1.0 - 1e-9) {
        result.clipped = true;
        scale_terms_contrast(terms, result.clip_scale);
        for (float & value : delta) value *= result.clip_scale;
    }
    install_terms_contrast(bank, event, terms);
    set_bank(weight, bank.a, bank.b);
    auto final_response = response_contrast(bank, negative.x);
    const double candidate_norm = l2_norm(final_response);
    if (candidate_norm > maximum) {
        result.final_state_scale = maximum/(candidate_norm + 1e-30)*0.99999;
        for (float & value : bank.b) value *= result.final_state_scale;
        set_bank(weight, bank.a, bank.b);
        final_response = response_contrast(bank, negative.x);
        result.clipped = true;
    }
    std::vector<float> actual_delta(final_response.size());
    for (size_t i = 0; i < actual_delta.size(); ++i) {
        actual_delta[i] = final_response[i] - original_response[i];
    }
    result.actual_update_rho = l2_norm(actual_delta)/(base_norm + 1e-30);
    result.final_state_rho = l2_norm(final_response)/(base_norm + 1e-30);
    if (result.final_state_rho > ratio_cap*(1.0 + 1e-5)) {
        throw std::runtime_error("accumulated contrast bank violated trust region");
    }
    result.state_frobenius_norm = bank_frobenius_contrast(bank);
    return result;
}

void write_measurement_contrast(
        std::ofstream & output,
        const llama_vocab * vocab,
        const std::string & layer,
        double epsilon,
        const std::string & condition,
        int32_t checkpoint,
        const std::string & split,
        const contrast_case & item,
        const contrast_trace & trace,
        const contrast_bank & bank) {
    const double ratio = l2_norm(response_contrast(bank, trace.x))/(l2_norm(trace.y) + 1e-30);
    output << "{\"row_type\":\"measurement\",\"layer\":\"" << json_escape_contrast(layer)
           << "\",\"epsilon\":" << epsilon << ",\"condition\":\"" << condition
           << "\",\"checkpoint\":" << checkpoint << ",\"split\":\"" << split
           << "\",\"item_id\":\"" << item.id << "\",\"first\":" << item.first
           << ",\"second\":" << item.second << ",\"target\":\"" << item.target
           << "\",\"target_token\":" << trace.target << ",\"greedy_token\":" << trace.greedy
           << ",\"greedy_text\":\"" << json_escape_contrast(token_piece_contrast(vocab, trace.greedy))
           << "\",\"correct\":" << (trace.greedy == trace.target ? "true" : "false")
           << ",\"target_probability\":" << trace.probability << ",\"target_rank\":" << trace.target_rank
           << ",\"target_margin\":" << trace.margin << ",\"nll\":" << trace.nll
           << ",\"plastic_base_ratio\":" << ratio << "}\n";
}

std::vector<contrast_trace> evaluate_cases_contrast(
        llama_context * ctx,
        const llama_vocab * vocab,
        activation_collector & collector,
        const std::vector<contrast_case> & cases) {
    std::vector<contrast_trace> result;
    result.reserve(cases.size());
    for (const auto & item : cases) {
        const llama_token target = one_token_contrast(vocab, item.target);
        result.push_back(capture_response_contrast(
            ctx, vocab, collector, negative_prompt_contrast(item), target, false));
    }
    return result;
}

struct unrelated_contrast_result {
    double mean_nll = 0;
    double mean_kl = 0;
};

unrelated_contrast_result evaluate_unrelated_contrast(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::vector<score_result> & baseline) {
    static const std::vector<std::pair<std::string, std::string>> prompts = {
        { "The capital of France is", " Paris" },
        { "Two plus two equals", " 4" },
        { "The color of a clear daytime sky is", " blue" },
        { "Water freezes at zero degrees", " Celsius" },
    };
    unrelated_contrast_result result;
    for (size_t i = 0; i < prompts.size(); ++i) {
        const auto current = score_continuation(ctx, vocab, prompts[i].first, prompts[i].second);
        result.mean_nll += current.mean_nll;
        result.mean_kl += kl_divergence(baseline[i].first_logits, current.first_logits);
    }
    result.mean_nll /= prompts.size();
    result.mean_kl /= prompts.size();
    return result;
}

std::vector<score_result> baseline_unrelated_contrast(llama_context * ctx, const llama_vocab * vocab) {
    static const std::vector<std::pair<std::string, std::string>> prompts = {
        { "The capital of France is", " Paris" },
        { "Two plus two equals", " 4" },
        { "The color of a clear daytime sky is", " blue" },
        { "Water freezes at zero degrees", " Celsius" },
    };
    std::vector<score_result> result;
    for (const auto & prompt : prompts) result.push_back(score_continuation(ctx, vocab, prompt.first, prompt.second));
    return result;
}

void run_correction_contrast(
        const options & opts,
        llama_context * ctx,
        const llama_vocab * vocab,
        activation_collector & collector,
        llama_adapter_lora * adapter,
        llama_adapter_lora_weight & weight) {
    namespace fs = std::filesystem;
    if (opts.rank != 8) throw std::runtime_error("correction contrast pilot requires --rank 8");
    const int64_t d_in = weight.a->ne[0];
    const int64_t d_out = weight.b->ne[1];
    const int64_t rank = weight.a->ne[1];
    if (rank != 8) throw std::runtime_error("correction contrast adapter must have rank 8");
    fs::create_directories(opts.correction_contrast_dir);
    fs::create_directories(fs::path(opts.correction_contrast_dir)/"raw_results");
    std::ofstream raw(fs::path(opts.correction_contrast_dir)/"raw_results"/"results.jsonl");
    if (!raw) throw std::runtime_error("cannot create correction result JSONL");
    raw << std::setprecision(12);

    std::vector<contrast_case> candidates;
    std::vector<contrast_case> heldout;
    for (const auto & item : load_contrast_cases(opts.correction_task_file)) {
        if (item.role == "train_candidate") candidates.push_back(item);
        else if (item.role == "heldout") heldout.push_back(item);
    }
    if (heldout.size() != 8) throw std::runtime_error("correction pilot requires exactly 8 held-out cases");

    reset_bank(weight);
    llama_adapter_lora * adapters[] = { adapter };
    float scales[] = { 1.0f };
    if (llama_set_adapters_lora(ctx, adapters, 1, scales) != 0) throw std::runtime_error("contrast adapter activation failed");
    const auto unrelated_baseline = baseline_unrelated_contrast(ctx, vocab);

    std::vector<contrast_case> training;
    std::vector<contrast_trace> negative;
    std::vector<contrast_trace> positive;
    for (const auto & item : candidates) {
        const llama_token target = one_token_contrast(vocab, item.target);
        auto neg = capture_response_contrast(ctx, vocab, collector, negative_prompt_contrast(item), target, false);
        auto pos = capture_response_contrast(ctx, vocab, collector, positive_prompt_contrast(item), target, true);
        const bool negative_wrong = neg.greedy != target;
        const bool corrected_prompt_predicts_target = pos.greedy == target;
        raw << "{\"row_type\":\"candidate_preflight\",\"layer\":\"" << opts.adapter_tensor
            << "\",\"item_id\":\"" << item.id << "\",\"negative_wrong\":"
            << (negative_wrong ? "true" : "false") << ",\"positive_greedy_correct\":"
            << (corrected_prompt_predicts_target ? "true" : "false") << "}\n";
        if (negative_wrong && corrected_prompt_predicts_target && training.size() < 4) {
            training.push_back(item);
            negative.push_back(std::move(neg));
            positive.push_back(std::move(pos));
        }
    }
    if (training.size() != 4) throw std::runtime_error("fewer than four valid wrong-to-corrected training cases");

    // Zero-adapter and reset checks use the first held-out decision logits.
    if (llama_set_adapters_lora(ctx, nullptr, 0, nullptr) != 0) throw std::runtime_error("cannot disable adapter");
    const auto no_adapter = capture_response_contrast(ctx, vocab, collector, negative_prompt_contrast(heldout[0]), one_token_contrast(vocab, heldout[0].target), false);
    if (llama_set_adapters_lora(ctx, adapters, 1, scales) != 0) throw std::runtime_error("cannot restore adapter");
    reset_bank(weight);
    const auto zero_adapter = capture_response_contrast(ctx, vocab, collector, negative_prompt_contrast(heldout[0]), one_token_contrast(vocab, heldout[0].target), false);
    const double zero_difference = maximum_difference(no_adapter.logits, zero_adapter.logits);

    std::ofstream geometry(fs::path(opts.correction_contrast_dir)/"activation_geometry.json");
    geometry << std::setprecision(12) << "{\n  \"layer\": \"" << opts.adapter_tensor << "\",\n"
             << "  \"d_in\": " << d_in << ",\n  \"d_out\": " << d_out << ",\n"
             << "  \"rank\": " << rank << ",\n  \"zero_adapter_max_logit_difference\": " << zero_difference << ",\n"
             << "  \"selected_training_ids\": [";
    for (size_t i = 0; i < training.size(); ++i) geometry << (i ? ", " : "") << "\"" << training[i].id << "\"";
    geometry << "],\n  \"contrast_cosine_matrix\": [\n";
    double cosine_sum = 0;
    double cosine_min = 1;
    double cosine_max = -1;
    int32_t cosine_count = 0;
    for (size_t i = 0; i < training.size(); ++i) {
        geometry << "    [";
        for (size_t j = 0; j < training.size(); ++j) {
            const double numerator = raw_update_inner_contrast(positive[i], negative[i], positive[j], negative[j]);
            const double norm_i = std::sqrt(std::max(0.0, raw_update_inner_contrast(positive[i], negative[i], positive[i], negative[i])));
            const double norm_j = std::sqrt(std::max(0.0, raw_update_inner_contrast(positive[j], negative[j], positive[j], negative[j])));
            const double cosine = numerator/(norm_i*norm_j + 1e-30);
            geometry << (j ? ", " : "") << cosine;
            if (i < j) {
                cosine_sum += cosine;
                cosine_min = std::min(cosine_min, cosine);
                cosine_max = std::max(cosine_max, cosine);
                ++cosine_count;
            }
        }
        geometry << "]" << (i + 1 == training.size() ? "\n" : ",\n");
    }
    geometry << "  ],\n  \"mean_same_task_contrast_cosine\": " << cosine_sum/std::max(1, cosine_count)
             << ",\n  \"min_same_task_contrast_cosine\": " << cosine_min
             << ",\n  \"max_same_task_contrast_cosine\": " << cosine_max << "\n}\n";

    const std::vector<double> epsilons = { 1e-5, 1e-4, 1e-3 };
    const std::vector<std::string> conditions = { "contrastive", "positive_only", "shuffled_correction", "sign_reversed" };
    constexpr double ratio_cap = 0.003;
    for (double epsilon : epsilons) {
        for (const auto & condition : conditions) {
            contrast_bank bank(d_in, d_out, rank);
            set_bank(weight, bank.a, bank.b);
            for (int32_t checkpoint = 0; checkpoint <= 4; ++checkpoint) {
                if (checkpoint > 0) {
                    const int32_t event = checkpoint - 1;
                    const int32_t shuffled = (event + 1) % 4;
                    const auto update = apply_correction_contrast(
                        condition, epsilon, ratio_cap, event, positive[event], negative[event], positive[shuffled], weight, bank);
                    raw << "{\"row_type\":\"update\",\"layer\":\"" << opts.adapter_tensor
                        << "\",\"epsilon\":" << epsilon << ",\"condition\":\"" << condition
                        << "\",\"checkpoint\":" << checkpoint << ",\"event_id\":\"" << training[event].id
                        << "\",\"raw_positive_norm\":" << update.raw_positive_norm
                        << ",\"raw_negative_norm\":" << update.raw_negative_norm
                        << ",\"raw_contrast_norm\":" << update.raw_contrast_norm
                        << ",\"positive_negative_cosine\":" << update.positive_negative_cosine
                        << ",\"normalized_contrast_norm\":" << update.normalized_contrast_norm
                        << ",\"control_match_scale\":" << update.control_match_scale
                        << ",\"requested_update_rho\":" << update.requested_update_rho
                        << ",\"actual_update_rho\":" << update.actual_update_rho
                        << ",\"final_state_rho\":" << update.final_state_rho
                        << ",\"trust_cap\":" << ratio_cap << ",\"trust_clipped\":" << (update.clipped ? "true" : "false")
                        << ",\"clip_scale\":" << update.clip_scale
                        << ",\"preexisting_state_scale\":" << update.preexisting_state_scale
                        << ",\"final_state_scale\":" << update.final_state_scale
                        << ",\"plastic_state_frobenius_norm\":" << update.state_frobenius_norm << "}\n";
                }
                const auto heldout_metrics = evaluate_cases_contrast(ctx, vocab, collector, heldout);
                for (size_t i = 0; i < heldout.size(); ++i) {
                    write_measurement_contrast(raw, vocab, opts.adapter_tensor, epsilon, condition, checkpoint, "heldout", heldout[i], heldout_metrics[i], bank);
                }
                const auto training_metrics = evaluate_cases_contrast(ctx, vocab, collector, training);
                for (size_t i = 0; i < training.size(); ++i) {
                    write_measurement_contrast(raw, vocab, opts.adapter_tensor, epsilon, condition, checkpoint, "training", training[i], training_metrics[i], bank);
                }
                const auto unrelated = evaluate_unrelated_contrast(ctx, vocab, unrelated_baseline);
                double baseline_nll = 0;
                for (const auto & item : unrelated_baseline) baseline_nll += item.mean_nll;
                baseline_nll /= unrelated_baseline.size();
                raw << "{\"row_type\":\"unrelated\",\"layer\":\"" << opts.adapter_tensor
                    << "\",\"epsilon\":" << epsilon << ",\"condition\":\"" << condition
                    << "\",\"checkpoint\":" << checkpoint << ",\"mean_kl\":" << unrelated.mean_kl
                    << ",\"baseline_mean_nll\":" << baseline_nll << ",\"adapted_mean_nll\":" << unrelated.mean_nll
                    << ",\"nll_change\":" << unrelated.mean_nll - baseline_nll << "}\n";
                raw.flush();
            }
        }
    }

    reset_bank(weight);
    const auto reset = capture_response_contrast(ctx, vocab, collector, negative_prompt_contrast(heldout[0]), one_token_contrast(vocab, heldout[0].target), false);
    const double reset_difference = maximum_difference(zero_adapter.logits, reset.logits);
    std::ofstream meta(fs::path(opts.correction_contrast_dir)/"run_metadata.json");
    meta << std::setprecision(12)
         << "{\n  \"status\": \"completed\",\n  \"layer\": \"" << opts.adapter_tensor << "\",\n"
         << "  \"input_tensor\": \"" << opts.input_tensor << "\",\n  \"output_tensor\": \"" << opts.output_tensor << "\",\n"
         << "  \"d_in\": " << d_in << ",\n  \"d_out\": " << d_out << ",\n  \"rank\": 8,\n"
         << "  \"plastic_memory_bytes\": " << 4*rank*(d_in + d_out) << ",\n"
         << "  \"plastic_macs_per_token\": " << rank*(d_in + d_out) << ",\n"
         << "  \"trust_cap\": " << ratio_cap << ",\n  \"normalization\": \"unit x and y per outer-product term\",\n"
         << "  \"zero_adapter_max_logit_difference\": " << zero_difference << ",\n"
         << "  \"reset_max_logit_difference\": " << reset_difference << ",\n"
         << "  \"training_count\": 4,\n  \"heldout_count\": 8\n}\n";
    std::fprintf(stderr, "correction contrast completed layer=%s output=%s\n",
        opts.adapter_tensor.c_str(), opts.correction_contrast_dir.c_str());
}
