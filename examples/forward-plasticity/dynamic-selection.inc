// Isolated dynamic module-selection PoC. Included by forward-plasticity.cpp so
// it can reuse the mutable LoRA tensors and activation callback infrastructure.

struct dynamic_case {
    std::string id;
    std::string role;
    std::string target;
    std::string problem;
    std::string correction;
};

struct dynamic_site {
    std::string name;
    std::string type;
    std::string input_name;
    std::string output_name;
    int32_t block = -1;
    llama_adapter_lora_weight * weight = nullptr;
    int64_t d_in = 0;
    int64_t d_out = 0;
    int64_t rank = 0;
    int64_t next_slot = 0;
    int32_t selection_count = 0;
    std::vector<float> a;
    std::vector<float> b;
};

struct dynamic_module_trace {
    std::vector<float> x;
    std::vector<float> y;
};

struct dynamic_trace {
    std::unordered_map<std::string, dynamic_module_trace> modules;
    std::vector<float> logits;
    llama_token target = -1;
    llama_token greedy = -1;
    llama_token response = -1;
    double target_probability = 0;
    double target_nll = 0;
    int32_t target_rank = 0;
    double target_margin = 0;
};

struct dynamic_output_matrix {
    int64_t width = 0;
    int64_t rows = 0;
    std::vector<float> values;
    const float * row(int64_t index) const { return values.data() + index*width; }
};

struct dynamic_ranking {
    int32_t site = -1;
    double relative_change = 0;
    double alignment = 0;
    double incremental_alignment = 0;
    bool incremental_alignment_defined = false;
    double eligibility = 0;
};

struct dynamic_probe {
    int32_t site = -1;
    double baseline_nll = 0;
    double probe_nll = 0;
    double utility = 0;
    double raw_update_rho = 0;
    double applied_update_rho = 0;
    double final_state_rho = 0;
    double scale = 0;
    bool clipped = false;
    double restore_max_difference = 0;
};

std::string dynamic_json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int) ch << std::dec;
                else out << ch;
        }
    }
    return out.str();
}

std::vector<std::string> dynamic_split_tsv(const std::string & line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) fields.push_back(field);
    return fields;
}

std::vector<dynamic_case> dynamic_load_cases(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open dynamic task TSV");
    std::vector<dynamic_case> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto fields = dynamic_split_tsv(line);
        if (fields.size() != 5) throw std::runtime_error("dynamic task row must have five tab-separated fields");
        if (fields[1] != "adaptation" && fields[1] != "heldout") throw std::runtime_error("invalid dynamic task role");
        result.push_back({fields[0], fields[1], fields[2], fields[3], fields[4]});
    }
    const auto adaptations = std::count_if(result.begin(), result.end(), [](const auto & row) { return row.role == "adaptation"; });
    const auto heldout = std::count_if(result.begin(), result.end(), [](const auto & row) { return row.role == "heldout"; });
    if (adaptations != 8 || heldout != 24) throw std::runtime_error("dynamic task must contain exactly 8 adaptation and 24 heldout rows");
    return result;
}

double dynamic_dot(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("dynamic vector size mismatch");
    double result = 0;
    for (size_t i = 0; i < left.size(); ++i) result += (double) left[i]*right[i];
    return result;
}

double dynamic_norm(const std::vector<float> & values) {
    return std::sqrt(std::max(0.0, dynamic_dot(values, values)));
}

double dynamic_cosine(const std::vector<float> & left, const std::vector<float> & right) {
    const double denominator = dynamic_norm(left)*dynamic_norm(right);
    return denominator > 0 ? dynamic_dot(left, right)/denominator : 0;
}

std::vector<float> dynamic_subtract(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("dynamic subtraction size mismatch");
    std::vector<float> result(left.size());
    for (size_t i = 0; i < left.size(); ++i) result[i] = left[i] - right[i];
    return result;
}

double dynamic_max_difference(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) throw std::runtime_error("dynamic comparison size mismatch");
    double result = 0;
    for (size_t i = 0; i < left.size(); ++i) result = std::max(result, std::abs((double) left[i] - right[i]));
    return result;
}

dynamic_output_matrix dynamic_load_output_matrix(const llama_model & model) {
    const ggml_tensor * tensor = model.output;
    if (!tensor || tensor->ne[2] != 1 || tensor->ne[3] != 1) throw std::runtime_error("invalid output tensor");
    const auto * traits = ggml_get_type_traits(tensor->type);
    if (!traits || !traits->to_float) throw std::runtime_error("output tensor cannot be dequantized");
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    dynamic_output_matrix result;
    result.width = tensor->ne[0];
    result.rows = tensor->ne[1];
    result.values.resize((size_t) result.width*result.rows);
    for (int64_t row = 0; row < result.rows; ++row) {
        traits->to_float(raw.data() + row*tensor->nb[1], result.values.data() + row*result.width, result.width);
    }
    return result;
}

std::vector<float> dynamic_probabilities(const std::vector<float> & logits) {
    const double normalizer = logsumexp(logits.data(), logits.size());
    std::vector<float> result(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) result[i] = std::exp(logits[i] - normalizer);
    return result;
}

std::vector<float> dynamic_exact_q(
        const dynamic_output_matrix & output, const std::vector<float> & logits, llama_token target) {
    if ((int64_t) logits.size() != output.rows || target < 0 || target >= output.rows) {
        throw std::runtime_error("dynamic output matrix/logit mismatch");
    }
    const auto probabilities = dynamic_probabilities(logits);
    const int32_t n_threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::vector<float>> partial(n_threads, std::vector<float>(output.width, 0.0f));
    std::vector<std::thread> workers;
    for (int32_t thread = 0; thread < n_threads; ++thread) {
        workers.emplace_back([&, thread]() {
            const int64_t begin = output.rows*thread/n_threads;
            const int64_t end = output.rows*(thread + 1)/n_threads;
            auto & sum = partial[thread];
            for (int64_t row = begin; row < end; ++row) {
                const float scale = probabilities[row];
                const float * weight = output.row(row);
                for (int64_t column = 0; column < output.width; ++column) sum[column] += scale*weight[column];
            }
        });
    }
    for (auto & worker : workers) worker.join();
    std::vector<float> result(output.width);
    for (int64_t column = 0; column < output.width; ++column) {
        double expected = 0;
        for (int32_t thread = 0; thread < n_threads; ++thread) expected += partial[thread][column];
        result[column] = output.row(target)[column] - expected;
    }
    return result;
}

dynamic_trace dynamic_score_prompt(
        llama_context * ctx, const llama_vocab * vocab, const std::string & prompt, llama_token target) {
    llama_memory_clear(llama_get_memory(ctx), true);
    auto tokens = tokenize(vocab, prompt, true);
    if (tokens.empty()) throw std::runtime_error("empty dynamic prompt");
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) throw std::runtime_error("dynamic prompt decode failed");
    llama_synchronize(ctx);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * raw = llama_get_logits_ith(ctx, -1);
    if (!raw) throw std::runtime_error("missing dynamic logits");
    dynamic_trace result;
    result.target = target;
    result.logits.assign(raw, raw + n_vocab);
    const double normalizer = logsumexp(raw, n_vocab);
    result.target_nll = normalizer - raw[target];
    result.target_probability = std::exp(-result.target_nll);
    result.greedy = 0;
    result.target_rank = 1;
    float other_max = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        if (raw[token] > raw[result.greedy]) result.greedy = token;
        if (token != target) other_max = std::max(other_max, raw[token]);
        if (raw[token] > raw[target]) ++result.target_rank;
    }
    result.target_margin = raw[target] - other_max;
    return result;
}

std::vector<float> dynamic_pooled(
        const activation_collector & collector, const std::string & name, int64_t expected) {
    const auto sum = collector.dynamic_sums.find(name);
    const auto count = collector.dynamic_counts.find(name);
    if (sum == collector.dynamic_sums.end() || count == collector.dynamic_counts.end() || count->second <= 0) {
        throw std::runtime_error("missing dynamic activation: " + name);
    }
    if ((int64_t) sum->second.size() != expected) throw std::runtime_error("dynamic activation dimension mismatch: " + name);
    std::vector<float> result(expected);
    for (int64_t i = 0; i < expected; ++i) result[i] = sum->second[i]/count->second;
    return result;
}

dynamic_trace dynamic_capture_response(
        llama_context * ctx, const llama_vocab * vocab, activation_collector & collector,
        const std::vector<dynamic_site> & sites, const std::string & prompt, llama_token target) {
    collector.enabled = false;
    collector.dynamic_mode = true;
    dynamic_trace result = dynamic_score_prompt(ctx, vocab, prompt, target);
    collector.reset();
    collector.dynamic_names.clear();
    for (const auto & site : sites) {
        collector.dynamic_names.insert(site.input_name);
        collector.dynamic_names.insert(site.output_name);
    }
    collector.enabled = true;
    result.response = result.greedy;
    if (llama_decode(ctx, llama_batch_get_one(&result.response, 1)) != 0) throw std::runtime_error("dynamic response decode failed");
    llama_synchronize(ctx);
    collector.enabled = false;
    for (const auto & site : sites) {
        result.modules.emplace(site.name, dynamic_module_trace{
            dynamic_pooled(collector, site.input_name, site.d_in),
            dynamic_pooled(collector, site.output_name, site.d_out),
        });
    }
    return result;
}

std::vector<dynamic_site> dynamic_discover_sites(llama_adapter_lora * adapter, activation_collector & collector) {
    std::vector<dynamic_site> result;
    auto add = [&](int32_t block, const std::string & type, const std::string & weight_name,
                   const std::string & input_name, const std::string & output_name) {
        auto found = adapter->ab_map.find(weight_name);
        if (found == adapter->ab_map.end()) return;
        auto & weight = found->second;
        dynamic_site site;
        site.name = weight_name;
        site.type = type;
        site.input_name = input_name;
        site.output_name = output_name;
        site.block = block;
        site.weight = &weight;
        site.d_in = weight.a->ne[0];
        site.rank = weight.a->ne[1];
        site.d_out = weight.b->ne[1];
        if (site.rank < 8 || weight.b->ne[0] != site.rank) throw std::runtime_error("dynamic adapter requires rank >= 8");
        site.a.assign(ggml_nelements(weight.a), 0.0f);
        site.b.assign(ggml_nelements(weight.b), 0.0f);
        set_bank(weight, site.a, site.b);
        collector.dynamic_names.insert(input_name);
        collector.dynamic_names.insert(output_name);
        result.push_back(std::move(site));
    };
    for (int32_t block = 0; block < 32; ++block) {
        add(block, "mlp", "blk." + std::to_string(block) + ".ffn_down.weight",
            "ffn_swiglu-" + std::to_string(block), "ffn_out-" + std::to_string(block));
        const std::string attention = "blk." + std::to_string(block) + ".attn_output.weight";
        if (adapter->ab_map.count(attention)) {
            add(block, "attention", attention, "attn_gated-" + std::to_string(block), "attn_output-" + std::to_string(block));
        } else {
            add(block, "deltanet", "blk." + std::to_string(block) + ".ssm_out.weight",
                "final_output-" + std::to_string(block), "linear_attn_out-" + std::to_string(block));
        }
    }
    if (result.size() < 32) throw std::runtime_error("too few dynamic candidate sites discovered");
    return result;
}

std::vector<float> dynamic_site_response(const dynamic_site & site, const std::vector<float> & x) {
    if ((int64_t) x.size() != site.d_in) throw std::runtime_error("dynamic response input mismatch");
    std::vector<float> result(site.d_out, 0.0f);
    for (int64_t slot = 0; slot < site.rank; ++slot) {
        double coefficient = 0;
        for (int64_t i = 0; i < site.d_in; ++i) coefficient += site.a[i + site.d_in*slot]*x[i];
        for (int64_t j = 0; j < site.d_out; ++j) result[j] += site.b[slot + site.rank*j]*coefficient;
    }
    return result;
}

std::vector<float> dynamic_contrast_response(
        const dynamic_module_trace & negative, const dynamic_module_trace & positive,
        const std::vector<float> & x, double coefficient) {
    std::vector<float> result(negative.y.size());
    const double positive_dot = dynamic_dot(positive.x, x);
    const double negative_dot = dynamic_dot(negative.x, x);
    for (size_t j = 0; j < result.size(); ++j) {
        result[j] = coefficient*(positive.y[j]*positive_dot - negative.y[j]*negative_dot);
    }
    return result;
}

double dynamic_trust_scale(
        const std::vector<float> & current, const std::vector<float> & delta, double maximum) {
    if (dynamic_norm(current) > maximum*(1 + 1e-8)) return 0;
    std::vector<float> candidate(current.size());
    auto norm_at = [&](double scale) {
        for (size_t i = 0; i < current.size(); ++i) candidate[i] = current[i] + scale*delta[i];
        return dynamic_norm(candidate);
    };
    if (norm_at(1.0) <= maximum) return 1.0;
    double low = 0;
    double high = 1;
    for (int iteration = 0; iteration < 80; ++iteration) {
        const double middle = (low + high)/2;
        if (norm_at(middle) <= maximum) low = middle;
        else high = middle;
    }
    return low;
}

dynamic_probe dynamic_install_or_probe(
        llama_context * ctx, const llama_vocab * vocab, dynamic_site & site,
        const dynamic_module_trace & negative, const dynamic_module_trace & positive,
        const std::string & probe_prompt, llama_token target, double epsilon, double ratio_cap, bool persist) {
    dynamic_probe result;
    const auto before_a = site.a;
    const auto before_b = site.b;
    const int64_t before_slot = site.next_slot;
    const auto baseline_result = dynamic_score_prompt(ctx, vocab, probe_prompt, target);
    result.baseline_nll = baseline_result.target_nll;
    const auto current = dynamic_site_response(site, negative.x);
    const auto raw_delta = dynamic_contrast_response(negative, positive, negative.x, epsilon);
    const double base_norm = dynamic_norm(negative.y);
    result.raw_update_rho = base_norm > 0 ? dynamic_norm(raw_delta)/base_norm : 0;
    const double maximum = ratio_cap*base_norm;
    result.scale = dynamic_trust_scale(current, raw_delta, maximum);
    result.clipped = result.scale < 1.0 - 1e-10;
    if (site.next_slot + 2 > site.rank) {
        result.scale = 0;
        result.clipped = true;
    }
    if (result.scale > 0) {
        const int64_t positive_slot = site.next_slot++;
        const int64_t negative_slot = site.next_slot++;
        for (int64_t i = 0; i < site.d_in; ++i) {
            site.a[i + site.d_in*positive_slot] = positive.x[i];
            site.a[i + site.d_in*negative_slot] = negative.x[i];
        }
        for (int64_t j = 0; j < site.d_out; ++j) {
            site.b[positive_slot + site.rank*j] = epsilon*result.scale*positive.y[j];
            site.b[negative_slot + site.rank*j] = -epsilon*result.scale*negative.y[j];
        }
        set_bank(*site.weight, site.a, site.b);
    }
    const auto after = dynamic_site_response(site, negative.x);
    const auto applied = dynamic_subtract(after, current);
    result.applied_update_rho = base_norm > 0 ? dynamic_norm(applied)/base_norm : 0;
    result.final_state_rho = base_norm > 0 ? dynamic_norm(after)/base_norm : 0;
    result.probe_nll = dynamic_score_prompt(ctx, vocab, probe_prompt, target).target_nll;
    result.utility = result.baseline_nll - result.probe_nll;
    if (!persist) {
        site.a = before_a;
        site.b = before_b;
        site.next_slot = before_slot;
        set_bank(*site.weight, site.a, site.b);
        const auto restored = dynamic_score_prompt(ctx, vocab, probe_prompt, target);
        result.restore_max_difference = dynamic_max_difference(restored.logits, baseline_result.logits);
    }
    return result;
}

double dynamic_kl(const std::vector<float> & base, const std::vector<float> & adapted) {
    if (base.size() != adapted.size()) throw std::runtime_error("dynamic KL size mismatch");
    const double z_base = logsumexp(base.data(), base.size());
    const double z_adapted = logsumexp(adapted.data(), adapted.size());
    double result = 0;
    for (size_t i = 0; i < base.size(); ++i) {
        const double probability = std::exp(base[i] - z_base);
        if (probability > 0) result += probability*((base[i] - z_base) - (adapted[i] - z_adapted));
    }
    return result;
}

void dynamic_write_heldout(
        std::ofstream & rows, std::ofstream & generated, llama_context * ctx, const llama_vocab * vocab,
        const std::vector<dynamic_case> & cases, const std::string & condition, int32_t checkpoint) {
    for (const auto & item : cases) {
        if (item.role != "heldout") continue;
        const auto target_tokens = tokenize(vocab, " " + item.target, false);
        const auto result = dynamic_score_prompt(ctx, vocab, item.problem, target_tokens[0]);
        rows << "{\"condition\":\"" << condition << "\",\"checkpoint\":" << checkpoint
             << ",\"id\":\"" << dynamic_json_escape(item.id) << "\",\"target\":\"" << item.target
             << "\",\"correct\":" << (result.greedy == target_tokens[0] ? "true" : "false")
             << ",\"target_probability\":" << result.target_probability << ",\"target_nll\":" << result.target_nll
             << ",\"target_rank\":" << result.target_rank << ",\"target_margin\":" << result.target_margin << "}\n";
        generated << "{\"condition\":\"" << condition << "\",\"checkpoint\":" << checkpoint
                  << ",\"id\":\"" << dynamic_json_escape(item.id) << "\",\"target_token\":" << target_tokens[0]
                  << ",\"generated_token\":" << result.greedy << "}\n";
    }
    rows.flush();
    generated.flush();
}

void dynamic_write_training(
        std::ofstream & rows, llama_context * ctx, const llama_vocab * vocab,
        const std::vector<dynamic_case> & cases, const std::string & condition, int32_t checkpoint) {
    for (const auto & item : cases) {
        if (item.role != "adaptation") continue;
        const auto target_tokens = tokenize(vocab, " " + item.target, false);
        const auto result = dynamic_score_prompt(ctx, vocab, item.problem, target_tokens[0]);
        rows << "{\"condition\":\"" << condition << "\",\"checkpoint\":" << checkpoint
             << ",\"id\":\"" << dynamic_json_escape(item.id) << "\",\"target\":\"" << item.target
             << "\",\"correct\":" << (result.greedy == target_tokens[0] ? "true" : "false")
             << ",\"target_probability\":" << result.target_probability << ",\"target_nll\":" << result.target_nll
             << ",\"target_rank\":" << result.target_rank << ",\"target_margin\":" << result.target_margin << "}\n";
    }
    rows.flush();
}

void run_dynamic_selection(
        const options & opts, const llama_model & model, llama_context * ctx, const llama_vocab * vocab,
        activation_collector & collector, llama_adapter_lora * adapter) {
    namespace fs = std::filesystem;
    if (opts.ratio_cap <= 0 || opts.ratio_cap > 0.001000001f) {
        throw std::runtime_error("dynamic PoC requires --ratio-cap in (0, 0.001]");
    }
    fs::create_directories(opts.dynamic_selection_dir);
    auto cases = dynamic_load_cases(opts.dynamic_selection_task_file);
    collector.dynamic_mode = true;
    auto sites = dynamic_discover_sites(adapter, collector);
    llama_adapter_lora * adapters[] = {adapter};
    float scales[] = {1.0f};
    if (llama_set_adapters_lora(ctx, adapters, 1, scales) != 0) throw std::runtime_error("dynamic adapter activation failed");

    for (const auto & item : cases) {
        const auto tokens = tokenize(vocab, " " + item.target, false);
        if (tokens.size() != 1) throw std::runtime_error("target is not exactly one token: " + item.target);
    }

    std::ofstream rankings(fs::path(opts.dynamic_selection_dir)/"candidate_rankings.jsonl");
    std::ofstream probes(fs::path(opts.dynamic_selection_dir)/"causal_probe_results.jsonl");
    std::ofstream heldout(fs::path(opts.dynamic_selection_dir)/"heldout_results.jsonl");
    std::ofstream training(fs::path(opts.dynamic_selection_dir)/"training_event_results.jsonl");
    std::ofstream generated(fs::path(opts.dynamic_selection_dir)/"generated_outputs.jsonl");
    if (!rankings || !probes || !heldout || !training || !generated) throw std::runtime_error("cannot open dynamic output files");

    const std::vector<std::string> unrelated_prompts = {
        "The capital of France is", "Two plus two equals", "Water freezes at", "A common greeting is",
    };
    const std::vector<std::string> unrelated_targets = {" Paris", " four", " zero", " hello"};
    std::vector<dynamic_trace> unrelated_baseline;
    for (size_t i = 0; i < unrelated_prompts.size(); ++i) {
        const auto target = tokenize(vocab, unrelated_targets[i], false);
        unrelated_baseline.push_back(dynamic_score_prompt(ctx, vocab, unrelated_prompts[i], target[0]));
    }

    dynamic_write_heldout(heldout, generated, ctx, vocab, cases, opts.dynamic_condition, 0);
    dynamic_write_training(training, ctx, vocab, cases, opts.dynamic_condition, 0);

    struct prepared_event {
        const dynamic_case * item = nullptr;
        llama_token target = -1;
        dynamic_trace negative;
        dynamic_trace positive;
        std::string positive_prompt;
    };
    std::vector<prepared_event> events;
    for (const auto & item : cases) {
        if (item.role != "adaptation") continue;
        const llama_token target = tokenize(vocab, " " + item.target, false)[0];
        const auto baseline = dynamic_score_prompt(ctx, vocab, item.problem, target);
        if (baseline.greedy == target) continue;
        const std::string positive_prompt = item.problem + "\n\nCorrection:\n" + item.correction +
            "\nNow solve the original problem correctly. Answer with exactly one relation label:";
        auto negative = dynamic_capture_response(ctx, vocab, collector, sites, item.problem, target);
        auto positive = dynamic_capture_response(ctx, vocab, collector, sites, positive_prompt, target);
        if (positive.target_probability <= negative.target_probability && positive.greedy != target) continue;
        events.push_back({&item, target, std::move(negative), std::move(positive), positive_prompt});
        if (events.size() == 4) break;
    }
    if (events.empty()) throw std::runtime_error("no usable failed correction events");

    const auto output_matrix = dynamic_load_output_matrix(model);
    std::mt19937 random(opts.seed);
    size_t probe_count = 0;
    size_t update_count = 0;
    double unrelated_kl_sum = 0;
    double unrelated_nll_change_sum = 0;
    for (size_t event_index = 0; event_index < events.size(); ++event_index) {
        auto & event = events[event_index];
        const auto & positive = opts.dynamic_condition == "shuffled_correction"
            ? events[(event_index + 1)%events.size()].positive : event.positive;
        const auto q = dynamic_exact_q(output_matrix, event.negative.logits, event.target);
        std::vector<dynamic_ranking> ranked;
        ranked.reserve(sites.size());
        for (size_t site_index = 0; site_index < sites.size(); ++site_index) {
            const auto & site = sites[site_index];
            const auto & negative_module = event.negative.modules.at(site.name);
            const auto & positive_module = positive.modules.at(site.name);
            const auto difference = dynamic_subtract(positive_module.y, negative_module.y);
            const double relative = dynamic_norm(difference)/(dynamic_norm(negative_module.y) + 1e-12);
            const double alignment = difference.size() == q.size() ? dynamic_cosine(difference, q) : 0;
            bool incremental_defined = false;
            double incremental_alignment = 0;
            if (negative_module.x.size() == difference.size() && positive_module.x.size() == difference.size() && difference.size() == q.size()) {
                const auto input_difference = dynamic_subtract(positive_module.x, negative_module.x);
                const auto incremental = dynamic_subtract(difference, input_difference);
                incremental_alignment = dynamic_cosine(incremental, q);
                incremental_defined = true;
            }
            ranked.push_back({(int32_t) site_index, relative, alignment,
                incremental_alignment, incremental_defined, relative*std::max(0.0, alignment)});
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto & left, const auto & right) {
            if (left.eligibility != right.eligibility) return left.eligibility > right.eligibility;
            return left.site < right.site;
        });
        for (size_t rank = 0; rank < ranked.size(); ++rank) {
            const auto & score = ranked[rank];
            const auto & site = sites[score.site];
            rankings << "{\"condition\":\"" << opts.dynamic_condition << "\",\"event\":" << event_index + 1
                     << ",\"association_id\":\"" << dynamic_json_escape(event.item->id) << "\",\"rank\":" << rank + 1
                     << ",\"module\":\"" << site.name << "\",\"block\":" << site.block
                     << ",\"module_type\":\"" << site.type << "\",\"relative_change\":" << score.relative_change
                     << ",\"alignment\":" << score.alignment
                     << ",\"incremental_alignment\":" << score.incremental_alignment
                     << ",\"incremental_alignment_defined\":" << (score.incremental_alignment_defined ? "true" : "false")
                     << ",\"eligibility\":" << score.eligibility << "}\n";
        }
        rankings.flush();

        std::vector<dynamic_probe> event_probes;
        if (opts.dynamic_condition == "dynamic_causal" || opts.dynamic_condition == "shuffled_correction") {
            const size_t top = std::min<size_t>(4, ranked.size());
            for (size_t index = 0; index < top; ++index) {
                auto & site = sites[ranked[index].site];
                auto probe = dynamic_install_or_probe(ctx, vocab, site,
                    event.negative.modules.at(site.name), positive.modules.at(site.name),
                    event.positive_prompt, event.target, 1e-5, opts.ratio_cap, false);
                probe.site = ranked[index].site;
                event_probes.push_back(probe);
                ++probe_count;
                probes << "{\"condition\":\"" << opts.dynamic_condition << "\",\"event\":" << event_index + 1
                       << ",\"module\":\"" << site.name << "\",\"baseline_nll\":" << probe.baseline_nll
                       << ",\"probe_nll\":" << probe.probe_nll << ",\"utility\":" << probe.utility
                       << ",\"raw_update_rho\":" << probe.raw_update_rho << ",\"actual_update_rho\":" << probe.applied_update_rho
                       << ",\"final_state_rho\":" << probe.final_state_rho << ",\"scale\":" << probe.scale
                       << ",\"clipped\":" << (probe.clipped ? "true" : "false")
                       << ",\"restore_max_difference\":" << probe.restore_max_difference << "}\n";
            }
            std::sort(event_probes.begin(), event_probes.end(), [](const auto & left, const auto & right) {
                return left.utility > right.utility;
            });
        }

        std::vector<int32_t> selected;
        if (opts.dynamic_condition == "dynamic_causal" || opts.dynamic_condition == "shuffled_correction") {
            for (const auto & probe : event_probes) {
                if (probe.utility > 0 && selected.size() < 2) selected.push_back(probe.site);
            }
        } else if (opts.dynamic_condition == "eligibility_only") {
            selected.push_back(ranked.front().site);
        } else if (opts.dynamic_condition == "random_module") {
            std::uniform_int_distribution<int32_t> distribution(0, sites.size() - 1);
            selected.push_back(distribution(random));
        } else if (opts.dynamic_condition == "fixed_module") {
            auto fixed = std::find_if(sites.begin(), sites.end(), [](const auto & site) {
                return site.name == "blk.27.ffn_down.weight";
            });
            selected.push_back(fixed == sites.end() ? ranked.front().site : std::distance(sites.begin(), fixed));
        }

        for (int32_t site_index : selected) {
            auto & site = sites[site_index];
            auto applied = dynamic_install_or_probe(ctx, vocab, site,
                event.negative.modules.at(site.name), positive.modules.at(site.name),
                event.positive_prompt, event.target, 1e-5, opts.ratio_cap, true);
            applied.site = site_index;
            ++site.selection_count;
            ++update_count;
            probes << "{\"condition\":\"" << opts.dynamic_condition << "\",\"event\":" << event_index + 1
                   << ",\"module\":\"" << site.name << "\",\"persistent\":true,\"utility\":" << applied.utility
                   << ",\"raw_update_rho\":" << applied.raw_update_rho << ",\"actual_update_rho\":" << applied.applied_update_rho
                   << ",\"final_state_rho\":" << applied.final_state_rho << ",\"clipped\":" << (applied.clipped ? "true" : "false") << "}\n";
        }
        probes.flush();

        dynamic_write_heldout(heldout, generated, ctx, vocab, cases, opts.dynamic_condition, event_index + 1);
        dynamic_write_training(training, ctx, vocab, cases, opts.dynamic_condition, event_index + 1);
        for (size_t i = 0; i < unrelated_prompts.size(); ++i) {
            const auto target = tokenize(vocab, unrelated_targets[i], false);
            const auto adapted = dynamic_score_prompt(ctx, vocab, unrelated_prompts[i], target[0]);
            unrelated_kl_sum += dynamic_kl(unrelated_baseline[i].logits, adapted.logits);
            unrelated_nll_change_sum += adapted.target_nll - unrelated_baseline[i].target_nll;
        }
    }

    size_t plastic_bytes = 0;
    for (const auto & site : sites) plastic_bytes += (site.a.size() + site.b.size())*sizeof(float);
    std::ofstream manifest(fs::path(opts.dynamic_selection_dir)/"manifest.json");
    manifest << "{\n  \"condition\": \"" << opts.dynamic_condition << "\",\n"
             << "  \"candidate_sites\": " << sites.size() << ",\n"
             << "  \"usable_corrections\": " << events.size() << ",\n"
             << "  \"heldout_examples\": 24,\n"
             << "  \"epsilon\": 1e-5,\n"
             << "  \"ratio_cap\": " << opts.ratio_cap << ",\n"
             << "  \"adapter_rank\": 8,\n"
             << "  \"plastic_bytes\": " << plastic_bytes << "\n}\n";
    std::ofstream summary(fs::path(opts.dynamic_selection_dir)/"summary.json");
    const double denominator = std::max<size_t>(1, events.size()*unrelated_prompts.size());
    summary << "{\n  \"condition\": \"" << opts.dynamic_condition << "\",\n"
            << "  \"events\": " << events.size() << ",\n"
            << "  \"probe_count\": " << probe_count << ",\n"
            << "  \"update_count\": " << update_count << ",\n"
            << "  \"mean_unrelated_kl\": " << unrelated_kl_sum/denominator << ",\n"
            << "  \"mean_unrelated_nll_change\": " << unrelated_nll_change_sum/denominator << "\n}\n";
    std::fprintf(stderr, "dynamic selection complete: condition=%s sites=%zu corrections=%zu probes=%zu updates=%zu\n",
        opts.dynamic_condition.c_str(), sites.size(), events.size(), probe_count, update_count);
}
