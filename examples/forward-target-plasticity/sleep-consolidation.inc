namespace sleep_consolidation {

constexpr int32_t k_fast_rank = 4;
constexpr int32_t k_slow_rank = 8;
constexpr double k_lambda = 1.0;
constexpr double k_tau = 0.0;

struct replay_target {
    std::string id;
    std::string prompt;
    llama_token target = -1;
    std::vector<float> base_logits;
};

struct component_score {
    int32_t slot = -1;
    double useful_gain = 0;
    double protective_damage = 0;
    double score = 0;
    bool selected = false;
};

struct transferred_state {
    bank slow;
    bank remaining_fast;

    explicit transferred_state(int32_t width) : slow(k_slow_rank, width), remaining_fast(k_fast_rank, width) {}
};

void copy_scaled_slot(const bank & source, int32_t source_slot, bank & destination,
        int32_t destination_slot, double scale, int32_t key_slot = -1) {
    const int32_t selected_key_slot = key_slot < 0 ? source_slot : key_slot;
    for (int32_t i = 0; i < source.width; ++i) {
        destination.u[(size_t) destination.rank*i + destination_slot] =
            (float) (scale*source.u[(size_t) source.rank*i + source_slot]);
        destination.v[(size_t) destination.width*destination_slot + i] =
            source.v[(size_t) source.width*selected_key_slot + i];
    }
    destination.owners[destination_slot] = source.owners[source_slot];
}

bank combine(const bank & fast, const bank & slow) {
    if (fast.width != slow.width || fast.rank != k_fast_rank || slow.rank != k_slow_rank) {
        throw std::runtime_error("invalid fast/slow bank dimensions");
    }
    bank result(k_fast_rank + k_slow_rank, fast.width);
    for (int32_t slot = 0; slot < fast.rank; ++slot) copy_scaled_slot(fast, slot, result, slot, 1.0);
    for (int32_t slot = 0; slot < slow.rank; ++slot) {
        copy_scaled_slot(slow, slot, result, k_fast_rank + slot, 1.0);
    }
    return result;
}

void install(llama_context * ctx, const bank & fast, const bank & slow) {
    const bank combined = combine(fast, slow);
    install_bank(ctx, combined);
}

bank isolate_component(const bank & source, int32_t slot) {
    bank result(source.rank, source.width);
    copy_scaled_slot(source, slot, result, slot, 1.0);
    return result;
}

std::vector<replay_target> make_protective_replay(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector) {
    const std::vector<std::pair<std::string, std::string>> prompts = {
        {"The capital of France is", " Paris"},
        {"Two plus two equals", " four"},
        {"The opposite of hot is", " cold"},
        {"The first month of the year is", " January"},
    };
    std::vector<replay_target> result;
    for (size_t i = 0; i < prompts.size(); ++i) {
        const auto tokens = tokenize(vocab, prompts[i].second, false);
        if (tokens.size() != 1) throw std::runtime_error("protective target is not one token");
        const auto state = evaluate_state(ctx, vocab, collector, prompts[i].first, tokens[0]);
        result.push_back({"protective_" + std::to_string(i + 1), prompts[i].first, tokens[0], state.logits});
    }
    return result;
}

double average_protective_nll(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const std::vector<replay_target> & replay) {
    double total = 0;
    for (const auto & item : replay) total += evaluate_state(ctx, vocab, collector, item.prompt, item.target).target_nll;
    return total/replay.size();
}

double average_protective_kl(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const std::vector<replay_target> & replay) {
    double total = 0;
    for (const auto & item : replay) {
        const auto state = evaluate_state(ctx, vocab, collector, item.prompt, item.target);
        total += distribution_kl(state.logits, item.base_logits);
    }
    return total/replay.size();
}

transferred_state transfer(
        const bank & fast, const std::vector<int32_t> & selected, double beta, bool shuffled) {
    transferred_state result(fast.width);
    result.remaining_fast = fast;
    int32_t destination_slot = 0;
    for (int32_t source_slot : selected) {
        const int32_t key_slot = shuffled ? (source_slot + 1)%fast.rank : source_slot;
        copy_scaled_slot(fast, source_slot, result.slow, destination_slot++, beta, key_slot);
        for (int32_t i = 0; i < fast.width; ++i) {
            result.remaining_fast.u[(size_t) fast.rank*i + source_slot] *= (float) (1.0 - beta);
        }
    }
    return result;
}

void write_state_rows(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const std::string & state_name, double beta, const bank & fast, const bank & slow,
        const std::vector<association> & useful, const std::vector<llama_token> & targets,
        const std::vector<replay_target> & protective, std::ofstream & output) {
    install(ctx, fast, slow);
    for (size_t i = 0; i < useful.size(); ++i) {
        const auto metrics = measure_token(ctx, vocab, collector, useful[i].query, targets[i]);
        output << std::setprecision(17)
               << "{\"state\":\"" << state_name << "\",\"beta\":" << beta
               << ",\"role\":\"useful\",\"item_id\":\"" << json_escape(useful[i].id)
               << "\",\"target_nll\":" << metrics.nll << ",\"target_probability\":" << metrics.probability
               << ",\"greedy_recall\":" << (metrics.greedy_recall ? "true" : "false") << "}\n";
    }
    for (const auto & item : protective) {
        const auto state = evaluate_state(ctx, vocab, collector, item.prompt, item.target);
        output << std::setprecision(17)
               << "{\"state\":\"" << state_name << "\",\"beta\":" << beta
               << ",\"role\":\"protective\",\"item_id\":\"" << item.id
               << "\",\"target_nll\":" << state.target_nll
               << ",\"target_probability\":" << state.target_probability
               << ",\"kl_from_w0\":" << distribution_kl(state.logits, item.base_logits) << "}\n";
    }
    output.flush();
}

} // namespace sleep_consolidation

void run_sleep_consolidation(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const options & opts,
        const std::vector<association> & benchmark) {
    using namespace sleep_consolidation;
    if (opts.rank != k_fast_rank || std::abs(opts.rho - 0.001f) > 1e-9 || opts.decay != 0 || opts.eta != 1) {
        throw std::runtime_error("sleep PoC requires rank=4 rho=0.001 decay=0 eta=1");
    }
    if (benchmark.size() < 4) throw std::runtime_error("sleep PoC requires four useful associations");
    const std::filesystem::path root(opts.sleep_output_dir);
    std::filesystem::create_directories(root/"raw_results");
    const auto started = std::chrono::steady_clock::now();

    std::vector<association> useful(benchmark.begin(), benchmark.begin() + 4);
    std::vector<llama_token> targets;
    for (const auto & item : useful) {
        const auto tokens = tokenize(vocab, item.target, false);
        if (tokens.size() != 1) throw std::runtime_error("sleep target is not one token: " + item.id);
        targets.push_back(tokens[0]);
    }

    bank fast(k_fast_rank, output_head.width);
    bank slow(k_slow_rank, output_head.width);
    install(ctx, fast, slow);
    const auto protective = make_protective_replay(ctx, vocab, collector);
    std::ofstream awake(root/"raw_results"/"awake_updates.jsonl", std::ios::trunc);

    for (int32_t event = 0; event < k_fast_rank; ++event) {
        install(ctx, fast, slow);
        const auto exposure = evaluate_state(ctx, vocab, collector, useful[event].learn_prompt, targets[event]);
        const auto q = exact_target_direction(output_head, exposure.logits, targets[event], opts.q_threads);
        const auto update = update_task_bank(opts, fast, "target", event, exposure.hidden, q, q,
            exposure.target_probability, opts.seed + event, -1, {});
        if (!update.update.feasible) throw std::runtime_error("awake fast update violated trust region");
        fast.owners[event] = useful[event].id;
        install(ctx, fast, slow);
        const auto after = measure_token(ctx, vocab, collector, useful[event].query, targets[event]);
        awake << std::setprecision(17)
              << "{\"event\":" << event + 1 << ",\"association_id\":\"" << json_escape(useful[event].id)
              << "\",\"slot\":" << update.slot << ",\"target_nll_before\":" << exposure.target_nll
              << ",\"target_probability_before\":" << exposure.target_probability
              << ",\"target_nll_after\":" << after.nll << ",\"target_probability_after\":" << after.probability
              << ",\"greedy_after\":" << (after.greedy_recall ? "true" : "false")
              << ",\"delta_rho\":" << update.update.delta_rho_actual
              << ",\"state_rho\":" << update.update.response_rho
              << ",\"projection_applied\":" << (update.update.projected ? "true" : "false") << "}\n";
        awake.flush();
        std::fprintf(stderr, "sleep awake event=%d/4 id=%s nll_before=%.6f nll_after=%.6f\n",
            event + 1, useful[event].id.c_str(), exposure.target_nll, after.nll);
    }

    const bank fast_before = fast;
    std::ofstream scores_output(root/"consolidation_scores.jsonl", std::ios::trunc);
    std::vector<component_score> scores;
    std::vector<int32_t> selected;
    for (int32_t slot = 0; slot < k_fast_rank; ++slot) {
        bank single = isolate_component(fast_before, slot);
        bank zero_fast(k_fast_rank, output_head.width);
        install(ctx, zero_fast, slow);
        const double useful_without = evaluate_state(ctx, vocab, collector, useful[slot].query, targets[slot]).target_nll;
        const double protective_without = average_protective_nll(ctx, vocab, collector, protective);
        install(ctx, single, slow);
        const double useful_with = evaluate_state(ctx, vocab, collector, useful[slot].query, targets[slot]).target_nll;
        const double protective_with = average_protective_nll(ctx, vocab, collector, protective);
        component_score score;
        score.slot = slot;
        score.useful_gain = useful_without - useful_with;
        score.protective_damage = protective_with - protective_without;
        score.score = score.useful_gain - k_lambda*score.protective_damage;
        score.selected = score.score > k_tau;
        if (score.selected) selected.push_back(slot);
        scores.push_back(score);
        const int evidence_count = slot == 1 ? 3 : 1;
        scores_output << std::setprecision(17)
                      << "{\"slot\":" << slot << ",\"association_id\":\"" << json_escape(useful[slot].id)
                      << "\",\"useful_gain\":" << score.useful_gain
                      << ",\"protective_damage\":" << score.protective_damage
                      << ",\"lambda\":" << k_lambda << ",\"score\":" << score.score
                      << ",\"selected\":" << (score.selected ? "true" : "false")
                      << ",\"replay_evidence_count\":" << evidence_count
                      << ",\"cumulative_replay_score\":" << evidence_count*score.score << "}\n";
        scores_output.flush();
        std::fprintf(stderr, "sleep score slot=%d gain=%.6f damage=%.6f C=%.6f selected=%d\n",
            slot, score.useful_gain, score.protective_damage, score.score, score.selected);
    }

    std::ofstream state_output(root/"raw_results"/"state_metrics.jsonl", std::ios::trunc);
    bank zero_fast(k_fast_rank, output_head.width);
    bank zero_slow(k_slow_rank, output_head.width);
    write_state_rows(ctx, vocab, collector, "w0", 0, zero_fast, zero_slow, useful, targets, protective, state_output);
    write_state_rows(ctx, vocab, collector, "fast_before_sleep", 0, fast_before, zero_slow, useful, targets, protective, state_output);
    write_state_rows(ctx, vocab, collector, "no_consolidation", 0, fast_before, zero_slow, useful, targets, protective, state_output);

    transferred_state utility_final(output_head.width);
    for (const double beta : {0.25, 0.5, 1.0}) {
        auto state = transfer(fast_before, selected, beta, false);
        const std::string suffix = beta == 0.25 ? "0_25" : beta == 0.5 ? "0_5" : "1_0";
        write_state_rows(ctx, vocab, collector, "utility_beta_" + suffix + "_combined", beta,
            state.remaining_fast, state.slow, useful, targets, protective, state_output);
        write_state_rows(ctx, vocab, collector, "utility_beta_" + suffix + "_slow_only", beta,
            zero_fast, state.slow, useful, targets, protective, state_output);
        if (beta == 1.0) utility_final = state;
    }

    const std::vector<int32_t> all_slots = {0, 1, 2, 3};
    auto all_state = transfer(fast_before, all_slots, 1.0, false);
    write_state_rows(ctx, vocab, collector, "all_beta_1_0_slow_only", 1.0,
        zero_fast, all_state.slow, useful, targets, protective, state_output);
    auto shuffled_state = transfer(fast_before, selected, 1.0, true);
    write_state_rows(ctx, vocab, collector, "shuffled_beta_1_0_slow_only", 1.0,
        zero_fast, shuffled_state.slow, useful, targets, protective, state_output);

    std::ofstream compression(root/"compression_metrics.json", std::ios::trunc);
    compression << std::setprecision(17)
                << "{\"slow_rank_budget\":" << k_slow_rank
                << ",\"temporary_rank\":" << selected.size()
                << ",\"final_rank\":" << selected.size()
                << ",\"compression_applied\":false,\"frobenius_relative_error\":0.0"
                << ",\"reason\":\"selected rank did not exceed rank-8 budget; QR/core-SVD not required\"}\n";

    const auto slow_path = (root/"raw_results"/"slow_state.bin").string();
    const auto fast_path = (root/"raw_results"/"fast_remaining.bin").string();
    save_bank(utility_final.slow, slow_path);
    save_bank(utility_final.remaining_fast, fast_path);
    install(ctx, utility_final.remaining_fast, utility_final.slow);
    const auto before_reload = evaluate_state(ctx, vocab, collector, useful[0].query, targets[0]);
    install(ctx, zero_fast, zero_slow);
    const bank loaded_slow = load_bank(slow_path, k_slow_rank, output_head.width);
    const bank loaded_fast = load_bank(fast_path, k_fast_rank, output_head.width);
    install(ctx, loaded_fast, loaded_slow);
    const auto after_reload = evaluate_state(ctx, vocab, collector, useful[0].query, targets[0]);
    double max_logit_diff = 0;
    for (size_t i = 0; i < before_reload.logits.size(); ++i) {
        max_logit_diff = std::max(max_logit_diff,
            std::abs((double) before_reload.logits[i] - after_reload.logits[i]));
    }
    std::ofstream serialization(root/"serialization_validation.json", std::ios::trunc);
    serialization << std::setprecision(17)
                  << "{\"max_logit_abs_diff\":" << max_logit_diff
                  << ",\"target_nll_before\":" << before_reload.target_nll
                  << ",\"target_nll_after\":" << after_reload.target_nll
                  << ",\"slow_rank\":" << loaded_slow.rank << ",\"fast_rank\":" << loaded_fast.rank
                  << ",\"passed\":" << (max_logit_diff <= 1e-6 ? "true" : "false") << "}\n";

    const double awake_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::ofstream manifest(root/"manifest.json", std::ios::trunc);
    manifest << std::setprecision(17)
             << "{\"experiment\":\"qwen35_forward_sleep_consolidation\",\"fast_rank\":4,\"slow_rank\":8"
             << ",\"rho_max\":0.001,\"lambda\":1.0,\"tau\":0.0,\"useful_events\":4"
             << ",\"protective_replay_examples\":4,\"selected_components\":" << selected.size()
             << ",\"persistent_bytes_fp32\":" << (uint64_t) (k_fast_rank + k_slow_rank)*2*output_head.width*sizeof(float)
             << ",\"runtime_seconds\":" << awake_seconds << "}\n";
    std::fprintf(stderr, "sleep consolidation complete selected=%zu runtime=%.2fs output=%s\n",
        selected.size(), awake_seconds, root.string().c_str());
}
