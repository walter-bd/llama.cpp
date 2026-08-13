struct task_episode {
    std::string id;
    std::string family;
    uint32_t seed = 0;
    std::vector<task_example> adaptation;
    std::vector<task_example> heldout;
};

void write_float_array(std::ostream & output, const std::vector<float> & values) {
    output << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) output << ',';
        output << values[i];
    }
    output << ']';
}

llama_token task_target_token(const llama_vocab * vocab, const task_example & item) {
    const auto tokens = tokenize(vocab, item.target, false);
    if (tokens.size() != 1) throw std::runtime_error("task target is not one token: " + item.id);
    return tokens[0];
}

std::vector<task_episode> group_task_episodes(const std::vector<task_example> & rows) {
    std::map<std::string, task_episode> grouped;
    for (const auto & row : rows) {
        auto & episode = grouped[row.episode_id];
        if (episode.id.empty()) {
            episode.id = row.episode_id;
            episode.family = row.task_family;
            episode.seed = row.task_seed;
        } else if (episode.family != row.task_family || episode.seed != row.task_seed) {
            throw std::runtime_error("inconsistent task episode metadata");
        }
        (row.split == "adaptation" ? episode.adaptation : episode.heldout).push_back(row);
    }
    std::vector<task_episode> result;
    for (auto & entry : grouped) {
        if (entry.second.adaptation.size() < 16 || entry.second.heldout.size() < 32) {
            throw std::runtime_error("task episode requires >=16 adaptation and >=32 heldout examples");
        }
        result.push_back(std::move(entry.second));
    }
    return result;
}

void write_task_measurements(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const bank & state, const std::string & condition,
        const task_episode & episode, const std::string & role, int32_t checkpoint,
        const std::vector<task_example> & rows, std::ofstream & measurements,
        std::ofstream & vectors, bool emit_vectors) {
    install_bank(ctx, state);
    for (const auto & item : rows) {
        const llama_token target = task_target_token(vocab, item);
        const auto metrics = measure_token(ctx, vocab, collector, item.prompt, target);
        const auto key = normalized(metrics.hidden);
        const auto response = state.apply(key);
        const double response_rho = l2_norm(response)/l2_norm(metrics.hidden);
        double alignment = 0;
        std::vector<float> q;
        if (condition == "target") {
            q = exact_target_direction(output_head, metrics.logits, target, 16);
            alignment = l2_norm(response) > 0 ? cosine(response, q) : 0;
        }
        measurements << std::setprecision(17)
            << "{\"episode_id\":\"" << json_escape(episode.id)
            << "\",\"task_family\":\"" << json_escape(episode.family)
            << "\",\"task_seed\":" << episode.seed
            << ",\"condition\":\"" << condition << "\",\"checkpoint\":" << checkpoint
            << ",\"dataset_role\":\"" << role << "\",\"example_id\":\"" << json_escape(item.id)
            << "\",\"target_token_id\":" << target << ",\"generated_token_id\":" << metrics.greedy_token
            << ",\"greedy_exact_recall\":" << (metrics.greedy_recall ? "true" : "false")
            << ",\"target_probability\":" << metrics.probability << ",\"target_rank\":" << metrics.rank
            << ",\"target_margin\":" << metrics.margin << ",\"target_nll\":" << metrics.nll
            << ",\"response_rho\":" << response_rho << ",\"plastic_q_cosine\":" << alignment
            << ",\"matching_semantics\":\"incremental\"}\n";
        if (emit_vectors) {
            vectors << std::setprecision(9)
                << "{\"episode_id\":\"" << json_escape(episode.id)
                << "\",\"task_family\":\"" << json_escape(episode.family)
                << "\",\"task_seed\":" << episode.seed << ",\"checkpoint\":" << checkpoint
                << ",\"dataset_role\":\"" << role << "\",\"example_id\":\"" << json_escape(item.id)
                << "\",\"vector_kind\":\"key_and_q\",\"key\":";
            write_float_array(vectors, key);
            vectors << ",\"q\":";
            write_float_array(vectors, q);
            vectors << "}\n";
        }
    }
    measurements.flush();
    vectors.flush();
}

void write_task_checkpoint(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const bank & state, const std::string & condition,
        const task_episode & episode, int32_t checkpoint, std::ofstream & measurements,
        std::ofstream & vectors, std::ofstream & stages,
        const std::vector<unrelated_reference> & unrelated) {
    const size_t seen = std::min<size_t>(checkpoint, episode.adaptation.size());
    const std::vector<task_example> adaptation_seen(
        episode.adaptation.begin(), episode.adaptation.begin() + seen);
    if (!adaptation_seen.empty()) {
        write_task_measurements(ctx, vocab, collector, output_head, state, condition, episode,
            "adaptation_seen", checkpoint, adaptation_seen, measurements, vectors, false);
    }
    write_task_measurements(ctx, vocab, collector, output_head, state, condition, episode,
        "heldout", checkpoint, episode.heldout, measurements, vectors,
        condition == "target" && checkpoint == 0);
    install_bank(ctx, state);
    const auto unrelated_metrics = measure_unrelated(ctx, vocab, collector, unrelated);
    stages << std::setprecision(17)
        << "{\"episode_id\":\"" << json_escape(episode.id) << "\",\"task_family\":\""
        << json_escape(episode.family) << "\",\"task_seed\":" << episode.seed
        << ",\"condition\":\"" << condition << "\",\"checkpoint\":" << checkpoint
        << ",\"unrelated_nll\":" << unrelated_metrics.first
        << ",\"unrelated_kl\":" << unrelated_metrics.second << "}\n";
    stages.flush();
}

llama_token different_task_target(
        const llama_vocab * vocab, const task_episode & episode, size_t index) {
    const llama_token own = task_target_token(vocab, episode.adaptation[index]);
    for (size_t offset = 1; offset < episode.adaptation.size(); ++offset) {
        const llama_token candidate = task_target_token(
            vocab, episode.adaptation[(index + offset)%episode.adaptation.size()]);
        if (candidate != own) return candidate;
    }
    throw std::runtime_error("task has no distinct target for control");
}

void run_task_condition(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const options & opts, const task_episode & episode,
        const std::string & condition, std::vector<double> & target_schedule,
        const std::vector<unrelated_reference> & unrelated, std::ofstream & measurements,
        std::ofstream & updates, std::ofstream & vectors, std::ofstream & stages) {
    bank state(opts.rank, output_head.width);
    ctx->clear_fast_weights();
    const std::vector<int32_t> checkpoints = {0, 1, 2, 4, 8, 16};
    write_task_checkpoint(ctx, vocab, collector, output_head, state, condition, episode, 0,
        measurements, vectors, stages, unrelated);
    std::vector<float> previous_q;
    size_t next_checkpoint = 1;
    for (int32_t index = 0; index < 16; ++index) {
        const auto & item = episode.adaptation[index];
        const llama_token target = task_target_token(vocab, item);
        install_bank(ctx, state);
        const auto before = evaluate_state(ctx, vocab, collector, item.prompt, target);
        const auto target_q = exact_target_direction(output_head, before.logits, target, opts.q_threads);
        std::vector<float> control_q = target_q;
        if (condition == "label_permuted") {
            const llama_token wrong = different_task_target(vocab, episode, index);
            control_q = exact_target_direction(output_head, before.logits, wrong, opts.q_threads);
        } else if (condition == "shuffled") {
            const auto & source = episode.adaptation[(index + 7)%episode.adaptation.size()];
            const llama_token source_target = task_target_token(vocab, source);
            const auto source_state = evaluate_state(ctx, vocab, collector, source.prompt, source_target);
            control_q = exact_target_direction(output_head, source_state.logits, source_target, opts.q_threads);
        }
        const double matched = condition == "target" ? -1 : target_schedule.at(index);
        auto result = update_task_bank(opts, state, condition, index, before.hidden,
            target_q, control_q, before.target_probability,
            (uint32_t) std::hash<std::string>{}(episode.id + item.id), matched, previous_q);
        if (!result.update.feasible && result.assignment_reason == "matched_control_infeasible") {
            updates << std::setprecision(17)
                << "{\"episode_id\":\"" << json_escape(episode.id) << "\",\"task_family\":\""
                << json_escape(episode.family) << "\",\"task_seed\":" << episode.seed
                << ",\"condition\":\"" << condition << "\",\"example_id\":\""
                << json_escape(item.id) << "\",\"exposure_number\":" << index + 1
                << ",\"slot_id\":" << result.slot
                << ",\"slot_assignment_reason\":\"matched_control_infeasible\""
                << ",\"target_token_id\":" << target
                << ",\"target_probability_before\":" << before.target_probability
                << ",\"requested_update_rho\":" << result.update.delta_rho_requested
                << ",\"actual_update_rho\":null"
                << ",\"final_state_rho\":" << result.update.response_rho
                << ",\"projection_applied\":false"
                << ",\"projection_direction_change_deg\":null"
                << ",\"matching_semantics\":\"incremental\""
                << ",\"status\":\"control_match_infeasible\"}\n";
            updates.flush();
            stages << "{\"episode_id\":\"" << json_escape(episode.id)
                   << "\",\"task_family\":\"" << json_escape(episode.family)
                   << "\",\"task_seed\":" << episode.seed
                   << ",\"condition\":\"" << condition
                   << "\",\"checkpoint\":" << index + 1
                   << ",\"status\":\"control_match_infeasible\"}\n";
            stages.flush();
            return;
        }
        if (!result.update.feasible || result.update.response_rho > opts.rho + 5e-6) {
            throw std::runtime_error("task trust-region violation");
        }
        if (condition == "target") target_schedule[index] = result.update.delta_rho_actual;
        install_bank(ctx, state);
        const auto after = evaluate_state(ctx, vocab, collector, item.prompt, target);
        result.update.target_probability_after = after.target_probability;
        updates << std::setprecision(17)
            << "{\"episode_id\":\"" << json_escape(episode.id) << "\",\"task_family\":\""
            << json_escape(episode.family) << "\",\"task_seed\":" << episode.seed
            << ",\"condition\":\"" << condition << "\",\"example_id\":\"" << json_escape(item.id)
            << "\",\"exposure_number\":" << index + 1 << ",\"slot_id\":" << result.slot
            << ",\"slot_assignment_reason\":\"" << result.assignment_reason
            << "\",\"target_token_id\":" << target
            << ",\"target_probability_before\":" << before.target_probability
            << ",\"target_probability_after\":" << after.target_probability
            << ",\"requested_update_rho\":" << result.update.delta_rho_requested
            << ",\"actual_update_rho\":" << result.update.delta_rho_actual
            << ",\"final_state_rho\":" << result.update.response_rho
            << ",\"projection_applied\":" << (result.update.projected ? "true" : "false")
            << ",\"projection_direction_change_deg\":" << result.projection_angle_deg
            << ",\"q_previous_cosine\":" << result.update.q_previous_cosine
            << ",\"q_response_cosine\":" << result.update.q_response_cosine
            << ",\"matching_semantics\":\"incremental\",\"status\":\"completed\"}\n";
        updates.flush();
        vectors << std::setprecision(9)
            << "{\"episode_id\":\"" << json_escape(episode.id) << "\",\"task_family\":\""
            << json_escape(episode.family) << "\",\"task_seed\":" << episode.seed
            << ",\"condition\":\"" << condition << "\",\"checkpoint\":" << index + 1
            << ",\"dataset_role\":\"adaptation_update\",\"example_id\":\"" << json_escape(item.id)
            << "\",\"vector_kind\":\"update\",\"key\":";
        write_float_array(vectors, normalized(before.hidden));
        vectors << ",\"q\":";
        write_float_array(vectors, target_q);
        vectors << ",\"delta\":";
        write_float_array(vectors, result.actual_delta);
        vectors << "}\n";
        vectors.flush();
        previous_q = condition == "target" ? target_q : control_q;
        if (next_checkpoint < checkpoints.size() && index + 1 == checkpoints[next_checkpoint]) {
            write_task_checkpoint(ctx, vocab, collector, output_head, state, condition, episode,
                index + 1, measurements, vectors, stages, unrelated);
            ++next_checkpoint;
        }
    }
}

void write_cross_task_checkpoint(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const bank & state,
        const task_episode & evaluation_episode, const task_episode & adaptation_episode,
        int32_t checkpoint, std::ofstream & measurements, std::ofstream & vectors,
        std::ofstream & stages, const std::vector<unrelated_reference> & unrelated) {
    write_task_measurements(ctx, vocab, collector, output_head, state, "unrelated_adaptation",
        evaluation_episode, "heldout", checkpoint, evaluation_episode.heldout,
        measurements, vectors, false);
    install_bank(ctx, state);
    const auto unrelated_metrics = measure_unrelated(ctx, vocab, collector, unrelated);
    stages << std::setprecision(17)
        << "{\"episode_id\":\"" << json_escape(evaluation_episode.id)
        << "\",\"task_family\":\"" << json_escape(evaluation_episode.family)
        << "\",\"task_seed\":" << evaluation_episode.seed
        << ",\"adaptation_episode_id\":\"" << json_escape(adaptation_episode.id)
        << "\",\"adaptation_task_family\":\"" << json_escape(adaptation_episode.family)
        << "\",\"condition\":\"unrelated_adaptation\",\"checkpoint\":" << checkpoint
        << ",\"unrelated_nll\":" << unrelated_metrics.first
        << ",\"unrelated_kl\":" << unrelated_metrics.second << "}\n";
    stages.flush();
}

void run_cross_task_condition(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const options & opts,
        const task_episode & evaluation_episode, const task_episode & adaptation_episode,
        const std::vector<unrelated_reference> & unrelated, std::ofstream & measurements,
        std::ofstream & updates, std::ofstream & vectors, std::ofstream & stages) {
    bank state(opts.rank, output_head.width);
    ctx->clear_fast_weights();
    const std::vector<int32_t> checkpoints = {0, 1, 2, 4, 8, 16};
    write_cross_task_checkpoint(ctx, vocab, collector, output_head, state,
        evaluation_episode, adaptation_episode, 0, measurements, vectors, stages, unrelated);
    std::vector<float> previous_q;
    size_t next_checkpoint = 1;
    for (int32_t index = 0; index < 16; ++index) {
        const auto & item = adaptation_episode.adaptation[index];
        const llama_token target = task_target_token(vocab, item);
        install_bank(ctx, state);
        const auto before = evaluate_state(ctx, vocab, collector, item.prompt, target);
        const auto q = exact_target_direction(output_head, before.logits, target, opts.q_threads);
        auto result = update_task_bank(opts, state, "target", index, before.hidden,
            q, q, before.target_probability,
            (uint32_t) std::hash<std::string>{}(adaptation_episode.id + item.id), -1, previous_q);
        if (!result.update.feasible || result.update.response_rho > opts.rho + 5e-6) {
            throw std::runtime_error("cross-task trust-region violation");
        }
        install_bank(ctx, state);
        const auto after = evaluate_state(ctx, vocab, collector, item.prompt, target);
        updates << std::setprecision(17)
            << "{\"episode_id\":\"" << json_escape(evaluation_episode.id)
            << "\",\"task_family\":\"" << json_escape(evaluation_episode.family)
            << "\",\"task_seed\":" << evaluation_episode.seed
            << ",\"adaptation_episode_id\":\"" << json_escape(adaptation_episode.id)
            << "\",\"adaptation_task_family\":\"" << json_escape(adaptation_episode.family)
            << "\",\"condition\":\"unrelated_adaptation\",\"example_id\":\""
            << json_escape(item.id) << "\",\"exposure_number\":" << index + 1
            << ",\"slot_id\":" << result.slot
            << ",\"slot_assignment_reason\":\"" << result.assignment_reason
            << "\",\"target_token_id\":" << target
            << ",\"target_probability_before\":" << before.target_probability
            << ",\"target_probability_after\":" << after.target_probability
            << ",\"requested_update_rho\":" << result.update.delta_rho_requested
            << ",\"actual_update_rho\":" << result.update.delta_rho_actual
            << ",\"final_state_rho\":" << result.update.response_rho
            << ",\"projection_applied\":" << (result.update.projected ? "true" : "false")
            << ",\"projection_direction_change_deg\":" << result.projection_angle_deg
            << ",\"q_previous_cosine\":" << result.update.q_previous_cosine
            << ",\"q_response_cosine\":" << result.update.q_response_cosine
            << ",\"matching_semantics\":\"incremental\",\"status\":\"completed\"}\n";
        updates.flush();
        previous_q = q;
        if (next_checkpoint < checkpoints.size() && index + 1 == checkpoints[next_checkpoint]) {
            write_cross_task_checkpoint(ctx, vocab, collector, output_head, state,
                evaluation_episode, adaptation_episode, index + 1,
                measurements, vectors, stages, unrelated);
            ++next_checkpoint;
        }
    }
}

void run_task_benchmark(
        llama_context * ctx, const llama_vocab * vocab, final_state_collector & collector,
        const output_matrix & output_head, const options & opts,
        const std::vector<task_example> & rows) {
    if (opts.rank != 4 || std::abs(opts.rho - 0.001f) > 1e-9 || opts.decay != 0 || opts.eta != 1) {
        throw std::runtime_error("task benchmark requires rank=4 rho=0.001 decay=0 eta=1");
    }
    const auto episodes = group_task_episodes(rows);
    const std::filesystem::path root(opts.task_output_dir);
    std::filesystem::create_directories(root);
    std::ofstream measurements(root/"measurements.jsonl", std::ios::trunc);
    std::ofstream updates(root/"updates.jsonl", std::ios::trunc);
    std::ofstream vectors(root/"geometry_vectors.jsonl", std::ios::trunc);
    std::ofstream stages(root/"stage_metrics.jsonl", std::ios::trunc);
    std::ofstream validation(root/"tokenizer_validation.jsonl", std::ios::trunc);
    std::ofstream progress(root/"progress.jsonl", std::ios::trunc);
    if (!measurements || !updates || !vectors || !stages || !validation || !progress) {
        throw std::runtime_error("cannot create task benchmark outputs");
    }

    for (const auto & row : rows) {
        validation << "{\"episode_id\":\"" << json_escape(row.episode_id)
            << "\",\"example_id\":\"" << json_escape(row.id) << "\",\"split\":\"" << row.split
            << "\",\"target\":\"" << json_escape(row.target) << "\",\"token_id\":"
            << task_target_token(vocab, row) << ",\"token_count\":1}\n";
    }
    validation.flush();
    ctx->clear_fast_weights();
    const auto unrelated = make_unrelated_reference(ctx, vocab, collector);

    if (opts.task_mode == "unrelated") {
        for (size_t index = 0; index < episodes.size(); ++index) {
            const auto & evaluation_episode = episodes[index];
            const task_episode * adaptation_episode = nullptr;
            for (size_t offset = 1; offset < episodes.size(); ++offset) {
                const auto & candidate = episodes[(index + offset)%episodes.size()];
                if (candidate.seed == evaluation_episode.seed &&
                        candidate.family != evaluation_episode.family) {
                    adaptation_episode = &candidate;
                    break;
                }
            }
            if (adaptation_episode == nullptr) {
                throw std::runtime_error("cross-task control requires another family with the same seed");
            }
            std::fprintf(stderr, "task benchmark start eval=%s adapt=%s condition=unrelated_adaptation\n",
                evaluation_episode.id.c_str(), adaptation_episode->id.c_str());
            run_cross_task_condition(ctx, vocab, collector, output_head, opts,
                evaluation_episode, *adaptation_episode, unrelated,
                measurements, updates, vectors, stages);
            progress << "{\"episode_id\":\"" << json_escape(evaluation_episode.id)
                << "\",\"adaptation_episode_id\":\"" << json_escape(adaptation_episode->id)
                << "\",\"condition\":\"unrelated_adaptation\",\"status\":\"complete\"}\n";
            progress.flush();
        }
        return;
    }

    for (const auto & episode : episodes) {
        std::vector<double> target_schedule(16, -1);
        for (const std::string condition : {"target", "random", "shuffled", "label_permuted"}) {
            std::fprintf(stderr, "task benchmark start episode=%s condition=%s\n",
                episode.id.c_str(), condition.c_str());
            run_task_condition(ctx, vocab, collector, output_head, opts, episode, condition,
                target_schedule, unrelated, measurements, updates, vectors, stages);
            progress << "{\"episode_id\":\"" << json_escape(episode.id) << "\",\"condition\":\""
                << condition << "\",\"status\":\"complete\"}\n";
            progress.flush();
            std::fprintf(stderr, "task benchmark complete episode=%s condition=%s\n",
                episode.id.c_str(), condition.c_str());
        }
    }
}
