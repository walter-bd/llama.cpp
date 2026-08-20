#pragma once

#include "llama.h"
#include "server-sidecar-profile.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

struct server_sidecar_runtime_step {
    std::vector<float> direction;
    double raw_norm = 0.0;
    double budget = 0.0;
    std::string source;
};

// The manager owns portable sidecar state and retrieval/write lifecycle.  The
// server owns token orchestration; retrieval deliberately returns a normalized
// direction instead of installing a context-global cvec.
class server_sidecar_manager {
public:
    server_sidecar_manager() = default;
    ~server_sidecar_manager() = default;

    server_sidecar_manager(const server_sidecar_manager &) = delete;
    server_sidecar_manager & operator=(const server_sidecar_manager &) = delete;

    void configure(llama_model * model, const std::string & sidecar_root);
    void clear_context(llama_context * ctx) const;
    void invalidate_episodes();

    bool model_ready() const;
    bool has(const std::string & id) const;

    bool capture_query(
            const std::string & id,
            llama_context * ctx,
            int32_t output_index,
            int32_t width,
            std::vector<float> & query,
            std::string & error);

    nlohmann::ordered_json status(const std::string & id) const;
    nlohmann::ordered_json create(const std::string & id, bool learning_enabled);
    nlohmann::ordered_json load(const std::string & id, const std::string & path, bool learning_enabled);
    nlohmann::ordered_json save(const std::string & id, const std::string & path);
    nlohmann::ordered_json reset(const std::string & id);
    nlohmann::ordered_json unload(const std::string & id);
    nlohmann::ordered_json erase(const std::string & id);
    nlohmann::ordered_json set_learning(const std::string & id, bool enabled);

    // Experience path: derive the validated Step-30 direction from the
    // observed token and the clean prepass greedy token captured in the
    // episode.
    nlohmann::ordered_json observe(
            const std::string & id,
            const std::string & episode_id,
            llama_token observed_token,
            const std::string & observed_target);

    // Explicit research/debug path.  Normal clients must not manufacture the
    // model-width write direction.
    nlohmann::ordered_json observe_debug(
            const std::string & id,
            const std::string & episode_id,
            const std::string & observed_target,
            const std::string & source,
            const std::vector<float> & write_direction,
            double novelty,
            double surprise,
            double recurrence,
            double consistency);

    // Compatibility path for the legacy cvec API.  Functional Step-30
    // requests use prepare_runtime_step() plus explicit replay operations.
    bool apply_query(
            const std::string & id,
            llama_context * ctx,
            const float * query,
            int32_t width,
            std::string * episode_id,
            std::string & error);

    bool prepare_runtime_step(
            const std::string & id,
            const float * query,
            int32_t width,
            std::string * episode_id,
            llama_token baseline_greedy,
            server_sidecar_runtime_step & step,
            std::string & error);

    bool apply_runtime_hidden(
            const std::string & id,
            llama_context * ctx,
            const float * direction,
            int32_t width,
            const std::array<double, 2> & alphas,
            std::string & error);

    bool clear_runtime_hidden(
            llama_context * ctx,
            int32_t width,
            std::string & error) const;

    bool apply_runtime_step30(
            const std::string & id,
            llama_context * ctx,
            const float * direction,
            int32_t width,
            const std::array<double, 3> & rho,
            std::string & error);

    void clear_runtime_step30(llama_context * ctx) const;

    bool validate_request(
            const std::string & id,
            int32_t n_parallel,
            int32_t n_completions,
            int32_t n_predict,
            bool speculative,
            std::string & error) const;

private:
    struct sidecar;

    std::shared_ptr<sidecar> get(const std::string & id) const;
    std::string package_path(const std::string & path, bool for_write) const;

    mutable std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<sidecar>> sidecars;
    llama_model * model = nullptr;
    int32_t model_width = 0;
    int32_t model_layers = 0;
    const server_sidecar_profile * injection_profile = nullptr;
    std::string profile_error;
    std::string root;
};
