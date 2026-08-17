#pragma once

#include "llama.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// These hooks name graph boundaries, not model families.  A profile maps a
// boundary to the public/backend capability that can actually implement it.
enum class server_sidecar_hook {
    result_norm,
    residual_post_layer,
    logits_pre_head,
};

enum class server_sidecar_index_space {
    public_cvec,
};

struct server_sidecar_injection_site {
    server_sidecar_hook hook = server_sidecar_hook::residual_post_layer;
    server_sidecar_index_space index_space = server_sidecar_index_space::public_cvec;
    int32_t graph_layer = -1;
    int32_t public_layer = -1;
    double scale = 0.0;
    bool supported = false;
};

struct server_sidecar_profile {
    std::string id;
    std::string architecture;
    int32_t hidden_size = 0;
    int32_t minimum_layers = 0;
    server_sidecar_hook query_hook = server_sidecar_hook::result_norm;
    std::string adapter_id;
    std::string adapter_version;

    // Sites implemented by the current public llama.cpp backend.
    std::vector<server_sidecar_injection_site> supported_sites;

    // The complete validated topology is recorded for status and diagnostics.
    std::vector<server_sidecar_injection_site> research_sites;
    bool final_logit_injection_supported = false;
    bool recursive_state_path_supported = false;
};

struct server_sidecar_dispatch_telemetry {
    uint64_t injection_sites = 0;
    uint64_t graph_layer_24 = 0;
    uint64_t graph_layer_28 = 0;
};

class server_sidecar_validated_topology {
public:
    static constexpr double block24_scale() { return 0.23072; }
    static constexpr double block28_scale() { return 0.38322; }
    static constexpr double final_scale() { return 0.89438; }

    static std::array<double, 3> normalized_runtime_weights();
};

class server_sidecar_authority_controller {
public:
    static constexpr double max_budget() { return 0.01; }
    static constexpr double reference_norm() { return 0.05; }

    static double budget(double response_norm);
};

class server_sidecar_profile_registry {
public:
    static const server_sidecar_profile * by_id(const std::string & id);
    static const server_sidecar_profile * for_model(
            const llama_model * model,
            const std::string & requested_id,
            std::string & error);
    static std::string model_architecture(const llama_model * model);
};

class server_sidecar_representation_adapter {
public:
    static bool validate_identity(
            const server_sidecar_profile & profile,
            int32_t width,
            std::string & error);
};

class server_sidecar_injection_engine {
public:
    // Applies the profile's supported public cvec sites using the historical
    // budget-scaled coefficients.  This remains available for compatibility
    // with the lifecycle/API smoke path.
    static bool apply_internal_cvec(
            llama_context * ctx,
            const server_sidecar_profile & profile,
            const float * direction,
            int32_t width,
            int32_t model_layers,
            double budget,
            server_sidecar_dispatch_telemetry & telemetry,
            std::string & error);

    // Apply explicit per-site coefficients.  The functional Step-30 path
    // uses locally calibrated coefficients rather than budget * site.scale.
    static bool apply_internal_cvec_alphas(
            llama_context * ctx,
            const server_sidecar_profile & profile,
            const float * direction,
            int32_t width,
            int32_t model_layers,
            const std::array<double, 2> & alphas,
            server_sidecar_dispatch_telemetry & telemetry,
            std::string & error);

    static bool clear_internal_cvec(
            llama_context * ctx,
            int32_t width,
            std::string & error);
};
