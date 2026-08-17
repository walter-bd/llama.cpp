#include "server-sidecar-profile.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace {

const server_sidecar_profile & qwen35_step30_profile() {
    static const server_sidecar_profile profile = {
        "qwen35-4b-step30-v1",
        "qwen35",
        2560,
        32,
        server_sidecar_hook::result_norm,
        "identity",
        "v1",
        {
            {
                server_sidecar_hook::residual_post_layer,
                server_sidecar_index_space::public_cvec,
                24,
                23,
                server_sidecar_validated_topology::block24_scale(),
                true,
            },
            {
                server_sidecar_hook::residual_post_layer,
                server_sidecar_index_space::public_cvec,
                28,
                27,
                server_sidecar_validated_topology::block28_scale(),
                true,
            },
        },
        {
            {
                server_sidecar_hook::residual_post_layer,
                server_sidecar_index_space::public_cvec,
                24,
                23,
                server_sidecar_validated_topology::block24_scale(),
                true,
            },
            {
                server_sidecar_hook::residual_post_layer,
                server_sidecar_index_space::public_cvec,
                28,
                27,
                server_sidecar_validated_topology::block28_scale(),
                true,
            },
            {
                server_sidecar_hook::logits_pre_head,
                server_sidecar_index_space::public_cvec,
                -1,
                -1,
                server_sidecar_validated_topology::final_scale(),
                true,
            },
        },
        true,
        true,
    };
    return profile;
}

bool same_string(const std::string & left, const char * right) {
    return left == right;
}

} // namespace

std::array<double, 3> server_sidecar_validated_topology::normalized_runtime_weights() {
    const double norm = std::sqrt(
        block24_scale() * block24_scale() +
        block28_scale() * block28_scale() +
        final_scale() * final_scale());
    return {
        block24_scale() / norm,
        block28_scale() / norm,
        final_scale() / norm,
    };
}

double server_sidecar_authority_controller::budget(double response_norm) {
    return max_budget() * std::min(1.0, std::max(0.0, response_norm / reference_norm()));
}

const server_sidecar_profile * server_sidecar_profile_registry::by_id(const std::string & id) {
    if (id == "qwen35-4b-step30-v1") {
        return &qwen35_step30_profile();
    }
    return nullptr;
}

std::string server_sidecar_profile_registry::model_architecture(const llama_model * model) {
    if (model == nullptr) {
        return {};
    }
    std::array<char, 128> buffer{};
    if (llama_model_meta_val_str(model, "general.architecture", buffer.data(), buffer.size()) <= 0) {
        return {};
    }
    return std::string(buffer.data());
}

const server_sidecar_profile * server_sidecar_profile_registry::for_model(
        const llama_model * model,
        const std::string & requested_id,
        std::string & error) {
    if (model == nullptr) {
        error = "sidecar_injection_profile_not_supported: model is unavailable";
        return nullptr;
    }

    const std::string architecture = model_architecture(model);
    const server_sidecar_profile * profile = nullptr;
    if (requested_id.empty()) {
        if (same_string(architecture, "qwen35")) {
            profile = &qwen35_step30_profile();
        }
    } else {
        profile = by_id(requested_id);
    }

    if (profile == nullptr) {
        error = "sidecar_injection_profile_not_supported: no profile for architecture '" +
            architecture + "'";
        return nullptr;
    }
    if (!requested_id.empty() && !same_string(architecture, profile->architecture.c_str())) {
        error = "sidecar_injection_profile_not_supported: profile architecture mismatch (model='" +
            architecture + "', profile='" + profile->architecture + "')";
        return nullptr;
    }
    if (llama_model_n_embd(model) != profile->hidden_size) {
        error = "sidecar_injection_profile_not_supported: hidden dimension mismatch (model=" +
            std::to_string(llama_model_n_embd(model)) + ", profile=" +
            std::to_string(profile->hidden_size) + ")";
        return nullptr;
    }
    if (llama_model_n_layer(model) < profile->minimum_layers) {
        error = "sidecar_injection_profile_not_supported: insufficient model layers (model=" +
            std::to_string(llama_model_n_layer(model)) + ", profile minimum=" +
            std::to_string(profile->minimum_layers) + ")";
        return nullptr;
    }
    error.clear();
    return profile;
}

bool server_sidecar_representation_adapter::validate_identity(
        const server_sidecar_profile & profile,
        int32_t width,
        std::string & error) {
    if (profile.adapter_id != "identity" || profile.adapter_version != "v1") {
        error = "sidecar_representation_adapter_not_supported: " + profile.adapter_id +
            " " + profile.adapter_version;
        return false;
    }
    if (width != profile.hidden_size) {
        error = "sidecar_representation_dimension_mismatch: expected " +
            std::to_string(profile.hidden_size) + ", got " + std::to_string(width);
        return false;
    }
    return true;
}

bool server_sidecar_injection_engine::apply_internal_cvec(
        llama_context * ctx,
        const server_sidecar_profile & profile,
        const float * direction,
        int32_t width,
        int32_t model_layers,
        double budget_value,
        server_sidecar_dispatch_telemetry & telemetry,
        std::string & error) {
    if (ctx == nullptr || direction == nullptr) {
        error = "sidecar injection context or direction is unavailable";
        return false;
    }
    if (!server_sidecar_representation_adapter::validate_identity(profile, width, error)) {
        return false;
    }
    if (model_layers < profile.minimum_layers) {
        error = "sidecar injection profile requires more model layers than are available";
        return false;
    }
    if (budget_value < 0.0 || budget_value > server_sidecar_authority_controller::max_budget() + 1e-9) {
        error = "sidecar injection authority is outside the validated range";
        return false;
    }
    if (profile.supported_sites.empty()) {
        error = "sidecar_injection_profile_not_supported: no supported injection sites";
        return false;
    }

    const std::array<double, 2> alphas = {
        budget_value * server_sidecar_validated_topology::block24_scale(),
        budget_value * server_sidecar_validated_topology::block28_scale(),
    };
    return apply_internal_cvec_alphas(
        ctx, profile, direction, width, model_layers, alphas, telemetry, error);
}

bool server_sidecar_injection_engine::apply_internal_cvec_alphas(
        llama_context * ctx,
        const server_sidecar_profile & profile,
        const float * direction,
        int32_t width,
        int32_t model_layers,
        const std::array<double, 2> & alphas,
        server_sidecar_dispatch_telemetry & telemetry,
        std::string & error) {
    if (ctx == nullptr || direction == nullptr) {
        error = "sidecar injection context or direction is unavailable";
        return false;
    }
    if (!server_sidecar_representation_adapter::validate_identity(profile, width, error)) {
        return false;
    }
    if (model_layers < profile.minimum_layers) {
        error = "sidecar injection profile requires more model layers than are available";
        return false;
    }
    if (profile.supported_sites.size() != alphas.size()) {
        error = "sidecar_injection_profile_not_supported: expected two hidden sites";
        return false;
    }

    int32_t first_layer = model_layers;
    int32_t last_layer = -1;
    std::vector<float> cvec(size_t(model_layers - 1) * size_t(width), 0.0f);
    for (size_t site_index = 0; site_index < profile.supported_sites.size(); ++site_index) {
        const auto & site = profile.supported_sites[site_index];
        if (!site.supported || site.hook != server_sidecar_hook::residual_post_layer ||
            site.index_space != server_sidecar_index_space::public_cvec ||
            site.graph_layer <= 0 || site.public_layer <= 0 ||
            site.public_layer >= model_layers || site.graph_layer != site.public_layer + 1) {
            error = "sidecar_injection_profile_not_supported: invalid public cvec site mapping";
            return false;
        }
        first_layer = std::min(first_layer, site.public_layer);
        last_layer = std::max(last_layer, site.public_layer);
        const size_t offset = size_t(site.public_layer - 1) * size_t(width);
        for (int32_t i = 0; i < width; ++i) {
            cvec[offset + size_t(i)] = float(alphas[site_index] * double(direction[size_t(i)]));
        }
    }
    if (first_layer > last_layer || llama_set_adapter_cvec(
            ctx, cvec.data(), cvec.size(), width, first_layer, last_layer) != 0) {
        error = "llama_set_adapter_cvec rejected profile injection";
        return false;
    }

    for (const auto & site : profile.supported_sites) {
        ++telemetry.injection_sites;
        if (site.graph_layer == 24) {
            ++telemetry.graph_layer_24;
        } else if (site.graph_layer == 28) {
            ++telemetry.graph_layer_28;
        }
    }
    return true;
}

bool server_sidecar_injection_engine::clear_internal_cvec(
        llama_context * ctx,
        int32_t width,
        std::string & error) {
    if (ctx == nullptr || width <= 0) {
        error = "sidecar injection context or width is unavailable";
        return false;
    }
    if (llama_set_adapter_cvec(ctx, nullptr, 0, width, -1, -1) != 0) {
        error = "llama_set_adapter_cvec rejected cvec clear";
        return false;
    }
    return true;
}
