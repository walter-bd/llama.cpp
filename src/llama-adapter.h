#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

// Context-local Qwen3.5 Step-30 intervention. The three tensors contain the
// normalized retrieved direction pre-scaled by the validated per-site rho.
// Local hidden norms and the final output projection remain in the graph.
struct llama_adapter_step30 {
    bool init(const llama_model & model);
    bool set(
            const float * direction,
                   size_t direction_count,
                    float rho_shallow,
                    float rho_deep,
                    float rho_final);

    ggml_tensor * apply_hidden(ggml_context * ctx, ggml_tensor * cur, int il) const;
    ggml_tensor * apply_output(
            ggml_context * ctx,
            ggml_tensor * logits,
            ggml_tensor * final_hidden,
            ggml_tensor * output) const;

    int32_t get_n_embd() const { return n_embd; }

private:
    int32_t n_embd = 0;
    ggml_tensor * shallow = nullptr;
    ggml_tensor * deep    = nullptr;
    ggml_tensor * final   = nullptr;
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};

using llama_adapter_step30_ptr = std::unique_ptr<llama_adapter_step30>;

// Context-local low-rank residual applied after the final model norm.
// Tensor layout follows GGML mul_mat conventions:
//   v: [n_embd, rank], u: [rank, n_embd].
struct llama_adapter_fast_weights {
    bool init(const llama_model & model, int32_t rank);
    bool set(const float * u_data, size_t u_count, const float * v_data, size_t v_count);
    void clear();

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur) const;

    int32_t get_rank() const { return rank; }
    int32_t get_n_embd() const { return n_embd; }

private:
    int32_t rank = 0;
    int32_t n_embd = 0;
    ggml_tensor * u = nullptr;
    ggml_tensor * v = nullptr;
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};

using llama_adapter_fast_weights_ptr = std::unique_ptr<llama_adapter_fast_weights>;

// External AttnRes-inspired reader over eight frozen Qwen residual-stream taps.
// The reader owns only a pseudo-query and a context-local low-rank map. It does
// not replace or mutate the model's residual connections.
enum llama_attnres_route_mode {
    LLAMA_ATTNRES_ROUTE_QUERY = 0,
    LLAMA_ATTNRES_ROUTE_FINAL_QUERY,
    LLAMA_ATTNRES_ROUTE_ONE_HOT,
};

struct llama_adapter_attnres_result {
    ggml_tensor * output     = nullptr;
    ggml_tensor * routed     = nullptr;
    ggml_tensor * scores     = nullptr;
    ggml_tensor * weights    = nullptr;
    ggml_tensor * correction = nullptr;
};

struct llama_adapter_attnres_sidecar {
    static constexpr int32_t N_TAPS = 8;

    bool init(
            const llama_model & model,
            int32_t rank,
            llama_attnres_route_mode mode,
            int32_t one_hot_tap,
            bool scale_scores);
    bool set(
            const float * u_data, size_t u_count,
            const float * v_data, size_t v_count);
    bool set_query(const float * query_data, size_t query_count);
    void clear_map();

    llama_adapter_attnres_result apply_to(
            ggml_context * ctx,
            ggml_tensor * cur,
            const std::array<ggml_tensor *, N_TAPS> & taps,
            float rms_eps) const;

    bool matches(
            int32_t requested_rank,
            llama_attnres_route_mode requested_mode,
            int32_t requested_one_hot_tap,
            bool requested_scale_scores) const;

    int32_t get_rank() const { return rank; }
    int32_t get_n_embd() const { return n_embd; }
    llama_attnres_route_mode get_mode() const { return mode; }
    int32_t get_one_hot_tap() const { return one_hot_tap; }
    bool get_scale_scores() const { return scale_scores; }

private:
    int32_t rank = 0;
    int32_t n_embd = 0;
    llama_attnres_route_mode mode = LLAMA_ATTNRES_ROUTE_QUERY;
    int32_t one_hot_tap = -1;
    bool scale_scores = true;

    ggml_tensor * u = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * query = nullptr;
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};

using llama_adapter_attnres_sidecar_ptr = std::unique_ptr<llama_adapter_attnres_sidecar>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;
