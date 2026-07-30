#pragma once

#include <cstdint>
#include <memory>
#include <string>

struct ggml_tensor;
struct llama_model;

struct llama_sparse_residual_entry {
    ggml_tensor * row_ptr = nullptr;
    ggml_tensor * col_idx = nullptr;
    ggml_tensor * values  = nullptr;

    int64_t n_rows = 0;
    int64_t n_cols = 0;
};

class llama_sparse_residual {
public:
    ~llama_sparse_residual();

    static std::unique_ptr<llama_sparse_residual> load(
            const std::string & path,
            const llama_model & model);

    const llama_sparse_residual_entry * get(const ggml_tensor * weight) const;

private:
    struct impl;

    explicit llama_sparse_residual(std::unique_ptr<impl> pimpl);

    std::unique_ptr<impl> pimpl;
};
