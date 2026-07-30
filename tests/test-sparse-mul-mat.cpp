#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return 1;
    }

    ggml_tensor * row_ptr = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    ggml_tensor * col_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I16, 3);
    ggml_tensor * values = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 3);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * output = ggml_sparse_mul_mat(ctx, row_ptr, col_idx, values, input, 2, 3);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }

    const int32_t rows[] = {0, 2, 3};
    const int16_t cols[] = {0, 2, 1};
    const ggml_fp16_t vals[] = {
        ggml_fp32_to_fp16(1.5f),
        ggml_fp32_to_fp16(-2.0f),
        ggml_fp32_to_fp16(0.5f),
    };
    const float inputs[] = {
        2.0f, 3.0f, 4.0f,
        -1.0f, 2.0f, 0.5f,
    };

    ggml_backend_tensor_set(row_ptr, rows, 0, sizeof(rows));
    ggml_backend_tensor_set(col_idx, cols, 0, sizeof(cols));
    ggml_backend_tensor_set(values, vals, 0, sizeof(vals));
    ggml_backend_tensor_set(input, inputs, 0, sizeof(inputs));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> actual(4);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    const float expected[] = {-5.0f, 1.5f, -2.5f, 1.0f};

    bool ok = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-5f) {
            std::fprintf(
                stderr,
                "output[%zu] mismatch: got %.8f expected %.8f\n",
                i,
                actual[i],
                expected[i]);
            ok = false;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (!ok) {
        return 1;
    }

    std::puts("sparse_mul_mat CPU test passed");
    return 0;
}
