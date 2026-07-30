#include "sparse-mul-mat.cuh"

#include <type_traits>

template <typename input_t>
static __global__ void sparse_mul_mat_f32(
        const int32_t * __restrict__ row_ptr,
        const uint16_t * __restrict__ col_idx,
        const half * __restrict__ values,
        const char * __restrict__ src,
        float * __restrict__ dst,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        uint64_t src_nb0,
        uint64_t src_nb1,
        uint64_t src_nb2,
        uint64_t src_nb3,
        uint64_t dst_nb0,
        uint64_t dst_nb1,
        uint64_t dst_nb2,
        uint64_t dst_nb3) {
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= ne0 * ne1 * ne2 * ne3) {
        return;
    }

    int64_t rem = ir;
    const int64_t row = rem % ne0;
    rem /= ne0;
    const int64_t i1 = rem % ne1;
    rem /= ne1;
    const int64_t i2 = rem % ne2;
    const int64_t i3 = rem / ne2;

    float sum = 0.0f;
    for (int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        const char * xp = src
            + (uint64_t) col_idx[p] * src_nb0
            + (uint64_t) i1 * src_nb1
            + (uint64_t) i2 * src_nb2
            + (uint64_t) i3 * src_nb3;
        if constexpr (std::is_same_v<input_t, half>) {
            sum += __half2float(values[p]) * __half2float(*(const half *) xp);
        } else {
            sum += __half2float(values[p]) * *(const float *) xp;
        }
    }

    *(float *) ((char *) dst
        + (uint64_t) row * dst_nb0
        + (uint64_t) i1 * dst_nb1
        + (uint64_t) i2 * dst_nb2
        + (uint64_t) i3 * dst_nb3) = sum;
}

void ggml_cuda_op_sparse_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * row_ptr = dst->src[0];
    const ggml_tensor * col_idx = dst->src[1];
    const ggml_tensor * values  = dst->src[2];
    const ggml_tensor * src     = dst->src[3];

    GGML_ASSERT(row_ptr->type == GGML_TYPE_I32);
    GGML_ASSERT(col_idx->type == GGML_TYPE_I16);
    GGML_ASSERT(values->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(row_ptr->ne[0] == dst->ne[0] + 1);

    const int64_t n = ggml_nelements(dst);
    const int threads = 256;
    const int blocks = (int) ((n + threads - 1) / threads);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());

    if (src->type == GGML_TYPE_F32) {
        ggml_cuda_kernel_launch(sparse_mul_mat_f32<float>, launch_params,
            (const int32_t *) row_ptr->data,
            (const uint16_t *) col_idx->data,
            (const half *) values->data,
            (const char *) src->data,
            (float *) dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            (uint64_t) src->nb[0], (uint64_t) src->nb[1],
            (uint64_t) src->nb[2], (uint64_t) src->nb[3],
            (uint64_t) dst->nb[0], (uint64_t) dst->nb[1],
            (uint64_t) dst->nb[2], (uint64_t) dst->nb[3]);
    } else {
        ggml_cuda_kernel_launch(sparse_mul_mat_f32<half>, launch_params,
            (const int32_t *) row_ptr->data,
            (const uint16_t *) col_idx->data,
            (const half *) values->data,
            (const char *) src->data,
            (float *) dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            (uint64_t) src->nb[0], (uint64_t) src->nb[1],
            (uint64_t) src->nb[2], (uint64_t) src->nb[3],
            (uint64_t) dst->nb[0], (uint64_t) dst->nb[1],
            (uint64_t) dst->nb[2], (uint64_t) dst->nb[3]);
    }
}
