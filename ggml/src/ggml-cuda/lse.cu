#include "lse.cuh"
#include "common.cuh"

// Per-row log-sum-exp + argmax kernel.
// x:         [nrows, ncols] row-major float input
// lse_out:   [nrows] float output — lse_out[row] = max + log(sum(exp(x - max)))
// argmax_out:[nrows] int32 output — argmax_out[row] = index of max element in row
static __global__ void lse_argmax_f32(
        const float * __restrict__ x,
              float * __restrict__ lse_out,
              int32_t * __restrict__ argmax_out,
        const int64_t ncols) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;

    const float * rowx = x + row * ncols;

    // Pass 1: find row max + argmax
    float     max_val = -FLT_MAX;
    int32_t   max_idx = -1;

    for (int64_t col = tid; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > max_val) {
            max_val = val;
            max_idx = (int32_t)col;
        }
    }

    // Warp-level reduce max + argmax
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        const float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, offset, WARP_SIZE);
        const int   other_idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, offset, WARP_SIZE);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    // Block-level reduce max + argmax (multi-warp)
    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    __shared__ float   shared_max[32];
    __shared__ int32_t shared_idx[32];

    if (n_warps > 1) {
        if (lane_id == 0) {
            shared_max[warp_id] = max_val;
            shared_idx[warp_id] = max_idx;
        }
        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                max_val = shared_max[lane_id];
                max_idx = shared_idx[lane_id];
            } else {
                max_val = -FLT_MAX;
            }
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                const float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, offset, WARP_SIZE);
                const int   other_idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, offset, WARP_SIZE);
                if (other_val > max_val) {
                    max_val = other_val;
                    max_idx = other_idx;
                }
            }
        }
    }

    // Broadcast max_val to all threads (needed for pass 2)
    __shared__ float block_max;
    if (warp_id == 0 && lane_id == 0) {
        block_max = max_val;
    }
    __syncthreads();
    max_val = block_max;

    // Pass 2: compute sum(exp(x - max))
    float sum_exp = 0.0f;
    for (int64_t col = tid; col < ncols; col += blockDim.x) {
        sum_exp += expf(rowx[col] - max_val);
    }

    // Block-level sum reduce
    __shared__ float shared_sum[32];
    sum_exp = block_reduce<block_reduce_method::SUM>(sum_exp, shared_sum);

    // Write output
    if (warp_id == 0 && lane_id == 0) {
        lse_out[row]    = max_val + logf(sum_exp);
        argmax_out[row] = max_idx;
    }
}

static void lse_argmax_launch(const float * x, float * lse_out, int32_t * argmax_out,
                               int64_t nrows, int64_t ncols, cudaStream_t stream) {
    const int num_threads = std::min<int64_t>(1024, (ncols + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num((unsigned int)nrows, 1, 1);
    lse_argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(x, lse_out, argmax_out, ncols);
}

void ggml_cuda_lse_argmax(const float * x, float * lse_out, int32_t * argmax_out,
                           int64_t nrows, int64_t ncols, void * stream_ptr) {
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : 0;
    lse_argmax_launch(x, lse_out, argmax_out, nrows, ncols, stream);
}

// Launch LSE+argmax kernel using the ggml backend's CUDA context and stream.
// This avoids the implicit synchronization caused by using the default CUDA stream (0).
void ggml_cuda_lse_argmax_on_backend(ggml_backend_t backend,
                                       const float * x, float * lse_out, int32_t * argmax_out,
                                       int64_t nrows, int64_t ncols) {
    // Get the ggml_backend_cuda_context from the backend's device
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        // Fallback to default stream
        lse_argmax_launch(x, lse_out, argmax_out, nrows, ncols, 0);
        return;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) {
        lse_argmax_launch(x, lse_out, argmax_out, nrows, ncols, 0);
        return;
    }

    // Get the CUDA context via proc address
    using get_ctx_fn_t = ggml_backend_cuda_context * (*)(ggml_backend_t);
    auto get_ctx_fn = (get_ctx_fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_get_context");
    if (get_ctx_fn) {
        ggml_backend_cuda_context * ctx = get_ctx_fn(backend);
        cudaStream_t stream = ctx ? ctx->stream() : 0;
        lse_argmax_launch(x, lse_out, argmax_out, nrows, ncols, stream);
    } else {
        lse_argmax_launch(x, lse_out, argmax_out, nrows, ncols, 0);
    }
}

void ggml_cuda_lse_argmax_alloc(int64_t n_batch, float ** dev_lse, int32_t ** dev_argmax) {
    cudaMalloc(dev_lse, n_batch * sizeof(float));
    cudaMalloc(dev_argmax, n_batch * sizeof(int32_t));
}

void ggml_cuda_lse_argmax_free(float * dev_lse, int32_t * dev_argmax) {
    cudaFree(dev_lse);
    cudaFree(dev_argmax);
}

void ggml_cuda_lse_argmax_download(const float * dev_lse, const int32_t * dev_argmax,
                                    float * host_lse, int32_t * host_argmax,
                                    int64_t n_tokens) {
    cudaMemcpy(host_lse, dev_lse, n_tokens * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_argmax, dev_argmax, n_tokens * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

// Per-row log-sum-exp kernel (LSE only, no argmax).
// x:       [nrows, ncols] row-major float input
// lse_out: [nrows] float output — lse_out[row] = max + log(sum(exp(x - max)))
static __global__ void lse_f32(
        const float * __restrict__ x,
              float * __restrict__ lse_out,
        const int64_t ncols) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;

    const float * rowx = x + row * ncols;

    // Pass 1: find row max
    float max_val = -FLT_MAX;
    for (int64_t col = tid; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Warp-level reduce max
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        const float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, offset, WARP_SIZE);
        if (other_val > max_val) {
            max_val = other_val;
        }
    }

    // Block-level reduce max (multi-warp)
    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    __shared__ float shared_max[32];
    if (n_warps > 1) {
        if (lane_id == 0) {
            shared_max[warp_id] = max_val;
        }
        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                max_val = shared_max[lane_id];
            } else {
                max_val = -FLT_MAX;
            }
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                const float other_val = __shfl_xor_sync(0xFFFFFFFF, max_val, offset, WARP_SIZE);
                if (other_val > max_val) {
                    max_val = other_val;
                }
            }
        }
    }

    // Broadcast max_val to all threads
    __shared__ float block_max;
    if (warp_id == 0 && lane_id == 0) {
        block_max = max_val;
    }
    __syncthreads();
    max_val = block_max;

    // Pass 2: compute sum(exp(x - max))
    float sum_exp = 0.0f;
    for (int64_t col = tid; col < ncols; col += blockDim.x) {
        sum_exp += expf(rowx[col] - max_val);
    }

    // Block-level sum reduce
    __shared__ float shared_sum[32];
    sum_exp = block_reduce<block_reduce_method::SUM>(sum_exp, shared_sum);

    // Write output
    if (warp_id == 0 && lane_id == 0) {
        lse_out[row] = max_val + logf(sum_exp);
    }
}

void ggml_cuda_lse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks  = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    lse_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
