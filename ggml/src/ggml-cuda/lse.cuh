#pragma once

#include "common.cuh"

void ggml_cuda_lse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_lse_argmax(const float * x, float * lse_out, int32_t * argmax_out,
                           int64_t nrows, int64_t ncols, void * stream_ptr);

// Launch LSE+argmax kernel using the ggml backend's CUDA stream.
// This avoids the implicit synchronization caused by using the default CUDA stream.
void ggml_cuda_lse_argmax_on_backend(ggml_backend_t backend,
                                       const float * x, float * lse_out, int32_t * argmax_out,
                                       int64_t nrows, int64_t ncols);

// Allocate/free/download device buffers for LSE+argmax results.
void ggml_cuda_lse_argmax_alloc(int64_t n_batch, float ** dev_lse, int32_t ** dev_argmax);
void ggml_cuda_lse_argmax_free(float * dev_lse, int32_t * dev_argmax);
void ggml_cuda_lse_argmax_download(const float * dev_lse, const int32_t * dev_argmax,
                                    float * host_lse, int32_t * host_argmax,
                                    int64_t n_tokens);
