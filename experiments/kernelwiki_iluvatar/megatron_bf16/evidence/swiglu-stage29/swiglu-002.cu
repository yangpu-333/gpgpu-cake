#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

__global__ void swiglu_forward_kernel(const at::BFloat16* __restrict__ x,
                                      at::BFloat16* __restrict__ out,
                                      int64_t N, int64_t H) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= N) return;
  const int64_t row = i / H;
  const int64_t col = i - row * H;
  const int64_t gateidx = row * (2 * H) + col;
  const int64_t upidx = gateidx + H;
  const float a = static_cast<float>(x[gateidx]);
  const float b = static_cast<float>(x[upidx]);
  const float sig = 1.0f / (1.0f + expf(-a));
  const float o = a * sig * b;
  out[i] = at::BFloat16(o);
}

__global__ void swiglu_backward_kernel(const at::BFloat16* __restrict__ x,
                                       const at::BFloat16* __restrict__ grad,
                                       at::BFloat16* __restrict__ dx,
                                       int64_t N, int64_t H) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= N) return;
  const int64_t row = i / H;
  const int64_t col = i - row * H;
  const int64_t gateidx = row * (2 * H) + col;
  const int64_t upidx = gateidx + H;
  const float a = static_cast<float>(x[gateidx]);
  const float b = static_cast<float>(x[upidx]);
  const float g = static_cast<float>(grad[i]);
  const float sig = 1.0f / (1.0f + expf(-a));
  const float dgate = g * sig * (1.0f + a * (1.0f - sig)) * b;
  const float dup = g * (a * sig);
  dx[gateidx] = at::BFloat16(dgate);
  dx[upidx] = at::BFloat16(dup);
}

void swiglu_forward_launch(const at::Tensor& x, at::Tensor& out,
                           int64_t rows, int64_t H) {
  const int64_t N = rows * H;
  const int threads = 256;
  const int64_t blocks = (N + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  swiglu_forward_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      x.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(), N, H);
}

void swiglu_backward_launch(const at::Tensor& x, const at::Tensor& grad,
                            at::Tensor& dx, int64_t rows, int64_t H) {
  const int64_t N = rows * H;
  const int threads = 256;
  const int64_t blocks = (N + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  swiglu_backward_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      x.data_ptr<at::BFloat16>(), grad.data_ptr<at::BFloat16>(),
      dx.data_ptr<at::BFloat16>(), N, H);
}
