// swiglu-003.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void swiglu_forward_kernel(
    const at::BFloat16* __restrict__ x,
    at::BFloat16* __restrict__ out,
    long n, long H) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long row = i / H;
  long col = i - row * H;
  long gateidx = row * (2 * H) + col;
  long upidx = gateidx + H;
  float a = static_cast<float>(x[gateidx]);
  float b = static_cast<float>(x[upidx]);
  float sig = 1.0f / (1.0f + expf(-a));
  float silu = a * sig;
  out[i] = at::BFloat16(silu * b);
}

__global__ void swiglu_backward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ grad,
    at::BFloat16* __restrict__ dx,
    long n, long H) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long row = i / H;
  long col = i - row * H;
  long gateidx = row * (2 * H) + col;
  long upidx = gateidx + H;
  float a = static_cast<float>(x[gateidx]);
  float b = static_cast<float>(x[upidx]);
  float g = static_cast<float>(grad[i]);
  float sig = 1.0f / (1.0f + expf(-a));
  float dgate = g * sig * (1.0f + a * (1.0f - sig)) * b;
  float dup = g * (a * sig);
  dx[gateidx] = at::BFloat16(dgate);
  dx[upidx] = at::BFloat16(dup);
}

at::Tensor swiglu_forward_cuda(const at::Tensor& x) {
  const at::cuda::CUDAGuard guard(x.device());
  auto sizes = x.sizes().vec();
  long H = sizes.back() / 2;
  sizes.back() = H;
  auto out = at::empty(sizes, x.options());
  long n = out.numel();
  if (n == 0) return out;
  const int threads = 256;
  const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  swiglu_forward_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(), n, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor swiglu_backward_cuda(const at::Tensor& x, const at::Tensor& grad) {
  const at::cuda::CUDAGuard guard(x.device());
  auto dx = at::empty_like(x);
  long H = x.size(-1) / 2;
  long n = grad.numel();
  if (n == 0) return dx;
  const int threads = 256;
  const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  swiglu_backward_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<at::BFloat16>(), grad.data_ptr<at::BFloat16>(),
      dx.data_ptr<at::BFloat16>(), n, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dx;
}
