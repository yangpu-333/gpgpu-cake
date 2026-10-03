#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <vector>
#include <cstdint>

namespace {

__global__ void fused_affine_forward_kernel(
    const c10::BFloat16* __restrict__ x,
    const c10::BFloat16* __restrict__ r,
    const c10::BFloat16* __restrict__ w,
    c10::BFloat16* __restrict__ y,
    c10::BFloat16* __restrict__ t,
    int64_t numel, int64_t N) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  int64_t row = i / N;
  int64_t col = i % N;
  float xv = static_cast<float>(x[i]);
  float rv = static_cast<float>(r[row]);
  float wv = static_cast<float>(w[col]);
  // First BF16 product, rounded and materialized before the second product.
  c10::BFloat16 tb = c10::BFloat16(xv * rv);
  float tf = static_cast<float>(tb);
  t[i] = tb;
  y[i] = c10::BFloat16(tf * wv);
}

__global__ void fused_affine_backward_kernel(
    const c10::BFloat16* __restrict__ g,
    const c10::BFloat16* __restrict__ x,
    const c10::BFloat16* __restrict__ r,
    const c10::BFloat16* __restrict__ w,
    const c10::BFloat16* __restrict__ t,
    c10::BFloat16* __restrict__ dx,
    c10::BFloat16* __restrict__ dr_terms,
    c10::BFloat16* __restrict__ dw_terms,
    int64_t numel, int64_t N) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  int64_t row = i / N;
  int64_t col = i % N;
  float gv = static_cast<float>(g[i]);
  float wv = static_cast<float>(w[col]);
  float rv = static_cast<float>(r[row]);
  float xv = static_cast<float>(x[i]);
  float tv = static_cast<float>(t[i]);
  // Native BF16 order; gw rounded once then reused.
  c10::BFloat16 gw_b = c10::BFloat16(gv * wv);
  float gwf = static_cast<float>(gw_b);
  dx[i] = c10::BFloat16(gwf * rv);
  dr_terms[i] = c10::BFloat16(gwf * xv);
  dw_terms[i] = c10::BFloat16(gv * tv);
}

}  // namespace

std::vector<at::Tensor> fused_affine_forward_cuda(const at::Tensor& x, const at::Tensor& r, const at::Tensor& w) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto y = at::empty_like(x);
  auto t = at::empty_like(x);
  int64_t numel = x.numel();
  int64_t N = x.size(-1);
  if (numel == 0) return {y, t};
  const int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_affine_forward_kernel<<<blocks, threads, 0, stream>>>(
      x.data_ptr<c10::BFloat16>(), r.data_ptr<c10::BFloat16>(), w.data_ptr<c10::BFloat16>(),
      y.data_ptr<c10::BFloat16>(), t.data_ptr<c10::BFloat16>(), numel, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, t};
}

std::vector<at::Tensor> fused_affine_backward_cuda(const at::Tensor& g, const at::Tensor& x, const at::Tensor& r, const at::Tensor& w, const at::Tensor& t) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto dx = at::empty_like(x);
  auto dr_terms = at::empty_like(x);
  auto dw_terms = at::empty_like(x);
  int64_t numel = x.numel();
  int64_t N = x.size(-1);
  if (numel == 0) return {dx, dr_terms, dw_terms};
  const int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_affine_backward_kernel<<<blocks, threads, 0, stream>>>(
      g.data_ptr<c10::BFloat16>(), x.data_ptr<c10::BFloat16>(), r.data_ptr<c10::BFloat16>(),
      w.data_ptr<c10::BFloat16>(), t.data_ptr<c10::BFloat16>(),
      dx.data_ptr<c10::BFloat16>(), dr_terms.data_ptr<c10::BFloat16>(), dw_terms.data_ptr<c10::BFloat16>(),
      numel, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dr_terms, dw_terms};
}
