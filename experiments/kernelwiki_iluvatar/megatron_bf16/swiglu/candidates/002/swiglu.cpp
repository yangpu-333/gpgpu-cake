#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <cstdint>

void swiglu_forward_launch(const at::Tensor& x, at::Tensor& out,
                           int64_t rows, int64_t H);
void swiglu_backward_launch(const at::Tensor& x, const at::Tensor& grad,
                            at::Tensor& dx, int64_t rows, int64_t H);

static void check_x(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "swiglu: x must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "swiglu: x must be BF16");
  TORCH_CHECK(x.is_contiguous(), "swiglu: x must be contiguous");
  const auto d = x.dim();
  TORCH_CHECK(d == 2 || d == 3, "swiglu: x must be 2D or 3D");
  TORCH_CHECK(x.size(d - 1) % 2 == 0, "swiglu: last dim must be even");
}

at::Tensor forward(const at::Tensor& x) {
  check_x(x);
  const at::cuda::CUDAGuard guard(x.device());
  const auto d = x.dim();
  const int64_t last = x.size(d - 1);
  const int64_t H = last / 2;
  const int64_t rows = (last == 0) ? 0 : (x.numel() / last);
  std::vector<int64_t> sizes = x.sizes().vec();
  sizes[d - 1] = H;
  auto out = at::empty(sizes, x.options());
  if (rows * H > 0) {
    swiglu_forward_launch(x, out, rows, H);
  }
  return out;
}

at::Tensor backward(const at::Tensor& x, const at::Tensor& grad) {
  check_x(x);
  TORCH_CHECK(grad.is_cuda(), "swiglu: grad must be CUDA");
  TORCH_CHECK(grad.scalar_type() == at::kBFloat16, "swiglu: grad must be BF16");
  TORCH_CHECK(grad.is_contiguous(), "swiglu: grad must be contiguous");
  const at::cuda::CUDAGuard guard(x.device());
  const auto d = x.dim();
  const int64_t last = x.size(d - 1);
  const int64_t H = last / 2;
  const int64_t rows = (last == 0) ? 0 : (x.numel() / last);
  auto dx = at::empty_like(x);
  if (rows * H > 0) {
    swiglu_backward_launch(x, grad, dx, rows, H);
  }
  return dx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "SwiGLU forward (BF16 CUDA)");
  m.def("backward", &backward, "SwiGLU backward (BF16 CUDA)");
}
