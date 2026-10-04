// swiglu-003.cpp
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

at::Tensor swiglu_forward_cuda(const at::Tensor& x);
at::Tensor swiglu_backward_cuda(const at::Tensor& x, const at::Tensor& grad);

using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

static void check_input(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "swiglu requires a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "swiglu requires a BF16 tensor");
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "swiglu requires a 2D or 3D tensor");
  TORCH_CHECK(x.size(-1) % 2 == 0, "swiglu requires an even last dimension");
  TORCH_CHECK(x.is_contiguous(), "swiglu requires a contiguous tensor");
}

struct Swiglu : public torch::autograd::Function<Swiglu> {
  static at::Tensor forward(AutogradContext* ctx, at::Tensor x) {
    check_input(x);
    ctx->save_for_backward({x});
    return swiglu_forward_cuda(x);
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto grad = grad_outputs[0].contiguous();
    auto dx = swiglu_backward_cuda(x, grad);
    return {dx};
  }
};

at::Tensor swiglu(const at::Tensor& x) {
  return Swiglu::apply(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu", &swiglu, "SwiGLU forward+backward (BI-V150 candidate003)");
}
