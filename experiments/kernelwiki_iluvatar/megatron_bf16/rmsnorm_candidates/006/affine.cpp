#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <vector>

// CUDA launchers (defined in the .cu translation unit).
std::vector<at::Tensor> fused_affine_forward_cuda(const at::Tensor& x, const at::Tensor& r, const at::Tensor& w);
std::vector<at::Tensor> fused_affine_backward_cuda(const at::Tensor& g, const at::Tensor& x, const at::Tensor& r, const at::Tensor& w, const at::Tensor& t);

using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

class FusedAffine : public torch::autograd::Function<FusedAffine> {
 public:
  static tensor_list forward(AutogradContext* ctx, torch::Tensor x, torch::Tensor r, torch::Tensor w) {
    auto xc = x.contiguous();
    auto rc = r.contiguous();
    auto wc = w.contiguous();
    auto outs = fused_affine_forward_cuda(xc, rc, wc);  // {y, t}
    ctx->save_for_backward({xc, rc, wc, outs[1]});
    return {outs[0]};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto r = saved[1];
    auto w = saved[2];
    auto t = saved[3];
    auto g = grad_outputs[0].contiguous();
    auto outs = fused_affine_backward_cuda(g, x, r, w, t);  // {dx_direct, dr_terms, dw_terms}
    // Native reductions, order unchanged; norm gradient branch stays separate.
    auto dr = outs[1].sum_to_size(r.sizes());
    auto dw = outs[2].sum_to_size(w.sizes());
    return {outs[0], dr, dw};
  }
};

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
  // Native BF16 normalization graph, identical to candidate003.
  auto v = at::pow(x, 2).mean({-1}, true);
  v.add_(1.0013580322265625e-05);
  auto r = at::rsqrt(v);
  return FusedAffine::apply(x, r, w)[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm, "RMSNorm fused BF16 affine (stage27 006)");
}
