#include <torch/extension.h>

torch::Tensor int8_gemv_forward(torch::Tensor W, torch::Tensor scale, torch::Tensor x,
                                c10::optional<torch::Tensor> bias);
std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps,
                                           c10::optional<torch::Tensor> residual);
void rope_forward(torch::Tensor x, torch::Tensor cos_table, torch::Tensor sin_table, torch::Tensor pos,
                  int64_t heads, int64_t head_dim);
torch::Tensor silu_mul_forward(torch::Tensor gate_up);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemv", &int8_gemv_forward, "INT8 GEMV for 1 to 8 input vectors, optional bias",
          py::arg("W"), py::arg("scale"), py::arg("x"), py::arg("bias") = py::none());
    m.def("rmsnorm", &rmsnorm_forward, "RMSNorm, optionally adding a residual first",
          py::arg("x"), py::arg("weight"), py::arg("eps"), py::arg("residual") = py::none());
    m.def("rope_", &rope_forward, "In-place RoPE on the first heads of each token");
    m.def("silu_mul", &silu_mul_forward, "silu(gate) * up from one gate_up tensor");
}
