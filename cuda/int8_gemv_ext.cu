#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

const int THREADS = 256;
const int ROWS_PER_BLOCK = 8;

__global__ void int8_gemv(const int8_t* __restrict__ W,
                          const float* __restrict__ scale,
                          const float* __restrict__ x,
                          float* __restrict__ y,
                          int rows, int cols) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp;
    if (row>=rows) return;
    const int8_t* w = W + (size_t)row * cols;
    const char4* w4 = reinterpret_cast<const char4*>(w);
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float sum = 0.0f;
    for (int j = lane; j < cols / 4; j += 32) {
        char4 wv = w4[j];
        float4 xv = x4[j];
        sum += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        y[row] = scale[row] * sum;
    }

}

torch::Tensor int8_gemv_forward(torch::Tensor W, torch::Tensor scale, torch::Tensor x) {
    TORCH_CHECK(W.is_cuda() && scale.is_cuda() && x.is_cuda(), "all tensors must be on the GPU");
    TORCH_CHECK(W.dtype() == torch::kInt8, "W must be int8");
    TORCH_CHECK(scale.dtype() == torch::kFloat32 && x.dtype() == torch::kFloat32,
                "scale and x must be float32");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (rows x cols)");
    TORCH_CHECK(W.is_contiguous() && scale.is_contiguous() && x.is_contiguous(),
                "all tensors must be contiguous");

    int rows = W.size(0);
    int cols = W.size(1);
    TORCH_CHECK(cols % 4 == 0, "cols must be a multiple of 4");
    TORCH_CHECK(scale.numel() == rows, "scale must have one value per row");
    TORCH_CHECK(x.numel() == cols, "x must have cols elements");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");

    auto y = torch::empty({rows}, x.options());
    int blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    int8_gemv<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        W.data_ptr<int8_t>(), scale.data_ptr<float>(),
        x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemv", &int8_gemv_forward, "Fused INT8 GEMV (CUDA)");
}