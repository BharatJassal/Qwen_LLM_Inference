#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

const int THREADS = 256;
const int ROWS_PER_BLOCK = 8;
const int MAX_BATCH = 8;

// one warp per row, the row is read once for all B inputs
template <int B>
__global__ void int8_gemv(const int8_t* __restrict__ W,
                          const float* __restrict__ scale,
                          const float* __restrict__ x,
                          float* __restrict__ y,
                          int rows, int cols) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp;
    if (row >= rows) return;
    const char4* w4 = reinterpret_cast<const char4*>(W + (size_t)row * cols);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n = cols / 4;

    float sum[B] = {};
    for (int j = lane; j < n; j += 32) {
        char4 wv = w4[j];
        #pragma unroll
        for (int b = 0; b < B; b++) {
            float4 xv = x4[b * n + j];
            sum[b] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    }
    #pragma unroll
    for (int b = 0; b < B; b++) {
        for (int offset = 16; offset > 0; offset /= 2) {
            sum[b] += __shfl_down_sync(0xffffffff, sum[b], offset);
        }
        if (lane == 0) {
            y[(size_t)b * rows + row] = scale[row] * sum[b];
        }
    }
}

using Kernel = void (*)(const int8_t*, const float*, const float*, float*, int, int);
const Kernel KERNELS[MAX_BATCH] = {int8_gemv<1>, int8_gemv<2>, int8_gemv<3>, int8_gemv<4>,
                                   int8_gemv<5>, int8_gemv<6>, int8_gemv<7>, int8_gemv<8>};

torch::Tensor int8_gemv_forward(torch::Tensor W, torch::Tensor scale, torch::Tensor x) {
    TORCH_CHECK(W.is_cuda() && scale.is_cuda() && x.is_cuda(), "all tensors must be on the GPU");
    TORCH_CHECK(W.dtype() == torch::kInt8, "W must be int8");
    TORCH_CHECK(scale.dtype() == torch::kFloat32 && x.dtype() == torch::kFloat32,
                "scale and x must be float32");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (rows x cols)");
    TORCH_CHECK(x.dim() == 1 || x.dim() == 2, "x must be (cols) or (batch x cols)");
    TORCH_CHECK(W.is_contiguous() && scale.is_contiguous() && x.is_contiguous(),
                "all tensors must be contiguous");

    int rows = W.size(0);
    int cols = W.size(1);
    int batch = x.dim() == 1 ? 1 : x.size(0);
    TORCH_CHECK(cols % 4 == 0, "cols must be a multiple of 4");
    TORCH_CHECK(scale.numel() == rows, "scale must have one value per row");
    TORCH_CHECK(x.size(-1) == cols, "x must have cols elements per row");
    TORCH_CHECK(batch >= 1 && batch <= MAX_BATCH, "batch must be between 1 and ", MAX_BATCH);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");

    auto y = x.dim() == 1 ? torch::empty({rows}, x.options()) : torch::empty({batch, rows}, x.options());
    int blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    KERNELS[batch - 1]<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        W.data_ptr<int8_t>(), scale.data_ptr<float>(),
        x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemv", &int8_gemv_forward, "Fused INT8 GEMV for 1 to 8 input vectors (CUDA)");
}
