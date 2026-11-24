#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

const int NORM_THREADS = 256;
const int THREADS = 256;

static void check_gpu_float(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kFloat32 && t.is_contiguous(),
                name, " must be a contiguous float32 GPU tensor");
}

// one block per row. With a residual, also writes the sum so the next layer can use it
template <bool ADD>
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ residual,
                               const float* __restrict__ weight, float* __restrict__ sum_out,
                               float* __restrict__ out, int n, float eps) {
    size_t base = (size_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[base + i];
        if (ADD) {
            v += residual[base + i];
            sum_out[base + i] = v;
        }
        ss += v * v;
    }

    __shared__ float warp_sums[NORM_THREADS / 32];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) ss += __shfl_down_sync(0xffffffff, ss, offset);
    if (lane == 0) warp_sums[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = lane < NORM_THREADS / 32 ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) ss += __shfl_down_sync(0xffffffff, ss, offset);
        if (lane == 0) warp_sums[0] = ss;
    }
    __syncthreads();

    float rms = sqrtf(warp_sums[0] / n + eps);
    const float* src = ADD ? sum_out : x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[base + i] = src[base + i] / rms * weight[i];
    }
}

std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps,
                                           c10::optional<torch::Tensor> residual) {
    check_gpu_float(x, "x");
    check_gpu_float(weight, "weight");
    int n = x.size(-1);
    int rows = x.numel() / n;
    TORCH_CHECK(weight.numel() == n, "weight must match the last dimension of x");
    auto out = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (residual) {
        check_gpu_float(*residual, "residual");
        TORCH_CHECK(residual->sizes() == x.sizes(), "residual must have the same shape as x");
        auto sum = torch::empty_like(x);
        rmsnorm_kernel<true><<<rows, NORM_THREADS, 0, stream>>>(
            x.data_ptr<float>(), residual->data_ptr<float>(), weight.data_ptr<float>(),
            sum.data_ptr<float>(), out.data_ptr<float>(), n, (float)eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {sum, out};
    }
    rmsnorm_kernel<false><<<rows, NORM_THREADS, 0, stream>>>(
        x.data_ptr<float>(), nullptr, weight.data_ptr<float>(), nullptr, out.data_ptr<float>(), n, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out};
}

// rotates the first `heads` heads of each token in place, matching x * cos + rotate_half(x) * sin.
// _rn intrinsics stop the compiler from fusing into FMAs so results match PyTorch exactly
__global__ void rope_kernel(float* __restrict__ x, const float* __restrict__ cos_table,
                            const float* __restrict__ sin_table, const int64_t* __restrict__ pos,
                            int tokens, int heads, int row_stride, int head_dim) {
    int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tokens * heads * half) return;
    int j = idx % half;
    int h = (idx / half) % heads;
    int t = idx / (half * heads);
    float* v = x + (size_t)t * row_stride + h * head_dim;
    const float* c = cos_table + pos[t] * head_dim;
    const float* s = sin_table + pos[t] * head_dim;
    float a = v[j], b = v[j + half];
    v[j] = __fadd_rn(__fmul_rn(a, c[j]), __fmul_rn(-b, s[j]));
    v[j + half] = __fadd_rn(__fmul_rn(b, c[j + half]), __fmul_rn(a, s[j + half]));
}

void rope_forward(torch::Tensor x, torch::Tensor cos_table, torch::Tensor sin_table, torch::Tensor pos,
                  int64_t heads, int64_t head_dim) {
    check_gpu_float(x, "x");
    check_gpu_float(cos_table, "cos_table");
    check_gpu_float(sin_table, "sin_table");
    TORCH_CHECK(pos.is_cuda() && pos.dtype() == torch::kInt64 && pos.is_contiguous(), "pos must be int64 on the GPU");
    TORCH_CHECK(cos_table.size(-1) == head_dim && sin_table.sizes() == cos_table.sizes(), "tables must be (len, head_dim)");
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    int row_stride = x.size(-1);
    int tokens = x.numel() / row_stride;
    TORCH_CHECK(pos.numel() == tokens, "pos must have one position per token");
    TORCH_CHECK(heads * head_dim <= row_stride, "x has fewer than heads * head_dim values per token");
    int total = tokens * heads * (head_dim / 2);
    rope_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), cos_table.data_ptr<float>(), sin_table.data_ptr<float>(), pos.data_ptr<int64_t>(),
        tokens, heads, row_stride, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// gate_up holds gate in the first half of each row and up in the second
__global__ void silu_mul_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int rows, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * n) return;
    int r = idx / n, i = idx % n;
    float g = gate_up[(size_t)r * 2 * n + i];
    float u = gate_up[(size_t)r * 2 * n + n + i];
    out[idx] = g / (1.0f + expf(-g)) * u;
}

torch::Tensor silu_mul_forward(torch::Tensor gate_up) {
    check_gpu_float(gate_up, "gate_up");
    TORCH_CHECK(gate_up.size(-1) % 2 == 0, "last dimension must hold gate and up halves");
    int n = gate_up.size(-1) / 2;
    int rows = gate_up.numel() / (2 * n);
    auto sizes = gate_up.sizes().vec();
    sizes.back() = n;
    auto out = torch::empty(sizes, gate_up.options());
    silu_mul_kernel<<<(rows * n + THREADS - 1) / THREADS, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        gate_up.data_ptr<float>(), out.data_ptr<float>(), rows, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
