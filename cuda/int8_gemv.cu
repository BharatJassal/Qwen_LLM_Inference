// $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -version "[17.0,18.0)" -property installationPath
// & "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
// Fused INT8 GEMV: y[i] = scale[i] * sum_j W[i][j] * x[j]
//
// W is int8 (rows x cols, row-major), scale is one float per row (from
// src/quantize.py), x and y are float. This file is a standalone test harness:
// it checks the kernel against a CPU reference and measures achieved memory
// bandwidth on the three projection shapes used by Qwen2.5-0.5B.
//
// Build (in a terminal with the VS 2022 compiler loaded):
//   nvcc -O3 -arch=sm_61 -o int8_gemv.exe int8_gemv.cu
// Run:
//   .\int8_gemv.exe

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_),       \
                   __FILE__, __LINE__);                                        \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

const float PEAK_GBPS = 484.0f;  // Tailored to personal Nvidia GTX 1080 Ti memory bandwidth
const int THREADS = 256;         
const int ROWS_PER_BLOCK = 8;    
const int ITERS = 200;           

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

// Timing it with the same launch shape gives the floor, the cost
// of launching and scheduling the blocks, before any memory is read.
__global__ void empty_kernel() {}

// Time per launch in microseconds: TRIALS batches of ITERS back to back launches, reporting the median batch so one noisy batch can't skew it.
const int TRIALS = 9;
template <typename Launch>
double time_us(Launch launch) {
    for (int it = 0; it < 50; it++) launch();  // let GPU clocks ramp
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    std::vector<double> trials;
    for (int t = 0; t < TRIALS; t++) {
        CHECK(cudaEventRecord(start));
        for (int it = 0; it < ITERS; it++) launch();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        trials.push_back(1000.0 * ms / ITERS);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    std::sort(trials.begin(), trials.end());
    return trials[TRIALS / 2];
}


void gemv_cpu(const std::vector<int8_t>& W, const std::vector<float>& scale,
              const std::vector<float>& x, std::vector<double>& y,
              std::vector<double>& mag, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double acc = 0.0, abs_acc = 0.0;
        for (int j = 0; j < cols; j++) {
            double t = (double)W[(size_t)i * cols + j] * (double)x[j];
            acc += t;
            abs_acc += fabs(t);
        }
        y[i] = acc * scale[i];
        mag[i] = abs_acc * scale[i];
    }
}

bool run_shape(cublasHandle_t handle, const char* name, int rows, int cols) {
    // test data
    std::vector<int8_t> hW((size_t)rows * cols);
    std::vector<float> hScale(rows), hX(cols), hY(rows);
    std::vector<double> hRef(rows), hMag(rows);
    for (auto& w : hW) w = (int8_t)(rand() % 255 - 127);
    for (auto& s : hScale) s = 0.001f + 0.01f * (rand() / (float)RAND_MAX);
    for (auto& v : hX) v = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

    int8_t* dW;
    float *dScale, *dX, *dY;
    CHECK(cudaMalloc(&dW, hW.size()));
    CHECK(cudaMalloc(&dScale, rows * sizeof(float)));
    CHECK(cudaMalloc(&dX, cols * sizeof(float)));
    CHECK(cudaMalloc(&dY, rows * sizeof(float)));
    CHECK(cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dScale, hScale.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dX, hX.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dY, 0, rows * sizeof(float)));

    int blocks = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    // Correctness: one launch, compare against the CPU.
    int8_gemv<<<blocks, THREADS>>>(dW, dScale, dX, dY, rows, cols);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hY.data(), dY, rows * sizeof(float), cudaMemcpyDeviceToHost));
    gemv_cpu(hW, hScale, hX, hRef, hMag, rows, cols);

    double max_err = 0.0;
    for (int i = 0; i < rows; i++) {
        double err = fabs((double)hY[i] - hRef[i]) / (hMag[i] + 1e-30);
        if (std::isnan(err) || err > max_err) max_err = err;
    }
    bool pass = max_err < 1e-4;

    // Timing.
    double us = time_us([&] { int8_gemv<<<blocks, THREADS>>>(dW, dScale, dX, dY, rows, cols); });
    double floor_us = time_us([&] { empty_kernel<<<blocks, THREADS>>>(); });

    // Bytes that the kernel moves at a minimum, weights + scales + x + y.
    double bytes = (double)rows * cols + rows * 4.0 + cols * 4.0 + rows * 4.0;
    double ideal_us = bytes / (PEAK_GBPS * 1e9) * 1e6;  // time at peak bandwidth
    double gbps = bytes / (us * 1e-6) / 1e9;

    printf("%-8s %4d x %-4d  %s  err %.1e  kernel %6.2f us  floor %5.2f us  ideal %5.2f us"
           "  %5.1f GB/s  %4.1f%% of peak\n",
           name, rows, cols, pass ? "PASS" : "FAIL", max_err, us, floor_us, ideal_us,
           gbps, 100.0 * gbps / PEAK_GBPS);
    if (!pass) printf("         (GB/s is meaningless until the shape passes)\n");
    float* dWf;
    CHECK(cudaMalloc(&dWf, (size_t)rows * cols * sizeof(float)));
    CHECK(cudaMemset(dWf, 0, (size_t)rows * cols * sizeof(float)));
    const float alpha = 1.0f, beta = 0.0f;
    double cublas_us = time_us([&] {
        cublasSgemv(handle, CUBLAS_OP_T, cols, rows, &alpha, dWf, cols, dX, 1, &beta, dY, 1);
    });
    double cublas_bytes = (double)rows * cols * 4.0 + cols * 4.0 + rows * 4.0;
    double cublas_gbps = cublas_bytes / (cublas_us * 1e-6) / 1e9;
    printf("         cuBLAS FP32 %6.2f us  %5.1f GB/s  %4.1f%% of peak  ->  INT8 kernel is %.2fx faster\n",
           cublas_us, cublas_gbps, 100.0 * cublas_gbps / PEAK_GBPS, cublas_us / us);
    CHECK(cudaFree(dWf));
    CHECK(cudaFree(dW));
    CHECK(cudaFree(dScale));
    CHECK(cudaFree(dX));
    CHECK(cudaFree(dY));
    return pass;
}

int main() {
    srand(0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    bool ok = true;
    ok &= run_shape(handle, "gate/up", 4864, 896);
    ok &= run_shape(handle, "down", 896, 4864);
    ok &= run_shape(handle, "q/o", 896, 896);
    cublasDestroy(handle);
    printf(ok ? "\nAll shapes PASS\n" : "\nSome shapes FAIL\n");
    return ok ? 0 : 1;
}
