#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <string>
#include <cuda_runtime.h>

#include "lenses.h"
#include "arrayff.hxx"

const float WL  = 2.0f;
const float XL1 = -WL;
const float YL1 = -WL;

// GPU version of the shoot function
__device__ void d_shoot(float& xs, float& ys, float xl, float yl,
                        const float* __restrict__ xlens,
                        const float* __restrict__ ylens,
                        const float* __restrict__ eps, int nlenses)
{
    float dx, dy, dr;
    xs = xl;
    ys = yl;
    for (int p = 0; p < nlenses; ++p) {
        dx = xl - xlens[p];
        dy = yl - ylens[p];
        dr = dx * dx + dy * dy;
        xs -= eps[p] * dx / dr;
        ys -= eps[p] * dy / dr;
    }
}

// CUDA Kernel for Inverse Ray Tracing
__global__ void ray_trace_kernel(float* lensim, int npixx, int npixy, float lens_scale,
                                 float rsrc2, float xsrc, float ysrc, float ldc,
                                 const float* __restrict__ xlens,
                                 const float* __restrict__ ylens,
                                 const float* __restrict__ eps, int nlenses)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < npixx && iy < npixy) {
        float xl = XL1 + ix * lens_scale;
        float yl = YL1 + iy * lens_scale;

        float xs, ys;
        d_shoot(xs, ys, xl, yl, xlens, ylens, eps, nlenses);

        float xd = xs - xsrc;
        float yd = ys - ysrc;
        float sep2 = xd * xd + yd * yd;

        int idx = iy * npixx + ix;

        if (sep2 < rsrc2) {
            float mu = sqrtf(1.0f - sep2 / rsrc2);
            lensim[idx] = 1.0f - ldc * (1.0f - mu);
        } else {
            lensim[idx] = 0.0f;
        }
    }
}

int main(int argc, char* argv[])
{
    // Usage: ./lens_cuda [lens_scale] [nlenses_mode] [run_cpu (0 or 1)] [out_filename]
    float lens_scale = (argc > 1) ? std::atof(argv[1]) : 0.005f;
    int lens_mode    = (argc > 2) ? std::atoi(argv[2]) : 3;
    int run_cpu      = (argc > 3) ? std::atoi(argv[3]) : 1;
    std::string out_file = (argc > 4) ? argv[4] : "lens.fit";

    float* xlens_h;
    float* ylens_h;
    float* eps_h;

    srand(42); // Fixed seed for reproducible multi-lens configurations
    int nlenses = 0;
    if (lens_mode == 1) nlenses = set_example_1(&xlens_h, &ylens_h, &eps_h);
    else if (lens_mode == 2) nlenses = set_example_2(&xlens_h, &ylens_h, &eps_h);
    else if (lens_mode == 3) nlenses = set_example_3(&xlens_h, &ylens_h, &eps_h);
    else nlenses = set_example_n(lens_mode, &xlens_h, &ylens_h, &eps_h);

    const float rsrc  = 0.1f;
    const float ldc   = 0.5f;
    const float xsrc  = 0.0f;
    const float ysrc  = 0.0f;
    const float rsrc2 = rsrc * rsrc;

    const int npixx = static_cast<int>(round((WL * 2.0f) / lens_scale)) + 1;
    const int npixy = static_cast<int>(round((WL * 2.0f) / lens_scale)) + 1;

    Array<float, 2> lensim(npixy, npixx);

    // Allocate Device Memory
    float *xlens_d, *ylens_d, *eps_d, *lensim_d;
    size_t lens_bytes = nlenses * sizeof(float);
    size_t img_bytes  = static_cast<size_t>(npixx) * npixy * sizeof(float);

    cudaEvent_t total_start, total_stop, k_start, k_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_stop);

    cudaFree(0); // Warm up CUDA context
    cudaFree(0); // Warm up CUDA context
    cudaEventRecord(total_start);

    cudaMalloc(&xlens_d, lens_bytes);
    cudaMalloc(&ylens_d, lens_bytes);
    cudaMalloc(&eps_d, lens_bytes);
    cudaMalloc(&lensim_d, img_bytes);

    cudaMemcpy(xlens_d, xlens_h, lens_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ylens_d, ylens_h, lens_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(eps_d, eps_h, lens_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((npixx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (npixy + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(k_start);
    ray_trace_kernel<<<numBlocks, threadsPerBlock>>>(
        lensim_d, npixx, npixy, lens_scale, rsrc2, xsrc, ysrc, ldc, xlens_d, ylens_d, eps_d, nlenses
    );
    cudaEventRecord(k_stop);
    cudaEventSynchronize(k_stop);

    cudaMemcpy(lensim.buffer, lensim_d, img_bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);

    float kernel_ms = 0.0f, total_gpu_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, k_start, k_stop);
    cudaEventElapsedTime(&total_gpu_ms, total_start, total_stop);

    double cpu_ms = 0.0;
    if (run_cpu) {
        Array<float, 2> lensim_cpu(npixy, npixx);
        auto c_start = std::chrono::high_resolution_clock::now();
        for (int iy = 0; iy < npixy; ++iy) {
            for (int ix = 0; ix < npixx; ++ix) {
                float xl = XL1 + ix * lens_scale;
                float yl = YL1 + iy * lens_scale;
                float xs, ys;
                shoot(xs, ys, xl, yl, xlens_h, ylens_h, eps_h, nlenses);
                float xd = xs - xsrc;
                float yd = ys - ysrc;
                float sep2 = xd * xd + yd * yd;
                if (sep2 < rsrc2) {
                    float mu = std::sqrt(1.0f - sep2 / rsrc2);
                    lensim_cpu(iy, ix) = 1.0f - ldc * (1.0f - mu);
                } else {
                    lensim_cpu(iy, ix) = 0.0f;
                }
            }
        }
        auto c_stop = std::chrono::high_resolution_clock::now();
        cpu_ms = std::chrono::duration<double, std::milli>(c_stop - c_start).count();
    }

    std::cout << "Scale: " << lens_scale
              << " | Grid: " << npixx << "x" << npixy
              << " | Lenses: " << nlenses
              << " | GPU_Kernel_ms: " << kernel_ms
              << " | GPU_Total_ms: " << total_gpu_ms;
    if (run_cpu) {
        std::cout << " | CPU_ms: " << cpu_ms
                  << " | Speedup_Kernel: " << (cpu_ms / kernel_ms) << "x"
                  << " | Speedup_Total: " << (cpu_ms / total_gpu_ms) << "x";
    }
    std::cout << std::endl;

    if (out_file != "none") {
        dump_array<float, 2>(lensim, out_file);
    }

    cudaFree(xlens_d);
    cudaFree(ylens_d);
    cudaFree(eps_d);
    cudaFree(lensim_d);
    delete[] xlens_h;
    delete[] ylens_h;
    delete[] eps_h;

    return 0;
}
