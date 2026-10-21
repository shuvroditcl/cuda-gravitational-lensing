#!/bin/bash
#SBATCH --job-name=ass4_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

# Compile the CUDA code alongside the C++ helper files
nvcc -O3 -arch=sm_75 -arch=sm_75 lens_cuda.cu lenses.cpp fitsfile.cpp -o lens_cuda -lcfitsio

echo "=== PART 1: VARYING IMAGE RESOLUTION (3 LENSES) ==="
./lens_cuda 0.005   3 1 lens_3_801.fit
./lens_cuda 0.0025  3 1 none
./lens_cuda 0.001   3 1 lens.fit
./lens_cuda 0.0005  3 1 none

echo "=== PART 2: VARYING LENS COUNT (801x801 IMAGE) ==="
./lens_cuda 0.005   1 1 lens_1.fit
./lens_cuda 0.005   2 1 lens_2.fit
./lens_cuda 0.005   3 1 none
./lens_cuda 0.005  10 1 lens_10.fit
./lens_cuda 0.005  50 1 none
./lens_cuda 0.005 100 1 none
./lens_cuda 0.005 500 1 none

echo "=== PART 3: HIGH-RES MULTI-LENS (4001x4001 IMAGE, GPU VS CPU) ==="
./lens_cuda 0.001   1 1 none
./lens_cuda 0.001  10 1 none
./lens_cuda 0.001  50 1 none
./lens_cuda 0.001 100 1 none
