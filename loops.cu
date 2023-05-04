#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

extern "C" {
    void launch_cuda_loop(double *data, int n);
    void launch_cuda_dgemm(int m, int n, int k, double *A, double *B, double *C);
}

/**
 * @brief Increments each element of the input data array by 1.0.
 *
 * This kernel runs on the GPU and is executed by many threads in parallel.
 * Each thread calculates its unique index within the grid of threads and
 * increments the corresponding element of the data array by 1.0.
 *
 * @param data Pointer to the input data array (on the device).
 * @param n Size of the input data array.
 */
__global__ void sum_loop_kernel(double *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0;
    }
}

/**
 * @brief Launches the sum_loop_kernel CUDA kernel with a specified block size and grid size.
 *
 * This function allocates device memory for the input data array, copies the data
 * from the host to the device, launches the my_kernel CUDA kernel, copies the
 * results back to the host, and frees the device memory.
 *
 * @param data Pointer to the input data array (on the host).
 * @param n Size of the input data array.
 */
void launch_cuda_loop(double *data, int n) {
    double *d_data; // Copy of the array on the device
    cudaMalloc((void **)&d_data, n * sizeof(double));
    cudaMemcpy(d_data, data, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256; // Should be a multiple of the warp size = 32, never less than 4 * 32 = 128
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launches the sum_loop_kernel with the specified grid size and block size
    sum_loop_kernel<<<gridSize, blockSize>>>(d_data, n);

    cudaMemcpy(data, d_data, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

/**
 * @brief Performs matrix multiplication using cuBLAS DGEMM function on the GPU.
 *
 * This function multiplies two matrices A and B, and stores the result in matrix C.
 * The matrix dimensions are specified by the parameters m, n, and k.
 * It uses the cuBLAS library to perform the DGEMM operation on the GPU.
 * Device memory is allocated for the input and output matrices, and data is copied
 * between the host and device as needed.
 *
 * @param m Number of rows in matrix A and matrix C.
 * @param n Number of columns in matrix B and matrix C.
 * @param k Number of columns in matrix A and number of rows in matrix B.
 * @param A Pointer to the input matrix A (on the host).
 * @param B Pointer to the input matrix B (on the host).
 * @param C Pointer to the output matrix C (on the host).
 */
void launch_cuda_dgemm(int m, int n, int k, double *A, double *B, double *C) {
    cublasHandle_t handle; // Created only once for all cuBLAS operations
    cublasCreate(&handle);

    double alpha = 1.0;
    double beta = 0.0;

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double), cudaMemcpyHostToDevice);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}
