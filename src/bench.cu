#include "bf16_gemm.cuh"
#include <cublas_v2.h>
#define CHECK_CUBLAS(call)                                                                   \
    {                                                                                        \
        cublasStatus_t err = call;                                                           \
        if (err != CUBLAS_STATUS_SUCCESS)                                                    \
        {                                                                                    \
            fprintf(stderr, "CUBLAS error in %s at line %d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    }
template <typename T>
struct CUDABuffer
{
    T *data;
    size_t size;
    CUDABuffer(size_t size) : size(size)
    {
        CHECK_CUDA(cudaMalloc(&data, size * sizeof(T)));
    }
    ~CUDABuffer()
    {
        cudaFree(data);
    }
    void copy_from_host(const T *host_data, cudaStream_t stream = nullptr)
    {
        CHECK_CUDA(cudaMemcpyAsync(data, host_data, size * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    void copy_to_host(T *host_data, cudaStream_t stream = nullptr)
    {
        CHECK_CUDA(cudaMemcpyAsync(host_data, data, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }
};
void rand_bf16(bf16 *data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        float val = (static_cast<float>(rand()) / RAND_MAX);
        data[i] = __float2bfloat16(val);
    }
}
void cublas_gemm(cublasHandle_t handle, uint32_t M, uint32_t N, uint32_t K, const bf16 *X, const bf16 *W, bf16 *Y)
{
    // X: MxK row-major, W: KxN col-major, Y: MxN row-major
    // cuBLAS is column-major, so we compute Y^T = W^T * X^T
    // Y^T (NxM col-major) = W^T (NxK) * X^T (KxM)
    // X row-major MxK = X^T col-major KxM
    // W col-major KxN = W as-is
    // We need W^T * X^T = (NxK) * (KxM) = NxM
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha_f,
        W, CUDA_R_16BF, K, // W is KxN col-major; W^T is NxK
        X, CUDA_R_16BF, K, // X row-major MxK = X^T col-major KxM
        &beta_f,
        Y, CUDA_R_16BF, N, // Y row-major MxN = Y^T col-major NxM
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}
float compare_result(const bf16 *Y, const bf16 *Y_ref, size_t size)
{
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        float y = __bfloat162float(Y[i]);
        float y_ref = __bfloat162float(Y_ref[i]);
        float abs_error = fabsf(y - y_ref);
        if (abs_error > max_abs_error)
        {
            max_abs_error = abs_error;
        }
    }
    return max_abs_error;
}
int main()
{
    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    constexpr int BM = 128, BN = 128, BK = 64, NUM_STAGES = 2, CWG = 1;
    // int M = 4096, N = 1024, K = 512;
    std::vector<std::tuple<int, int, int>> test_cases = {
        {128, 128, 128},
        {1024, 128, 128},
        {1024, 256, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
    };
    for (auto &&[M, N, K] : test_cases)
    {
        printf("\n=== Testing M=%d N=%d K=%d ===\n", M, N, K);
        // Allocate host data
        std::vector<bf16> h_X(M * K), h_W(K * N), h_Y(M * N), h_Y_ref(M * N);
        srand(42);
        rand_bf16(h_X.data(), M * K);
        rand_bf16(h_W.data(), K * N);

        // Allocate device data
        CUDABuffer<bf16> d_X(M * K), d_W(K * N), d_Y(M * N), d_Y_ref(M * N);
        d_X.copy_from_host(h_X.data());
        d_W.copy_from_host(h_W.data());

        // cuBLAS reference
        CHECK_CUDA(cudaMemsetAsync(d_Y_ref.data, 0, M * N * sizeof(bf16), stream));
        cublas_gemm(handle, M, N, K, d_X.data, d_W.data, d_Y_ref.data);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Our SIMT kernel
        CHECK_CUDA(cudaMemsetAsync(d_Y.data, 0, M * N * sizeof(bf16), stream));
        BF16GemmSIMT<BM, BN, BK, NUM_STAGES, CWG>::run(M, N, K, d_X.data, d_W.data, d_Y.data, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Compare SIMT
        d_Y.copy_to_host(h_Y.data(), stream);
        d_Y_ref.copy_to_host(h_Y_ref.data(), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        float max_err = compare_result(h_Y.data(), h_Y_ref.data(), M * N);
        printf("[SIMT] Max abs error vs cuBLAS: %f\n", max_err);
        float tol = K * 0.01f;  // scale tolerance with K (accumulation order differences)
        if (max_err < tol)
        {
            printf("[SIMT] PASSED\n");
        }
        else
        {
            printf("[SIMT] FAILED\n");
        }

        // Our MMA kernel
        CHECK_CUDA(cudaMemsetAsync(d_Y.data, 0, M * N * sizeof(bf16), stream));
        BF16GemmMMA<BM, BN, BK, NUM_STAGES, CWG>::run(M, N, K, d_X.data, d_W.data, d_Y.data, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Compare MMA
        d_Y.copy_to_host(h_Y.data(), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        max_err = compare_result(h_Y.data(), h_Y_ref.data(), M * N);
        printf("[MMA]  Max abs error vs cuBLAS: %f\n", max_err);
        if (max_err < tol)
        {
            printf("[MMA]  PASSED\n");
        }
        else
        {
            printf("[MMA]  FAILED\n");
            int count = 0;
            for (int i = 0; i < M * N && count < 10; i++)
            {
                float y = __bfloat162float(h_Y[i]);
                float y_ref = __bfloat162float(h_Y_ref[i]);
                if (fabsf(y - y_ref) > 0.5f)
                {
                    printf("  [%d] ours=%f ref=%f\n", i, y, y_ref);
                    count++;
                }
            }
            return 1;
        }

        // ── Benchmark ──────────────────────────────────────────────────
        double flops = 2.0 * M * N * K;

        double cublas_ms = bench_ms([&]()
                                    { cublas_gemm(handle, M, N, K, d_X.data, d_W.data, d_Y_ref.data); }, stream);

        double simt_ms = bench_ms([&]()
                                  { BF16GemmSIMT<BM, BN, BK, NUM_STAGES, CWG>::run(M, N, K, d_X.data, d_W.data, d_Y.data, stream); }, stream);

        double mma_ms = bench_ms([&]()
                                 { BF16GemmMMA<BM, BN, BK, NUM_STAGES, CWG>::run(M, N, K, d_X.data, d_W.data, d_Y.data, stream); }, stream);

        double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;
        double simt_tflops = flops / (simt_ms * 1e-3) / 1e12;
        double mma_tflops = flops / (mma_ms * 1e-3) / 1e12;

        printf("\n── Benchmark (M=%d N=%d K=%d) ──\n", M, N, K);
        printf("[cuBLAS] %.4f ms  %.4f TFLOPS\n", cublas_ms, cublas_tflops);
        printf("[SIMT]   %.4f ms  %.4f TFLOPS  (%.1f%% of cuBLAS)\n", simt_ms, simt_tflops, 100.0 * simt_tflops / cublas_tflops);
        printf("[MMA]    %.4f ms  %.4f TFLOPS  (%.1f%% of cuBLAS)\n", mma_ms, mma_tflops, 100.0 * mma_tflops / cublas_tflops);
    }
    return 0;
}