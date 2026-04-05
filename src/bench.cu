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
        float val = (static_cast<float>(rand()) / RAND_MAX) * 2.0 - 1.0;
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
// Naive fp32 reference GEMM: Y = X * W
// X: row-major M×K (bf16), W: col-major K×N (bf16), Y: row-major M×N (f32)
__global__ void naive_gemm_f32_kernel(int M, int N, int K,
                                      const bf16 *__restrict__ X,
                                      const bf16 *__restrict__ W,
                                      float *__restrict__ Y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
    {
        float x = __bfloat162float(X[row * K + k]);
        float w = __bfloat162float(W[k + col * K]); // col-major
        acc += x * w;
    }
    Y[row * N + col] = acc;
}

void naive_gemm_f32(int M, int N, int K,
                    const bf16 *X, const bf16 *W, float *Y,
                    cudaStream_t stream = nullptr)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_gemm_f32_kernel<<<grid, block, 0, stream>>>(M, N, K, X, W, Y);
    CHECK_CUDA(cudaGetLastError());
}

std::pair<float, float> compare_result_f32(const bf16 *Y, const float *Y_ref, size_t size)
{
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        float y = __bfloat162float(Y[i]);
        float y_ref = Y_ref[i];
        float abs_error = fabsf(y - y_ref);
        float denom = fmaxf(fmaxf(fabsf(y), fabsf(y_ref)), 1.0f);
        float rel_error = abs_error / denom;
        max_abs_error = std::max<float>(max_abs_error, abs_error);
        max_rel_error = std::max<float>(max_rel_error, rel_error);
    }
    return {max_abs_error, max_rel_error};
}

std::pair<float, float> compare_result(const bf16 *Y, const bf16 *Y_ref, size_t size)
{
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        float y = __bfloat162float(Y[i]);
        float y_ref = __bfloat162float(Y_ref[i]);
        float abs_error = fabsf(y - y_ref);
        float denom = fmaxf(fmaxf(fabsf(y), fabsf(y_ref)), 1.0f);
        float rel_error = abs_error / denom;
        max_abs_error = std::max<float>(max_abs_error, abs_error);
        max_rel_error = std::max<float>(max_rel_error, rel_error);
    }
    return {max_abs_error, max_rel_error};
}

bool check_result(const char *label, const bf16 *Y, const bf16 *Y_ref, const float *Y_f32_ref, int size, float tol)
{
    auto [abs_err, rel_err] = compare_result(Y, Y_ref, size);
    auto [abs_err_f32, rel_err_f32] = compare_result_f32(Y, Y_f32_ref, size);
    auto [cublas_abs, cublas_rel] = compare_result_f32(Y_ref, Y_f32_ref, size);
    printf("[%s] vs cuBLAS:    abs=%f rel=%f\n", label, abs_err, rel_err);
    printf("[%s] vs fp32 ref:  abs=%f rel=%f\n", label, abs_err_f32, rel_err_f32);
    printf("[%s] cuBLAS vs ref: abs=%f rel=%f\n", label, cublas_abs, cublas_rel);
    if (rel_err_f32 < tol)
    {
        printf("[%s] PASSED\n", label);
        return true;
    }
    printf("[%s] FAILED\n", label);
    int count = 0;
    for (int i = 0; i < size && count < 10; i++)
    {
        float y = __bfloat162float(Y[i]);
        float y_ref = Y_f32_ref[i];
        if (fabsf(y - y_ref) / fmaxf(fmaxf(fabsf(y), fabsf(y_ref)), 1.0f) > tol)
        {
            printf("  [%d] ours=%f ref=%f\n", i, y, y_ref);
            count++;
        }
    }
    return false;
}

int main()
{
    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    constexpr int BM = 128, BN = 128, BK = 64, NUM_STAGES = 2, CWG = 1;
    constexpr int MMA_CWG = 2, MMA_WARP_M = 32, MMA_WARP_N = 64;
    constexpr int SPLIT_K = 2;
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
        {8192, 8192, 8192},
        // llama3 8b shapes
        {4096, 14336 * 2, 4096}, // upgate
        {4096, 4096, 14336},     // downproj
        {128, 14336 * 2, 4096},  // upgate
        {128, 4096, 14336}       // downproj
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

        d_Y_ref.copy_to_host(h_Y_ref.data(), stream);

        // Naive fp32 reference
        CUDABuffer<float> d_Y_f32(M * N);
        naive_gemm_f32(M, N, K, d_X.data, d_W.data, d_Y_f32.data, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> h_Y_f32(M * N);
        CHECK_CUDA(cudaMemcpyAsync(h_Y_f32.data(), d_Y_f32.data, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Our MMA kernel
        CHECK_CUDA(cudaMemsetAsync(d_Y.data, 0, M * N * sizeof(bf16), stream));
        BF16GemmMMA<BM, BN, BK, NUM_STAGES, MMA_CWG, MMA_WARP_M, MMA_WARP_N>::run(M, N, K, d_X.data, d_W.data, d_Y.data, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Compare MMA
        d_Y.copy_to_host(h_Y.data(), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        float tol = 0.1f;
        if (!check_result("MMA", h_Y.data(), h_Y_ref.data(), h_Y_f32.data(), M * N, tol))
            return 1;

        // Our MMA Split-K kernel
        CUDABuffer<float> d_workspace(SPLIT_K * M * N);
        CHECK_CUDA(cudaMemsetAsync(d_Y.data, 0, M * N * sizeof(bf16), stream));
        BF16GemmMMASplitK<BM, BN, BK, NUM_STAGES, MMA_CWG, MMA_WARP_M, MMA_WARP_N, SPLIT_K>::run(
            M, N, K, d_X.data, d_W.data, d_Y.data, d_workspace.data, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        d_Y.copy_to_host(h_Y.data(), stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        if (!check_result("MMA-SK", h_Y.data(), h_Y_ref.data(), h_Y_f32.data(), M * N, tol))
            return 1;

        // ── Benchmark (rotate buffers to flush L2 cache) ────────────
        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        size_t l2_size;
        {
            int l2_bytes;
            CHECK_CUDA(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
            l2_size = static_cast<size_t>(l2_bytes);
        }
        size_t per_set_bytes = (size_t(M) * K + size_t(K) * N + size_t(M) * N) * sizeof(bf16);
        int NUM_BUFS = std::max(1, (int)((2 * l2_size + per_set_bytes - 1) / per_set_bytes));
        printf("[bench] L2 cache: %zu KB, per-set: %zu KB, NUM_BUFS: %d\n",
               l2_size / 1024, per_set_bytes / 1024, NUM_BUFS);
        std::vector<CUDABuffer<bf16> *> bench_X, bench_W, bench_Y;
        for (int b = 0; b < NUM_BUFS; b++)
        {
            bench_X.push_back(new CUDABuffer<bf16>(M * K));
            bench_W.push_back(new CUDABuffer<bf16>(K * N));
            bench_Y.push_back(new CUDABuffer<bf16>(M * N));
            bench_X.back()->copy_from_host(h_X.data(), stream);
            bench_W.back()->copy_from_host(h_W.data(), stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        int buf_idx = 0;
        double flops = 2.0 * M * N * K;

        double cublas_ms = bench_ms([&]()
                                    { int b = buf_idx++ % NUM_BUFS;
                                      cublas_gemm(handle, M, N, K, bench_X[b]->data, bench_W[b]->data, bench_Y[b]->data); }, stream);

        double mma_ms = bench_ms([&]()
                                 { int b = buf_idx++ % NUM_BUFS;
                                   BF16GemmMMA<BM, BN, BK, NUM_STAGES, MMA_CWG, MMA_WARP_M, MMA_WARP_N>::run(M, N, K, bench_X[b]->data, bench_W[b]->data, bench_Y[b]->data, stream); }, stream);

        double splitk_ms = bench_ms([&]()
                                    { int b = buf_idx++ % NUM_BUFS;
                                      BF16GemmMMASplitK<BM, BN, BK, NUM_STAGES, MMA_CWG, MMA_WARP_M, MMA_WARP_N, SPLIT_K>::run(
                                          M, N, K, bench_X[b]->data, bench_W[b]->data, bench_Y[b]->data, d_workspace.data, stream); }, stream);

        double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;
        double mma_tflops = flops / (mma_ms * 1e-3) / 1e12;
        double splitk_tflops = flops / (splitk_ms * 1e-3) / 1e12;

        printf("\n── Benchmark (M=%d N=%d K=%d) ──\n", M, N, K);
        printf("[cuBLAS] %.4f ms  %.4f TFLOPS\n", cublas_ms, cublas_tflops);
        printf("[MMA]    %.4f ms  %.4f TFLOPS  (%.1f%% of cuBLAS)\n", mma_ms, mma_tflops, 100.0 * mma_tflops / cublas_tflops);
        printf("[MMA-SK] %.4f ms  %.4f TFLOPS  (%.1f%% of cuBLAS)\n", splitk_ms, splitk_tflops, 100.0 * splitk_tflops / cublas_tflops);

        for (int b = 0; b < NUM_BUFS; b++)
        {
            delete bench_X[b];
            delete bench_W[b];
            delete bench_Y[b];
        }
    }
    return 0;
}