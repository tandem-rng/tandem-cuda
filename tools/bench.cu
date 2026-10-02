// Throughput of the device fills into device memory, next to cuRAND Philox4x32-10.
// Build and run on a GPU host: make bench
#include <cstdio>
#include <cstdlib>

#include <curand.h>

#include "../tandem.cuh"

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess) {                                                                    \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);    \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static const size_t N = (size_t)1 << 28;

template <class F> static double best_gibs(size_t bytes, F body) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < 3; i++) body();
    float best = 1e30f;
    for (int rep = 0; rep < 7; rep++) {
        CUDA_CHECK(cudaEventRecord(t0));
        body();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaGetLastError());
    return (double)bytes / (best * 1e-3) / (1024.0 * 1024.0 * 1024.0);
}

int main() {
    void *buf;
    CUDA_CHECK(cudaMalloc(&buf, N * 8));
    const uint32_t key[4] = {1, 2, 3, 4};
    auto u32 = static_cast<uint32_t *>(buf);
    auto u64 = static_cast<uint64_t *>(buf);
    auto f32 = static_cast<float *>(buf);
    auto f64 = static_cast<double *>(buf);

    std::printf("%-34s %10s\n", "", "GiB/s");
    std::printf("%-34s %10.0f\n", "tandem fill_u32",
                best_gibs(N * 4, [&] { tandem::fill_u32(key, 0, 32, u32, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_u64",
                best_gibs(N * 8, [&] { tandem::fill_u64(key, 0, 32, u64, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_f32",
                best_gibs(N * 4, [&] { tandem::fill_f32(key, 0, 32, f32, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_f64",
                best_gibs(N * 8, [&] { tandem::fill_f64(key, 0, 32, f64, N); }));

    curandGenerator_t g;
    curandCreateGenerator(&g, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(g, 42);
    std::printf("%-34s %10.0f\n", "cuRAND Philox4x32-10 u32",
                best_gibs(N * 4, [&] { curandGenerate(g, u32, N); }));
    std::printf("%-34s %10.0f\n", "cuRAND Philox4x32-10 f32",
                best_gibs(N * 4, [&] { curandGenerateUniform(g, f32, N); }));
    std::printf("%-34s %10.0f\n", "cuRAND Philox4x32-10 f64",
                best_gibs(N * 8, [&] { curandGenerateUniformDouble(g, f64, N); }));
    curandDestroyGenerator(g);
    CUDA_CHECK(cudaFree(buf));
    return 0;
}
