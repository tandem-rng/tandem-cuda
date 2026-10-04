// Throughput of the device fills into device memory, next to cuRAND Philox4x32-10.
// Minimum of 21 cudaEvent timings per row after a half-second warm-up.
// Build and run on a GPU host: make bench
#include <cstdio>
#include <cstdlib>

#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "../tandem_thrust.cuh"

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
    for (int rep = 0; rep < 21; rep++) {
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

    // Half a second of fills first, so the clocks have ramped before anything is timed.
    {
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (float ms = 0; ms < 500;) {
            tandem::fill_u32(key, 0, 32, u32, N);
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        }
    }
    std::printf("%-34s %10s\n", "", "GiB/s");
    std::printf("%-34s %10.0f\n", "tandem fill_u32, direct kernel",
                best_gibs(N * 4, [&] { tandem::detail::fill<uint32_t>(key, 0, 32u, u32, N, 0, false); }));
    std::printf("%-34s %10.0f\n", "tandem fill_u64, direct kernel",
                best_gibs(N * 8, [&] { tandem::detail::fill<uint64_t>(key, 0, 32u, u64, N, 0, false); }));
    std::printf("%-34s %10.0f\n", "tandem fill_f32, direct kernel",
                best_gibs(N * 4, [&] { tandem::detail::fill<float>(key, 0, 32u, f32, N, 0, false); }));
    std::printf("%-34s %10.0f\n", "tandem fill_f64, direct kernel",
                best_gibs(N * 8, [&] { tandem::detail::fill<double>(key, 0, 32u, f64, N, 0, false); }));
    std::printf("%-34s %10.0f\n", "tandem fill_u32, tile kernel",
                best_gibs(N * 4, [&] { tandem::fill_u32(key, 0, 32, u32, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_u64, tile kernel",
                best_gibs(N * 8, [&] { tandem::fill_u64(key, 0, 32, u64, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_f32, tile kernel",
                best_gibs(N * 4, [&] { tandem::fill_f32(key, 0, 32, f32, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_f64, tile kernel",
                best_gibs(N * 8, [&] { tandem::fill_f64(key, 0, 32, f64, N); }));

    // The Thrust functors reach each block by random access, so they are far slower than a fill.
    {
        auto idx0 = thrust::counting_iterator<uint64_t>(0), idx1 = thrust::counting_iterator<uint64_t>(N);
        tandem::uniform<uint32_t> fu(key, 32);
        tandem::uniform<double> fd(key, 32);
        std::printf("%-34s %10.0f\n", "thrust::transform uniform<u32>",
                    best_gibs(N * 4, [&] { thrust::transform(idx0, idx1, thrust::device_pointer_cast(u32), fu); }));
        std::printf("%-34s %10.0f\n", "thrust::transform uniform<f64>",
                    best_gibs(N * 8, [&] { thrust::transform(idx0, idx1, thrust::device_pointer_cast(f64), fd); }));
    }
    auto u16 = static_cast<uint16_t *>(buf);
    auto u8 = static_cast<uint8_t *>(buf);
    auto b8 = static_cast<bool *>(buf);
    std::printf("%-34s %10.0f\n", "tandem fill_u16, tile kernel",
                best_gibs(N * 2, [&] { tandem::fill_u16(key, 0, 32, u16, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_f16_bits, tile kernel",
                best_gibs(N * 2, [&] { tandem::fill_f16_bits(key, 0, 32, u16, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_u8, tile kernel",
                best_gibs(N, [&] { tandem::fill_u8(key, 0, 32, u8, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_bool, shared-mem kernel",
                best_gibs(N, [&] { tandem::fill_bool(key, 0, 32, b8, N); }));

    std::printf("%-34s %10.0f\n", "tandem fill_u32_below(1000)",
                best_gibs(N * 4, [&] { tandem::fill_u32_below(key, 0, 32, 1000u, u32, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_u32_below(2^32 - 2)",
                best_gibs(N * 4, [&] { tandem::fill_u32_below(key, 0, 32, 0xfffffffeu, u32, N); }));
#ifndef TANDEM_BENCH_OLD
    std::printf("%-34s %10.0f\n", "tandem fill_u32_below, low, i32",
                best_gibs(N * 4, [&] { tandem::fill_u32_below(key, 0, 32, 0xfffffffeu, (int32_t)-7, reinterpret_cast<int32_t *>(u32), N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_u32_below, low, i64",
                best_gibs(N * 8, [&] { tandem::fill_u32_below(key, 0, 32, 1000u, (int64_t)-7, reinterpret_cast<int64_t *>(u64), N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_u64_below, low, i64",
                best_gibs(N * 8, [&] { tandem::fill_u64_below(key, 0, 32, ~0ull - 1, (int64_t)-7, reinterpret_cast<int64_t *>(u64), N); }));
#endif
    std::printf("%-34s %10.0f\n", "tandem fill_u64_below(1000)",
                best_gibs(N * 8, [&] { tandem::fill_u64_below(key, 0, 32, 1000u, u64, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_u64_below(2^64 - 2)",
                best_gibs(N * 8, [&] { tandem::fill_u64_below(key, 0, 32, ~0ull - 1, u64, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_normal_f32",
                best_gibs(N * 4, [&] { tandem::fill_normal_f32(key, 0, 32, f32, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_normal_f64",
                best_gibs(N * 8, [&] { tandem::fill_normal_f64(key, 0, 32, f64, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_normal_f64, odd start",
                best_gibs(N * 8, [&] { tandem::fill_normal_f64(key, 64, 32, f64, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_exponential_f32",
                best_gibs(N * 4, [&] { tandem::fill_exponential_f32(key, 0, 32, f32, N); }));
    std::printf("%-34s %10.0f\n", "tandem fill_exponential_f64",
                best_gibs(N * 8, [&] { tandem::fill_exponential_f64(key, 0, 32, f64, N); }));

    curandGenerator_t g;
    curandCreateGenerator(&g, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(g, 42);
    std::printf("%-34s %10.0f\n", "cuRAND Philox4x32-10 u32",
                best_gibs(N * 4, [&] { curandGenerate(g, u32, N); }));
    std::printf("%-34s %10.0f\n", "cuRAND Philox4x32-10 f32",
                best_gibs(N * 4, [&] { curandGenerateUniform(g, f32, N); }));
    std::printf("%-34s %10.0f\n", "cuRAND Philox4x32-10 f64",
                best_gibs(N * 8, [&] { curandGenerateUniformDouble(g, f64, N); }));
    std::printf("%-34s %10.0f\n", "cuRAND Philox4x32-10 normal f32",
                best_gibs(N * 4, [&] { curandGenerateNormal(g, f32, N, 0.f, 1.f); }));
    std::printf("%-34s %10.0f\n", "cuRAND Philox4x32-10 normal f64",
                best_gibs(N * 8, [&] { curandGenerateNormalDouble(g, f64, N, 0., 1.); }));
    curandDestroyGenerator(g);
    CUDA_CHECK(cudaFree(buf));
    return 0;
}
