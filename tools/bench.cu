// Throughput of the device fills into device memory, next to cuRAND Philox4x32-10.
// Each row first runs its own fill for WARM_MS, so the card reaches that fill's steady clocks and
// power whatever ran before, then reports the median and the fastest of 21 cudaEvent timings.
// docs/speed.md gives the median: the fastest once hid a slow allocation in most calls. A row run
// alone gives the figures it gives in the table. The fills that hit the 250 W power cap are measured at
// the capped steady state.
// Build and run on a GPU host: make bench. `tools/bench exponential` runs only the rows whose
// name contains "exponential".
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
static const float WARM_MS = 2000;
static const char *filter = nullptr;

struct Rate {
    double median, best; /* GiB/s */
};

template <class F> static Rate rate(size_t bytes, F body) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (float ms = 0; ms < WARM_MS;) {
        body();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    }
    std::vector<float> ms(21);
    for (float &m : ms) {
        CUDA_CHECK(cudaEventRecord(t0));
        body();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&m, t0, t1));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    std::sort(ms.begin(), ms.end());
    auto gibs = [&](float t) { return (double)bytes / (t * 1e-3) / (1024.0 * 1024.0 * 1024.0); };
    return Rate{gibs(ms[10]), gibs(ms[0])};
}

template <class F> static void row(const char *name, size_t bytes, F body) {
    if (filter && !std::strstr(name, filter)) return;
    Rate r = rate(bytes, body);
    std::printf("%-34s %10.0f %10.0f\n", name, r.median, r.best);
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc > 1) filter = argv[1];
    void *buf;
    CUDA_CHECK(cudaMalloc(&buf, N * 8));
    const uint32_t key[4] = {1, 2, 3, 4};
    auto u32 = static_cast<uint32_t *>(buf);
    auto u64 = static_cast<uint64_t *>(buf);
    auto f32 = static_cast<float *>(buf);
    auto f64 = static_cast<double *>(buf);

    std::printf("%-34s %10s %10s\n", "GiB/s", "median", "fastest");
    row("tandem fill_u32, direct kernel", N * 4,
        [&] { tandem::detail::fill<uint32_t>(key, 0, 32u, u32, N, 0, false); });
    row("tandem fill_u64, direct kernel", N * 8,
        [&] { tandem::detail::fill<uint64_t>(key, 0, 32u, u64, N, 0, false); });
    row("tandem fill_f32, direct kernel", N * 4,
        [&] { tandem::detail::fill<float>(key, 0, 32u, f32, N, 0, false); });
    row("tandem fill_f64, direct kernel", N * 8,
        [&] { tandem::detail::fill<double>(key, 0, 32u, f64, N, 0, false); });
    row("tandem fill_u32, tile kernel", N * 4, [&] { tandem::fill_u32(key, 0, 32, u32, N); });
    row("tandem fill_u64, tile kernel", N * 8, [&] { tandem::fill_u64(key, 0, 32, u64, N); });
    row("tandem fill_f32, tile kernel", N * 4, [&] { tandem::fill_f32(key, 0, 32, f32, N); });
    row("tandem fill_f64, tile kernel", N * 8, [&] { tandem::fill_f64(key, 0, 32, f64, N); });

    // The Thrust functors reach each block by random access, so they are far slower than a fill.
    {
        auto idx0 = thrust::counting_iterator<uint64_t>(0), idx1 = thrust::counting_iterator<uint64_t>(N);
        tandem::uniform<uint32_t> fu(key, 32);
        tandem::uniform<double> fd(key, 32);
        row("thrust::transform uniform<u32>", N * 4,
            [&] { thrust::transform(idx0, idx1, thrust::device_pointer_cast(u32), fu); });
        row("thrust::transform uniform<f64>", N * 8,
            [&] { thrust::transform(idx0, idx1, thrust::device_pointer_cast(f64), fd); });
    }
    auto u16 = static_cast<uint16_t *>(buf);
    auto u8 = static_cast<uint8_t *>(buf);
    auto b8 = static_cast<bool *>(buf);
    row("tandem fill_u16, tile kernel", N * 2, [&] { tandem::fill_u16(key, 0, 32, u16, N); });
    row("tandem fill_f16_bits, tile kernel", N * 2, [&] { tandem::fill_f16_bits(key, 0, 32, u16, N); });
    row("tandem fill_u8, tile kernel", N, [&] { tandem::fill_u8(key, 0, 32, u8, N); });
    row("tandem fill_bool, shared-mem kernel", N, [&] { tandem::fill_bool(key, 0, 32, b8, N); });

    row("tandem fill_u32_below(1000)", N * 4, [&] { tandem::fill_u32_below(key, 0, 32, 1000u, u32, N); });
    row("tandem fill_u32_below(2^32 - 2)", N * 4,
        [&] { tandem::fill_u32_below(key, 0, 32, 0xfffffffeu, u32, N); });
#ifndef TANDEM_BENCH_OLD
    row("tandem fill_u32_below, low, i32", N * 4, [&] {
        tandem::fill_u32_below(key, 0, 32, 0xfffffffeu, (int32_t)-7, reinterpret_cast<int32_t *>(u32), N);
    });
    row("tandem fill_u32_below, low, i64", N * 8, [&] {
        tandem::fill_u32_below(key, 0, 32, 1000u, (int64_t)-7, reinterpret_cast<int64_t *>(u64), N);
    });
    row("tandem fill_u64_below, low, i64", N * 8, [&] {
        tandem::fill_u64_below(key, 0, 32, ~0ull - 1, (int64_t)-7, reinterpret_cast<int64_t *>(u64), N);
    });
#endif
    row("tandem fill_u64_below(1000)", N * 8, [&] { tandem::fill_u64_below(key, 0, 32, 1000u, u64, N); });
    row("tandem fill_u64_below(2^64 - 2)", N * 8,
        [&] { tandem::fill_u64_below(key, 0, 32, ~0ull - 1, u64, N); });
    row("tandem fill_normal_f32", N * 4, [&] { tandem::fill_normal_f32(key, 0, 32, f32, N); });
    row("tandem fill_normal_f64", N * 8, [&] { tandem::fill_normal_f64(key, 0, 32, f64, N); });
    row("tandem fill_normal_f64, odd start", N * 8, [&] { tandem::fill_normal_f64(key, 64, 32, f64, N); });
    // Word 6 is draw 3: the shuffled pairs sit 16 bytes off the 128-byte lines.
    row("tandem fill_normal_f64, word 6", N * 8, [&] { tandem::fill_normal_f64(key, 192, 32, f64, N); });
    for (int lg : {24, 20}) {
        size_t m = (size_t)1 << lg;
        char name[64];
        std::snprintf(name, sizeof name, "tandem fill_normal_f64, 2^%d", lg);
        row(name, m * 8, [&] { tandem::fill_normal_f64(key, 0, 32, f64, m); });
        std::snprintf(name, sizeof name, "tandem fill_normal_f64, 2^%d, odd", lg);
        row(name, m * 8, [&] { tandem::fill_normal_f64(key, 64, 32, f64, m); });
    }
    row("tandem fill_exponential_f32", N * 4, [&] { tandem::fill_exponential_f32(key, 0, 32, f32, N); });
    row("tandem fill_exponential_f64", N * 8, [&] { tandem::fill_exponential_f64(key, 0, 32, f64, N); });

    curandGenerator_t g;
    curandCreateGenerator(&g, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(g, 42);
    row("cuRAND Philox4x32-10 u32", N * 4, [&] { curandGenerate(g, u32, N); });
    row("cuRAND Philox4x32-10 f32", N * 4, [&] { curandGenerateUniform(g, f32, N); });
    row("cuRAND Philox4x32-10 f64", N * 8, [&] { curandGenerateUniformDouble(g, f64, N); });
    row("cuRAND Philox4x32-10 normal f32", N * 4, [&] { curandGenerateNormal(g, f32, N, 0.f, 1.f); });
    row("cuRAND Philox4x32-10 normal f64", N * 8,
        [&] { curandGenerateNormalDouble(g, f64, N, 0., 1.); });
    curandDestroyGenerator(g);
    CUDA_CHECK(cudaFree(buf));
    return 0;
}
