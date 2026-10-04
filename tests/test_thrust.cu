// The Thrust adapters: element i of a functor or iterator equals element i of the fill, and
// matches the stream dumps. Run on a GPU host: make thrust
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cub/device/device_reduce.cuh>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "../tandem_thrust.cuh"

static int failures;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                            \
        }                                                                                          \
    } while (0)

template <class T> using dvec = thrust::device_vector<T>;

template <class T> static std::vector<T> to_host(const dvec<T> &d) {
    std::vector<T> h(d.size());
    thrust::copy(d.begin(), d.end(), h.begin());
    return h;
}

template <class F, class T> static std::vector<T> transformed(F f, size_t n) {
    dvec<T> d(n);
    thrust::transform(thrust::counting_iterator<uint64_t>(0),
                      thrust::counting_iterator<uint64_t>(n), d.begin(), f);
    return to_host(d);
}

template <class T, class Fill> static std::vector<T> filled(Fill fill, size_t n) {
    dvec<T> d(n);
    fill(thrust::raw_pointer_cast(d.data()), n);
    cudaDeviceSynchronize();
    return to_host(d);
}

// Normals differ by the sin/cos path of the fill kernel, so they match to a tolerance.
template <class T> static bool close(const std::vector<T> &a, const std::vector<T> &b, double rel) {
    for (size_t i = 0; i < a.size(); i++)
        if (!(std::fabs((double)a[i] - (double)b[i]) <= rel * (1.0 + std::fabs((double)b[i])) + 1e-6 * (sizeof(T) == 4)))
            return false;
    return true;
}

template <class T> static std::vector<T> slurp(const char *dir, const char *name) {
    char path[512];
    std::snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::printf("FAIL cannot open %s\n", path);
        failures++;
        return {};
    }
    std::fseek(f, 0, SEEK_END);
    size_t len = (size_t)std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<T> v(len / sizeof(T));
    if (std::fread(v.data(), 1, len, f) != len) failures++;
    std::fclose(f);
    return v;
}

static const uint32_t KEY1234[4] = {1, 2, 3, 4};

static void test_dumps(const char *dir) {
    uint32_t s42[4];
    tandem::seed_key(42, 0, s42);
    auto u32 = slurp<uint32_t>(dir, "k1234_K32_u32.bin");
    CHECK(!u32.empty() && (transformed<tandem::uniform<uint32_t>, uint32_t>(tandem::uniform<uint32_t>(KEY1234, 32), u32.size()) == u32));
    auto u32k8 = slurp<uint32_t>(dir, "k1234_K8_u32.bin");
    CHECK(!u32k8.empty() && (transformed<tandem::uniform<uint32_t>, uint32_t>(tandem::uniform<uint32_t>(KEY1234, 8), u32k8.size()) == u32k8));
    auto u64 = slurp<uint64_t>(dir, "k1234_K32_u64.bin");
    CHECK(!u64.empty() && (transformed<tandem::uniform<uint64_t>, uint64_t>(tandem::uniform<uint64_t>(KEY1234, 32), u64.size()) == u64));
    auto f64 = slurp<double>(dir, "seed42_K32_f64.bin");
    CHECK(!f64.empty() && (transformed<tandem::uniform<double>, double>(tandem::uniform<double>(s42, 32), f64.size()) == f64));
    auto f32 = slurp<float>(dir, "seed42_K32_f32.bin");
    CHECK(!f32.empty() && (transformed<tandem::uniform<float>, float>(tandem::uniform<float>(s42, 32), f32.size()) == f32));
}

// Functors against the fills at random keys, K, positions and lengths.
static void test_against_fills() {
    std::mt19937_64 gen(4242);
    for (int trial = 0; trial < 30; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = gen() % (1u << 20);
        size_t n = (size_t)(gen() % 3000) + (trial % 5 == 0);
        auto fu32 = [&](uint32_t *p, size_t m) { tandem::fill_u32(key, pos, K, p, m); };
        auto fu64 = [&](uint64_t *p, size_t m) { tandem::fill_u64(key, pos, K, p, m); };
        auto ff32 = [&](float *p, size_t m) { tandem::fill_f32(key, pos, K, p, m); };
        auto ff64 = [&](double *p, size_t m) { tandem::fill_f64(key, pos, K, p, m); };
        CHECK((transformed<tandem::uniform<uint32_t>, uint32_t>(tandem::uniform<uint32_t>(key, K, pos), n) == filled<uint32_t>(fu32, n)));
        CHECK((transformed<tandem::uniform<uint64_t>, uint64_t>(tandem::uniform<uint64_t>(key, K, pos), n) == filled<uint64_t>(fu64, n)));
        CHECK((transformed<tandem::uniform<float>, float>(tandem::uniform<float>(key, K, pos), n) == filled<float>(ff32, n)));
        CHECK((transformed<tandem::uniform<double>, double>(tandem::uniform<double>(key, K, pos), n) == filled<double>(ff64, n)));

        // Ranges that reject often, rarely and never.
        for (uint32_t r : {6u, 1000u, 1u << 20, 3000000000u, 0x80000001u}) {
            auto fb = [&](uint32_t *p, size_t m) { tandem::fill_u32_below(key, pos, K, r, p, m); };
            CHECK((transformed<tandem::below<uint32_t>, uint32_t>(tandem::below<uint32_t>(key, r, K, pos), n) == filled<uint32_t>(fb, n)));
        }
        for (uint64_t r : {6ull, 1000000007ull, 0xc000000000003039ull}) {
            auto fb = [&](uint64_t *p, size_t m) { tandem::fill_u64_below(key, pos, K, r, p, m); };
            CHECK((transformed<tandem::below<uint64_t>, uint64_t>(tandem::below<uint64_t>(key, r, K, pos), n) == filled<uint64_t>(fb, n)));
        }

        // Normals: iterator element 2j, 2j + 1 are the pair of uniforms 2j, 2j + 1.
        auto fn64 = [&](double *p, size_t m) { tandem::fill_normal_f64(key, pos, K, p, m); };
        auto fn32 = [&](float *p, size_t m) { tandem::fill_normal_f32(key, pos, K, p, m); };
        CHECK(close(transformed<tandem::normal<double>, double>(tandem::normal<double>(key, K, pos), n), filled<double>(fn64, n), 1e-12));
        CHECK(close(transformed<tandem::normal<float>, float>(tandem::normal<float>(key, K, pos), n), filled<float>(fn32, n), 16 * 0x1p-23));
    }
}

// Iterators feed reductions and copies without a buffer.
static void test_iterators() {
    const uint32_t key[4] = {9, 8, 7, 6};
    const size_t n = 100000;
    auto it = tandem::make_iterator(tandem::uniform<uint64_t>(key, 32, 0));
    dvec<uint64_t> d(n);
    tandem::fill_u64(key, 0, 32, thrust::raw_pointer_cast(d.data()), n);
    uint64_t want = thrust::reduce(d.begin(), d.end(), (uint64_t)0);
    CHECK(thrust::reduce(it, it + n, (uint64_t)0) == want);

    dvec<uint64_t> copy(n);
    thrust::copy_n(it, n, copy.begin());
    CHECK(to_host(copy) == to_host(d));

    // CUB takes the iterator as well.
    dvec<uint64_t> out(1);
    void *tmp = nullptr;
    size_t bytes = 0;
    cub::DeviceReduce::Sum(tmp, bytes, it, thrust::raw_pointer_cast(out.data()), (int)n);
    cudaMalloc(&tmp, bytes);
    cub::DeviceReduce::Sum(tmp, bytes, it, thrust::raw_pointer_cast(out.data()), (int)n);
    cudaDeviceSynchronize();
    cudaFree(tmp);
    CHECK(to_host(out)[0] == want);

    // An iterator that starts later reads the same elements.
    auto later = tandem::make_iterator(tandem::uniform<uint64_t>(key, 32, 0), 1000);
    dvec<uint64_t> tail(500);
    thrust::copy_n(later, 500, tail.begin());
    auto h = to_host(d), t = to_host(tail);
    CHECK(std::equal(t.begin(), t.end(), h.begin() + 1000));
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/data";
    test_dumps(dir);
    test_against_fills();
    test_iterators();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::puts("thrust: ok");
    return 0;
}
