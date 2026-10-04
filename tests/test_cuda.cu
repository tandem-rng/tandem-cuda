// Spec vectors, the Julia stream dumps, and agreement with the C library at random keys and
// positions. Run on a GPU host: make test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../tandem.cuh"
#include "vectors.h"

extern "C" {
#include "tandem.h"
}

static int failures;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            failures++;                                                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                            \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess) {                                                                    \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);    \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static bool words_equal(const uint32_t a[4], const uint32_t b[4]) {
    return std::memcmp(a, b, 16) == 0;
}

// ---- Device helpers -------------------------------------------------------------------------

template <class T, T (tandem::device_rng::*draw)()>
__global__ void draw_kernel(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, uint64_t pos,
                            uint32_t K, T *out, size_t n) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng r = tandem::device_rng::from_key(key, pos, K);
    for (size_t i = 0; i < n; i++) out[i] = (r.*draw)();
}

// std::vector<bool> has no data(), so bool draws come back as bytes.
__global__ void bool_kernel(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, uint64_t pos,
                            uint32_t K, uint8_t *out, size_t n) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng r = tandem::device_rng::from_key(key, pos, K);
    for (size_t i = 0; i < n; i++) out[i] = r.next_bool();
}

template <class T> struct dev {
    T *p = nullptr;
    size_t n;
    explicit dev(size_t n) : n(n) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T) + 16)); }
    ~dev() { cudaFree(p); }
    std::vector<T> host() const {
        std::vector<T> v(n);
        CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

template <class T, T (tandem::device_rng::*draw)()>
static std::vector<T> device_draws(const uint32_t key[4], uint64_t pos, uint32_t K, size_t n) {
    dev<T> d(n);
    draw_kernel<T, draw><<<1, 1>>>(key[0], key[1], key[2], key[3], pos, K, d.p, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    return d.host();
}

static std::vector<uint8_t> device_bools(const uint32_t key[4], uint64_t pos, uint32_t K,
                                         size_t n) {
    dev<uint8_t> d(n);
    bool_kernel<<<1, 1>>>(key[0], key[1], key[2], key[3], pos, K, d.p, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    return d.host();
}

// T is the output type, E the fill kind where several kinds share an output type.
template <class T, class E = T>
static std::vector<T> device_fill(const uint32_t key[4], uint64_t pos, uint32_t K, size_t n,
                                  size_t byte_shift = 0, bool tile = true) {
    dev<T> d(n + 2);
    T *out = reinterpret_cast<T *>(reinterpret_cast<char *>(d.p) + byte_shift);
    tandem::detail::fill<E>(key, pos, K, reinterpret_cast<typename tandem::detail::elem<E>::out_t *>(out),
                            n, 0, tile);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<T> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), out, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

// ---- Vectors ------------------------------------------------------------------------------

static void test_vectors() {
    for (const auto &v : VEC_T) {
        uint32_t o[4], h[4];
        std::memcpy(o, v.o, 16);
        std::memcpy(h, v.h, 16);
        tandem::T(o, h);
        CHECK(words_equal(o, v.o_out) && words_equal(h, v.h_out));
    }
    for (const auto &v : VEC_F) {
        uint32_t o[4], h[4];
        tandem::F_keyed(VEC_KEY, v.counter, tandem::DOMAIN_STREAM, tandem::AUX_STREAM, o, h);
        CHECK(words_equal(o, v.o) && words_equal(h, v.h));
    }
    auto u32 = device_fill<uint32_t>(VEC_KEY, 0, VEC_K, 64);
    for (const auto &s : VEC_STREAM) {
        CHECK(std::memcmp(&u32[s.first_word], s.words, 16) == 0);
        uint64_t row = (s.first_word * 32) >> 10, lane = ((s.first_word * 32) >> 7) & 7;
        uint32_t b[4];
        tandem::block(VEC_KEY, 8 * (row / VEC_K) + lane, (uint32_t)(row % VEC_K), b);
        CHECK(words_equal(b, s.words));
    }
    auto f64 = device_fill<double>(VEC_KEY, 0, VEC_K, 32);
    for (const auto &v : VEC_F64) CHECK(f64[v.index] == v.value);
    auto f32 = device_fill<float>(VEC_KEY, 0, VEC_K, 32);
    for (const auto &v : VEC_F32) CHECK(f32[v.index] == v.value);
    auto bits = device_bools(VEC_KEY, 0, VEC_K, 129);
    for (const auto &v : VEC_BOOL) CHECK(bits[v.index] == (v.value != 0));

    tandem::device_rng r = tandem::device_rng::from_key(VEC_KEY, 0, VEC_K);
    CHECK(words_equal(r.split(0).key, VEC_SPLIT0));
    CHECK(words_equal(r.split(1).key, VEC_SPLIT1));
    CHECK(words_equal(r.sub(7).key, VEC_PURPOSE7));
    CHECK(words_equal(r.child(0, tandem::DOMAIN_FORK, 0, false).key, VEC_FORK0));

    tandem::device_rng s = tandem::device_rng::seed(VEC_SEED, 0, VEC_K);
    CHECK(words_equal(s.key, VEC_SEED_KEY));
    auto sf64 = device_fill<double>(s.key, 0, VEC_K, 32);
    for (const auto &v : VEC_SEED_F64) CHECK(sf64[v.index] == v.value);
    auto su32 = device_fill<uint32_t>(s.key, 0, VEC_K, 32);
    for (const auto &v : VEC_SEED_U32) CHECK(su32[v.index] == v.value);
}

// ---- Dumps --------------------------------------------------------------------------------

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

template <class T, T (tandem::device_rng::*draw)()>
static void check_dump(const char *dir, const char *name, const uint32_t key[4], uint32_t K) {
    std::vector<T> want = slurp<T>(dir, name);
    if (want.empty()) return;
    std::vector<T> fill = device_fill<T>(key, 0, K, want.size());
    std::vector<T> draws = device_draws<T, draw>(key, 0, K, want.size());
    for (size_t i = 0; i < want.size(); i++) {
        if (fill[i] != want[i]) {
            std::printf("FAIL %s: fill differs at element %zu\n", name, i);
            failures++;
            break;
        }
    }
    for (size_t i = 0; i < want.size(); i++) {
        if (draws[i] != want[i]) {
            std::printf("FAIL %s: device draw differs at element %zu\n", name, i);
            failures++;
            break;
        }
    }
}

// Public launchers fill whatever pointer they get, so the dump comparison needs no type.
template <class T, class L> static std::vector<T> public_fill(L launch, size_t n) {
    dev<T> d(n + 2);
    launch(d.p, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    return d.host();
}

static uint64_t fill_bool_bytes(const uint32_t key[4], uint64_t pos, uint32_t K, uint8_t *out,
                                size_t n, cudaStream_t stream) {
    return tandem::fill_bool(key, pos, K, reinterpret_cast<bool *>(out), n, stream);
}

template <class T, uint64_t (*launch)(const uint32_t *, uint64_t, uint32_t, T *, size_t, cudaStream_t)>
static void check_public_dump(const char *dir, const char *name, const uint32_t key[4]) {
    std::vector<T> want = slurp<T>(dir, name);
    if (want.empty()) return;
    auto got = public_fill<T>([&](T *p, size_t n) { launch(key, 0, 32, p, n, 0); }, want.size());
    got.resize(want.size()); // the device buffer has two spare elements
    if (got != want) {
        size_t i = 0;
        while (i < want.size() && got[i] == want[i]) i++;
        std::printf("FAIL %s: public fill differs from the dump at element %zu of %zu\n", name, i,
                    want.size());
        failures++;
    }
}

static void test_dumps(const char *dir) {
    tandem::device_rng s42 = tandem::device_rng::seed(42, 0, 32);
    check_dump<uint32_t, &tandem::device_rng::next_u32>(dir, "k1234_K32_u32.bin", KEY1234, 32);
    check_dump<uint64_t, &tandem::device_rng::next_u64>(dir, "k1234_K32_u64.bin", KEY1234, 32);
    check_dump<uint32_t, &tandem::device_rng::next_u32>(dir, "k1234_K8_u32.bin", KEY1234, 8);
    check_dump<double, &tandem::device_rng::next_f64>(dir, "seed42_K32_f64.bin", s42.key, 32);
    check_dump<float, &tandem::device_rng::next_f32>(dir, "seed42_K32_f32.bin", s42.key, 32);
    std::vector<uint8_t> bools = slurp<uint8_t>(dir, "seed42_K32_bool.bin");
    auto got = device_bools(s42.key, 0, 32, bools.size());
    for (size_t i = 0; i < bools.size(); i++)
        if (got[i] != bools[i]) {
            std::printf("FAIL bool draws differ at element %zu\n", i);
            failures++;
            break;
        }
    // The sub-word fills, through the public launchers.
    check_public_dump<uint8_t, fill_bool_bytes>(dir, "seed42_K32_bool.bin", s42.key);
    check_public_dump<uint8_t, tandem::fill_u8>(dir, "seed42_K32_u8.bin", s42.key);
    check_public_dump<uint16_t, tandem::fill_f16_bits>(dir, "seed42_K32_f16bits.bin", s42.key);
}

// ---- Against the C library at random keys and positions -------------------------------------

// Fills are compared at every K, alignment and tile choice. Scalar draws are compared where the
// device generator has them.
template <class T, class E, void (*cfill)(tandem_rng *, T *, size_t),
          T (tandem::device_rng::*draw)() = nullptr>
static void check_against_c(std::mt19937_64 &gen, const char *label) {
    for (int trial = 0; trial < 40; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = gen() % (1u << 20);
        size_t n = (size_t)(gen() % (trial < 30 ? 5000 : 300000));
        size_t shift = (gen() % 2) ? 0 : (gen() % 4) * sizeof(T) % 16;

        tandem_rng c = tandem_from_key(key, pos, K);
        std::vector<T> want(n);
        cfill(&c, want.data(), n);
        for (bool tile : {true, false}) {
            std::vector<T> got = device_fill<T, E>(key, pos, K, n, shift, tile);
            if (want != got) {
                size_t i = 0;
                while (i < n && want[i] == got[i]) i++;
                std::printf("FAIL %s %s fill vs C at trial %d (K=%u pos=%llu n=%zu) element %zu\n",
                            label, tile ? "tile" : "direct", trial, K, (unsigned long long)pos,
                            n, i);
                failures++;
            }
        }
        if (draw == nullptr || n > 512) continue;
        if constexpr (draw != nullptr) {
            std::vector<T> draws = device_draws<T, draw>(key, pos, K, n);
            if (want != draws) {
                std::printf("FAIL %s device draws vs C at trial %d\n", label, trial);
                failures++;
            }
        }
    }
}

// The C fills take bool* and write 0 or 1, the device bool fill writes the same bytes.
static void c_fill_bool(tandem_rng *r, uint8_t *out, size_t n) {
    tandem_fill_bool(r, reinterpret_cast<bool *>(out), n);
}

static void test_against_c() {
    std::mt19937_64 gen(2026);
    check_against_c<uint32_t, uint32_t, tandem_fill_u32, &tandem::device_rng::next_u32>(gen, "u32");
    check_against_c<uint64_t, uint64_t, tandem_fill_u64, &tandem::device_rng::next_u64>(gen, "u64");
    check_against_c<float, float, tandem_fill_f32, &tandem::device_rng::next_f32>(gen, "f32");
    check_against_c<double, double, tandem_fill_f64, &tandem::device_rng::next_f64>(gen, "f64");
    check_against_c<uint8_t, uint8_t, tandem_fill_u8>(gen, "u8");
    check_against_c<uint16_t, uint16_t, tandem_fill_u16>(gen, "u16");
    check_against_c<uint16_t, tandem::detail::f16_bits, tandem_fill_f16_bits>(gen, "f16 bits");
    check_against_c<uint8_t, tandem::detail::bool_bits, c_fill_bool>(gen, "bool");

    // Mixed widths through one device generator agree with the C generator.
    const uint32_t key[4] = {9, 8, 7, 6};
    tandem::device_rng r = tandem::device_rng::from_key(key, 5, 32);
    tandem_rng c = tandem_from_key(key, 5, 32);
    for (int i = 0; i < 2000; i++) {
        CHECK(r.next_bool() == tandem_next_bool(&c));
        CHECK(r.next_u32() == tandem_next_u32(&c));
        CHECK(r.next_f64() == tandem_next_f64(&c));
        CHECK(r.next_u64() == tandem_next_u64(&c));
        CHECK(r.next_f32() == tandem_next_f32(&c));
    }
    CHECK(r.pos == tandem_position(&c));
}

// A signed fill is the unsigned fill of the same width read in two's complement, and it returns
// the same position.
template <class S, class U, uint64_t (*sfill)(const uint32_t *, uint64_t, uint32_t, S *, size_t, cudaStream_t),
          void (*cfill)(tandem_rng *, U *, size_t)>
static void check_signed(std::mt19937_64 &gen, const char *label) {
    for (int trial = 0; trial < 10; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = gen() % (1u << 20);
        size_t n = (size_t)(gen() % 20000);
        tandem_rng c = tandem_from_key(key, pos, K);
        std::vector<U> want(n);
        cfill(&c, want.data(), n);
        dev<S> d(n + 2);
        uint64_t end = sfill(key, pos, K, d.p, n, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<S> got = d.host();
        CHECK(std::memcmp(got.data(), want.data(), n * sizeof(S)) == 0);
        CHECK(end == tandem_position(&c));
        if (std::memcmp(got.data(), want.data(), n * sizeof(S)) != 0)
            std::printf("FAIL %s signed fill differs at trial %d\n", label, trial);
    }
}

static void test_signed() {
    std::mt19937_64 gen(77);
    check_signed<int8_t, uint8_t, tandem::fill_i8, tandem_fill_u8>(gen, "i8");
    check_signed<int16_t, uint16_t, tandem::fill_i16, tandem_fill_u16>(gen, "i16");
    check_signed<int32_t, uint32_t, tandem::fill_i32, tandem_fill_u32>(gen, "i32");
    check_signed<int64_t, uint64_t, tandem::fill_i64, tandem_fill_u64>(gen, "i64");
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/data";
    test_vectors();
    test_dumps(dir);
    test_against_c();
    test_signed();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::puts("cuda: ok");
    return 0;
}
