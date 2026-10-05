// Spec vectors, the Julia stream dumps, and agreement with the C library at random keys and
// positions. Run on a GPU host: make test
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../tandem.cuh"
#include "vectors.h"
#include "cross_fill_below.h"
#include "cross_fill_exponential.h"
#include "cross_fill_normal.h"
// tandem-c's, through -I$(TANDEM_C)
#include "tests/cross_exponential.h"
#include "tests/cross_normal.h"

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

// ---- Device generator beyond the scalar draws ---------------------------------------------------

constexpr size_t API_N = 600, API_AT = 24, API_CHILD = 5;

struct ApiOut {
    uint32_t below32[API_N];
    uint64_t below64[API_N];
    double normal[API_N];
    uint8_t u8[API_N];
    uint16_t u16[API_N], f16[API_N];
    uint32_t at32[API_AT];
    uint64_t at64[API_AT];
    float atf32[API_AT];
    double atf64[API_AT];
    float normalf[API_N], pairf[2 * API_N];
    double pair[2 * API_N];
    double expo[API_N];
    float expof[API_N];
    uint64_t fork[API_CHILD], split, sub, pos;
};

__global__ void api_kernel(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, uint64_t pos,
                           uint32_t K, uint32_t range32, uint64_t range64, ApiOut *o) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng r = tandem::device_rng::from_key(key, pos, K);
    for (size_t i = 0; i < API_N; i++) {
        o->below32[i] = r.urand(range32);
        o->below64[i] = r.urand64(range64);
        o->normal[i] = r.normal();
        o->normalf[i] = r.normalf();
        tandem::Pair2<double> p2 = r.normal2();
        o->pair[2 * i] = p2.z0, o->pair[2 * i + 1] = p2.z1;
        tandem::Pair2<float> pf = r.normalf2();
        o->pairf[2 * i] = pf.z0, o->pairf[2 * i + 1] = pf.z1;
        o->u8[i] = r.next_u8();
        o->u16[i] = r.next_u16();
        o->f16[i] = r.next_f16_bits();
        o->expo[i] = r.exponential();
        o->expof[i] = r.exponentialf();
    }
    for (size_t i = 0; i < API_AT; i++) {
        o->at32[i] = r.at_urand(i);
        o->at64[i] = r.at_urand64(i);
        o->atf32[i] = r.at_frand(i);
        o->atf64[i] = r.at_drand(i);
    }
    tandem::device_rng kids[API_CHILD];
    r.fork(kids, API_CHILD);
    for (size_t i = 0; i < API_CHILD; i++) o->fork[i] = kids[i].next_u64();
    o->split = r.split(3).next_u64();
    o->sub = r.sub(9).next_u64();
    o->pos = r.pos;
}

// The device generator matches the C library on every draw it shares, in one mixed sequence, and
// the host core on the f64 normals. f32 normals agree to libm precision, everything else bit for
// bit.
static void check_device_api(const uint32_t key[4], uint64_t pos, uint32_t K, uint32_t range32,
                             uint64_t range64) {
    ApiOut *d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(ApiOut)));
    api_kernel<<<1, 1>>>(key[0], key[1], key[2], key[3], pos, K, range32, range64, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    ApiOut g;
    CUDA_CHECK(cudaMemcpy(&g, d, sizeof g, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));

    tandem_rng c = tandem_from_key(key, pos, K);
    int bad = 0;
    for (size_t i = 0; i < API_N; i++) {
        bad += g.below32[i] != tandem_u32_below(&c, range32);
        bad += g.below64[i] != tandem_u64_below(&c, range64);
        // An f64 normal is one UInt64 draw at the generator's position.
        auto host = [&] { return tandem::Rng::from_key(tandem::Key{{key[0], key[1], key[2], key[3]}}, tandem_position(&c), K); };
        double z = host().normal();
        tandem_next_u64(&c);
        bad += std::memcmp(&g.normal[i], &z, 8) != 0;
        float fa = tandem_next_f32(&c), fb = tandem_next_f32(&c);
        float zf = sqrtf(-2.0f * logf(1.0f - fa)) * cosf(2.0f * 3.14159265358979323846f * fb);
        bad += !(std::fabs(g.normalf[i] - zf) <= 4 * 0x1p-23f * (1.0f + std::fabs(zf)));
        {
            auto p2 = host().normal2();
            tandem_next_u64(&c), tandem_next_u64(&c);
            bad += std::memcmp(&g.pair[2 * i], &p2.z0, 8) != 0 || std::memcmp(&g.pair[2 * i + 1], &p2.z1, 8) != 0;
            float qa = tandem_next_f32(&c), qb = tandem_next_f32(&c);
            auto pf = tandem::box_muller2_f32(qa, qb);
            bad += !(std::fabs(g.pairf[2 * i] - pf.z0) <= 4 * 0x1p-23f * (1.0f + std::fabs(pf.z0)));
            bad += !(std::fabs(g.pairf[2 * i + 1] - pf.z1) <=
                     4 * 0x1p-23f * (1.0f + std::fabs(pf.z1)));
        }
        bad += g.u8[i] != tandem_next_u8(&c);
        bad += g.u16[i] != tandem_next_u16(&c);
        bad += g.f16[i] != tandem_next_f16_bits(&c);
        double e = tandem_exponential_f64(&c);
        float ef = tandem_exponential_f32(&c);
        bad += std::memcmp(&g.expo[i], &e, 8) != 0 || std::memcmp(&g.expof[i], &ef, 4) != 0;
    }
    for (size_t i = 0; i < API_AT; i++) {
        bad += g.at32[i] != tandem_at_u32(&c, i);
        bad += g.at64[i] != tandem_at_u64(&c, i);
        bad += g.atf32[i] != tandem_at_f32(&c, i);
        bad += g.atf64[i] != tandem_at_f64(&c, i);
    }
    tandem_rng kids[API_CHILD];
    tandem_fork(&c, kids, API_CHILD);
    for (size_t i = 0; i < API_CHILD; i++) bad += g.fork[i] != tandem_next_u64(&kids[i]);
    tandem_rng sp = tandem_split(&c, 3), sb = tandem_sub(&c, 9);
    bad += g.split != tandem_next_u64(&sp);
    bad += g.sub != tandem_next_u64(&sb);
    bad += g.pos != tandem_position(&c);
    if (bad) std::printf("FAIL device_rng vs C: %d differences (K=%u pos=%llu range=%u)\n", bad, K,
                         (unsigned long long)pos, range32);
    failures += bad != 0;
}

static void test_device_api() {
    const uint32_t key[4] = {11, 22, 33, 44};
    // A small range rarely rejects, a range above 2^31 rejects often and exercises the loop.
    check_device_api(key, 0, 32, 7u, 1000003u);
    check_device_api(key, 77, 8, 3000000000u, 0xc000000000003039ull);
    check_device_api(key, 4097, 1, 0x80000001u, 0x8000000000000001ull);
}

// ---- Bounded and normal fills -----------------------------------------------------------------

// Without a rejection a bounded fill equals the sequential C calls. With one, it follows the
// contract in core.hpp, written out again here on the C library's own generators. g is the
// global draw index, the aligned start over the draw width plus the element index.
static uint32_t ref_below32(const tandem_rng *root, uint32_t u, uint32_t range, uint64_t g) {
    uint64_t m = (uint64_t)u * range;
    if ((uint32_t)m < range) {
        uint32_t t = (0u - range) % range;
        if ((uint32_t)m < t) {
            tandem_rng sb = tandem_sub(root, 0x424c573332ull), f = tandem_split(&sb, g);
            do m = (uint64_t)tandem_next_u32(&f) * range; while ((uint32_t)m < t);
        }
    }
    return (uint32_t)(m >> 32);
}

static uint64_t ref_below64(const tandem_rng *root, uint64_t x, uint64_t range, uint64_t g) {
    unsigned __int128 m = (unsigned __int128)x * range;
    if ((uint64_t)m < range) {
        uint64_t t = (0u - range) % range;
        if ((uint64_t)m < t) {
            tandem_rng sb = tandem_sub(root, 0x424c573634ull), f = tandem_split(&sb, g);
            do m = (unsigned __int128)tandem_next_u64(&f) * range; while ((uint64_t)m < t);
        }
    }
    return (uint64_t)(m >> 64);
}

static void c_fill_below(tandem_rng *g, uint32_t *out, size_t n, uint32_t range) {
    tandem_fill_u32_below(g, out, n, range);
}
static void c_fill_below(tandem_rng *g, uint64_t *out, size_t n, uint64_t range) {
    tandem_fill_u64_below(g, out, n, range);
}

template <class T, class Launch, class Ref>
static void check_below(std::mt19937_64 &gen, const char *label, T range, Launch launch,
                        Ref ref, void (*cfill)(tandem_rng *, T *, size_t)) {
    constexpr unsigned bits = sizeof(T) * 8;
    for (int trial = 0; trial < 12; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = gen() % (1u << 20);
        size_t n = (size_t)(gen() % 30000);
        tandem_rng root = tandem_from_key(key, 0, K), c = tandem_from_key(key, pos, K);
        std::vector<T> raw(n), want(n), cwant(n);
        cfill(&c, raw.data(), n);
        uint64_t g0 = tandem::align_pos(pos, bits) / bits;
        for (size_t e = 0; e < n; e++) want[e] = ref(&root, raw[e], range, g0 + e);
        // The C library's bounded fill implements the same contract.
        c = tandem_from_key(key, pos, K);
        c_fill_below(&c, cwant.data(), n, range);
        CHECK(cwant == want);
        dev<T> d(n + 2);
        uint64_t end = launch(key, pos, K, range, d.p, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<T> got = d.host();
        got.resize(n);
        CHECK(end == tandem::align_pos(pos, bits) + (uint64_t)n * bits);
        if (got != want) {
            size_t i = 0;
            while (i < n && got[i] == want[i]) i++;
            std::printf("FAIL %s below(%llu) trial %d (K=%u pos=%llu n=%zu) element %zu\n", label,
                        (unsigned long long)range, trial, K, (unsigned long long)pos, n, i);
            failures++;
        }
    }
}

// The same fill where no draw rejects must also equal the sequential C bounded draws.
template <class T, class Launch, T (*cbelow)(tandem_rng *, T)>
static void check_below_sequential(std::mt19937_64 &gen, const char *label, T range,
                                   Launch launch) {
    for (int trial = 0; trial < 12; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = (gen() % (1u << 20)) & ~(uint64_t)63;
        size_t n = (size_t)(gen() % 30000);
        tandem_rng c = tandem_from_key(key, pos, K);
        std::vector<T> want(n);
        for (auto &v : want) v = cbelow(&c, range);
        dev<T> d(n + 2);
        launch(key, pos, K, range, d.p, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<T> got = d.host();
        got.resize(n);
        if (got != want) {
            std::printf("FAIL %s below(%llu) vs sequential C, trial %d\n", label,
                        (unsigned long long)range, trial);
            failures++;
        }
    }
}

// A low bound adds to the draw in the output type, wrapping, and a wider output stores the
// 32-bit draw zero-extended. The draws and the position are those of the plain fill.
template <class O, class W, class Plain, class Low>
static void check_below_low(std::mt19937_64 &gen, const char *label, W range, O low, Plain plain,
                            Low launch) {
    for (int trial = 0; trial < 12; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = gen() % (1u << 20);
        size_t n = (size_t)(gen() % 30000);
        std::vector<W> base = plain(key, pos, K, range, n);
        using U = std::make_unsigned_t<O>;
        dev<O> d(n + 2);
        uint64_t end = launch(key, pos, K, range, low, d.p, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<O> got = d.host();
        got.resize(n);
        bool ok = true;
        for (size_t e = 0; e < n; e++) ok &= got[e] == (O)(U)((U)low + (U)base[e]);
        CHECK(ok);
        CHECK(end == (n ? tandem::align_pos(pos, sizeof(W) * 8) + (uint64_t)n * sizeof(W) * 8 : pos));
        if (!ok) std::printf("FAIL %s below_low(%llu) trial %d\n", label, (unsigned long long)range, trial);
    }
}

static void test_below_low() {
    std::mt19937_64 gen(1999);
    auto plain32 = [](const uint32_t *k, uint64_t p, uint32_t K, uint32_t r, size_t n) {
        dev<uint32_t> d(n + 2);
        tandem::fill_u32_below(k, p, K, r, d.p, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto h = d.host();
        h.resize(n);
        return h;
    };
    auto plain64 = [](const uint32_t *k, uint64_t p, uint32_t K, uint64_t r, size_t n) {
        dev<uint64_t> d(n + 2);
        tandem::fill_u64_below(k, p, K, r, d.p, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto h = d.host();
        h.resize(n);
        return h;
    };
    auto l32 = [](const uint32_t *k, uint64_t p, uint32_t K, uint32_t r, auto low, auto *o, size_t n) {
        return tandem::fill_u32_below(k, p, K, r, low, o, n);
    };
    auto l64 = [](const uint32_t *k, uint64_t p, uint32_t K, uint64_t r, auto low, auto *o, size_t n) {
        return tandem::fill_u64_below(k, p, K, r, low, o, n);
    };
    // Ranges that reject often (3e9), rarely (1000) and the full range minus one.
    for (uint32_t r : {1000u, 3000000000u, 0xfffffffeu}) {
        check_below_low<uint32_t>(gen, "u32->u32", r, 4000000000u, plain32, l32);
        check_below_low<int32_t>(gen, "u32->i32", r, (int32_t)-1000, plain32, l32);
        check_below_low<uint64_t>(gen, "u32->u64", r, 5000000000ull, plain32, l32);
        check_below_low<int64_t>(gen, "u32->i64", r, (int64_t)-5000000000ll, plain32, l32);
    }
    for (uint64_t r : {1000ull, 0xc000000000003039ull}) {
        check_below_low<uint64_t>(gen, "u64->u64", r, 0xfffffffffffffff0ull, plain64, l64);
        check_below_low<int64_t>(gen, "u64->i64", r, (int64_t)-7, plain64, l64);
    }
    // Outputs narrower than the draw, as the JAX FFI stores int8 and int16 bounded draws, through
    // the tile kernel (K >= 8) and the direct kernel (K < 8).
    auto narrow = [](const uint32_t *k, uint64_t p, uint32_t K, uint32_t r, auto low, auto *o, size_t n) {
        using O = std::remove_pointer_t<decltype(o)>;
        if (n == 0) return p;
        return tandem::detail::fill<tandem::detail::below32<O>>(
            k, p, K, o, n, 0, true,
            tandem::detail::Bound{r, (uint64_t)(int64_t)low, tandem::below_threshold_u32(r)});
    };
    check_below_low<uint16_t>(gen, "u32->u16", 1000u, (uint16_t)60000, plain32, narrow);
    check_below_low<int8_t>(gen, "u32->i8", 100u, (int8_t)-50, plain32, narrow);
    // The threshold is computed once: range 0 returns the low bound and consumes the draws.
    dev<int32_t> z(8);
    CHECK(tandem::fill_u32_below(KEY1234, 0, 32, 0, (int32_t)5, z.p, 8) == 8 * 32);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int32_t v : z.host()) CHECK(v == 5 || v == 0 /* the two spare elements */);
}

static void test_below() {
    std::mt19937_64 gen(5150);
    auto l32 = [](const uint32_t *k, uint64_t p, uint32_t K, uint32_t r, uint32_t *o, size_t n) {
        return tandem::fill_u32_below(k, p, K, r, o, n);
    };
    auto l64 = [](const uint32_t *k, uint64_t p, uint32_t K, uint64_t r, uint64_t *o, size_t n) {
        return tandem::fill_u64_below(k, p, K, r, o, n);
    };
    // Ranges that reject often, rarely, and never (powers of two), and the full range.
    for (uint32_t r : {1u, 2u, 6u, 1000u, 65537u, 1u << 20, 3000000000u, 0x80000001u, 0xffffffffu})
        check_below<uint32_t>(gen, "u32", r, l32, ref_below32, tandem_fill_u32);
    for (uint64_t r : {1ull, 6ull, 1000000007ull, 1ull << 40, 0xc000000000003039ull,
                       0x8000000000000001ull, ~0ull})
        check_below<uint64_t>(gen, "u64", r, l64, ref_below64, tandem_fill_u64);
    for (uint32_t r : {1u, 2u, 6u, 1000u, 1u << 20})
        check_below_sequential<uint32_t, decltype(l32), tandem_u32_below>(gen, "u32", r, l32);
    for (uint64_t r : {1ull, 6ull, 1000000007ull, 1ull << 40})
        check_below_sequential<uint64_t, decltype(l64), tandem_u64_below>(gen, "u64", r, l64);
}

// The fixtures other ports match, which include the fallback stream of rejected draws.
static void test_cross_below() {
    CHECK(words_equal(CROSS_FILL_KEY, tandem::device_rng::seed(42, 0, 32).key));
    for (const auto &f : CROSS_BELOW32) {
        dev<uint32_t> d(64 + 2);
        tandem::fill_u32_below(CROSS_FILL_KEY, 0, 32, f.range, d.p, 64);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto got = d.host();
        CHECK(std::memcmp(got.data(), f.out, sizeof f.out) == 0);
    }
    for (const auto &f : CROSS_BELOW64) {
        dev<uint64_t> d(64 + 2);
        tandem::fill_u64_below(CROSS_FILL_KEY, 0, 32, f.range, d.p, 64);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto got = d.host();
        CHECK(std::memcmp(got.data(), f.out, sizeof f.out) == 0);
    }
    for (const auto &f : CROSS_BELOW32_AT) {
        dev<uint32_t> d(64 + 2);
        tandem::fill_u32_below(CROSS_FILL_KEY, f.start, 32, f.range, d.p, 64);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto got = d.host();
        CHECK(std::memcmp(got.data(), f.out, sizeof f.out) == 0);
    }
    for (const auto &f : CROSS_BELOW64_AT) {
        dev<uint64_t> d(64 + 2);
        tandem::fill_u64_below(CROSS_FILL_KEY, f.start, 32, f.range, d.p, 64);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto got = d.host();
        CHECK(std::memcmp(got.data(), f.out, sizeof f.out) == 0);
    }
}

// A bounded fill cut at any element boundary equals the whole fill, rejected draws included,
// because the fallback is keyed by the global draw index. Each piece starts where the last
// ended. The fused low-bound and wider-output kernels are covered through `launch`.
template <class O, class Launch>
static void check_cut_below(std::mt19937_64 &gen, const char *label, Launch launch) {
    for (int trial = 0; trial < 8; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = gen() % (1u << 20);
        size_t n = 1000 + (size_t)(gen() % 20000), k = 1 + (size_t)(gen() % (n - 1));
        dev<O> whole(n), cut(n);
        uint64_t end = launch(key, pos, K, whole.p, n);
        uint64_t mid = launch(key, pos, K, cut.p, k);
        uint64_t end2 = launch(key, mid, K, cut.p + k, n - k);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<O> a = whole.host(), b = cut.host();
        CHECK(end == end2);
        if (a != b) {
            size_t i = 0;
            while (a[i] == b[i]) i++;
            std::printf("FAIL %s cut fill (K=%u pos=%llu n=%zu cut=%zu) element %zu\n", label, K,
                        (unsigned long long)pos, n, k, i);
            failures++;
        }
    }
}

static void test_cut_below() {
    std::mt19937_64 gen(2718);
    // Ranges just above 2^31 and 2^63 reject about half the draws.
    check_cut_below<uint32_t>(gen, "u32", [](const uint32_t *k, uint64_t p, uint32_t K, uint32_t *o, size_t n) {
        return tandem::fill_u32_below(k, p, K, 0x80000001u, o, n);
    });
    check_cut_below<uint64_t>(gen, "u64", [](const uint32_t *k, uint64_t p, uint32_t K, uint64_t *o, size_t n) {
        return tandem::fill_u64_below(k, p, K, 0x8000000000000001ull, o, n);
    });
    check_cut_below<int32_t>(gen, "u32->i32", [](const uint32_t *k, uint64_t p, uint32_t K, int32_t *o, size_t n) {
        return tandem::fill_u32_below(k, p, K, 0x80000001u, (int32_t)-9, o, n);
    });
    check_cut_below<int64_t>(gen, "u32->i64", [](const uint32_t *k, uint64_t p, uint32_t K, int64_t *o, size_t n) {
        return tandem::fill_u32_below(k, p, K, 0x80000001u, (int64_t)-5000000000ll, o, n);
    });
    check_cut_below<int64_t>(gen, "u64->i64", [](const uint32_t *k, uint64_t p, uint32_t K, int64_t *o, size_t n) {
        return tandem::fill_u64_below(k, p, K, 0x8000000000000001ull, (int64_t)-7, o, n);
    });
    // The generator handle cuts the same way, each call starting where the last ended.
    for (int trial = 0; trial < 4; trial++) {
        size_t n = 5000, k = 1 + (size_t)(gen() % (n - 1));
        tandem::generator w = tandem::generator::from_key(KEY1234, 4321 + (uint64_t)trial, 32), c = w;
        dev<uint32_t> a(n), b(n);
        w.fill_u32_below(0x80000001u, a.p, n);
        c.fill_u32_below(0x80000001u, b.p, k);
        c.fill_u32_below(0x80000001u, b.p + k, n - k);
        CUDA_CHECK(cudaDeviceSynchronize());
        CHECK(a.host() == b.host());
        CHECK(w.pos == c.pos);
    }
}

// The absolute floor of the f32 fill's tolerance. Its angle goes through __sincosf, whose
// absolute error on [-pi, pi] is at most 2^-21.41, times a radius of at most sqrt(-2 ln 2^-24) =
// 5.77. That passes the spec's 1e-6 for about one value in 10^6: a = 0x1.fff256p-1 and
// b = 0x1.040388p-2 give -0.10545215 against the host's -0.105453432.
constexpr double F32_FILL_ABS = 2.1e-6;

// This repository's normal fixtures, at even and odd starts.
template <class T, class F, uint64_t (*launch)(const uint32_t *, uint64_t, uint32_t, T *, size_t, cudaStream_t)>
static void check_cross_normal(const F &f, double tol) {
    dev<T> d(f.n + 2);
    launch(CROSS_FILL_KEY, f.pos, 32, d.p, f.n, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto got = d.host();
    size_t bad = 0;
    double max_abs = 0, max_ulps = 0; // reported when tuning the float step
    for (size_t i = 0; i < f.n; i++) {
        double want = f.out[i], dv = std::fabs((double)got[i] - want);
        double ulp = std::fabs(want) * 0x1p-23;
#ifndef TANDEM_PRECISE_F32_NORMAL
        bool ok = sizeof(T) == 4 ? dv <= 16 * ulp + F32_FILL_ABS : dv <= tol * (1.0 + std::fabs(want));
#else
        bool ok = dv <= tol * (1.0 + std::fabs(want));
#endif
        bad += !ok;
        max_abs = std::fmax(max_abs, dv);
        if (std::fabs(want) > 1e-2) max_ulps = std::fmax(max_ulps, dv / ulp);
    }
    if (sizeof(T) == 4)
        std::printf("f32 fixture pos %llu: max deviation %.3g abs, %.3g ulps (|x| > 1e-2)\n",
                    (unsigned long long)f.pos, max_abs, max_ulps);
    if (bad) {
        std::printf("FAIL normal fixture at pos %llu: %zu elements differ\n",
                    (unsigned long long)f.pos, bad);
        failures++;
    }
}

static void test_cross_normal() {
    for (const auto &f : CROSS_NORMAL64)
        check_cross_normal<double, cross_normal64, tandem::fill_normal_f64>(f, 0.0); // exact
    for (const auto &f : CROSS_NORMAL32)
        check_cross_normal<float, cross_normal32, tandem::fill_normal_f32>(f, 4 * 0x1p-23);
    // tandem-c's fixture: normalf2 calls of Rng(42) after one bit draw, which is the fill from
    // position 1, to the device fill's 16 ulps + F32_FILL_ABS.
    const size_t n = 2 * CROSS_NORMAL_COUNT;
    dev<float> df(n);
    CHECK(tandem::fill_normal_f32(CROSS_FILL_KEY, 1, 32, df.p, n) == CROSS_NORMALF_END_POS);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got = df.host();
    for (size_t i = 0; i < n; i++)
        CHECK(std::fabs(got[i] - CROSS_NORMALF[i]) <= 16 * 0x1p-23f * std::fabs(CROSS_NORMALF[i]) + (float)F32_FILL_ABS);
}

// Float32 normal fills: pair j is one Box-Muller step of the uniform draws 2j and 2j + 1, cos half
// first, and an odd n drops the last sin half but still consumes both draws. The reference runs the
// host step on the C library's uniform fills, at every start slot, including starts at an odd draw
// where each pair spans two blocks. The device's sincos and logf match to 16 ulps + F32_FILL_ABS.
template <class T, void (*cfill)(tandem_rng *, T *, size_t), unsigned W,
          tandem::Pair2<T> (*step)(T, T),
          uint64_t (*launch)(const uint32_t *, uint64_t, uint32_t, T *, size_t, cudaStream_t)>
static void check_normal(std::mt19937_64 &gen, const char *label, double tol) {
    double max_abs = 0, max_ulps = 0; // largest deviation from the host step, for the report
    for (int trial = 0; trial < 80; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = (gen() % (1u << 20)) & ~(uint64_t)127;
        pos += (W / 4) * (trial % (128 / W)) + (trial % 3 == 0 ? 7 : 0); // every draw slot
        size_t n = (size_t)(gen() % (trial < 60 ? 3000 : 100000));
        if (trial % 5 == 0) n |= 1; // odd n
        size_t np = (n + 1) / 2;
        tandem_rng c = tandem_from_key(key, pos, K);
        std::vector<T> u(2 * np), want(n);
        cfill(&c, u.data(), 2 * np);
        for (size_t j = 0; j < np; j++) {
            auto z = step(u[2 * j], u[2 * j + 1]);
            want[2 * j] = z.z0;
            if (2 * j + 1 < n) want[2 * j + 1] = z.z1;
        }
        dev<T> d(n + 2);
        uint64_t end = launch(key, pos, K, d.p, n, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<T> got = d.host();
        CHECK(end == tandem::align_pos(pos, W) + (uint64_t)np * 2u * W);
        CHECK(end == tandem_position(&c));
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) {
            double dv = std::fabs((double)got[i] - (double)want[i]);
            double ulp = std::fabs((double)want[i]) * 0x1p-23;
#ifndef TANDEM_PRECISE_F32_NORMAL
            bool ok = W == 32 ? dv <= 16 * ulp + F32_FILL_ABS : dv <= tol * (1.0 + std::fabs((double)want[i]));
#else
            bool ok = dv <= tol * (1.0 + std::fabs((double)want[i]));
#endif
            bad += !ok;
            if (W == 32) {
                max_abs = std::fmax(max_abs, dv);
                if (std::fabs((double)want[i]) > 1e-2) max_ulps = std::fmax(max_ulps, dv / ulp);
            }
        }
        if (trial == 79 && W == 32)
            std::printf("f32 normal fill, max deviation: %.3g abs, %.3g ulps (|x| > 1e-2)\n", max_abs, max_ulps);
        if (bad) {
            std::printf("FAIL %s normal fill: %zu elements differ (trial %d K=%u pos=%llu n=%zu)\n",
                        label, bad, trial, K, (unsigned long long)pos, n);
            failures++;
        }
    }
}

static void test_normal() {
    // Trial 70 of this seed holds the input past the spec's 1e-6, see F32_FILL_ABS.
    std::mt19937_64 gen(8675);
    check_normal<float, tandem_fill_f32, 32, tandem::box_muller2_f32, tandem::fill_normal_f32>(
        gen, "f32", 4 * 0x1p-23);
}

// Float64 normal fills equal the C library's fill byte for byte, end position included: element e is
// the ziggurat of UInt64 draw e, its misses continued on the fallback keyed by the global draw
// index. Every K and start slot, outputs on and 8 bytes off 16-byte alignment, short fills (one
// kernel) and long ones (two kernels), with misses and tail values among them.
static void test_normal64() {
    std::mt19937_64 gen(1618);
    size_t misses = 0, tails = 0;
    for (int trial = 0; trial < 60; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = (gen() % (1u << 20)) & ~(uint64_t)127;
        pos += 64 * (trial % 2) + (trial % 3 == 0 ? 7 : 0); // even and odd draws, unaligned
        size_t n = trial < 4 ? (size_t)1 << 21 : (size_t)(gen() % (trial < 40 ? 3000 : 200000));
        unsigned shift = (trial / 2) % 2;
        tandem_rng c = tandem_from_key(key, pos, K);
        std::vector<uint64_t> raw(n);
        tandem_fill_u64(&c, raw.data(), n);
        for (uint64_t r : raw) {
            bool hit;
            tandem::normal_f64_fast(r, hit);
            misses += !hit;
            tails += !hit && (r & 1023u) == 0;
        }
        c = tandem_from_key(key, pos, K);
        std::vector<double> want(n);
        tandem_fill_normal_f64(&c, want.data(), n);
        dev<double> d(n + 2);
        uint64_t end = tandem::fill_normal_f64(key, pos, K, d.p + shift, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> got(n);
        CUDA_CHECK(cudaMemcpy(got.data(), d.p + shift, n * 8, cudaMemcpyDeviceToHost));
        CHECK(end == tandem_position(&c));
        if (std::memcmp(got.data(), want.data(), n * 8) != 0) {
            size_t i = 0;
            while (std::memcmp(&got[i], &want[i], 8) == 0) i++;
            std::printf("FAIL f64 normal fill (trial %d K=%u pos=%llu n=%zu shift=%u) element %zu\n",
                        trial, K, (unsigned long long)pos, n, shift, i);
            failures++;
        }
    }
    std::printf("f64 normal fills: %zu misses, %zu in the tail\n", misses, tails);
    CHECK(tails > 0);
}

// The table pass writes octets from the fill's first draw d0: every d0 % 16, which sets the
// block's lane and the draw within it, then K = 1, whose every octet takes the next group's first
// row, and outputs on 32-byte addresses and 8 and 16 bytes off. Two-kernel fills, bit for bit.
static void test_normal64_octets() {
    const size_t n = 70001;
    const uint32_t key[4] = {8, 9, 10, 11};
    for (uint64_t d0 : {0ull, 1ull, 2ull, 3ull, 4ull, 5ull, 6ull, 7ull, 8ull, 9ull, 10ull, 11ull, 12ull,
                        13ull, 14ull, 15ull, (1ull << 30) + 3})
        for (uint32_t K : {1u, 32u}) {
            tandem_rng c = tandem_from_key(key, 64 * d0, K);
            std::vector<double> want(n), got(n);
            tandem_fill_normal_f64(&c, want.data(), n);
            for (unsigned shift : {0u, 1u, 2u}) {
                dev<double> d(n + 2);
                CHECK(tandem::fill_normal_f64(key, 64 * d0, K, d.p + shift, n) == tandem_position(&c));
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(got.data(), d.p + shift, n * 8, cudaMemcpyDeviceToHost));
                if (std::memcmp(got.data(), want.data(), n * 8) != 0) {
                    std::printf("FAIL f64 normal octets (d0=%llu K=%u shift=%u)\n",
                                (unsigned long long)d0, K, shift);
                    failures++;
                }
            }
        }
}

static uint64_t fnv(const void *p, size_t n) {
    const unsigned char *b = static_cast<const unsigned char *>(p);
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

// The f64 normals of a Python implementation written from the text of spec Appendix A, as
// tandem-c's tests/test_normal_bits.c hashes them: 2e5 from key {1, 2, 3, 4}, K = 32, at bits 0
// and 2373. And tandem-c's fixture rows, whose last rows hold a wedge accept, a wedge reject and a
// tail value.
static void test_normal_reference() {
    const size_t n = 200000;
    const struct {
        uint64_t start, hash, end;
    } ref[] = {{0, 0x0c4059ed409d578dull, 12800000}, {2373, 0x30ce40c86b295193ull, 12802432}};
    for (const auto &f : ref) {
        dev<double> d(n);
        CHECK(tandem::fill_normal_f64(KEY1234, f.start, 32, d.p, n) == f.end);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> got = d.host();
        CHECK(fnv(got.data(), n * 8) == f.hash);
    }
    for (const auto &f : CROSS_NORMAL) {
        dev<double> d(CROSS_NORMAL_COUNT);
        CHECK(tandem::fill_normal_f64(CROSS_FILL_KEY, f.start, 32, d.p, CROSS_NORMAL_COUNT) == f.end_pos);
        CUDA_CHECK(cudaDeviceSynchronize());
        CHECK(std::memcmp(d.host().data(), f.want, sizeof f.want) == 0);
    }
}

// A normal fill cut at any element equals the whole fill: at an odd element, at a missed element
// and just after it, with the two-kernel whole fill against short and long pieces.
static void test_cut_normal() {
    const size_t n = 300000;
    const uint64_t pos = 77;
    tandem_rng c = tandem_from_key(KEY1234, pos, 32);
    std::vector<uint64_t> raw(n);
    tandem_fill_u64(&c, raw.data(), n);
    size_t m = 1;
    for (bool hit = true; hit; m++) tandem::normal_f64_fast(raw[m], hit);
    m--;
    for (size_t a : {(size_t)12345, m, m + 1}) {
        tandem::generator w = tandem::generator::from_key(KEY1234, pos, 32), v = w;
        dev<double> whole(n), cut(n);
        w.fill_normal_f64(whole.p, n);
        v.fill_normal_f64(cut.p, a);
        v.fill_normal_f64(cut.p + a, n - a);
        CUDA_CHECK(cudaDeviceSynchronize());
        CHECK(whole.host() == cut.host());
        CHECK(w.pos == v.pos && w.pos == tandem_position(&c));
    }
}

// Two-kernel normal fills on two streams at once, whose miss lists come from one pool. Each round
// frees a list that the other stream's next fill can take.
static void test_normal_streams() {
    const size_t n = (size_t)1 << 20;
    const uint32_t keys[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    std::vector<double> want[2];
    for (int s = 0; s < 2; s++) {
        tandem_rng c = tandem_from_key(keys[s], 64 * s, 32);
        want[s].resize(n);
        tandem_fill_normal_f64(&c, want[s].data(), n);
    }
    cudaStream_t st[2];
    for (auto &x : st) CUDA_CHECK(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
    dev<double> out(8 * n); /* fill s of round r at (4 s + r) n */
    for (int r = 0; r < 4; r++)
        for (int s = 0; s < 2; s++)
            tandem::fill_normal_f64(keys[s], 64 * s, 32, out.p + (4 * s + r) * n, n, st[s]);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> got = out.host();
    for (int s = 0; s < 2; s++)
        for (int r = 0; r < 4; r++)
            CHECK(std::memcmp(got.data() + (4 * s + r) * n, want[s].data(), n * 8) == 0);
    for (auto x : st) CUDA_CHECK(cudaStreamDestroy(x));
}

// Exponential fills: element i is -ln(1 - u) of uniform draw i, the same polynomial arithmetic as
// tandem-c's fill, so the device output equals tandem_fill_exponential_* byte for byte at every
// start slot, K and length, the end positions included.
template <class T, void (*cfill)(tandem_rng *, T *, size_t), unsigned W,
          uint64_t (*launch)(const uint32_t *, uint64_t, uint32_t, T *, size_t, cudaStream_t)>
static void check_exponential(std::mt19937_64 &gen, const char *label) {
    for (int trial = 0; trial < 60; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = (gen() % (1u << 20)) & ~(uint64_t)127;
        pos += (W / 4) * (trial % (128 / W)) + (trial % 3 == 0 ? 7 : 0); // every draw slot
        size_t n = trial == 0 ? (size_t)1 << 22 : (size_t)(gen() % (trial < 40 ? 3000 : 100000));
        tandem_rng c = tandem_from_key(key, pos, K);
        std::vector<T> want(n);
        cfill(&c, want.data(), n);
        dev<T> d(n + 2);
        uint64_t end = launch(key, pos, K, d.p, n, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<T> got = d.host();
        CHECK(end == tandem_position(&c));
        if (std::memcmp(got.data(), want.data(), n * sizeof(T)) != 0) {
            std::printf("FAIL %s exponential fill differs (trial %d K=%u pos=%llu n=%zu)\n", label,
                        trial, K, (unsigned long long)pos, n);
            failures++;
        }
    }
}

static void test_exponential() {
    std::mt19937_64 gen(2718);
    check_exponential<double, tandem_fill_exponential_f64, 64, tandem::fill_exponential_f64>(gen, "f64");
    check_exponential<float, tandem_fill_exponential_f32, 32, tandem::fill_exponential_f32>(gen, "f32");
}

// The exponential fixtures of this repository and of tandem-c, bit for bit on the device.
template <class T, class F, uint64_t (*launch)(const uint32_t *, uint64_t, uint32_t, T *, size_t, cudaStream_t)>
static void check_cross_exponential(const F &f) {
    dev<T> d(f.n);
    launch(CROSS_FILL_KEY, f.pos, 32, d.p, f.n, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(std::memcmp(d.host().data(), f.out, f.n * sizeof(T)) == 0);
}

static void test_cross_exponential() {
    for (const auto &f : CROSS_EXP64)
        check_cross_exponential<double, cross_exp64, tandem::fill_exponential_f64>(f);
    for (const auto &f : CROSS_EXP32)
        check_cross_exponential<float, cross_exp32, tandem::fill_exponential_f32>(f);
    // tandem-c's tests/cross_exponential.h: the Rng(42) exponentials from each start position.
    for (const auto &f : CROSS_EXPONENTIAL) {
        dev<double> d(CROSS_EXPONENTIAL_COUNT);
        CHECK(tandem::fill_exponential_f64(CROSS_FILL_KEY, f.start, 32, d.p, CROSS_EXPONENTIAL_COUNT) == f.end_pos);
        CUDA_CHECK(cudaDeviceSynchronize());
        CHECK(std::memcmp(d.host().data(), f.want, sizeof f.want) == 0);
    }
    for (const auto &f : CROSS_EXPONENTIALF) {
        dev<float> d(CROSS_EXPONENTIAL_COUNT);
        CHECK(tandem::fill_exponential_f32(CROSS_FILL_KEY, f.start, 32, d.p, CROSS_EXPONENTIAL_COUNT) == f.end_pos);
        CUDA_CHECK(cudaDeviceSynchronize());
        CHECK(std::memcmp(d.host().data(), f.want, sizeof f.want) == 0);
    }
}

// Successive generator fills continue one stream: the same values and positions as the C
// generator makes through the same sequence of fills, with every width mixed.
static void test_generator() {
    const uint32_t key[4] = {5, 6, 7, 8};
    tandem::generator g = tandem::generator::from_key(key, 3, 16);
    tandem_rng c = tandem_from_key(key, 3, 16);
    const size_t n = 1000;

    dev<uint32_t> d32(n + 2);
    dev<uint64_t> d64(n + 2);
    dev<uint8_t> d8(n + 2);
    dev<double> dz(n + 2);
    dev<float> df(n + 2);
    g.fill_u32(d32.p, 37);
    g.fill_u8(d8.p, 13);
    g.fill_f64(dz.p, 21);
    g.fill_u64(d64.p, 5);
    g.fill_f32(df.p, 9);
    g.fill_u32_below(6, d32.p + 100, 50);
    g.fill_normal_f64(dz.p + 100, 77);
    g.fill_exponential_f32(df.p + 100, 19);
    g.fill_exponential_f64(dz.p + 200, 23);
    g.fill_u16(reinterpret_cast<uint16_t *>(d8.p + 64), 11);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> w32(n);
    std::vector<uint64_t> w64(n);
    std::vector<uint8_t> w8(n);
    std::vector<double> wz(n);
    std::vector<float> wf(n);
    std::vector<uint16_t> w16(n);
    tandem_fill_u32(&c, w32.data(), 37);
    tandem_fill_u8(&c, w8.data(), 13);
    tandem_fill_f64(&c, wz.data(), 21);
    tandem_fill_u64(&c, w64.data(), 5);
    tandem_fill_f32(&c, wf.data(), 9);
    tandem_fill_u32_below(&c, w32.data() + 100, 50, 6);
    {   // 77 normals from 77 UInt64 draws
        uint64_t g0 = tandem::align_pos(tandem_position(&c), 64) >> 6;
        std::vector<uint64_t> u(77);
        tandem_fill_u64(&c, u.data(), 77);
        for (size_t e = 0; e < 77; e++) wz[100 + e] = tandem::normal_f64(u[e], key, 16, g0 + e);
    }
    tandem_fill_exponential_f32(&c, wf.data() + 100, 19);
    tandem_fill_exponential_f64(&c, wz.data() + 200, 23);
    tandem_fill_u16(&c, w16.data(), 11);

    auto h32 = d32.host();
    auto h8 = d8.host();
    auto h64 = d64.host();
    auto hz = dz.host();
    auto hf = df.host();
    CHECK(std::memcmp(h32.data(), w32.data(), 37 * 4) == 0);
    CHECK(std::memcmp(h32.data() + 100, w32.data() + 100, 50 * 4) == 0);
    CHECK(std::memcmp(h8.data(), w8.data(), 13) == 0);
    CHECK(std::memcmp(h64.data(), w64.data(), 5 * 8) == 0);
    CHECK(std::memcmp(hf.data(), wf.data(), 9 * 4) == 0);
    CHECK(std::memcmp(hz.data(), wz.data(), 21 * 8) == 0);
    CHECK(std::memcmp(h8.data() + 64, w16.data(), 22) == 0);
    CHECK(std::memcmp(hz.data() + 100, wz.data() + 100, 77 * 8) == 0);
    CHECK(std::memcmp(hf.data() + 100, wf.data() + 100, 19 * 4) == 0);
    CHECK(std::memcmp(hz.data() + 200, wz.data() + 200, 23 * 8) == 0);
    CHECK(g.pos == tandem_position(&c));
    CHECK(g.K == 16);

    // A seeded generator has the key of device_rng::seed.
    tandem::generator s = tandem::generator::seed(42, 0, 32);
    CHECK(words_equal(s.key, tandem::device_rng::seed(42, 0, 32).key));
}

// An empty f32 normal, exponential or bounded fill leaves an unaligned position alone. An empty f64
// normal fill aligns it to 64 bits, as an empty uniform fill does (spec section 5).
static void test_empty_fills() {
    const uint32_t key[4] = {1, 2, 3, 4};
    dev<float> f(4);
    dev<double> d(4);
    dev<uint32_t> u(4);
    dev<uint64_t> w(4);
    for (uint64_t pos : {0ull, 1ull, 33ull, 64ull, 65ull, 1001ull}) {
        CHECK(tandem::fill_normal_f64(key, pos, 32, d.p, 0) == tandem::align_pos(pos, 64));
        CHECK(tandem::fill_normal_f32(key, pos, 32, f.p, 0) == pos);
        CHECK(tandem::fill_exponential_f64(key, pos, 32, d.p, 0) == pos);
        CHECK(tandem::fill_exponential_f32(key, pos, 32, f.p, 0) == pos);
        CHECK(tandem::fill_u32_below(key, pos, 32, 6, u.p, 0) == pos);
        CHECK(tandem::fill_u64_below(key, pos, 32, 6, w.p, 0) == pos);
    }
    tandem::generator g = tandem::generator::from_key(key, 65, 32);
    g.fill_exponential_f32(f.p, 0);
    g.fill_u32_below(6, u.p, 0);
    CHECK(g.pos == 65);
    g.fill_normal_f64(d.p, 0);
    CHECK(g.pos == 128);
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
    test_device_api();
    test_below();
    test_below_low();
    test_cross_below();
    test_cut_below();
    test_normal();
    test_normal64();
    test_normal64_octets();
    test_normal_reference();
    test_cut_normal();
    test_normal_streams();
    test_cross_normal();
    test_exponential();
    test_cross_exponential();
    test_generator();
    test_empty_fills();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::puts("cuda: ok");
    return 0;
}
