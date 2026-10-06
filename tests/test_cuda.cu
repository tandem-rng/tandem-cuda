// Spec vectors, the spec's conformance cases in tests/conformance, and agreement with the C library
// at random keys and positions. Run on a GPU host: make test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../tandem.cuh"
#include "vectors.h"
#include "conformance.h"

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

static const uint32_t KEY1234[4] = {1, 2, 3, 4};

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

// Two-kernel normal fills on two streams at once, whose miss lists the library keeps. Each round
// leaves a list that the other stream's next fill may take, and the sizes grow and shrink, so
// fills take kept lists, lists in flight on their own stream, and new ones.
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
    const size_t len[4] = {n, n / 8, n, n / 2}; /* fill s of round r writes len[r] elements */
    for (int r = 0; r < 4; r++)
        for (int s = 0; s < 2; s++)
            tandem::fill_normal_f64(keys[s], 64 * s, 32, out.p + (4 * s + r) * n, len[r], st[s]);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> got = out.host();
    for (int s = 0; s < 2; s++)
        for (int r = 0; r < 4; r++)
            CHECK(std::memcmp(got.data() + (4 * s + r) * n, want[s].data(), len[r] * 8) == 0);
    for (auto x : st) CUDA_CHECK(cudaStreamDestroy(x));
}

// A fill captured into a graph takes its list from the stream-ordered allocator, so each launch of
// the graph has its own.
static void test_normal_capture() {
    const size_t n = (size_t)1 << 18;
    const uint32_t key[4] = {1, 2, 3, 4};
    tandem_rng c = tandem_from_key(key, 64, 32);
    std::vector<double> want(n);
    tandem_fill_normal_f64(&c, want.data(), n);
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    dev<double> out(2 * n);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    tandem::fill_normal_f64(key, 64, 32, out.p, n, st);
    tandem::fill_normal_f64(key, 64, 32, out.p + n, n, st);
    CUDA_CHECK(cudaStreamEndCapture(st, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    for (int launch = 0; launch < 2; launch++) {
        CUDA_CHECK(cudaMemsetAsync(out.p, 0, 2 * n * 8, st));
        CUDA_CHECK(cudaGraphLaunch(exec, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        std::vector<double> got = out.host();
        CHECK(std::memcmp(got.data(), want.data(), n * 8) == 0);
        CHECK(std::memcmp(got.data() + n, want.data(), n * 8) == 0);
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(st));
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

__global__ void exponential_f32_kernel(float *out) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    out[k] = tandem::exponential_f32(tandem::to_f32(k << 8));
}

// The f32 map of spec Appendix A as tandem-c writes it. The device's division without its range
// check and the folded powers of two must give its bits for every one of the 2^24 Float32 draws,
// on the device and on the host.
static void test_exponential_f32_all() {
    const uint32_t n = 1u << 24;
    dev<float> d(n);
    exponential_f32_kernel<<<n / 256, 256>>>(d.p);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got = d.host();
    uint32_t bad_dev = 0, bad_host = 0;
    for (uint32_t k = 0; k < n; k++) {
        float x = 1.0f - tandem::to_f32(k << 8), m, nk;
        uint32_t ix;
        std::memcpy(&ix, &x, 4);
        ix += 0x004afb0du;
        nk = (float)(127 - (int32_t)(ix >> 23));
        ix = (ix & 0x007fffffu) + 0x3f3504f3u;
        std::memcpy(&m, &ix, 4);
        float s = (m - 1.0f) / (m + 1.0f), zz = s * s;
        float p = std::fma(zz, std::fma(zz, std::fma(zz, 0.14275366f, 0.20000061f), 0.33333334f), 1.0f);
        float want = 0.5f * std::fma(nk, 2.857213530660374e-06f, std::fma(nk, 1.38629150390625f, (s * -4.0f) * p));
        bad_dev += std::memcmp(&got[k], &want, 4) != 0;
        float host = tandem::exponential_f32(tandem::to_f32(k << 8));
        bad_host += std::memcmp(&host, &want, 4) != 0;
    }
    CHECK(bad_dev == 0);
    CHECK(bad_host == 0);
}

__global__ void radius_sqrt_kernel(uint32_t *bad) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    float x = -2.0f * logf(1.0f - tandem::to_f32(k << 8));
    if (__float_as_uint(__fsqrt_rn(x)) != __float_as_uint(tandem::detail::sqrt_rn_nonneg(x)))
        atomicAdd(bad, 1u);
}

// The f32 normal's square root without the range check equals the IEEE square root on the radius
// of every one of the 2^24 Float32 draws, zero included.
static void test_radius_sqrt_all() {
    dev<uint32_t> bad(1);
    CUDA_CHECK(cudaMemset(bad.p, 0, 4));
    radius_sqrt_kernel<<<(1u << 24) / 256, 256>>>(bad.p);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(bad.host()[0] == 0);
}

static void test_exponential() {
    std::mt19937_64 gen(2718);
    check_exponential<double, tandem_fill_exponential_f64, 64, tandem::fill_exponential_f64>(gen, "f64");
    check_exponential<float, tandem_fill_exponential_f32, 32, tandem::fill_exponential_f32>(gen, "f32");
    test_exponential_f32_all();
}

// ---- Weighted choice ----------------------------------------------------------------------------

// A table built on the host, its arrays copied to the device, and the C library's table of the same
// weights.
struct DeviceChoice {
    std::vector<uint64_t> ccut;
    std::vector<uint32_t> calias;
    tandem_choice_table c;
    dev<uint64_t> cut;
    dev<uint32_t> alias;
    tandem::ChoiceTable t;

    explicit DeviceChoice(const std::vector<double> &w)
        : ccut(w.size()), calias(w.size()), cut(w.size()), alias(w.size()) {
        std::vector<uint64_t> hc(w.size());
        std::vector<uint32_t> ha(w.size());
        CHECK(tandem::choice_build(t, w.data(), w.size(), hc.data(), ha.data()));
        CHECK(tandem_choice_build(&c, w.data(), w.size(), ccut.data(), calias.data()));
        CUDA_CHECK(cudaMemcpy(cut.p, hc.data(), w.size() * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(alias.p, ha.data(), w.size() * 4, cudaMemcpyHostToDevice));
        t.cut = cut.p;
        t.alias = alias.p;
    }
};

__global__ void choice_kernel(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, uint64_t pos,
                              uint32_t K, tandem::ChoiceTable t, uint32_t *out, size_t n,
                              uint64_t *end) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng r = tandem::device_rng::from_key(key, pos, K);
    for (size_t i = 0; i < n; i++) out[i] = r.choice(t);
    *end = r.pos;
}

// Element i of a choice fill maps UInt64 draw i through the table, so the device fill equals
// tandem_fill_choice at every K and start, through the tile and the direct kernel, with outputs on
// and off 16-byte alignment, the end positions too. The device_rng draws equal tandem_choice.
static void test_choice() {
    std::mt19937_64 gen(4242);
    for (int trial = 0; trial < 40; trial++) {
        uint32_t key[4];
        for (auto &w : key) w = (uint32_t)gen();
        uint32_t K = 1u << (gen() % 8);
        uint64_t pos = gen() % (1u << 20);
        size_t n = (size_t)(gen() % (trial < 30 ? 5000 : 300000)), shift = gen() % 4;
        std::vector<double> w(1 + gen() % (trial % 2 ? 3000 : 12));
        for (auto &x : w) x = gen() % 4 ? (double)(gen() >> 11) * 0x1p-40 : 0.0;
        w[gen() % w.size()] = 0.5;
        DeviceChoice tb(w);
        tandem_rng c = tandem_from_key(key, pos, K);
        std::vector<uint32_t> want(n);
        tandem_fill_choice(&c, want.data(), n, &tb.c);
        for (bool tile : {true, false}) {
            dev<uint32_t> d(n + 4);
            uint64_t end = tandem::detail::fill<tandem::detail::choice_idx>(
                key, pos, K, d.p + shift, n, 0, tile, tandem::detail::Bound{0, 0, 0, tb.t});
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<uint32_t> got(n);
            CUDA_CHECK(cudaMemcpy(got.data(), d.p + shift, n * 4, cudaMemcpyDeviceToHost));
            CHECK(end == tandem_position(&c));
            if (got != want) {
                size_t i = 0;
                while (got[i] == want[i]) i++;
                std::printf("FAIL choice fill vs C (%s, K=%u pos=%llu n=%zu m=%zu) element %zu\n",
                            tile ? "tile" : "direct", K, (unsigned long long)pos, n, w.size(), i);
                failures++;
            }
        }
        if (n > 512) continue;
        dev<uint32_t> d(n + 1);
        dev<uint64_t> end(1);
        choice_kernel<<<1, 1>>>(key[0], key[1], key[2], key[3], pos, K, tb.t, d.p, n, end.p);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint32_t> got = d.host();
        got.resize(n);
        tandem_rng s = tandem_from_key(key, pos, K);
        std::vector<uint32_t> seq(n);
        for (auto &v : seq) v = tandem_choice(&s, &tb.c);
        CHECK(got == seq && end.host()[0] == tandem_position(&s));
    }
    // The generator handle cuts a fill anywhere into pieces that equal the whole fill.
    DeviceChoice tb({3, 0, 1, 7.5, 0.125});
    const size_t n = 100000;
    for (size_t k : {(size_t)1, (size_t)7, (size_t)20, (size_t)21, (size_t)65537, n - 1}) {
        tandem::generator w = tandem::generator::from_key(KEY1234, 4321, 32), v = w;
        dev<uint32_t> a(n), b(n);
        w.fill_choice(tb.t, a.p, n);
        v.fill_choice(tb.t, b.p, k);
        v.fill_choice(tb.t, b.p + k, n - k);
        CUDA_CHECK(cudaDeviceSynchronize());
        CHECK(a.host() == b.host() && w.pos == v.pos);
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
// normal or choice fill aligns it to 64 bits, as an empty uniform fill does (spec section 5).
static void test_empty_fills() {
    const uint32_t key[4] = {1, 2, 3, 4};
    dev<float> f(4);
    dev<double> d(4);
    dev<uint32_t> u(4);
    dev<uint64_t> w(4);
    DeviceChoice tb({1, 2});
    for (uint64_t pos : {0ull, 1ull, 33ull, 64ull, 65ull, 1001ull}) {
        CHECK(tandem::fill_normal_f64(key, pos, 32, d.p, 0) == tandem::align_pos(pos, 64));
        CHECK(tandem::fill_choice(key, pos, 32, tb.t, u.p, 0) == tandem::align_pos(pos, 64));
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

// ---- Conformance cases of the spec ---------------------------------------------------------------

// The scalar draws of the conformance kinds on a device_rng, as bit patterns, and its end position.
enum class Draw { below32, below64, normal64, normal32, exp64, exp32, choice };

__global__ void scalar_kernel(Draw kind, uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3,
                              uint64_t pos, uint32_t K, uint64_t range, tandem::ChoiceTable t,
                              uint64_t *out, size_t n, uint64_t *end) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng r = tandem::device_rng::from_key(key, pos, K);
    for (size_t i = 0; i < n; i++) {
        switch (kind) {
        case Draw::below32: out[i] = r.urand((uint32_t)range); break;
        case Draw::below64: out[i] = r.urand64(range); break;
        case Draw::normal64: out[i] = (uint64_t)__double_as_longlong(r.normal()); break;
        case Draw::normal32:
            if (i + 1 < n) {
                tandem::Pair2<float> p = r.normalf2();
                out[i] = __float_as_uint(p.z0);
                out[++i] = __float_as_uint(p.z1);
            } else {
                out[i] = __float_as_uint(r.normalf());
            }
            break;
        case Draw::exp64: out[i] = (uint64_t)__double_as_longlong(r.exponential()); break;
        case Draw::exp32: out[i] = __float_as_uint(r.exponentialf()); break;
        case Draw::choice: out[i] = r.choice(t); break;
        }
    }
    *end = r.pos;
}

static std::vector<uint64_t> device_scalars(Draw kind, const conf::Case &c, const tandem::ChoiceTable &t,
                                            uint64_t &end) {
    dev<uint64_t> d(c.n + 1), e(1);
    scalar_kernel<<<1, 1>>>(kind, c.key[0], c.key[1], c.key[2], c.key[3], c.start, c.K, c.range, t,
                            d.p, c.n, e.p);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = e.host()[0];
    std::vector<uint64_t> v = d.host();
    v.resize(c.n);
    return v;
}

static size_t elem_size(const std::string &kind) {
    return kind.find("64") != std::string::npos ? 8 : 4;
}

// n elements of the fill of case c on generator g into device memory.
static void case_fill(tandem::generator &g, const conf::Case &c, const tandem::ChoiceTable &t,
                      void *out, size_t n) {
    if (c.kind == "fill_below_u32") g.fill_u32_below((uint32_t)c.range, static_cast<uint32_t *>(out), n);
    else if (c.kind == "fill_below_u64") g.fill_u64_below(c.range, static_cast<uint64_t *>(out), n);
    else if (c.kind == "fill_normal_f64") g.fill_normal_f64(static_cast<double *>(out), n);
    else if (c.kind == "fill_normal_f32") g.fill_normal_f32(static_cast<float *>(out), n);
    else if (c.kind == "fill_exponential_f64") g.fill_exponential_f64(static_cast<double *>(out), n);
    else if (c.kind == "fill_exponential_f32") g.fill_exponential_f32(static_cast<float *>(out), n);
    else g.fill_choice(t, static_cast<uint32_t *>(out), n);
}

// The first n elements of a device buffer as bit patterns.
static std::vector<uint64_t> patterns(const dev<uint64_t> &d, size_t n, size_t size) {
    std::vector<unsigned char> b(n * size);
    CUDA_CHECK(cudaMemcpy(b.data(), d.p, b.size(), cudaMemcpyDeviceToHost));
    std::vector<uint64_t> v(n, 0);
    for (size_t i = 0; i < n; i++) std::memcpy(&v[i], &b[i * size], size);
    return v;
}

// Float32 normals within the spec's 16 ulps plus an absolute floor, everything else bit for bit.
static bool same_values(const conf::Case &c, const std::vector<uint64_t> &got, double abs) {
    if (got.size() != c.n) return false;
    for (size_t i = 0; i < c.n; i++) {
        if (c.kind != "fill_normal_f32") {
            if (got[i] != c.values[i]) return false;
            continue;
        }
        float g;
        uint32_t b = (uint32_t)got[i];
        std::memcpy(&g, &b, 4);
        double want = c.f32(i);
        if (!(std::fabs((double)g - want) <= 16 * 0x1p-23 * std::fabs(want) + abs)) return false;
    }
    return true;
}

// Every case of fill_below, normal, exponential and choice.json through the generator's device
// fills: values, end, and for n = 0 no write. The fallback of a rejected bounded element or a
// missed Float64 normal is keyed by the global draw index, which the shifted-start pairs show on
// the device's own output. Each case cut at elements 1, 7, 20, 21 and n - 1 equals the whole
// fill. A Float32 normal fill is cut only at pair boundaries, at 2, 8, 20 and the largest even
// element below n, since an odd cut drops a sin half. The scalar device_rng draws equal each normal, exponential and choice fill, end included.
static void test_conformance_fills(const char *dir) {
    std::vector<std::pair<std::string, std::vector<uint64_t>>> out; // device output by id
    for (const char *file : {"fill_below.json", "normal.json", "exponential.json", "choice.json"}) {
        for (const auto &c : conf::cases(dir, file)) {
            const size_t size = elem_size(c.kind);
            std::unique_ptr<DeviceChoice> tb;
            tandem::ChoiceTable t{};
            if (!c.weights.empty()) {
                tb.reset(new DeviceChoice(c.weights));
                t = tb->t;
            }
            tandem::generator g = tandem::generator::from_key(c.key, c.start, c.K);
            dev<uint64_t> whole(c.n + 1);
            CUDA_CHECK(cudaMemset(whole.p, 0xff, 8 * (c.n + 1)));
            case_fill(g, c, t, whole.p, c.n);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<uint64_t> got = patterns(whole, c.n, size);
            if (!same_values(c, got, F32_FILL_ABS) || (c.has_end && g.pos != c.end)) {
                std::printf("FAIL conformance %s: device fill\n", c.id.c_str());
                failures++;
            }
            if (c.n == 0) {
                CHECK(whole.host()[0] == ~0ull);
                continue;
            }
            const bool pairs = c.kind == "fill_normal_f32";
            const std::vector<size_t> cuts = pairs ? std::vector<size_t>{2, 8, 20, (c.n - 1) & ~(size_t)1}
                                                   : std::vector<size_t>{1, 7, 20, 21, c.n - 1};
            for (size_t k : cuts) {
                if (k == 0 || k >= c.n) continue;
                tandem::generator h = tandem::generator::from_key(c.key, c.start, c.K);
                dev<uint64_t> pieces(c.n + 1);
                case_fill(h, c, t, pieces.p, k);
                case_fill(h, c, t, reinterpret_cast<char *>(pieces.p) + k * size, c.n - k);
                CUDA_CHECK(cudaDeviceSynchronize());
                if (patterns(pieces, c.n, size) != got || h.pos != g.pos) {
                    std::printf("FAIL conformance %s: fill cut at %zu\n", c.id.c_str(), k);
                    failures++;
                }
            }
            Draw kind = c.kind == "fill_normal_f64" ? Draw::normal64
                        : c.kind == "fill_normal_f32" ? Draw::normal32
                        : c.kind == "fill_exponential_f64" ? Draw::exp64
                        : c.kind == "fill_exponential_f32" ? Draw::exp32
                        : c.kind == "fill_choice" ? Draw::choice : Draw::below32;
            if (kind != Draw::below32) { // scalar bounded draws discard a rejection, unlike the fill
                uint64_t end;
                std::vector<uint64_t> s = device_scalars(kind, c, t, end);
                if (!same_values(c, s, 1e-6) || end != g.pos) {
                    std::printf("FAIL conformance %s: device scalar draws\n", c.id.c_str());
                    failures++;
                }
            }
            out.emplace_back(c.id, got);
        }
    }
    auto of = [&](const char *suffix) -> const std::vector<uint64_t> & {
        for (const auto &o : out)
            if (o.first.size() >= std::strlen(suffix) &&
                o.first.compare(o.first.size() - std::strlen(suffix), std::string::npos, suffix) == 0)
                return o.second;
        std::printf("FAIL conformance: no case %s\n", suffix);
        std::exit(1);
    };
    // Element i of the later start equals element i + shift of the earlier one.
    const struct {
        const char *later, *earlier;
        size_t shift;
    } shifted[] = {{"CROSS_BELOW32_AT[4]", "CROSS_BELOW32[4]", 1}, {"CROSS_BELOW64_AT[6]", "CROSS_BELOW64[6]", 1},
                   {"CROSS_NORMAL[1]", "CROSS_NORMAL[0]", 1},      {"CROSS_NORMAL32[2]", "CROSS_NORMAL32[0]", 2},
                   {"CROSS_CHOICE[1]", "CROSS_CHOICE[0]", 1}};
    for (const auto &s : shifted) {
        const auto &a = of(s.later), &z = of(s.earlier);
        CHECK(std::equal(a.begin(), a.end() - s.shift, z.begin() + s.shift));
    }
    const auto &nf = of("CROSS_NORMALF"), &n1 = of("CROSS_NORMAL32[1]");
    CHECK(std::equal(n1.begin(), n1.end(), nf.begin()));
}

// below.json on device_rng. Width from range: a 32-bit range into 64-bit elements draws 32 bits per
// element and gives CROSS_BELOW32[3], where the u64 fill of the same range gives CROSS_BELOW64[3].
static void test_conformance_below(const char *dir) {
    for (const auto &c : conf::cases(dir, "below.json")) {
        uint64_t end;
        std::vector<uint64_t> s = device_scalars(c.kind == "below_u32" ? Draw::below32 : Draw::below64, c,
                                                 tandem::ChoiceTable{}, end);
        CHECK(s == c.values && end == c.end);
    }
    auto cs = conf::cases(dir, "fill_below.json");
    const auto &b32 = conf::find(cs, "CROSS_BELOW32[3]");
    dev<uint64_t> d(64);
    CHECK(tandem::fill_u32_below(b32.key, 0, 32, 1000u, (uint64_t)0, d.p, 64) == 64 * 32);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(b32.range == 1000 && d.host() == b32.values);
}

__global__ void at_kernel(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, uint64_t pos,
                          uint32_t K, uint32_t *a32, uint64_t *a64, float *af32, double *af64,
                          size_t n) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng r = tandem::device_rng::from_key(key, pos, K);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        a32[i] = r.at_urand(i);
        a64[i] = r.at_urand64(i);
        af32[i] = r.at_frand(i);
        af64[i] = r.at_drand(i);
    }
}

__global__ void top_kernel(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, uint64_t *out) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng r = tandem::device_rng::from_key(key, ((uint64_t)1 << 63) - 1u, 32);
    out[0] = r.next_u64();
    out[1] = r.pos;
}

// hashes.json on the device: every uniform stream type this library fills, from the public fills
// and from device_rng draws, and the Float64 normal and exponential dumps, whose fills run in
// order on one generator per start. The Float32 normal dump needs bit-exact Box-Muller, which only
// the host block gives, so tests/host_core.cpp checks it. Random access equals the sequential fill
// across blocks, rows and chunks, and a UInt64 draw at 2^63 - 1 aligns to 2^63 and ends at
// 2^63 + 64.
static void test_conformance_hashes(const char *dir) {
    const conf::Json hashes = conf::load(dir, "hashes.json");
    for (const conf::Json &s : hashes["streams"].a) {
        uint32_t key[4];
        for (int w = 0; w < 4; w++) key[w] = (uint32_t)s["key"].a[w].hex();
        const uint64_t pos = s["start"].u();
        const uint32_t K = (uint32_t)s["K"].u();
        const size_t n = (size_t)s["n"].u(), bytes = (size_t)s["bytes"].u();
        const std::string type = s["type"].s;
        dev<unsigned char> d(bytes);
        std::vector<unsigned char> scalar;
        auto from = [](const auto &v) {
            const unsigned char *p = reinterpret_cast<const unsigned char *>(v.data());
            return std::vector<unsigned char>(p, p + v.size() * sizeof(v[0]));
        };
        if (type == "UInt32") {
            tandem::fill_u32(key, pos, K, reinterpret_cast<uint32_t *>(d.p), n);
            scalar = from(device_draws<uint32_t, &tandem::device_rng::next_u32>(key, pos, K, n));
        } else if (type == "UInt64") {
            tandem::fill_u64(key, pos, K, reinterpret_cast<uint64_t *>(d.p), n);
            scalar = from(device_draws<uint64_t, &tandem::device_rng::next_u64>(key, pos, K, n));
        } else if (type == "Float64") {
            tandem::fill_f64(key, pos, K, reinterpret_cast<double *>(d.p), n);
            scalar = from(device_draws<double, &tandem::device_rng::next_f64>(key, pos, K, n));
        } else if (type == "Float32") {
            tandem::fill_f32(key, pos, K, reinterpret_cast<float *>(d.p), n);
            scalar = from(device_draws<float, &tandem::device_rng::next_f32>(key, pos, K, n));
        } else if (type == "UInt8") {
            tandem::fill_u8(key, pos, K, d.p, n);
            scalar = from(device_draws<uint8_t, &tandem::device_rng::next_u8>(key, pos, K, n));
        } else if (type == "Bool") {
            tandem::fill_bool(key, pos, K, reinterpret_cast<bool *>(d.p), n);
            scalar = device_bools(key, pos, K, n);
        } else if (type == "Float16") {
            tandem::fill_f16_bits(key, pos, K, reinterpret_cast<uint16_t *>(d.p), n);
            scalar = from(device_draws<uint16_t, &tandem::device_rng::next_f16_bits>(key, pos, K, n));
        } else {
            continue; // UInt128, Char and complex types have no fill here
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned char> fill = d.host();
        CHECK(conf::sha256(fill.data(), bytes) == s["sha256"].s);
        CHECK(scalar.size() == bytes && conf::sha256(scalar.data(), bytes) == s["sha256"].s);
    }
    for (const conf::Json &dm : hashes["dumps"].a) {
        bool f32_normal = false;
        for (const conf::Json &f : dm["draws"].a) f32_normal |= f["kind"].s == "fill_normal_f32";
        if (f32_normal) continue;
        uint32_t key[4];
        for (int w = 0; w < 4; w++) key[w] = (uint32_t)dm["key"].a[w].hex();
        uint64_t h = conf::FNV_BASIS, bytes = 0, end = 0;
        for (const conf::Json &st : dm["starts"].a) {
            tandem::generator g = tandem::generator::from_key(key, st.u(), (uint32_t)dm["K"].u());
            for (const conf::Json &f : dm["draws"].a) {
                const size_t n = (size_t)f["n"].u(), size = elem_size(f["kind"].s);
                dev<unsigned char> d(n * size);
                if (f["kind"].s == "fill_normal_f64") g.fill_normal_f64(reinterpret_cast<double *>(d.p), n);
                else if (f["kind"].s == "fill_exponential_f64") g.fill_exponential_f64(reinterpret_cast<double *>(d.p), n);
                else g.fill_exponential_f32(reinterpret_cast<float *>(d.p), n);
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<unsigned char> b = d.host();
                h = conf::fnv1a(h, b.data(), b.size());
                bytes += b.size();
            }
            end = g.pos;
        }
        CHECK(bytes == dm["bytes"].u() && h == dm["fnv1a"].hex());
        if (const conf::Json *e = dm.find("end")) CHECK(end == e->u());
    }

    // K = 8 groups span 8192 bits, so 3000 draws from just before a group cross every boundary.
    for (uint64_t pos : {8192ull - 100, 3ull * 8192 + 4000}) {
        const size_t n = 3000;
        dev<uint32_t> a32(n);
        dev<uint64_t> a64(n);
        dev<float> af32(n);
        dev<double> af64(n);
        at_kernel<<<12, 256>>>(KEY1234[0], KEY1234[1], KEY1234[2], KEY1234[3], pos, 8, a32.p, a64.p,
                               af32.p, af64.p, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CHECK(a32.host() == device_fill<uint32_t>(KEY1234, pos, 8, n));
        CHECK(a64.host() == device_fill<uint64_t>(KEY1234, pos, 8, n));
        CHECK(af32.host() == device_fill<float>(KEY1234, pos, 8, n));
        CHECK(af64.host() == device_fill<double>(KEY1234, pos, 8, n));
    }
    dev<uint64_t> top(2);
    top_kernel<<<1, 1>>>(KEY1234[0], KEY1234[1], KEY1234[2], KEY1234[3], top.p);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint64_t> t = top.host(), want = device_fill<uint64_t>(KEY1234, ((uint64_t)1 << 63) - 64, 32, 2);
    CHECK(t[0] == want[1] && t[1] == ((uint64_t)1 << 63) + 64);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/conformance";
    test_vectors();
    test_conformance_fills(dir);
    test_conformance_below(dir);
    test_conformance_hashes(dir);
    test_against_c();
    test_signed();
    test_device_api();
    test_below();
    test_below_low();
    test_cut_below();
    test_normal();
    test_radius_sqrt_all();
    test_normal64();
    test_normal64_octets();
    test_cut_normal();
    test_normal_streams();
    test_normal_capture();
    test_exponential();
    test_choice();
    test_generator();
    test_empty_fills();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::puts("cuda: ok");
    return 0;
}
