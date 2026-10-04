/* Tandem8x32 core: the step, the seeding function, the stream layout, the float mappings,
 * child keys, and a scalar generator. Portable C++17 without Kokkos or CUDA types, so CUDA,
 * HIP, SYCL and host code can share it. tandem.cuh builds on it.
 *
 * Implements https://github.com/tandem-rng/spec and produces the stream it defines, bit for
 * bit. Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(KOKKOS_VERSION)
#define TANDEM_FN KOKKOS_INLINE_FUNCTION
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define TANDEM_FN __host__ __device__ inline
#else
#define TANDEM_FN inline
#endif

namespace tandem {

constexpr uint32_t DEFAULT_K = 32;

constexpr uint32_t CLOCK_WEYL = 0x9e3779b9u;
constexpr uint32_t DOMAIN_STREAM = 0x9e3779b9u;
constexpr uint32_t DOMAIN_SPLIT = 0xbb67ae85u;
constexpr uint32_t DOMAIN_FORK = 0xd2511f53u;
constexpr uint32_t DOMAIN_FOLD = 0xcd9e8d57u;
constexpr uint32_t DOMAIN_SEED = 0xa54ff53au;
constexpr uint32_t AUX_STREAM = 0x94d049bbu;

TANDEM_FN constexpr uint32_t round_constant(int r) {
    constexpr uint32_t rc[8] = {0xd17cc1b7u, 0xa7220a94u, 0xfe13abe8u, 0xfa9a6ee0u,
                                0xedb14accu, 0x9e21c820u, 0xff28b1d5u, 0xef5de2b0u};
    return rc[r];
}

TANDEM_FN uint32_t rotl(uint32_t x, unsigned r) { return (x << r) | (x >> (32u - r)); }

/* The step T: mix the exposed half, clock the hidden half, feed o0 back into h0. */
TANDEM_FN void T(uint32_t o[4], uint32_t h[4]) {
    uint64_t p0 = (uint64_t)o[0] * (h[0] | 1u);
    uint64_t p1 = (uint64_t)o[2] * (h[1] | 1u);
    uint32_t lo0 = (uint32_t)p0, hi0 = (uint32_t)(p0 >> 32);
    uint32_t lo1 = (uint32_t)p1, hi1 = (uint32_t)(p1 >> 32);
    uint32_t n0 = o[1] ^ hi1 ^ lo1;
    uint32_t n1 = rotl(lo1, 16) ^ h[2];
    uint32_t n2 = o[3] ^ hi0 ^ lo0;
    uint32_t n3 = rotl(lo0, 16) ^ h[3];

    h[0] ^= rotl(h[1], 7);
    h[1] ^= rotl(h[2], 13);
    h[2] ^= rotl(h[3], 22);
    h[3] ^= rotl(h[0], 3);
    h[0] = (h[0] + CLOCK_WEYL) ^ n0;

    o[0] = n0;
    o[1] = n1;
    o[2] = n2;
    o[3] = n3;
}

/* The seeding function F: eight rounds of T, a round constant, and a half swap. */
TANDEM_FN void F(uint32_t o[4], uint32_t h[4]) {
    for (int r = 0; r < 8; r++) {
        T(o, h);
        o[0] ^= round_constant(r);
        for (int w = 0; w < 4; w++) {
            uint32_t t = o[w];
            o[w] = h[w];
            h[w] = t;
        }
    }
}

TANDEM_FN void F_keyed(const uint32_t key[4], uint64_t counter, uint32_t domain, uint32_t aux,
                       uint32_t o[4], uint32_t h[4]) {
    o[0] = (uint32_t)counter;
    o[1] = (uint32_t)(counter >> 32);
    o[2] = domain;
    o[3] = aux;
    for (int w = 0; w < 4; w++)
        h[w] = key[w];
    F(o, h);
}

/* Block B(c, j): the exposed half of chunk c after j + 1 steps. */
TANDEM_FN void block(const uint32_t key[4], uint64_t c, uint32_t j, uint32_t out[4]) {
    uint32_t h[4];
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, out, h);
    for (uint32_t s = 0; s <= j; s++)
        T(out, h);
}

TANDEM_FN uint64_t align_pos(uint64_t pos, unsigned w) {
    return (pos + w - 1u) & ~((uint64_t)w - 1u);
}

TANDEM_FN unsigned log2k(uint32_t K) {
    unsigned s = 0;
    while ((K >> s) > 1u)
        s++;
    return s;
}

/* Chunk of stream bit p: row r = p >> 10 sits in group r / K at step r mod K, and the lane is
 * bits 7..9 of p. */
TANDEM_FN uint64_t chunk_of(uint64_t p, uint32_t K) {
    return 8u * ((p >> 10) >> log2k(K)) + ((p >> 7) & 7u);
}

TANDEM_FN uint32_t step_of(uint64_t p, uint32_t K) { return (uint32_t)((p >> 10) & (K - 1u)); }

TANDEM_FN double to_f64(uint64_t raw) { return (double)(raw >> 11) * 0x1p-53; }
TANDEM_FN float to_f32(uint32_t raw) { return (float)(raw >> 8) * 0x1p-24f; }

/* (raw >> 5) * 2^-11 as binary16 bits. Every such value is zero or a normal half whose
 * significand is the 11-bit integer k = raw >> 5, so the encoding is exact. The binary32
 * of k is exact too and carries the same significand and exponent m, which gives the half's
 * exponent m + 4 after rebiasing, without a loop. */
TANDEM_FN uint16_t to_f16_bits(uint16_t raw) {
    uint32_t k = raw >> 5;
    if (k == 0)
        return 0;
    float f = (float)k;
    uint32_t b;
    std::memcpy(&b, &f, sizeof b);
    return (uint16_t)((b >> 13) - (123u << 10));
}

/* High word of a 64 x 64-bit product, from 32-bit halves. */
TANDEM_FN uint64_t mulhi64(uint64_t a, uint64_t b) {
    uint64_t a0 = (uint32_t)a, a1 = a >> 32, b0 = (uint32_t)b, b1 = b >> 32;
    uint64_t mid = a1 * b0 + ((a0 * b0) >> 32);
    uint64_t mid2 = a0 * b1 + (uint32_t)mid;
    return a1 * b1 + (mid >> 32) + (mid2 >> 32);
}

/* The eight chunks of group g, stepped together. After step j the exposed half of lane l is
 * block B(8g + l, j), and the eight blocks in lane order are stream row g K + j.
 *
 * Host compilers with GCC vector extensions (GCC 12+, clang) hold each word of four lanes in
 * one 128-bit vector, so a step compiles to NEON or SSE/AVX and a row leaves registers by a
 * 4x4 transpose. They do not vectorize the scalar loop below on their own. Device compilers,
 * and TANDEM_NO_SIMD, take the scalar loop. */
#if !defined(TANDEM_NO_SIMD) && !defined(__CUDACC__) && !defined(__HIPCC__) &&                     \
    !defined(SYCL_LANGUAGE_VERSION) &&                                                             \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 12))
#define TANDEM_ROW_SIMD 1
#endif

#ifdef TANDEM_ROW_SIMD
struct Row {
    typedef uint32_t u32x4 __attribute__((vector_size(16)));
    typedef uint64_t u64x4 __attribute__((vector_size(32)));

    u32x4 o[2][4], h[2][4]; /* [lanes 0-3 or 4-7][word] */

    static u32x4 rot(u32x4 x, unsigned r) { return (x << r) | (x >> (32u - r)); }

    static void quad_step(u32x4 *o, u32x4 *h) {
        u64x4 p0 = __builtin_convertvector(o[0], u64x4) * __builtin_convertvector(h[0] | 1u, u64x4);
        u64x4 p1 = __builtin_convertvector(o[2], u64x4) * __builtin_convertvector(h[1] | 1u, u64x4);
        u32x4 lo0 = __builtin_convertvector(p0, u32x4),
              hi0 = __builtin_convertvector(p0 >> 32, u32x4);
        u32x4 lo1 = __builtin_convertvector(p1, u32x4),
              hi1 = __builtin_convertvector(p1 >> 32, u32x4);
        u32x4 n0 = o[1] ^ hi1 ^ lo1;
        u32x4 n1 = rot(lo1, 16) ^ h[2];
        u32x4 n2 = o[3] ^ hi0 ^ lo0;
        u32x4 n3 = rot(lo0, 16) ^ h[3];
        h[0] ^= rot(h[1], 7);
        h[1] ^= rot(h[2], 13);
        h[2] ^= rot(h[3], 22);
        h[3] ^= rot(h[0], 3);
        h[0] = (h[0] + CLOCK_WEYL) ^ n0;
        o[0] = n0;
        o[1] = n1;
        o[2] = n2;
        o[3] = n3;
    }

    void step() {
        quad_step(o[0], h[0]);
        quad_step(o[1], h[1]);
    }

    /* F on the eight chunks at once. Lanes differ only in the low counter word, and 8g has
     * three zero low bits, so lane l's counter is (low word of 8g) + l without carry. */
    void seed(const uint32_t key[4], uint64_t g) {
        uint32_t lo = (uint32_t)(8u * g), hi = (uint32_t)((8u * g) >> 32);
        for (int q = 0; q < 2; q++) {
            uint32_t c = lo + 4u * q;
            u32x4 zero = {0, 0, 0, 0};
            o[q][0] = u32x4{c, c + 1u, c + 2u, c + 3u};
            o[q][1] = zero + hi;
            o[q][2] = zero + DOMAIN_STREAM;
            o[q][3] = zero + AUX_STREAM;
            for (int w = 0; w < 4; w++)
                h[q][w] = zero + key[w];
            for (int r = 0; r < 8; r++) {
                quad_step(o[q], h[q]);
                o[q][0] ^= round_constant(r);
                for (int w = 0; w < 4; w++) {
                    u32x4 t = o[q][w];
                    o[q][w] = h[q][w];
                    h[q][w] = t;
                }
            }
        }
    }

    /* The exposed half of lane l. */
    void lane(unsigned l, uint32_t w[4]) const {
        for (int k = 0; k < 4; k++)
            w[k] = o[l >> 2][k][l & 3u];
    }

    /* The row's eight blocks in stream order: per quad, a 4x4 word transpose. */
    void blocks(u32x4 b[8]) const {
        for (int q = 0; q < 2; q++) {
            const u32x4 *v = o[q];
            u32x4 t0 = __builtin_shufflevector(v[0], v[1], 0, 4, 1, 5);
            u32x4 t1 = __builtin_shufflevector(v[2], v[3], 0, 4, 1, 5);
            u32x4 t2 = __builtin_shufflevector(v[0], v[1], 2, 6, 3, 7);
            u32x4 t3 = __builtin_shufflevector(v[2], v[3], 2, 6, 3, 7);
            b[4 * q + 0] = __builtin_shufflevector(t0, t1, 0, 1, 4, 5);
            b[4 * q + 1] = __builtin_shufflevector(t0, t1, 2, 3, 6, 7);
            b[4 * q + 2] = __builtin_shufflevector(t2, t3, 0, 1, 4, 5);
            b[4 * q + 3] = __builtin_shufflevector(t2, t3, 2, 3, 6, 7);
        }
    }

    /* The row as 32 words, 32 Float32 or 16 Float64 values, written to unaligned memory. */
    void store(uint32_t *out) const {
        u32x4 b[8];
        blocks(b);
        __builtin_memcpy(out, b, sizeof b);
    }
    void store(float *out) const {
        typedef float f32x4 __attribute__((vector_size(16)));
        u32x4 b[8];
        f32x4 f[8];
        blocks(b);
        for (int i = 0; i < 8; i++)
            f[i] = __builtin_convertvector(b[i] >> 8, f32x4) * 0x1p-24f;
        __builtin_memcpy(out, f, sizeof f);
    }
    void store(double *out) const {
        typedef uint64_t u64x2 __attribute__((vector_size(16)));
        typedef double f64x2 __attribute__((vector_size(16)));
        u32x4 b[8];
        f64x2 f[8];
        blocks(b);
        for (int i = 0; i < 8; i++) {
            u64x2 w;
            __builtin_memcpy(&w, &b[i], 16);
            f[i] = __builtin_convertvector(w >> 11, f64x2) * 0x1p-53;
        }
        __builtin_memcpy(out, f, sizeof f);
    }
};
#else
struct Row {
    uint32_t o[4][8], h[4][8]; /* [word][lane] */

    TANDEM_FN void step() {
        for (int l = 0; l < 8; l++) {
            uint32_t a[4] = {o[0][l], o[1][l], o[2][l], o[3][l]};
            uint32_t b[4] = {h[0][l], h[1][l], h[2][l], h[3][l]};
            T(a, b);
            for (int w = 0; w < 4; w++) {
                o[w][l] = a[w];
                h[w][l] = b[w];
            }
        }
    }

    TANDEM_FN void seed(const uint32_t key[4], uint64_t g) {
        for (int l = 0; l < 8; l++) {
            uint32_t a[4], b[4];
            F_keyed(key, 8u * g + (uint64_t)l, DOMAIN_STREAM, AUX_STREAM, a, b);
            for (int w = 0; w < 4; w++) {
                o[w][l] = a[w];
                h[w][l] = b[w];
            }
        }
    }

    TANDEM_FN void lane(unsigned l, uint32_t w[4]) const {
        for (int k = 0; k < 4; k++)
            w[k] = o[k][l];
    }

    TANDEM_FN void store(uint32_t *out) const {
        uint32_t w[32];
        for (int l = 0; l < 8; l++)
            for (int k = 0; k < 4; k++)
                w[4 * l + k] = o[k][l];
        std::memcpy(out, w, sizeof w);
    }
    TANDEM_FN void store(float *out) const {
        for (int l = 0; l < 8; l++)
            for (int k = 0; k < 4; k++)
                out[4 * l + k] = to_f32(o[k][l]);
    }
    TANDEM_FN void store(double *out) const {
        for (int l = 0; l < 8; l++)
            for (int k = 0; k < 2; k++)
                out[2 * l + k] = to_f64(o[2 * k][l] | ((uint64_t)o[2 * k + 1][l] << 32));
    }
};
#endif

struct Key {
    uint32_t w[4];
};

TANDEM_FN bool operator==(const Key &a, const Key &b) {
    return a.w[0] == b.w[0] && a.w[1] == b.w[1] && a.w[2] == b.w[2] && a.w[3] == b.w[3];
}

/* The key of a 128-bit seed given as two halves, whitened as the specification requires. */
TANDEM_FN void seed_key(uint64_t seed_lo, uint64_t seed_hi, uint32_t key[4]) {
    uint32_t h[4] = {(uint32_t)seed_lo, (uint32_t)(seed_lo >> 32), (uint32_t)seed_hi,
                     (uint32_t)(seed_hi >> 32)};
    key[0] = 0;
    key[1] = 0;
    key[2] = DOMAIN_SEED;
    key[3] = 0;
    F(key, h);
}

/* A child key: one half of F_keyed under the parent key. */
TANDEM_FN void child_key(const uint32_t key[4], uint64_t counter, uint32_t domain, uint32_t aux,
                         bool hidden, uint32_t out[4]) {
    uint32_t o[4], h[4];
    F_keyed(key, counter, domain, aux, o, h);
    for (int w = 0; w < 4; w++)
        out[w] = hidden ? h[w] : o[w];
}

/* One cached chunk state, whose exposed half is block B(chunk, step). It is a pure function of
 * the key and K, so the scalar generators keep it beside their transport form. */
struct ChunkCache {
    uint64_t chunk;
    uint32_t live, step;
    uint32_t o[4], h[4];

    /* Bring the cache to the block of stream bit p. Stepping forward inside the cached chunk
     * costs one T per block, any other move reseeds. */
    TANDEM_FN void load(const uint32_t key[4], uint32_t K, uint64_t p) {
        uint64_t c = chunk_of(p, K);
        uint32_t j = step_of(p, K);
        if (!live || c != chunk || j < step) {
            F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
            T(o, h);
            chunk = c;
            step = 0;
            live = 1;
        }
        for (; step < j; step++)
            T(o, h);
    }

    /* w bits (a power of two, 1 to 64) at the aligned position p. */
    TANDEM_FN uint64_t read(const uint32_t key[4], uint32_t K, uint64_t p, unsigned w) {
        load(key, K, p);
        uint32_t lo = o[(p >> 5) & 3u];
        if (w == 64u)
            return lo | ((uint64_t)o[((p >> 5) & 3u) + 1u] << 32);
        return (lo >> (p & 31u)) & (0xffffffffu >> (32u - w));
    }

    /* The scalar rule: align pos to the width, read, advance pos. */
    TANDEM_FN uint64_t next(const uint32_t key[4], uint32_t K, uint64_t &pos, unsigned w) {
        uint64_t p = align_pos(pos, w);
        pos = p + w;
        return read(key, K, p, w);
    }
};

/* The transport form of a generator (key, bit position, chunk length K) plus one cached chunk
 * state. */
struct GenState {
    uint32_t key[4];
    uint64_t pos;
    uint32_t K;
    ChunkCache cache;

    TANDEM_FN void init(const uint32_t k[4], uint64_t p, uint32_t length) {
        for (int w = 0; w < 4; w++)
            key[w] = k[w];
        pos = p;
        K = length ? length : DEFAULT_K;
        cache = ChunkCache{};
    }
};

/* The draw API shared by every scalar generator. D derives from Draws<D> and offers
 * `GenState &st()` and its const form. Rng keeps the state private, device_rng in tandem.cuh
 * exposes it. */
template <class D> class Draws {
  public:
    static constexpr uint32_t MAX_URAND = 0xffffffffu;
    static constexpr uint64_t MAX_URAND64 = ~(uint64_t)0;
    static constexpr int32_t MAX_RAND = 0x7fffffff;
    static constexpr int64_t MAX_RAND64 = 0x7fffffffffffffff;

    /* Scalar draws: align the position to the width, read, advance. */
    TANDEM_FN bool bit() { return raw(1) != 0; }
    TANDEM_FN uint32_t urand() { return (uint32_t)raw(32); }
    TANDEM_FN uint64_t urand64() { return raw(64); }
    TANDEM_FN float frand() { return to_f32(urand()); }
    TANDEM_FN double drand() { return to_f64(urand64()); }

    /* Bounded draws, uniform on [0, range) by Lemire's multiply and reject. range > 0. These
     * and normal() are not part of the specification. */
    TANDEM_FN uint32_t urand(uint32_t range) {
        uint64_t m = (uint64_t)urand() * range;
        if ((uint32_t)m < range) {
            uint32_t t = (0u - range) % range;
            while ((uint32_t)m < t)
                m = (uint64_t)urand() * range;
        }
        return (uint32_t)(m >> 32);
    }
    TANDEM_FN uint64_t urand64(uint64_t range) {
        uint64_t x = urand64(), lo = x * range;
        if (lo < range) {
            uint64_t t = (0u - range) % range;
            while (lo < t) {
                x = urand64();
                lo = x * range;
            }
        }
        return mulhi64(x, range);
    }
    TANDEM_FN uint32_t urand(uint32_t start, uint32_t end) { return start + urand(end - start); }
    TANDEM_FN uint64_t urand64(uint64_t start, uint64_t end) {
        return start + urand64(end - start);
    }
    TANDEM_FN int32_t rand() { return (int32_t)(urand() >> 1); }
    TANDEM_FN int32_t rand(int32_t range) { return (int32_t)urand((uint32_t)range); }
    TANDEM_FN int32_t rand(int32_t start, int32_t end) {
        return (int32_t)((uint32_t)start + urand((uint32_t)end - (uint32_t)start));
    }
    TANDEM_FN int64_t rand64() { return (int64_t)(urand64() >> 1); }
    TANDEM_FN int64_t rand64(int64_t range) { return (int64_t)urand64((uint64_t)range); }
    TANDEM_FN int64_t rand64(int64_t start, int64_t end) {
        return (int64_t)((uint64_t)start + urand64((uint64_t)end - (uint64_t)start));
    }
    TANDEM_FN float frand(float range) { return range * frand(); }
    TANDEM_FN float frand(float start, float end) { return start + (end - start) * frand(); }
    TANDEM_FN double drand(double range) { return range * drand(); }
    TANDEM_FN double drand(double start, double end) { return start + (end - start) * drand(); }

    /* Standard normal by Box-Muller from two Float64 draws, the first mapped to (0, 1]. */
    TANDEM_FN double normal() {
        double u = 1.0 - drand(), v = drand();
        return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v);
    }
    TANDEM_FN double normal(double mean, double std_dev = 1.0) { return mean + std_dev * normal(); }

    /* Random access: element i of the fill that would start here, without advancing. */
    TANDEM_FN uint32_t at_urand(uint64_t i) const { return (uint32_t)at(i, 32); }
    TANDEM_FN uint64_t at_urand64(uint64_t i) const { return at(i, 64); }
    TANDEM_FN float at_frand(uint64_t i) const { return to_f32((uint32_t)at(i, 32)); }
    TANDEM_FN double at_drand(uint64_t i) const { return to_f64(at(i, 64)); }

    /* Children start at position 0 with the parent's K. split and sub read the key alone. */
    TANDEM_FN D child(uint64_t counter, uint32_t domain, uint32_t aux, bool hidden) const {
        uint32_t k[4];
        child_key(st().key, counter, domain, aux, hidden, k);
        D r;
        r.st().init(k, 0, st().K);
        return r;
    }
    TANDEM_FN D split(uint64_t index) const {
        return child(index >> 1, DOMAIN_SPLIT, 0, index & 1u);
    }
    TANDEM_FN D sub(uint64_t purpose) const { return child(purpose, DOMAIN_FOLD, 0, false); }
    /* n children from the current block, then the position moves past that block. */
    TANDEM_FN void fork(D *children, uint64_t n) {
        uint64_t b = st().pos >> 7;
        for (uint64_t i = 0; i < n; i++)
            children[i] = child(b, DOMAIN_FORK, (uint32_t)(i >> 1), i & 1u);
        st().pos = (b + 1u) << 7;
    }

  private:
    TANDEM_FN D &self() { return static_cast<D &>(*this); }
    TANDEM_FN const D &self() const { return static_cast<const D &>(*this); }
    TANDEM_FN GenState &st() { return self().st(); }
    TANDEM_FN const GenState &st() const { return self().st(); }

    TANDEM_FN uint64_t raw(unsigned w) {
        GenState &g = st();
        return g.cache.next(g.key, g.K, g.pos, w);
    }

    TANDEM_FN uint64_t at(uint64_t i, unsigned w) const {
        const GenState &g = st();
        uint64_t p = align_pos(g.pos, w) + i * w;
        uint32_t b[4];
        block(g.key, chunk_of(p, g.K), step_of(p, g.K), b);
        uint32_t lo = b[(p >> 5) & 3u];
        return w == 64u ? lo | ((uint64_t)b[((p >> 5) & 3u) + 1u] << 32) : lo;
    }
};

/* A generator: the transport form (key, bit position, chunk length K) plus one cached chunk
 * state, whose exposed half is block B(chunk, step). The cache is a pure function of the key
 * and K, so a copy draws the same values, and moving the position never invalidates it.
 * About 80 bytes, cheap to copy into a kernel. */
class Rng : public Draws<Rng> {
  public:
    /* From a 128-bit seed as two halves, whitened as the specification requires. K is the
     * chunk length, a power of two in [1, 65536], 0 for the default of 32. */
    TANDEM_FN explicit Rng(uint64_t seed_lo = 0, uint64_t seed_hi = 0, uint32_t K = 0) {
        uint32_t key[4];
        seed_key(seed_lo, seed_hi, key);
        s_.init(key, 0, K);
    }

    TANDEM_FN static Rng from_key(const Key &key, uint64_t pos = 0, uint32_t K = 0) {
        Rng r;
        r.s_.init(key.w, pos, K);
        return r;
    }

    TANDEM_FN Key key() const { return Key{{s_.key[0], s_.key[1], s_.key[2], s_.key[3]}}; }
    TANDEM_FN uint64_t position() const { return s_.pos; }
    TANDEM_FN uint32_t chunk_length() const { return s_.K; }
    TANDEM_FN void set_position(uint64_t p) { s_.pos = p; }

    TANDEM_FN friend bool operator==(const Rng &a, const Rng &b) {
        return a.key() == b.key() && a.s_.pos == b.s_.pos && a.s_.K == b.s_.K;
    }

  private:
    friend class Draws<Rng>;
    GenState s_;

    TANDEM_FN GenState &st() { return s_; }
    TANDEM_FN const GenState &st() const { return s_; }
};

} // namespace tandem
