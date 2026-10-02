/* Tandem8x32 for CUDA: a noncryptographic pseudorandom number generator, fast on CPUs and
 * GPUs alike. Header only, C++17.
 *
 * Implements https://github.com/tandem-rng/spec and produces the same stream, bit for bit,
 * the specification defines. Two entry points:
 *
 *   - tandem::fill_u32/u64/f32/f64: fill device memory from a key and stream position. One
 *     thread per chunk, blocks stored lane-interleaved so a warp writes whole 128-byte lines.
 *   - tandem::device_rng: a per-thread generator with one cached chunk state, for kernels
 *     that draw scalars. Its draws equal the C library's tandem_next_* calls.
 *
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace tandem {

constexpr uint32_t DEFAULT_K = 32;

constexpr uint32_t CLOCK_WEYL = 0x9e3779b9u;
constexpr uint32_t DOMAIN_STREAM = 0x9e3779b9u;
constexpr uint32_t DOMAIN_SPLIT = 0xbb67ae85u;
constexpr uint32_t DOMAIN_FORK = 0xd2511f53u;
constexpr uint32_t DOMAIN_FOLD = 0xcd9e8d57u;
constexpr uint32_t DOMAIN_SEED = 0xa54ff53au;
constexpr uint32_t AUX_STREAM = 0x94d049bbu;

__host__ __device__ constexpr uint32_t round_constant(int r) {
    constexpr uint32_t rc[8] = {0xd17cc1b7u, 0xa7220a94u, 0xfe13abe8u, 0xfa9a6ee0u,
                                0xedb14accu, 0x9e21c820u, 0xff28b1d5u, 0xef5de2b0u};
    return rc[r];
}

__host__ __device__ inline uint32_t rotl(uint32_t x, unsigned r) {
    return (x << r) | (x >> (32u - r));
}

__host__ __device__ inline uint32_t mulhi(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return (uint32_t)(((uint64_t)a * b) >> 32);
#endif
}

/* The step T: mix the exposed half, clock the hidden half, feed o0 back into h0. */
__host__ __device__ inline void T(uint32_t o[4], uint32_t h[4]) {
    uint32_t m0 = h[0] | 1u, m1 = h[1] | 1u;
    uint32_t lo0 = o[0] * m0, hi0 = mulhi(o[0], m0);
    uint32_t lo1 = o[2] * m1, hi1 = mulhi(o[2], m1);
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
__host__ __device__ inline void F(uint32_t o[4], uint32_t h[4]) {
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

__host__ __device__ inline void F_keyed(const uint32_t key[4], uint64_t counter,
                                        uint32_t domain, uint32_t aux, uint32_t o[4],
                                        uint32_t h[4]) {
    o[0] = (uint32_t)counter;
    o[1] = (uint32_t)(counter >> 32);
    o[2] = domain;
    o[3] = aux;
    for (int w = 0; w < 4; w++) h[w] = key[w];
    F(o, h);
}

/* Block B(c, j): the exposed half of chunk c after j + 1 steps. */
__host__ __device__ inline void block(const uint32_t key[4], uint64_t c, uint32_t j,
                                      uint32_t out[4]) {
    uint32_t h[4];
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, out, h);
    for (uint32_t s = 0; s <= j; s++) T(out, h);
}

__host__ __device__ inline uint64_t align_pos(uint64_t pos, unsigned w) {
    return (pos + w - 1u) & ~((uint64_t)w - 1u);
}

__host__ __device__ inline double to_f64(uint64_t raw) {
    return (double)(raw >> 11) * 0x1p-53;
}

__host__ __device__ inline float to_f32(uint32_t raw) {
    return (float)(raw >> 8) * 0x1p-24f;
}

__host__ __device__ inline unsigned log2k(uint32_t K) {
    unsigned s = 0;
    while ((K >> s) > 1u) s++;
    return s;
}

/* ---- Per-thread generator ---------------------------------------------------------------- */

/* Transport form plus one cached chunk: the state of chunk `chunk` after `step + 1` steps,
 * whose exposed half is block B(chunk, step). About 20 registers. */
struct device_rng {
    uint32_t key[4];
    uint64_t pos;
    uint32_t K;
    uint32_t live;
    uint64_t chunk;
    uint32_t step;
    uint32_t o[4], h[4];

    __host__ __device__ static device_rng from_key(const uint32_t key[4], uint64_t pos,
                                                   uint32_t K) {
        device_rng r;
        for (int w = 0; w < 4; w++) r.key[w] = key[w];
        r.pos = pos;
        r.K = K ? K : DEFAULT_K;
        r.live = 0;
        r.chunk = 0;
        r.step = 0;
        for (int w = 0; w < 4; w++) r.o[w] = r.h[w] = 0;
        return r;
    }

    __host__ __device__ static device_rng seed(uint64_t seed_lo, uint64_t seed_hi,
                                               uint32_t K) {
        uint32_t o[4] = {0, 0, DOMAIN_SEED, 0};
        uint32_t h[4] = {(uint32_t)seed_lo, (uint32_t)(seed_lo >> 32), (uint32_t)seed_hi,
                         (uint32_t)(seed_hi >> 32)};
        F(o, h);
        return from_key(o, 0, K);
    }

    __host__ __device__ void skip_to(uint64_t p) { pos = p; }

    /* Bring the cache to the block that holds stream bit p. Stepping forward inside the
     * cached chunk costs one T per block; any other move reseeds. */
    __host__ __device__ void load(uint64_t p) {
        uint64_t row = p >> 10;
        unsigned shift = log2k(K);
        uint32_t j = (uint32_t)(row & (K - 1u));
        uint64_t c = 8u * (row >> shift) + ((p >> 7) & 7u);
        if (!live || c != chunk || j < step) {
            F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
            T(o, h);
            chunk = c;
            step = 0;
            live = 1;
        }
        for (; step < j; step++) T(o, h);
    }

    /* w bits (a power of two, 1 to 64) at the aligned position p. */
    __host__ __device__ uint64_t read(uint64_t p, unsigned w) {
        load(p);
        uint32_t lo = o[(p >> 5) & 3u];
        if (w == 64u) return lo | ((uint64_t)o[((p >> 5) & 3u) + 1u] << 32);
        return (lo >> (p & 31u)) & (0xffffffffu >> (32u - w));
    }

    __host__ __device__ uint64_t next(unsigned w) {
        uint64_t p = align_pos(pos, w);
        pos = p + w;
        return read(p, w);
    }

    __host__ __device__ bool next_bool() { return next(1) != 0; }
    __host__ __device__ uint32_t next_u32() { return (uint32_t)next(32); }
    __host__ __device__ uint64_t next_u64() { return next(64); }
    __host__ __device__ float next_f32() { return to_f32(next_u32()); }
    __host__ __device__ double next_f64() { return to_f64(next_u64()); }

    __host__ __device__ device_rng child(uint64_t counter, uint32_t domain, uint32_t aux,
                                         bool hidden) const {
        uint32_t co[4], ch[4];
        F_keyed(key, counter, domain, aux, co, ch);
        return from_key(hidden ? ch : co, 0, K);
    }

    /* Child by index, from the key alone. */
    __host__ __device__ device_rng split(uint64_t index) const {
        return child(index >> 1, DOMAIN_SPLIT, 0, index & 1u);
    }

    /* Child for a purpose, from the key alone. */
    __host__ __device__ device_rng sub(uint64_t purpose) const {
        return child(purpose, DOMAIN_FOLD, 0, false);
    }
};

/* ---- Fills ------------------------------------------------------------------------------ */

namespace detail {

/* How an output type is made from the words of a block. */
template <class E> struct elem;

template <> struct elem<uint32_t> {
    static constexpr unsigned bits = 32;
    __device__ static uint32_t make(uint32_t lo, uint32_t) { return lo; }
};
template <> struct elem<float> {
    static constexpr unsigned bits = 32;
    __device__ static float make(uint32_t lo, uint32_t) { return to_f32(lo); }
};
template <> struct elem<uint64_t> {
    static constexpr unsigned bits = 64;
    __device__ static uint64_t make(uint32_t lo, uint32_t hi) {
        return lo | ((uint64_t)hi << 32);
    }
};
template <> struct elem<double> {
    static constexpr unsigned bits = 64;
    __device__ static double make(uint32_t lo, uint32_t hi) {
        return to_f64(lo | ((uint64_t)hi << 32));
    }
};

template <class E> struct vec4;
template <> struct vec4<uint32_t> { using type = uint4; };
template <> struct vec4<float> { using type = float4; };
template <> struct vec4<uint64_t> { using type = ulonglong2; };
template <> struct vec4<double> { using type = double2; };

/* Store the elements of one block that fall inside the output. `first` is the byte offset
 * of the block in the stream, `b0` and `b1` bound the output's bytes. With ALIGNED the
 * output's blocks sit at 16-byte addresses, and a block fully inside is one vector store. */
template <class E, bool ALIGNED>
__device__ __forceinline__ void store_block(E *out, uint64_t b0, uint64_t b1, uint64_t first,
                                            const uint32_t w[4]) {
    constexpr unsigned size = elem<E>::bits / 8;
    constexpr unsigned per_block = 16 / size;
    E v[per_block];
    for (unsigned i = 0; i < per_block; i++)
        v[i] = elem<E>::make(w[i * (size / 4)], w[i * (size / 4) + (size / 4) - 1]);
    char *dst = reinterpret_cast<char *>(out) + (first - b0);
    if (ALIGNED && first >= b0 && first + 16 <= b1) {
        *reinterpret_cast<typename vec4<E>::type *>(dst) =
            *reinterpret_cast<const typename vec4<E>::type *>(v);
        return;
    }
    for (unsigned i = 0; i < per_block; i++) {
        uint64_t at = first + i * size;
        if (at >= b0 && at + size <= b1) reinterpret_cast<E *>(dst)[i] = v[i];
    }
}

constexpr unsigned THREADS = 256;
constexpr unsigned TILE_STEPS = 8;

/* One thread per chunk, direct stores. The thread walks its K blocks and stores those inside
 * rows r0 .. r1 (inclusive). Rows are 128 bytes, blocks 16, lanes are the eight chunks of a
 * group, so the eight threads of a group store one line per step. Any K. */
template <class E, bool ALIGNED>
__global__ void fill_rows_kernel(uint32_t key0, uint32_t key1, uint32_t key2, uint32_t key3,
                                 uint32_t K, uint64_t g0, uint64_t r0, uint64_t r1, uint64_t b0,
                                 uint64_t b1, E *out) {
    uint64_t c = 8u * g0 + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t g = c >> 3, lane = c & 7u;
    if (g > r1 / K) return;
    const uint32_t key[4] = {key0, key1, key2, key3};
    uint32_t o[4], h[4];
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    uint64_t row = g * K;
    uint32_t j0 = row < r0 ? (uint32_t)(r0 - row) : 0u;
    uint32_t j1 = (uint32_t)(r1 - row < K - 1u ? r1 - row : K - 1u);
    for (uint32_t j = 0; j <= j1; j++) {
        T(o, h);
        if (j >= j0) store_block<E, ALIGNED>(out, b0, b1, (row + j) * 128u + lane * 16u, o);
    }
}

/* One thread per chunk, 32 groups per block, output staged through shared memory. Every
 * TILE_STEPS steps the block holds, for each of its groups, TILE_STEPS consecutive rows,
 * which are 1024 contiguous bytes of the stream. The write phase hands consecutive 16-byte
 * slots to consecutive threads, so a warp writes 512 contiguous bytes. Needs K >= TILE_STEPS. */
template <class E, bool ALIGNED>
__global__ void __launch_bounds__(THREADS)
    fill_tile_kernel(uint32_t key0, uint32_t key1, uint32_t key2, uint32_t key3, uint32_t K,
                     uint64_t g0, uint64_t r1, uint64_t b0, uint64_t b1, E *out) {
    constexpr unsigned GROUPS = THREADS / 8, SLOTS = GROUPS * TILE_STEPS * 8;
    __shared__ uint4 tile[SLOTS];
    uint64_t gb = g0 + blockIdx.x * (uint64_t)GROUPS; /* first group of this block */
    unsigned gi = threadIdx.x >> 3, lane = threadIdx.x & 7u;
    uint64_t c = 8u * (gb + gi) + lane;
    bool mine = gb + gi <= r1 / K;
    const uint32_t key[4] = {key0, key1, key2, key3};
    uint32_t o[4], h[4];
    F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM, o, h);
    uint64_t block_first = gb * K * 128u; /* stream byte of the block's first row */
    for (uint32_t jb = 0; jb < K; jb += TILE_STEPS) {
        if (block_first + jb * 128u > b1) break;
        if (mine) {
            for (unsigned j = 0; j < TILE_STEPS; j++) {
                T(o, h);
                tile[(gi * TILE_STEPS + j) * 8 + lane] = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();
        for (unsigned s = threadIdx.x; s < SLOTS; s += THREADS) {
            unsigned sg = s / (TILE_STEPS * 8), within = s % (TILE_STEPS * 8);
            uint64_t first = ((gb + sg) * K + jb) * 128u + within * 16u;
            if (first >= b1) continue;
            uint4 v = tile[s];
            uint32_t w[4] = {v.x, v.y, v.z, v.w};
            store_block<E, ALIGNED>(out, b0, b1, first, w);
        }
        __syncthreads();
    }
}

/* `tile` selects the shared-memory kernel where K allows; the tests and the bench also run
 * the direct kernel at every K. */
template <class E>
inline uint64_t fill(const uint32_t key[4], uint64_t pos, uint32_t K, E *out, size_t n,
                     cudaStream_t stream, bool tile = true) {
    constexpr unsigned bits = elem<E>::bits;
    K = K ? K : DEFAULT_K;
    uint64_t p0 = align_pos(pos, bits), p1 = p0 + (uint64_t)n * bits;
    if (n == 0) return p1;
    uint64_t r0 = p0 >> 10, r1 = (p1 - 1) >> 10;
    uint64_t g0 = r0 / K, g1 = r1 / K;
    uint64_t b0 = p0 / 8, b1 = p1 / 8;
    /* Blocks land on 16-byte addresses when the output's first byte and the fill's first
     * stream byte agree modulo 16. */
    bool aligned = ((reinterpret_cast<uintptr_t>(out) - b0) & 15u) == 0;
    uint64_t groups = g1 - g0 + 1u;
    if (tile && K >= TILE_STEPS) {
        unsigned blocks = (unsigned)((groups + THREADS / 8 - 1) / (THREADS / 8));
        if (aligned)
            fill_tile_kernel<E, true><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r1, b0, b1, out);
        else
            fill_tile_kernel<E, false><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r1, b0, b1, out);
    } else {
        unsigned blocks = (unsigned)((8u * groups + THREADS - 1) / THREADS);
        if (aligned)
            fill_rows_kernel<E, true><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r0, r1, b0, b1, out);
        else
            fill_rows_kernel<E, false><<<blocks, THREADS, 0, stream>>>(
                key[0], key[1], key[2], key[3], K, g0, r0, r1, b0, b1, out);
    }
    return p1;
}

} // namespace detail

/* Fill n elements of device memory with the draws that start at stream position pos, as
 * the C library's tandem_fill_* would. Return the position after the fill. */
inline uint64_t fill_u32(const uint32_t key[4], uint64_t pos, uint32_t K, uint32_t *out,
                         size_t n, cudaStream_t stream = 0) {
    return detail::fill(key, pos, K, out, n, stream);
}
inline uint64_t fill_u64(const uint32_t key[4], uint64_t pos, uint32_t K, uint64_t *out,
                         size_t n, cudaStream_t stream = 0) {
    return detail::fill(key, pos, K, out, n, stream);
}
inline uint64_t fill_f32(const uint32_t key[4], uint64_t pos, uint32_t K, float *out, size_t n,
                         cudaStream_t stream = 0) {
    return detail::fill(key, pos, K, out, n, stream);
}
inline uint64_t fill_f64(const uint32_t key[4], uint64_t pos, uint32_t K, double *out,
                         size_t n, cudaStream_t stream = 0) {
    return detail::fill(key, pos, K, out, n, stream);
}

} // namespace tandem
