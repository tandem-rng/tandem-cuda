/* Thrust and CUB adapters for tandem.cuh: functors from an element index to the value of the
 * fill at that index, and iterators over them, so the stream feeds thrust::transform,
 * thrust::reduce, thrust::copy_n or a CUB algorithm without being written to memory first.
 *
 * Element i equals element i of the fill that starts at the same key, K and position, for the
 * same type: tandem::uniform<T> is fill_u32/u64/f32/f64, tandem::below<T> is fill_u32_below and
 * fill_u64_below, tandem::normal<T> is fill_normal_f64 and fill_normal_f32. Each call reaches
 * its block by random access, which costs one seeding and up to K steps, so materialising the
 * stream with a fill is faster. Use these when the values are consumed once.
 *
 * Header only. Thrust and CUB ship with the CUDA toolkit.
 *
 * Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
 */
#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "tandem.cuh"

namespace tandem {

namespace detail {

/* The stream a functor reads: a key, a chunk length and the start position of the fill. */
struct stream_ref {
    uint32_t key[4];
    uint64_t pos;
    uint32_t K;

    stream_ref(const uint32_t k[4], uint32_t length, uint64_t p)
        : key{k[0], k[1], k[2], k[3]}, pos(p), K(length ? length : DEFAULT_K) {}

    __host__ __device__ device_rng rng() const { return device_rng::from_key(key, pos, K); }
};

} // namespace detail

/* Uniform draws of type T: uint32_t, uint64_t, float or double, as the fill of that type. */
template <class T> struct uniform : detail::stream_ref {
    uniform(const uint32_t key[4], uint32_t K = DEFAULT_K, uint64_t pos = 0)
        : detail::stream_ref(key, K, pos) {}

    __host__ __device__ T operator()(uint64_t i) const {
        device_rng r = rng();
        if constexpr (std::is_same<T, uint32_t>::value) return r.at_urand(i);
        else if constexpr (std::is_same<T, uint64_t>::value) return r.at_urand64(i);
        else if constexpr (std::is_same<T, float>::value) return r.at_frand(i);
        else return r.at_drand(i);
    }
};

/* Bounded draws on [0, range), uint32_t or uint64_t, as fill_u32_below and fill_u64_below. */
template <class T> struct below : detail::stream_ref {
    T range, thresh;
    uint64_t g0; /* global draw index of element 0, which keys the fallback as in the fills */

    below(const uint32_t key[4], T range, uint32_t K = DEFAULT_K, uint64_t pos = 0)
        : detail::stream_ref(key, K, pos), range(range), g0(align_pos(pos, sizeof(T) * 8) / (sizeof(T) * 8)) {
        if constexpr (std::is_same<T, uint32_t>::value) thresh = below_threshold_u32(range);
        else thresh = below_threshold_u64(range);
    }

    __host__ __device__ T operator()(uint64_t i) const {
        device_rng r = rng();
        if constexpr (std::is_same<T, uint32_t>::value)
            return below_u32_t(r.at_urand(i), range, thresh, key, K, g0 + i);
        else
            return below_u64_t(r.at_urand64(i), range, thresh, key, K, g0 + i);
    }
};

/* Standard normals, double or float, as fill_normal_f64 and fill_normal_f32. Element i of the
 * double version is the ziggurat of UInt64 draw i, with its fallback keyed by the global draw
 * index as in the fill, so it equals the fill bit for bit. Element 2j of the float version is the
 * cos half and 2j + 1 the sin half of the Box-Muller step of the uniforms 2j and 2j + 1, with the
 * precise step, so it can differ from the fill by a few ulps. */
template <class T> struct normal : detail::stream_ref {
    uint64_t g0; /* global draw index of element 0 of the double version */

    normal(const uint32_t key[4], uint32_t K = DEFAULT_K, uint64_t pos = 0)
        : detail::stream_ref(key, K, pos), g0(align_pos(pos, 64) >> 6) {}

    __host__ __device__ T operator()(uint64_t i) const {
        device_rng r = rng();
        if constexpr (std::is_same<T, float>::value) {
            uint64_t j = i >> 1;
            Pair2<float> z = box_muller2_f32(r.at_frand(2 * j), r.at_frand(2 * j + 1));
            return (i & 1u) ? z.z1 : z.z0;
        } else {
            return normal_f64(r.at_urand64(i), key, K, g0 + i);
        }
    }
};

/* An iterator whose element i is f(first + i). Random access, so it works with thrust::reduce,
 * thrust::copy_n and CUB. */
template <class F> using iterator = thrust::transform_iterator<F, thrust::counting_iterator<uint64_t>>;

template <class F> iterator<F> make_iterator(F f, uint64_t first = 0) {
    return iterator<F>(thrust::counting_iterator<uint64_t>(first), f);
}

} // namespace tandem
