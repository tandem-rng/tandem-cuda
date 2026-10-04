# API

- `tandem.cuh`: the CUDA fills and `device_rng`.
- `include/tandem/core.hpp`: the portable core that `tandem.cuh` builds on. It holds the
  step, the seeding function, the stream layout, the float mappings, child keys, an
  eight-lane row, the scalar generator `tandem::Rng` and the normal ziggurat, without CUDA
  types. `include/tandem/normal_tables.hpp` holds the ziggurat's tables, generated from the
  specification. Its functions are `TANDEM_FN`: `KOKKOS_INLINE_FUNCTION` under Kokkos,
  `__host__ __device__ inline` under nvcc and hipcc, `inline` otherwise.

- `tandem::fill_u32/u64/f32/f64(key, pos, K, device_ptr, n, stream)`: fill device memory
  from a key and stream position, as the specification's fill defines, and return the
  position after the fill. One thread per chunk walks its `K` blocks. For `K >= 8` a block
  of 32 groups stages eight steps in shared memory and writes 512 contiguous bytes per
  warp. For smaller `K` each group stores its own 128-byte line per step.
- `tandem::fill_u8/u16/f16_bits/bool` take the same arguments. `fill_f16_bits` writes the
  binary16 bit patterns of the specification's Float16 draws, `(raw >> 5) * 2^-11`, into a
  `uint16_t` buffer. `fill_bool` writes one byte, 0 or 1, per stream bit. The 8 and 16 bit
  fills use the tile kernel like the wider ones.
- `tandem::fill_i8/i16/i32/i64`: the unsigned fill of the same width read in two's
  complement, as the specification defines signed integers.
- `tandem::fill_u32_below(key, pos, K, range, out, n, stream)`, `fill_u64_below`: `n` draws
  uniform on `[0, range)`, as `Rng::urand(range)`. Not part of the specification.
- `tandem::fill_u32_below(key, pos, K, range, low, out, n, stream)` and `fill_u64_below` with a
  low bound: `out[e] = low + draw[e]`, summed in the output type and wrapping, so the output
  can be `uint32_t` or `int32_t`, and for a 32-bit range also `uint64_t` or `int64_t`, which
  widens the 32-bit draw into an 8-byte element. The draws and the consumed bits are those of the
  plain fill, and the offset and the widening are fused into the store, so there is no second
  pass. `fill_u64_below` takes a `uint64_t` or `int64_t` low bound and output.
- `tandem::fill_normal_f64(key, pos, K, out, n, stream)`: `n` standard normals by the 1024-layer
  ziggurat, one UInt64 draw each, as `Rng::normal()` calls, bit identical to tandem-c. From
  2^16 elements it takes `n / 8` bytes of scratch from `cudaMallocAsync` on `stream` for its list
  of misses. `fill_normal_f32`: both Box-Muller halves per two Float32 uniforms, as the flattened
  `Rng::normalf2()` calls. Appendix A of the specification.
- `tandem::fill_exponential_f64/f32(key, pos, K, out, n, stream)`: `n` standard exponentials
  `-ln(1 - u)`, one uniform each, as `Rng::exponential()` or `exponentialf()` calls, bit
  identical to tandem-c. Not part of the specification.
- `tandem::generator`: a host handle with public `key`, `pos` and `K`, built by `from_key` or
  `seed`. Its `fill_*` methods (every fill above, with the range first for the bounded ones) take
  the output pointer, `n` and a stream, fill from `pos`, and set `pos` to the position the
  fill returns, so successive fills continue one stream without the caller tracking positions.
- `tandem::device_rng`: a per-thread generator for kernels that draw scalars. It holds the
  transport form (public fields `key`, `pos`, `K`) and one cached chunk state, about 20
  registers. `from_key`, `seed`, `skip_to`, `next_bool/u8/u16/u32/u64/f16_bits/f32/f64`. Its
  draws follow the specification's scalar rule for the same key and position, mixed widths
  included. The rest of the draw API is `tandem::Rng`'s, shared through `Draws<D>` in
  `core.hpp`: bounded draws `urand(range)` and `urand64(range)` by Lemire's method, `normal()`
  (one ziggurat draw) and `normal2()` (two), `normalf()`/`normalf2()` by Box-Muller in float
  from two f32 uniforms, `exponential()` and `exponentialf()`, `at_urand/urand64/frand/drand(i)`,
  `child`, `split`, `sub` and `fork`. Bounded draws, normals and exponentials are not part of the specification.
  They match `tandem_u32_below`, `tandem_u64_below`, `tandem_normal_f64` and
  `tandem_exponential_f64` and `_f32` of the C library.
- `tandem_thrust.cuh`: Thrust and CUB adapters. `tandem::uniform<T>` (u32, u64, f32, f64),
  `tandem::below<T>` (u32, u64) and `tandem::normal<T>` (f32, f64) are functors from an index `i`
  to element `i` of the fill of the same type, built on the device generator's random access.
  `tandem::make_iterator(f, first)` wraps one in a `transform_iterator` over a
  `counting_iterator`, for `thrust::reduce`, `thrust::copy_n`, `thrust::transform` and CUB. A
  `normal<double>` yields the ziggurat of UInt64 draw `i`, and `normal<float>` the cos half at
  `2j` and the sin half at `2j + 1` of the uniforms `2j` and `2j + 1`. Header only, and Thrust
  ships with the toolkit.
- `tandem::normal_f64(r, key, K, g)`: the ziggurat normal of one UInt64 draw `r` with global draw
  index `g` under a generator's key and `K`, element `g - align(pos, 64) / 64` of a fill.
  `normal_f64_fast(r, hit)` is its table step alone.
- `tandem::T`, `F`, `F_keyed`, `block`: the specification's building blocks, host and device,
  from `core.hpp`.

## Use

Put `include/` on the include path, for example `nvcc -I<tandem-cuda>/include`.

```cpp
#include "tandem.cuh"

const uint32_t key[4] = {1, 2, 3, 4};
double *x;
cudaMalloc(&x, n * sizeof(double));
uint64_t pos = tandem::fill_f64(key, 0, 32, x, n);   // the spec's Float64 fill from position 0
uint32_t *die;                                       // 6-sided dice, n draws from the UInt32 stream
cudaMalloc(&die, n * sizeof(uint32_t));
pos = tandem::fill_u32_below(key, pos, 32, 6, die, n);
float *z;                                            // standard normals
cudaMalloc(&z, n * sizeof(float));
pos = tandem::fill_normal_f32(key, pos, 32, z, n);
pos = tandem::fill_exponential_f32(key, pos, 32, z, n); // standard exponentials, in place of z

tandem::generator g = tandem::generator::seed(42);   // the same, with the position kept for you
g.fill_f64(x, n);
g.fill_u32_below(6, die, n);

__global__ void kernel(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, float *out) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng rng = tandem::device_rng::from_key(key, 0, 32).split(blockIdx.x * blockDim.x + threadIdx.x);
    out[threadIdx.x] = rng.next_f32();
}
```

## Thrust

```cpp
#include "tandem_thrust.cuh"

auto it = tandem::make_iterator(tandem::uniform<uint64_t>(key, 32));     // element i of fill_u64
uint64_t sum = thrust::reduce(it, it + n, (uint64_t)0);                  // no buffer
```

## Parallel use

Element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.
