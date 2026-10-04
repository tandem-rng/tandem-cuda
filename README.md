<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .cu"></p>

# tandem-cuda

CUDA implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. Header only, C++17.
It produces the stream the specification defines, bit for bit.

- `tandem.cuh`: the CUDA fills and `device_rng`.
- `include/tandem/core.hpp`: the portable core that `tandem.cuh` builds on. It holds the
  step, the seeding function, the stream layout, the float mappings, child keys, an
  eight-lane row and the scalar generator `tandem::Rng`, without CUDA types. Its functions
  are `TANDEM_FN`: `KOKKOS_INLINE_FUNCTION` under Kokkos, `__host__ __device__ inline`
  under nvcc and hipcc, `inline` otherwise.

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
- `tandem::fill_normal_f64/f32(key, pos, K, out, n, stream)`: `n` standard normals by
  Box-Muller, as `Rng::normal()`. Not part of the specification.
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
  and `normalf()` by Box-Muller (`normalf` in float from two f32 uniforms), `at_urand/urand64/frand/drand(i)`, `child`, `split`, `sub` and `fork`.
  Bounded draws and normals are not part of the specification. They match `tandem_u32_below`,
  `tandem_u64_below` and `tandem_normal_f64` of the C library.
- `tandem::T`, `F`, `F_keyed`, `block`: the specification's building blocks, host and device,
  from `core.hpp`.

## Bounded and normal fills

These fills are not part of the specification. Other ports should follow the same contract,
and the C library's `tandem_fill_u32_below`, `tandem_fill_u64_below` and
`tandem_fill_normal_f64` do, up to the rare case below.

**Normals.** Element `i` is Box-Muller of the Float64 draws `2i` and `2i + 1` of the Float64
fill that starts at the same position: `u = 1 - d[2i]`, `v = d[2i + 1]`,
`sqrt(-2 ln u) cos(2 pi v)`. The fill starts at `pos` aligned up to 64 bits, consumes `128 n`
bits and returns that position, so it equals `n` calls of `Rng::normal()`. A start at an odd
Float64 draw makes every element span two blocks, and the kernel steps a second chunk per
thread to read them, at about 80% of the speed. The device `log` and `cos` can differ from the
host's in the last bits, so f64 normals agree across hosts and devices to about 1e-15 relative.

**Float normals.** `fill_normal_f32` is `Rng::normalf()`: element `i` is Box-Muller in float of
the Float32 draws `2i` and `2i + 1` of the Float32 fill that starts at the same position,
`sqrt(-2 ln u) cos(2 pi v)` with `u = 1 - d[2i]`, `v = d[2i + 1]`, all in `float` with the precise
`logf`, `cosf` and `sqrtf`. The fill starts at `pos` aligned up to 32 bits and consumes `64 n`
bits, so a block holds two elements and a start at an odd Float32 draw makes some span two
blocks. It does not round the f64 normal. Float normals agree across ports and devices to a few
ulps, not bit for bit, because libm float functions differ. The uniforms are exact.
Everything else in this library is bit for bit.

**Bounded integers.** Element `e` uses its own draw `d[e]` of the UInt32 (UInt64) fill and
Lemire's multiply and reject: `m = d * range`, accepted when the low word of `m` is at least
`2^32 mod range` (or its 64-bit analogue), result the high word. The fill consumes exactly `n`
draws, so it returns `align(pos, w) + w n` at once, without waiting for the device. A sequential
`Rng::urand(range)` loop would consume extra draws after a rejection, and a parallel fill
cannot know how many. So a rejected draw `e` retries on a fallback stream: draws `0, 1, ...` of
`split(e)` of `sub(P)` of the fill's generator at position 0 (same key and `K`), with
`P = 0x424c573332` for 32-bit and `0x424c573634` for 64-bit ranges, until one is accepted.
Those two purposes are reserved. A rejection has probability `(2^32 mod range) / 2^32`, so
ranges that are powers of two never reject, and a fill without rejections equals the sequential
loop. `range = 0` returns 0 and still consumes the draw. `tests/cross_fill_below.h` holds fixtures for
ports, bounded fills of 64 elements at several ranges from the key of seed 42, `K = 32` and
position 0, with the rejection counts. Regenerate it with `make cross`, which needs only a host
C++ compiler and `core.hpp`.

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

tandem::generator g = tandem::generator::seed(42);   // the same, with the position kept for you
g.fill_f64(x, n);
g.fill_u32_below(6, die, n);

__global__ void kernel(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, float *out) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng rng = tandem::device_rng::from_key(key, 0, 32).split(blockIdx.x * blockDim.x + threadIdx.x);
    out[threadIdx.x] = rng.next_f32();
}
```

## Tests

GitHub runners have no GPU, so CI compiles the tests and the bench for `sm_80` and checks
that `tests/vectors.h` matches the spec repository's `vectors.json`. Run the tests on a GPU
host:

```sh
make test TANDEM_C=../tandem-c      # or: pixi install && pixi run test
```

`tests/test_cuda.cu` checks every vector of the specification, compares device fills and
device scalar draws with reference stream dumps in `tests/data` (the bool, u8 and f16 dumps
through the public launchers), and compares device fills at random keys,
chunk lengths, positions, lengths and output alignments with the reference C implementation
compiled into the test (a checkout at `TANDEM_C`). One mixed sequence of draws on a device
generator, with bounded draws at small and at rejecting ranges, normals, `at_*`, `fork`,
`split` and `sub`, is compared with the C library, normals to 1e-12 and the rest bit for
bit. Bounded fills are compared at ranges that reject often, rarely and never, against a
reference of the contract above written on the C library's generators, and, where nothing
rejects, against the sequential C bounded draws. The f64 normal fill is compared with the C
fill at even and odd Float64 starts, the f32 one with a float reference on the C Float32 draws
at every start slot, to 4 ulps. A generator is checked by running mixed fills of every width through it and through the C
generator, which must agree in values and in the final position. The fill comparison covers u8, u16, u32, u64, f16 bits,
f32, f64 and bool. The signed fills are compared with the C unsigned fills and their
returned positions. `pixi.toml` provides a CUDA
12.8 toolchain from conda-forge for hosts without a system install.

## Speed

NVIDIA A100 40 GB (PCIe), CUDA 12.8, `make bench`: 2^28 elements into device memory,
minimum of 21 `cudaEvent` timings per row after a half-second warm-up. Both GPUs idle before
the run. Consecutive runs agreed within 1%, except `fill_normal_f64` within 4%. The elements are
2^28 of each type, so the rows for narrow types write fewer bytes.

| | GiB/s written |
|---|---|
| `tandem::fill_u32`, tile kernel (default for K >= 8) | 1383 |
| `tandem::fill_u64`, tile kernel | 1395 |
| `tandem::fill_f32`, tile kernel | 1383 |
| `tandem::fill_f64`, tile kernel | 1394 |
| `tandem::fill_u32`, direct kernel (K < 8) | 1309 |
| `tandem::fill_u64`, direct kernel | 1313 |
| `tandem::fill_f32`, direct kernel | 1309 |
| `tandem::fill_f64`, direct kernel | 1323 |
| `tandem::fill_u16`, tile kernel | 1364 |
| `tandem::fill_f16_bits`, tile kernel | 1368 |
| `tandem::fill_u8`, tile kernel | 1334 |
| `tandem::fill_bool` (one byte per bit) | 1227 |
| `tandem::fill_u32_below(1000)` | 1336 |
| `tandem::fill_u64_below(1000)` | 1348 |
| `tandem::fill_normal_f64` | 405 |
| `tandem::fill_normal_f64`, start at an odd Float64 draw | 348 |
| `tandem::fill_normal_f32` | 540 |
| cuRAND Philox4x32-10 `curandGenerate` | 1332 |
| cuRAND Philox4x32-10 `curandGenerateUniform` | 1307 |
| cuRAND Philox4x32-10 `curandGenerateUniformDouble` | 781 |
| cuRAND Philox4x32-10 `curandGenerateNormal` | 977 |
| cuRAND Philox4x32-10 `curandGenerateNormalDouble` | 597 |

The tile kernel stages eight steps of 32 groups in 32 KiB of shared memory and writes them
as 512 contiguous bytes per warp. It runs at the card's memory bandwidth, about 1.5 TB/s.
The direct kernel stores each block straight from registers, so the eight threads of a group
cover one 128-byte line per step. Both kernels pick a 16-byte vector store at compile time
when the output's blocks are 16-byte aligned. The bool fill expands each bit to a byte, so it
stages one step of 32 groups (32 KiB) and writes 1024 contiguous bytes per group. The bounded
fills run at the fill speed, because a rejection is rare and its retry runs out of line. The
normal fills are limited by `log`, `cos` and `sqrt`: double precision for `fill_normal_f64`, about
54 billion elements per second, and float for `fill_normal_f32`, about 70 billion.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
