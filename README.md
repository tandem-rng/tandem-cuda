<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .cu"></p>

# tandem-cuda

CUDA implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. Header only. `tandem.cuh` is built as C++23 with CUDA 13.4 and clang 20, and `include/tandem/core.hpp` stays valid C++17.
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
- `tandem::fill_normal_f64/f32(key, pos, K, out, n, stream)`: `n` standard normals, both
  Box-Muller halves per two uniforms, as the flattened `Rng::normal2()` or `normalf2()`
  calls. Not part of the specification.
- `tandem::generator`: a host handle with public `key`, `pos` and `K`, built by `from_key` or
  `seed`. Its `fill_*` methods (every fill above, with the range first for the bounded ones) take
  the output pointer, `n` and a stream, fill from `pos`, and set `pos` to the position the
  fill returns, so successive fills continue one stream without the caller tracking positions.
- `tandem::device_rng`: a per-thread generator for kernels that draw scalars. It holds the
  transport form (public fields `key`, `pos`, `K`) and one cached chunk state, about 20
  registers. `from_key`, `seed`, `skip_to`, `next_bool/u8/u16/u32/u64/f16_bits/f32/f64`. Its
  draws follow the specification's scalar rule for the same key and position, mixed widths
  included. The rest of the draw API is `tandem::Rng`'s, shared through `Draws<D>` in
  `core.hpp`: bounded draws `urand(range)` and `urand64(range)` by Lemire's method, `normal()`/`normal2()`
  and `normalf()`/`normalf2()` by Box-Muller (the `f` forms in float from two f32 uniforms), `at_urand/urand64/frand/drand(i)`, `child`, `split`, `sub` and `fork`.
  Bounded draws and normals are not part of the specification. They match `tandem_u32_below`,
  `tandem_u64_below` and `tandem_normal_f64` of the C library.
- `tandem::T`, `F`, `F_keyed`, `block`: the specification's building blocks, host and device,
  from `core.hpp`.

## Bounded and normal fills

These fills are not part of the specification. Other ports should follow the same contract,
and the C library's `tandem_fill_u32_below` and `tandem_fill_u64_below` do, up to the rare case
below.

**Normals.** One Box-Muller step turns two uniforms `a`, `b` into two normals,
`r = sqrt(-2 ln(1 - a))`, `z0 = r cos(2 pi b)`, `z1 = r sin(2 pi b)`. `Rng::normal2()` returns
the pair `(z0, z1)`, `Rng::normal()` its first half, and both consume two Float64 uniforms. A
fill of `n` normals is the flattened sequence of `normal2` calls: pair `j`, the elements `2j` and
`2j + 1`, comes from the Float64 draws `2j` and `2j + 1` of the Float64 fill that starts at the
same position. The fill starts at `pos` aligned up to 64 bits and consumes `2 ceil(n / 2)`
draws, so an odd `n` uses the cos half of its last pair and still advances past both draws. An empty
normal or bounded fill consumes nothing and returns `pos` unchanged, even when `pos` is unaligned. A
start at an odd Float64 draw makes every pair span two blocks, and the kernel steps a second
chunk per thread to read them, at about 80% of the speed. On a device the angle goes through
`sincospi(2b)`, on a host through `cos` and `sin`, and `log` differs in the last bits, so f64
normals agree across platforms to about 1e-15 relative, not bit for bit.

**Float normals.** `fill_normal_f32` and `Rng::normalf2()` are the same on Float32 draws, in
float: `u = 1 - d[2j]`, `v = d[2j + 1]`, precise `logf` and `sqrtf`. The device fill takes the
angle through the fast `__sincosf` on `2 pi (v - 0.5)`, which is accurate on `[-pi, pi]`, because
the precise `sincospif` made the fill compute bound at 1065 GiB/s against 1300 memory bound. The
result stays within 16 ulps + 1e-6 of the precise step: at most 1.5e-6 absolute over the random
fills and 7.2e-7 on the fixtures, up to 47 ulps for a value near 0.01, where the absolute error
dominates. `__logf` is not used, because its absolute error near 1 distorts small radii by
thousands of ulps. Define `TANDEM_PRECISE_F32_NORMAL` for `sincospif` and 4 ulps. The fill starts at `pos` aligned up to 32 bits and consumes `64 ceil(n / 2)` bits, so
a block holds two pairs and a start at an odd Float32 draw makes some span two blocks. It does
not round the f64 normal. Float normals agree across ports and devices to a few ulps, not bit for
bit, because libm float functions differ. The host version of the f32 step takes its angle in
double and rounds the results, because a float angle `2 pi b` is off by up to `2 pi b 2^-24`
where `sincospif` is not. The uniforms are exact, and everything else in this library is bit for
bit. `tests/cross_fill_normal.h` holds normal fixtures for ports at even and odd starts.

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

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

## Install

The library is headers, so copy `tandem.cuh` and `include/tandem/` or put them on the include path. The
`packaging/` directory holds a Spack recipe (`spack/package.py`) and a conda-forge style recipe
(`conda/recipe.yaml`) that install both. Neither is submitted to Spack or conda-forge yet, and both
build from the `main` branch.

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
rejects, against the sequential C bounded draws. Normal fills are compared with the host Box-Muller step on the C library's uniform fills, at
every start slot, odd and even `n`, to 1e-12 (f64) and 16 ulps + 1e-6 (f32, 4 ulps with `TANDEM_PRECISE_F32_NORMAL`), and with
`tests/cross_fill_normal.h`. A generator is checked by running mixed fills of every width through it and through the C
generator, which must agree in values and in the final position. The fill comparison covers u8, u16, u32, u64, f16 bits,
f32, f64 and bool. The signed fills are compared with the C unsigned fills and their
returned positions. `tests/host_core.cpp` (`make host`) builds `core.hpp` as C++17 with clang and
gcc and compares the scalar generator with the C library, because tandem-kokkos, tandem-fortran
and tandem-torch include it with their own standards.

**Toolchains.** `pixi.toml` provides conda-forge environments for hosts without a system install.
The default is CUDA 13.4 (nvcc 13.4.92) with clang 20 as the host compiler and `-std=c++23`, with
no warnings under `-Wall -Wextra`. `pixi run -e gcc` builds the same with gcc 14 as host compiler
(`make HOSTCXX=g++ HOSTCC=gcc`). `pixi run -e cuda12` is CUDA 12.8 (nvcc 12.8.93) with clang 19 and
`-std=c++20`, because nvcc 12.8 stops at C++20. CI builds all three nvcc environments.

Clang can also compile the CUDA sources itself (`clang++ -x cuda --cuda-gpu-arch=sm_80`), as Kokkos
builds do. `make clangcuda` does that for the tests in the `cuda12` environment, clang 19 with
CUDA 12.8 and C++20. It compiles without warnings, and on the A100 it passes the same suite with
the same f32 normal deviations as nvcc, so `__sincosf` behaves alike under both. CI compiles it.
Clang 20 does not build against the CUDA 13 headers yet, so that pairing is not offered. The GPU host batserv01 has
NVIDIA driver 570.124, which supports CUDA 12.8 at most, so its test suite and the speeds below run
in the `cuda12` environment. A CUDA 13 binary needs driver 580 or newer. The GPU suite passed there
with CUDA 12.8, clang 19 and C++20; the CUDA 13 environments are compiled but not run.

## Speed

NVIDIA A100 40 GB (PCIe), CUDA 12.8 built by clang 19 as the nvcc host compiler with `-std=c++20` (`pixi run -e cuda12 make bench`): 2^28 elements into device memory,
minimum of 21 `cudaEvent` timings per row after a half-second warm-up. Both GPUs idle before
the run. Consecutive runs agreed within 1%, except the f64 normal rows within 7%. The elements are
2^28 of each type, so the rows for narrow types write fewer bytes.

| | GiB/s written |
|---|---|
| `tandem::fill_u32`, tile kernel (default for K >= 8) | 1386 |
| `tandem::fill_u64`, tile kernel | 1394 |
| `tandem::fill_f32`, tile kernel | 1381 |
| `tandem::fill_f64`, tile kernel | 1392 |
| `tandem::fill_u32`, direct kernel (K < 8) | 1308 |
| `tandem::fill_u64`, direct kernel | 1315 |
| `tandem::fill_f32`, direct kernel | 1312 |
| `tandem::fill_f64`, direct kernel | 1323 |
| `tandem::fill_u16`, tile kernel | 1370 |
| `tandem::fill_f16_bits`, tile kernel | 1364 |
| `tandem::fill_u8`, tile kernel | 1330 |
| `tandem::fill_bool` (one byte per bit) | 1212 |
| `tandem::fill_u32_below(1000)` | 1336 |
| `tandem::fill_u64_below(1000)` | 1348 |
| `tandem::fill_normal_f64` | 750 |
| `tandem::fill_normal_f64`, start at an odd Float64 draw | 595 |
| `tandem::fill_normal_f32` | 1290 |
| cuRAND Philox4x32-10 `curandGenerate` | 1306 |
| cuRAND Philox4x32-10 `curandGenerateUniform` | 1311 |
| cuRAND Philox4x32-10 `curandGenerateUniformDouble` | 795 |
| cuRAND Philox4x32-10 `curandGenerateNormal` | 980 |
| cuRAND Philox4x32-10 `curandGenerateNormalDouble` | 597 |

The tile kernel stages eight steps of 32 groups in 32 KiB of shared memory and writes them
as 512 contiguous bytes per warp. It runs at the card's memory bandwidth, about 1.5 TB/s.
The direct kernel stores each block straight from registers, so the eight threads of a group
cover one 128-byte line per step. Both kernels pick a 16-byte vector store at compile time
when the output's blocks are 16-byte aligned. The bool fill expands each bit to a byte, so it
stages one step of 32 groups (32 KiB) and writes 1024 contiguous bytes per group. The bounded
fills run at the fill speed, because a rejection is rare and its retry runs out of line. The
f32 normal fill is memory bound, the f64 one is limited by double-precision `log`, `sincospi`
and `sqrt`.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
