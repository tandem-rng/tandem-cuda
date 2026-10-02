<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng"></p>

# tandem-cuda

CUDA implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. One header,
`tandem.cuh`, C++17. It produces the same stream, bit for bit, as the Julia reference
[TandemRNG.jl](https://github.com/tandem-rng/TandemRNG.jl), the C reference
[tandem-c](https://github.com/tandem-rng/tandem-c), the Rust crate
[tandem-rs](https://github.com/tandem-rng/tandem-rs) and the NumPy `BitGenerator`
[tandem-numpy](https://github.com/tandem-rng/tandem-numpy).

- `tandem::fill_u32/u64/f32/f64(key, pos, K, device_ptr, n, stream)`: fill device memory
  from a key and stream position, as the C library's `tandem_fill_*` would, and return the
  position after the fill. One thread per chunk walks its `K` blocks. Blocks are stored
  lane-interleaved, so the eight threads of a group write one 128-byte line per step and
  the fill runs at memory bandwidth.
- `tandem::device_rng`: a per-thread generator for kernels that draw scalars. It holds the
  transport form and one cached chunk state, about 20 registers. `from_key`, `seed`,
  `next_bool/u32/u64/f32/f64`, `split`, `sub`, `skip_to`. Its draws equal the C library's
  `tandem_next_*` for the same key and position, mixed widths included.
- `tandem::T`, `F`, `F_keyed`, `block`: the specification's building blocks, host and device.

## Use

```cpp
#include "tandem.cuh"

const uint32_t key[4] = {1, 2, 3, 4};
double *x;
cudaMalloc(&x, n * sizeof(double));
uint64_t pos = tandem::fill_f64(key, 0, 32, x, n);   // same values as the C fill from position 0

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
device scalar draws with dumps written by TandemRNG.jl (`tests/data`, shared with
tandem-c), and compares device fills at random keys, chunk lengths, positions, lengths and
output alignments with the C library compiled into the test. `pixi.toml` provides a CUDA
12.8 toolchain from conda-forge for hosts without a system install.

## Speed

NVIDIA A100 40 GB (PCIe), CUDA 12.8, `make bench`: 2^28 elements into device memory,
minimum of seven `cudaEvent` timings after three warm-up runs. GPU 1 idle before the run,
host load 11 from other users' CPU jobs.

| | GiB/s written |
|---|---|
| `tandem::fill_u32` | 1090 |
| `tandem::fill_u64` | 1098 |
| `tandem::fill_f32` | 1084 |
| `tandem::fill_f64` | 1097 |
| cuRAND Philox4x32-10 `curandGenerate` | 1016 |
| cuRAND Philox4x32-10 `curandGenerateUniform` | 986 |
| cuRAND Philox4x32-10 `curandGenerateUniformDouble` | 456 |

Every Tandem fill writes at about 85% of the A100's 1.3 TB/s. TandemRNG.jl's CUDA fills on
the same card reach 1257 to 1294 GiB/s with a shared-memory tile; that layout is the next
step for this header.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
