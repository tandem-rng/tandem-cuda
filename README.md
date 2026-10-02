<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .cu"></p>

# tandem-cuda

CUDA implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. One header,
`tandem.cuh`, C++17. It produces the stream the specification defines, bit for bit.

- `tandem::fill_u32/u64/f32/f64(key, pos, K, device_ptr, n, stream)`: fill device memory
  from a key and stream position, as the specification's fill defines, and return the
  position after the fill. One thread per chunk walks its `K` blocks. For `K >= 8` a block
  of 32 groups stages eight steps in shared memory and writes 512 contiguous bytes per
  warp. For smaller `K` each group stores its own 128-byte line per step.
- `tandem::device_rng`: a per-thread generator for kernels that draw scalars. It holds the
  transport form and one cached chunk state, about 20 registers. `from_key`, `seed`,
  `next_bool/u32/u64/f32/f64`, `split`, `sub`, `skip_to`. Its draws follow the specification's
  scalar rule for the same key and position, mixed widths included.
- `tandem::T`, `F`, `F_keyed`, `block`: the specification's building blocks, host and device.

## Use

```cpp
#include "tandem.cuh"

const uint32_t key[4] = {1, 2, 3, 4};
double *x;
cudaMalloc(&x, n * sizeof(double));
uint64_t pos = tandem::fill_f64(key, 0, 32, x, n);   // the spec's Float64 fill from position 0

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
device scalar draws with reference stream dumps in `tests/data`, and compares device fills at random keys,
chunk lengths, positions, lengths and output alignments with the reference C implementation
compiled into the test (a checkout at `TANDEM_C`). `pixi.toml` provides a CUDA
12.8 toolchain from conda-forge for hosts without a system install.

## Speed

NVIDIA A100 40 GB (PCIe), CUDA 12.8, `make bench`: 2^28 elements into device memory,
minimum of 21 `cudaEvent` timings per row after a half-second warm-up. Both GPUs idle before
the run. Two consecutive runs agreed within 1%.

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
| cuRAND Philox4x32-10 `curandGenerate` | 1332 |
| cuRAND Philox4x32-10 `curandGenerateUniform` | 1307 |
| cuRAND Philox4x32-10 `curandGenerateUniformDouble` | 781 |

The tile kernel stages eight steps of 32 groups in 32 KiB of shared memory and writes them
as 512 contiguous bytes per warp. It runs at the card's memory bandwidth, about 1.5 TB/s.
The direct kernel stores each block straight from registers, so the eight threads of a group
cover one 128-byte line per step. Both kernels pick a 16-byte vector store at compile time
when the output's blocks are 16-byte aligned.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
