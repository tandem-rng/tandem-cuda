# Speed

`pixi run -e cuda12 make bench` produces the figures.

## GPU

NVIDIA A100 40 GB (PCIe), CUDA 12.8 built by clang 19 as the nvcc host compiler with `-std=c++20` (`pixi run -e cuda12 make bench`): 2^28 elements into device memory,
minimum of 21 `cudaEvent` timings per row after a half-second warm-up. Both GPUs idle before
the run. Consecutive runs agreed within 1%, except the f64 normal rows within 3%. The Thrust rows are 11 to 21 times slower than the fills,
because each element reaches its block by random access. They are for values used once, without a
buffer. The elements are
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
| `tandem::fill_u32_below(2^32 - 2)`, `fill_u64_below(2^64 - 2)` (rejects about every draw's threshold check) | 1348, 1351 |
| `tandem::fill_u32_below` with a low bound into `int32_t` | 1348 |
| `tandem::fill_u32_below` with a low bound into `int64_t` (8-byte elements) | 1365 |
| `tandem::fill_u64_below` with a low bound into `int64_t` | 1352 |
| `tandem::fill_normal_f64` | 1065, 1096 |
| `tandem::fill_normal_f64`, start at an odd Float64 draw | 1102 |
| `tandem::fill_normal_f32` | 1290 |
| `tandem::fill_exponential_f64` | 948 |
| `tandem::fill_exponential_f32` | 1022 |
| `thrust::transform` of `tandem::uniform<uint32_t>` into a device vector | 65 |
| `thrust::transform` of `tandem::uniform<double>` into a device vector | 125 |

The tile kernel stages eight steps of 32 groups in 32 KiB of shared memory and writes them
as 512 contiguous bytes per warp. It runs at the card's memory bandwidth, about 1.5 TB/s.
The direct kernel stores each block straight from registers, so the eight threads of a group
cover one 128-byte line per step. Both kernels pick a 16-byte vector store at compile time
when the output's blocks are 16-byte aligned. The bool fill expands each bit to a byte, so it
stages one step of 32 groups (32 KiB) and writes 1024 contiguous bytes per group. The bounded
fills run at the fill speed, because a rejection is rare and its retry runs out of line. A
32-bit range into 8-byte elements gives each thread one 16-byte output slot from two draws, so a
warp still writes 512 contiguous bytes. Two stores per 16-byte block of draws gave 955 GiB/s. The
f32 normal fill is memory bound. The f64 ziggurat runs a table pass and a second kernel for the
0.43 % of misses, see [design](design.md). The Box-Muller fill it replaced ran at 833 and 679
GiB/s. The exponential fills run into the card's
250 W power cap on their division and logarithm per element, so they vary by up to 15 % between
runs.

## Other generators

cuRAND on the same card, with the same method:

| | GiB/s written |
|---|---|
| cuRAND Philox4x32-10 `curandGenerate` | 1306 |
| cuRAND Philox4x32-10 `curandGenerateUniform` | 1311 |
| cuRAND Philox4x32-10 `curandGenerateUniformDouble` | 795 |
| cuRAND Philox4x32-10 `curandGenerateNormal` | 980 |
| cuRAND Philox4x32-10 `curandGenerateNormalDouble` | 597 |
