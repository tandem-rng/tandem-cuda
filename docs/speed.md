# Speed

`pixi run -e cuda12 make bench` produces the figures.

## GPU

NVIDIA A100 40 GB (PCIe), CUDA 12.8 built by clang 19 as the nvcc host compiler with `-std=c++20` (`pixi run -e cuda12 make bench`): 2^28 elements into device memory.
Each row first runs its own fill for two seconds, then times 21 fills by `cudaEvent`.
`tools/bench` prints the median and the fastest of the 21, and the table gives the median. In the
last run the fastest lay at most 4 % above the median on every row. The fastest once hid an
allocation that slowed most fills, see [design](design.md). After the two seconds every fill runs at
the card's 250 W power cap, so the rows give the capped rate. A row run alone (`tools/bench <part of its name>`) gives the figure it gives
in the table. Rows that name a length write fewer elements. The GPU had no other process during
the runs, and two consecutive runs agreed within 2 %. The elements are 2^28 of each type, so the
rows for narrow types write fewer bytes.

The cuRAND column is Philox4x32-10 of cuRAND 10.3.9 from the same run, by the same method, through the
host API with the call named. It writes the row's output type where cuRAND has one. Where it has
none, the column gives the nearest call, marked "nearest": cuRAND has no 8-, 16- or 64-bit integer
output for Philox, so `curandGenerate` writes the same bytes as 32-bit words, and it has no half
floats, bools, bounded integers or exponentials. A normal row's start draw is a generator offset in
cuRAND. The Thrust rows of cuRAND use its documented device API idiom, `curand_init(seed, i, 0)`
per element.

| | GiB/s written | cuRAND Philox4x32-10 | cuRAND call |
|---|---|---|---|
| `tandem::fill_u32`, tile kernel (default for K >= 8) | 1377 | 1285 | `curandGenerate` |
| `tandem::fill_u64`, tile kernel | 1388 | 1293 | `curandGenerate`, nearest |
| `tandem::fill_f32`, tile kernel | 1377 | 1270 | `curandGenerateUniform` |
| `tandem::fill_f64`, tile kernel | 1389 | 781 | `curandGenerateUniformDouble` |
| `tandem::fill_u32`, direct kernel (K < 8) | 1309 | 1285 | `curandGenerate` |
| `tandem::fill_u64`, direct kernel | 1317 | 1293 | `curandGenerate`, nearest |
| `tandem::fill_f32`, direct kernel | 1307 | 1270 | `curandGenerateUniform` |
| `tandem::fill_f64`, direct kernel | 1317 | 781 | `curandGenerateUniformDouble` |
| `tandem::fill_u16`, tile kernel | 1356 | 1265 | `curandGenerate`, nearest |
| `tandem::fill_f16_bits`, tile kernel | 1341 | 1265 | `curandGenerate`, nearest |
| `tandem::fill_u8`, tile kernel | 1320 | 1246 | `curandGenerate`, nearest |
| `tandem::fill_bool` (one byte per bit) | 1227 | 1246 | `curandGenerate`, nearest |
| `tandem::fill_u32_below(1000)` | 1334 | 1285 | `curandGenerate`, nearest |
| `tandem::fill_u64_below(1000)` | 1354 | 1293 | `curandGenerate`, nearest |
| `tandem::fill_u32_below(2^32 - 2)` (rejects about every draw's threshold check) | 1327 | 1285 | `curandGenerate`, nearest |
| `tandem::fill_u64_below(2^64 - 2)` | 1283 | 1293 | `curandGenerate`, nearest |
| `tandem::fill_u32_below` with a low bound into `int32_t` | 1329 | 1285 | `curandGenerate`, nearest |
| `tandem::fill_u32_below` with a low bound into `int64_t` (8-byte elements) | 1362 | 1293 | `curandGenerate`, nearest |
| `tandem::fill_u64_below` with a low bound into `int64_t` | 1314 | 1293 | `curandGenerate`, nearest |
| `tandem::fill_normal_f64` | 1058 | 567 | `curandGenerateNormalDouble` |
| `tandem::fill_normal_f64`, start at an odd Float64 draw | 958 | 542 | `curandGenerateNormalDouble`, offset 1 |
| `tandem::fill_normal_f64`, start at word 6 (draw 3) | 942 | 547 | `curandGenerateNormalDouble`, offset 3 |
| `tandem::fill_normal_f64`, 2^24 elements, even and odd start | 803, 793 | 528, 513 | `curandGenerateNormalDouble`, offset 0 and 1 |
| `tandem::fill_normal_f64`, 2^20 elements, even and odd start | 246, 224 | 224, 206 | `curandGenerateNormalDouble`, offset 0 and 1 |
| `tandem::fill_normal_f32` | 1204 | 862 | `curandGenerateNormal` |
| `tandem::fill_exponential_f64` | 941 | 781 | `curandGenerateUniformDouble`, nearest |
| `tandem::fill_exponential_f32` | 1149 | 1270 | `curandGenerateUniform`, nearest |
| `thrust::transform` of `tandem::uniform<uint32_t>` into a device vector | 65 | 687 | `curand` |
| `thrust::transform` of `tandem::uniform<double>` into a device vector | 124 | 1229 | `curand_uniform_double` |

The tile kernel stages eight steps of 32 groups in 32 KiB of shared memory and writes them
as 512 contiguous bytes per warp. It runs at the card's memory bandwidth, about 1.5 TB/s.
The direct kernel stores each block straight from registers, so the eight threads of a group
cover one 128-byte line per step. Both kernels pick a 16-byte vector store at compile time
when the output's blocks are 16-byte aligned. The bool fill expands each bit to a byte, so it
stages one step of 32 groups (32 KiB) and writes 1024 contiguous bytes per group. The bounded
fills run at the fill speed, because a rejection is rare and its retry runs out of line. A
32-bit range into 8-byte elements gives each thread one 16-byte output slot from two draws, so a
warp still writes 512 contiguous bytes. Two stores per 16-byte block of draws gave 955 GiB/s. The
f64 ziggurat runs a table pass and a second kernel for the
0.43 % of misses, see [design](design.md). At 2^24 elements it reaches three quarters of the 2^28 rate: the
table pass alone runs at 1168 GiB/s, and the misses kernel takes 25 µs of the 150. The Box-Muller fill it replaced ran at 833 and 679
GiB/s. A start at word 6 takes the octet stores of tandem-sycl, which keep each group's 128
bytes on whole sectors, see [design](design.md). Before them it ran at 689.

The normal and exponential fills run at 68 % to 87 % of the uniform rate. Every fill draws the
card's 250 W, so the SM clock sets each fill's rate once its arithmetic per byte is high enough.
The uniform fills run at 1170 to 1260 MHz and are bound by memory. The f32 normal, the
exponentials and the f64 normal run at 1035 to 1110 MHz, where the arithmetic per element bounds
them: a logarithm, a square root, a sine and a cosine per f32 normal pair, the specification's f64
logarithm, about 22 f64 operations per element, or the ziggurat's table read and its misses
kernel, which takes 16 % of the f64 normal fill. See [design](design.md) for the costs that were removed.

cuRAND is faster on five rows. On the bool, the 2^64 - 2 bounded and the f32 exponential rows its
nearest call does less work than the row: no expansion of bits to bytes, no bounding, no
logarithm. The Thrust rows cannot match it: element i of a Tandem stream needs its chunk's seeding,
eight rounds of the step, and up to K - 1 steps, while a Philox element is one ten-round block of
its counter.
