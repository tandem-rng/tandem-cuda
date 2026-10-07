# Tests

Run the tests on a GPU host:

```sh
make test TANDEM_C=../tandem-c      # or: pixi install && pixi run test
make thrust                         # tests/test_thrust.cu
make host                           # tests/host_core.cpp
make stats                          # tools/normal_stats.cu, 1e8 f64 normals
make reject                         # tests/reject_widened.cu must not compile
```

## Suite

## Conformance

`tests/conformance` holds byte-identical copies of the `conformance/*.json` files of tandem-spec
at f420545, and a CI job compares them with the spec. `tests/conformance.h` reads them, and the
tests check each item of the spec's `conformance/CHECKLIST.md` at b31af72, whose JSON files equal
f420545's. The device runs the fills and the
`device_rng` draws, and the host (`tests/host_core.cpp`) computes each fill element from
`core.hpp`'s per-element maps or from the scalar draws that equal it.

| Checklist section | Host | Device |
|---|---|---|
| Fallback by global draw index | every `fill_below.json` and `normal.json` case, the shifted-start pairs | the same through the fills |
| Width from range | `urand`, `urand64` and `below` at range 1000 and 0, `below` at 2^32 and 2^32 + 1 | `device_rng::below`, `below<uint64_t>` of Thrust and a 32-bit range into `uint64_t` elements give `CROSS_BELOW32[3]`, the u64 fill `CROSS_BELOW64[3]` |
| n = 0 | (fills are device only) | the seven empty cases, nothing written |
| Odd n | `CROSS_NORMAL32[0..4]` and their ends | the same |
| Pair rule for Float32 Box-Muller | `CROSS_NORMALF`, the pair shift, `normalf` | the same, with `device_rng` |
| Weighted choice | tables of the vectors cases, every case, the shift, `m = 1`, invalid weights | every case, the shift, the empty fill |
| Cut fill | (scalar draws equal the fills) | every case cut at 1, 7, 20, 21 and `n - 1`, f32 normals at 2, 8, 20 and the largest even element below `n` |
| Block and 2^63 position boundaries | stream hashes, all five dumps, random access, `set_position` and `from_key` bounds, a draw at 2^63 - 1, `fill_end` | stream hashes from fills and draws, normal and exponential dumps, random access, a draw at 2^63 - 1, every fill to an end of 2^64 |

The device's Float32 normals are not bit exact, so the Float32 normal dump is a host check.
This library has no UInt128, Char or complex fills, so their stream hashes and the complex block
boundary are not checked. `device_rng` and `generator` take any start, so the start bounds are
checked on `Rng` alone. Every fill whose end reaches 2^64 throws `std::length_error` before it
launches: the device test runs each fill through a generator to an end of exactly 2^64 and checks
that the output and the position are untouched, and that one element less fills. The host test
checks `fill_end`, the check every entry point calls.

## Suite

`tests/test_cuda.cu` checks every vector of the specification and the conformance cases above,
and compares device fills at random keys,
chunk lengths, positions, lengths and output alignments with the reference C implementation
compiled into the test (a checkout at `TANDEM_C`). One mixed sequence of draws on a device
generator, with bounded draws at small and at rejecting ranges, normals, `at_*`, `fork`,
`split` and `sub`, is compared with the C library, f64 normals with the host core, f32 normals to
4 ulps and the rest bit for bit. Bounded fills are compared at ranges that reject often, rarely and never, against a
reference of the [design](design.md) contract written on the C library's generators, and, where nothing
rejects, against the sequential C bounded draws. The C library's bounded fills must equal that
reference too. A bounded fill cut at a random element, at ranges that reject about half the draws,
equals the whole fill for both widths, the fused low-bound and wider outputs and the generator.
Outputs narrower than the draw, 16 and 8 bits, equal the plain fill plus the low bound. f64 normal
fills equal tandem-c's `tandem_fill_normal_f64` byte for byte, end positions included, at random
keys, `K`, start slots, lengths up to 2^21, and outputs on and 8 bytes off 16-byte alignment, which
covers both kernels and both store paths, about 43000 misses and 570 tail values. A normal fill cut at an odd element, at a missed element and
just after it equals the whole fill. Fills from every first draw modulo 16, which covers the octet
stores, equal tandem-c too. Fills on two streams at once, of growing and shrinking lengths, and
fills captured into a graph and launched twice, equal it as well. f32 normal fills are compared with the host Box-Muller step
on the C library's uniform fills, at every start slot, odd and even `n`, to 16 ulps + 2.1e-6 (4 ulps
with `TANDEM_PRECISE_F32_NORMAL`). An empty f64 normal fill aligns the
position to 64 bits, the f32 one leaves it alone. Exponential fills equal tandem-c's
`tandem_fill_exponential_f64` and `_f32` byte for byte at random keys, `K`, start slots and
lengths up to 2^22. The f32 map equals tandem-c's arithmetic for all 2^24
Float32 draws on the device and on the host, which covers the device division without its range
check. The device generator's `exponential()` and `exponentialf()` equal the C library's scalar
draws. Choice fills equal tandem-c's `tandem_fill_choice` at random keys, `K`, starts, lengths,
tables and output alignments, through the tile and the direct kernel, and `device_rng::choice`
equals `tandem_choice`. The f32 normal's device square root equals the IEEE square root on the radius of every Float32 draw. A generator is checked by running mixed fills of every width through it and through the C
generator, which must agree in values and in the final position. The fill comparison covers u8, u16, u32, u64, f16 bits,
f32, f64 and bool. The signed fills are compared with the C unsigned fills and their
returned positions. `tests/host_core.cpp` (`make host`) builds `core.hpp` as C++17 with clang and
gcc and compares the scalar generator with the C library, because tandem-kokkos, tandem-fortran
and tandem-torch include it with their own standards. Beyond the conformance cases, `choice_build`
equals `tandem_choice_build` entry for entry for random weights that span zeros, subnormals and
`DBL_MAX`, and `Rng::choice` equals `tandem_choice`. `./tests/host_core --dump` writes the bytes of tandem-c's
`tools/dump_normals`. The hashes hold on the M4 with clang and gcc 16, and on x86 with clang 20 and
gcc 14 under `-mavx2 -mfma`, also with `-ffp-contract=fast` and `-march=native`.

`tools/normal_stats.cu` (`make stats`) checks 1e8 device f64 normals once per change of the normal
code, not in CI: moments 1 to 6 and the counts beyond 3, 3.5, 4, 4.5 and 5 at `|z| < 4`, and the
Kolmogorov-Smirnov and Anderson-Darling tests at `p > 0.001`. On the A100 the largest `|z|` was
1.32, KS gave `p = 0.96` and AD `p = 0.97`.

`tests/test_thrust.cu` (`make thrust`) checks that the functors and iterators equal the fills at
random keys, `K`, positions and lengths, for every type and for bounded ranges that reject, that
the uniform functors hash to `hashes.json`, and that `thrust::reduce`, `thrust::copy_n` and
`cub::DeviceReduce::Sum` read an iterator correctly. Normals match the fills to the tolerances in [design](design.md).

`tests/reject_widened.cu` (`make reject`) is a negative compile test. A kind with 16-bit draws into
32-bit outputs must stop at the `static_assert` of the widened tile store, which pairs two 32-bit
draws per 8 bytes and would leave half of each block unwritten. Its 32-bit control must compile.

## CI

GitHub runners have no GPU, so CI compiles the tests and the bench for `sm_80`, runs `make reject`
and `make host`, compares `tests/conformance` with tandem-spec f420545, and checks
that `tests/vectors.h` and `include/tandem/normal_tables.hpp` match the spec repository's
`vectors.json` and `tables/normal_f64_zig1024.json`.
