# Tests

GitHub runners have no GPU, so CI compiles the tests and the bench for `sm_80` and checks
that `tests/vectors.h` and `include/tandem/normal_tables.hpp` match the spec repository's
`vectors.json` and `tables/normal_f64_zig1024.json`. Run the tests on a GPU
host:

```sh
make test TANDEM_C=../tandem-c      # or: pixi install && pixi run test
```

`tests/test_thrust.cu` (`make thrust`) checks that the functors and iterators equal the fills at
random keys, `K`, positions and lengths, for every type and for bounded ranges that reject, that
they match the stream dumps, and that `thrust::reduce`, `thrust::copy_n` and
`cub::DeviceReduce::Sum` read an iterator correctly. Normals match the fills to the tolerances in [design](design.md).

`tests/test_cuda.cu` checks every vector of the specification, compares device fills and
device scalar draws with reference stream dumps in `tests/data` (the bool, u8 and f16 dumps
through the public launchers), and compares device fills at random keys,
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
covers both kernels and both store paths, about 43000 misses and 570 tail values. They also match
tandem-c's `tests/cross_normal.h` rows and its hashes of a Python implementation of Appendix A,
2e5 elements from bits 0 and 2373. A normal fill cut at an odd element, at a missed element and
just after it equals the whole fill. f32 normal fills are compared with the host Box-Muller step
on the C library's uniform fills, at every start slot, odd and even `n`, to 16 ulps + 1e-6 (4 ulps
with `TANDEM_PRECISE_F32_NORMAL`), and with tandem-c's fixture. `tests/cross_fill_normal.h`
matches exactly in f64 and to the same tolerance in f32. An empty f64 normal fill aligns the
position to 64 bits, the f32 one leaves it alone. Exponential fills equal tandem-c's
`tandem_fill_exponential_f64` and `_f32` byte for byte at random keys, `K`, start slots and
lengths up to 2^22, and match `tests/cross_fill_exponential.h` and tandem-c's
`tests/cross_exponential.h` exactly. The device generator's `exponential()` and `exponentialf()`
equal the C library's scalar draws. A generator is checked by running mixed fills of every width through it and through the C
generator, which must agree in values and in the final position. The fill comparison covers u8, u16, u32, u64, f16 bits,
f32, f64 and bool. The signed fills are compared with the C unsigned fills and their
returned positions. `tests/host_core.cpp` (`make host`) builds `core.hpp` as C++17 with clang and
gcc and compares the scalar generator with the C library, because tandem-kokkos, tandem-fortran
and tandem-torch include it with their own standards. It hashes 1e6 f64 and 2e6 - 1 f32 normals
from five start positions and checks the values of tandem-c's `tests/test_normal_bits.c`, checks
its Python-reference hashes, compares the f64 rows of `tests/cross_fill_normal.h` with tandem-c's
`tests/cross_normal.h` byte for byte, and hashes 1e6 f64 and 1e6 f32 exponentials against
`tests/test_exponential_bits.c`. `./tests/host_core --dump` writes the bytes of tandem-c's
`tools/dump_normals`. The hashes hold on the M4 with clang and gcc 16, and on x86 with clang 20 and
gcc 14 under `-mavx2 -mfma`, also with `-ffp-contract=fast` and `-march=native`.

`tools/normal_stats.cu` (`make stats`) checks 1e8 device f64 normals once per change of the normal
code, not in CI: moments 1 to 6 and the counts beyond 3, 3.5, 4, 4.5 and 5 at `|z| < 4`, and the
Kolmogorov-Smirnov and Anderson-Darling tests at `p > 0.001`. On the A100 the largest `|z|` was
1.32, KS gave `p = 0.96` and AD `p = 0.97`.
