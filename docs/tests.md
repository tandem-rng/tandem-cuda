# Tests

GitHub runners have no GPU, so CI compiles the tests and the bench for `sm_80` and checks
that `tests/vectors.h` matches the spec repository's `vectors.json`. Run the tests on a GPU
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
`split` and `sub`, is compared with the C library, f32 normals to 4 ulps and the rest bit for
bit. Bounded fills are compared at ranges that reject often, rarely and never, against a
reference of the [design](design.md) contract written on the C library's generators, and, where nothing
rejects, against the sequential C bounded draws. The C library's bounded fills must equal that
reference too. A bounded fill cut at a random element, at ranges that reject about half the draws,
equals the whole fill for both widths, the fused low-bound and wider outputs and the generator. Normal fills are compared with the host Box-Muller step on the C library's uniform fills, at
every start slot, odd and even `n`, bit for bit (f64) and to 16 ulps + 1e-6 (f32, 4 ulps with
`TANDEM_PRECISE_F32_NORMAL`), and with `tests/cross_fill_normal.h` and tandem-c's
`tests/cross_normal.h`, f64 bit for bit. Exponential fills equal tandem-c's
`tandem_fill_exponential_f64` and `_f32` byte for byte at random keys, `K`, start slots and
lengths up to 2^22, and match `tests/cross_fill_exponential.h` and tandem-c's
`tests/cross_exponential.h` exactly. The device generator's `exponential()` and `exponentialf()`
equal the C library's scalar draws. A generator is checked by running mixed fills of every width through it and through the C
generator, which must agree in values and in the final position. The fill comparison covers u8, u16, u32, u64, f16 bits,
f32, f64 and bool. The signed fills are compared with the C unsigned fills and their
returned positions. `tests/host_core.cpp` (`make host`) builds `core.hpp` as C++17 with clang and
gcc and compares the scalar generator with the C library, because tandem-kokkos, tandem-fortran
and tandem-torch include it with their own standards. It also hashes 1e6 pairs of f64 and f32
normals from five start positions and checks the value of tandem-c's `tests/test_normal_bits.c`,
and does the same for 1e6 f64 and 1e6 f32 exponentials against `tests/test_exponential_bits.c`.
`./tests/host_core --dump` writes the bytes of tandem-c's `tools/dump_normals`: they are identical
on the M4 with clang and on x86 with clang 19, gcc 11 and gcc 14, with and without `-mfma`.
