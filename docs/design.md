# Design of the bounded, normal and exponential fills

These fills are not part of the specification. Other ports should follow the same contract,
and the C library's `tandem_fill_u32_below` and `tandem_fill_u64_below` give the same values bit
for bit, rejected draws included.

**Normals.** One Box-Muller step turns two uniforms `a`, `b` into two normals,
`r = sqrt(-2 ln(1 - a))`, `z0 = r cos(2 pi b)`, `z1 = r sin(2 pi b)`. `Rng::normal2()` returns
the pair `(z0, z1)`, `Rng::normal()` its first half, and both consume two Float64 uniforms. A
fill of `n` normals is the flattened sequence of `normal2` calls: pair `j`, the elements `2j` and
`2j + 1`, comes from the Float64 draws `2j` and `2j + 1` of the Float64 fill that starts at the
same position. The fill starts at `pos` aligned up to 64 bits and consumes `2 ceil(n / 2)`
draws, so an odd `n` uses the cos half of its last pair and still advances past both draws. An empty
normal or bounded fill consumes nothing and returns `pos` unchanged, even when `pos` is unaligned. A
start at an odd Float64 draw makes every pair span two blocks, and the kernel steps a second
chunk per thread to read them, at about 80% of the speed. Devices and hosts run the same
polynomial step below, so f64 normals agree bit for bit on every platform, and with tandem-c.

`box_muller2` in `core.hpp`, on a device and on a host, and `box_muller2_f32` on a host do not
call libm. They are the polynomial form of tandem-c: the logarithm from the exponent bits and a short series, and the
sine and cosine from an exact quarter-turn reduction and polynomials, at most 9.9e-16 relative in
f64 and 3.3 ulps in f32 against libm. Every multiply-add is an explicit `std::fma` and contraction
is off, as in tandem-c, so every host compiler and target gives the C library's bits, and a scalar
`normal2()` equals a pair of a fill. The f64 step `tandem::normal_pair_f64` is inlined into the
device kernels too: no plain product feeds a plain sum except an exact one, so device contraction
cannot change the bits. It took the A100 f64 fill from 706 GiB/s with `log` and `sincospi` to 833. `tandem::normal_block_f64` and `normal_block_f32` turn arrays
of uniforms into normals for host fills, and clang vectorizes them on Arm and on x86 without
`-ffast-math` (`make hostvec` checks it). x86 needs `-mfma` (the Makefile passes `-mavx2 -mfma`),
because without it `std::fma` is a slow library call that gives the same bits.

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
bit. `tests/cross_fill_normal.h` holds normal fixtures for ports at even and odd starts, from the
host polynomial code: host builds match them exactly, devices to the tolerances above.

**Exponentials.** `fill_exponential_f64` writes element `i` as `-ln(1 - u)` of Float64 draw `i` of
the fill that starts at `pos` aligned up to 64 bits, and `fill_exponential_f32` the same in float
on Float32 draws aligned up to 32 bits, as Appendix A of the specification defines them. A fill
consumes `n` draws, equals the `Rng::exponential()` (`exponentialf()`) calls, and an empty fill
returns `pos` unchanged. `tandem::exponential_f64` and `exponential_f32` in `core.hpp` compute the
logarithm of the f64 and f32 Box-Muller steps on hosts and devices: no libm call, every multiply-add
an explicit `std::fma`, and no plain product feeding a plain sum. Device contraction therefore
cannot change the bits, and every host and device returns tandem-c's values, which the tests check
byte for byte. Building device code with `-use_fast_math` or `-prec-div=false` changes the
division and breaks that. The maximum error is 1.1e-15 relative in f64 and 2.8e-7 in f32. The
fills run the direct kernel, because the map makes them compute bound and the tile kernel's
separate write phase lost 7 to 13 % on the A100. `tests/cross_fill_exponential.h` holds fixtures
for ports at five start positions, unaligned ones included.

**Bounded integers.** Element `e` uses its own draw `d[e]` of the UInt32 (UInt64) fill and
Lemire's multiply and reject: `m = d * range`, accepted when the low word of `m` is at least
`2^32 mod range` (or its 64-bit analogue), result the high word. The threshold `2^32 mod range` is computed once per fill, not per element, which took
large ranges from 1107 to 1348 GiB/s (`u32`) and from 644 to 1351 (`u64`). The fill consumes exactly `n`
draws, so it returns `align(pos, w) + w n` at once, without waiting for the device. A sequential
`Rng::urand(range)` loop would consume extra draws after a rejection, and a parallel fill
cannot know how many. So a rejected element `e` retries on a fallback stream: draws `0, 1, ...`
of `split(g)` of `sub(P)` of the fill's generator at position 0 (same key and `K`), with
`P = 0x424c573332` for 32-bit and `0x424c573634` for 64-bit ranges, until one is accepted. `g` is
the global draw index `align(pos, w) / w + e` (spec Appendix A), so a fill cut at any element
boundary, each piece starting where the last ended, equals the whole fill.
Those two purposes are reserved. A rejection has probability `(2^32 mod range) / 2^32`, so
ranges that are powers of two never reject, and a fill without rejections equals the sequential
loop. `range = 0` returns 0 and still consumes the draw. `tests/cross_fill_below.h` holds fixtures for
ports, bounded fills of 64 elements at several ranges from the key of seed 42, `K = 32`, with the
rejection counts: `CROSS_BELOW32` and `CROSS_BELOW64` at position 0, where `g = e`, and
`CROSS_BELOW32_AT` and `CROSS_BELOW64_AT` at the bit positions 1 and 12345 of tandem-c's fixtures,
where `g` differs from `e`. Regenerate it with `make cross`, which needs only a host
C++ compiler and `core.hpp`.
