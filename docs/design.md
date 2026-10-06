# Design

These fills are not part of the specification. Other ports should follow the same contract,
and the C library's `tandem_fill_u32_below` and `tandem_fill_u64_below` give the same values bit
for bit, rejected draws included.

## Bounded integers

Element `e` uses its own draw `d[e]` of the UInt32 (UInt64) fill and
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

## Normals

### Float64

`fill_normal_f64` is the 1024-layer ziggurat of Appendix A of the
specification. Element `e` comes from UInt64 draw `e` of the fill that starts at `pos` aligned up
to 64 bits: bits 0-9 pick the layer `i`, bit 10 the sign, bits 11-63 the magnitude `ra`, and
`x = +-ra W[i]` is the normal when `ra < K[i]`, 99.57 % of the time. A miss continues on the draws
of `split(g)` of `sub(0x4e524d3634)` of the generator with the fill's key at position 0, where `g`
is the global draw index `align(pos, 64) / 64 + e`, through the wedge test `ln y < -x^2 / 2` and
Marsaglia's tail. Each element starts its own fallback, so a fill consumes `n` draws, a fill cut
at any element equals the whole fill, and `Rng::normal()` is element 0 of a fill.
`Rng::normal2()` is two such draws. An empty fill returns `pos` aligned up to 64 bits, as an empty
uniform fill does.

The tables `W`, `K` and `Y` come from the specification's `tables/normal_f64_zig1024.json`
through `tools/gen_normal_tables.py`, which checks the file's SHA-256, into
`include/tandem/normal_tables.hpp` (`make tables`). CI regenerates the header from the spec
repository and fails on any difference. `W[i]` and `K[i]` sit side by side, so the fast path reads
one 16-byte entry: two separate tables cost the A100 the L1 throughput of a second gather. Device
code reads its own copy in global memory through the read-only cache. `ln` is the reference
logarithm `-L(x) / 2`, the polynomial of the exponentials, every multiply-add an explicit
`std::fma`. Every other operation of the slow path rounds once: the wedge's product is an fma with
a zero addend and its sum `__dadd_rn` on a device, so no compiler fuses them. Host and device
therefore return tandem-c's values bit for bit, also under `-ffp-contract=fast`.

The fill runs two kernels. The table pass is the direct fill kernel with the fast path: one
thread per chunk, each block's two elements stored as one 16-byte vector. A miss goes to a queue
in shared memory, and the block appends its queue to a list in global memory with one atomic add.
The second kernel continues each listed miss, one thread per miss. The table pass makes no call,
which keeps its registers and its loop free of the slow path's setup. The list has room for
`n / 128` misses, twice the expected count, `n / 8` bytes. If it overflows, the second kernel
walks the whole fill again and continues every miss.

The library keeps the lists between fills, per device. A fill takes a list whose last fill is
done, or one last used on its own stream, which it waits for in stream order. Otherwise it
allocates one with `cudaMalloc`, rounded up to a power of two. So the lists grow to the number of
streams that fill at once, and they live to the end of the program. A list from
`cudaMallocAsync` in each fill went back to the default pool at every synchronize, after which
the next allocation waited about 0.7 ms on the host. With a synchronize after each fill, the A100
then wrote these medians (GiB/s):

| fill | `cudaMallocAsync` | kept list | `cudaMallocAsync`, release threshold raised |
|---|---|---|---|
| 2^28, even start | 588, 591 | 1059, 1061 | 1055, 1059 |
| 2^28, odd start | 578, 583 | 975, 988 | 975, 997 |
| 2^24 | 123 | 803, 808 | 808 |
| 2^20 | 9 | 231 | 231 |

Ten fills back to back gave 995 against 1070 at 2^28 and 67 against 273 at 2^20. The kept list
matched the raised threshold everywhere. An earlier comparison by the fastest of 21 fills found
caching no faster, because the fastest fill was one that found the pool's memory. A fill captured
into a graph still takes its list from `cudaMallocAsync`, so that each launch of the graph has
its own.
Fills below 2^16 elements, or without a list, run one kernel that continues each miss in place
and allocate nothing.

When the output and the stream differ by 8 bytes modulo 16, as for a start at an odd draw into an
aligned buffer, an element pair of one block straddles two 16-byte slots. A warp shuffle then
brings each thread the low element of the next lane, and the thread stores its high element with
it. Lane 7's partner is lane 0's next block, and the last lane of a group pairs with the next
group's first element, through the warp or, across warps, shared memory. Every group's 128 bytes
then cover whole 32-byte sectors. A row 8 bytes off the sectors halved the speed, even with
16-byte stores, and one lone element at each group's ends, a sector half written by one warp and
finished by another much later, cost 15 %. Both kernels store each block one step late, so that
the store does not wait for the step's table reads.

Those stores cover whole sectors only when the stream's rows start on a sector, or 8 bytes
before one for an odd draw. A start at draw 2 or 3 modulo 4 into a 32-byte aligned output put
each group's 128 bytes 16 bytes off the sectors, five sectors per group and step, and word 6
(draw 3) ran at 689 GiB/s against 990 for draw 1. Those starts take tandem-sycl's octet stores
(`fill_normal64_octets`): units of two elements from the fill's first draw, lane l of a group
writing unit 8m + l of octet m, 128 aligned bytes, with the values passed between lanes by warp
shuffles and stored one step late. Word 6 then runs at 931 to 954. The octets lost 3 to 11 % at
the other starts, so those keep the kernel above.

### Float32

`fill_normal_f32` is Box-Muller on Float32 draws, in float: pair `j`, the
elements `2j` (cos half) and `2j + 1` (sin half), comes from the draws `2j` and `2j + 1`, as the
flattened `Rng::normalf2()` calls, with `u = 1 - d[2j]`, `v = d[2j + 1]`, precise `logf` and
`sqrtf`. An odd `n` uses the cos half of its last pair and still advances past both draws, and an
empty fill returns `pos` unchanged. The device fill takes the
angle through the fast `__sincosf` on `2 pi (v - 0.5)`, which is accurate on `[-pi, pi]`, because
the precise `sincospif` made the fill compute bound at 1065 GiB/s against 1300 memory bound. The
result differs from the precise step by at most 1.73e-6 absolute over the random fills and 7.2e-7
on the fixtures, up to 47 ulps for a value near 0.01, where the absolute error dominates. The
bound is 16 ulps + 2.1e-6: `__sincosf` errs by up to 2^-21.41 absolute on `[-pi, pi]`, times a
radius of up to 5.77. That exceeds the 1e-6 floor of the specification's tolerance for about one
value in 10^6. The tests' seed holds one: `a = 0x1.fff256p-1`, `b = 0x1.040388p-2`, radius 5,
give -0.10545215 against the host's -0.105453432, 1.28e-6 apart. So the tests take 16 ulps +
2.1e-6 for the device fill. On the A100, `sincospif` made the fill compute bound at 1065 GiB/s. The
C reference's float polynomials, bit exact with tandem-c, gave 935, also with precise `logf`
for the radius, while `__sincosf` gave 1270 in the same runs. `__logf` is not used, because its absolute error near 1 distorts small radii by
thousands of ulps. Define `TANDEM_PRECISE_F32_NORMAL` for `sincospif` and 4 ulps. The fill starts at `pos` aligned up to 32 bits and consumes `64 ceil(n / 2)` bits, so
a block holds two pairs and a start at an odd Float32 draw makes some span two blocks. Float normals agree across ports and devices to a few ulps, not bit for
bit, because libm float functions differ. The host version of the f32 step takes its angle in
double and rounds the results, because a float angle `2 pi b` is off by up to `2 pi b 2^-24`
where `sincospif` is not. The uniforms are exact, and everything else in this library is bit for
bit. `tests/cross_fill_normal.h` holds normal fixtures for ports, from the host code: f64 ziggurat
fills at the six starts of tandem-c's `tests/cross_normal.h`, with misses of every kind, exact
everywhere, and f32 fills at even and odd starts, exact on hosts and to the tolerances above on
devices.

## Exponentials

`fill_exponential_f64` writes element `i` as `-ln(1 - u)` of Float64 draw `i` of
the fill that starts at `pos` aligned up to 64 bits, and `fill_exponential_f32` the same in float
on Float32 draws aligned up to 32 bits, as Appendix A of the specification defines them. A fill
consumes `n` draws, equals the `Rng::exponential()` (`exponentialf()`) calls, and an empty fill
returns `pos` unchanged. `tandem::exponential_f64` and `exponential_f32` in `core.hpp` compute the
reference logarithm and the logarithm of the f32 Box-Muller step on hosts and devices: no libm
call, every multiply-add
an explicit `std::fma`, and no plain product feeding a plain sum. Device contraction therefore
cannot change the bits, and every host and device returns tandem-c's values, which the tests check
byte for byte. Building device code with `-use_fast_math` or `-prec-div=false` changes the f64
division and breaks that. The maximum error is 1.1e-15 relative in f64 and 2.8e-7 in f32. The
fills run the direct kernel, because the map makes them compute bound and the tile kernel's
separate write phase lost 7 to 13 % on the A100. `tests/cross_fill_exponential.h` holds fixtures
for ports at five start positions, unaligned ones included.

The reference logarithm multiplies by -4 and halves the result. `neg_log_f64` and
`exponential_f32` move those powers of two into the operands instead: the numerator `2 - 2m`
gives `t = -2s`, the polynomial runs on `t^2` with each coefficient of `s^(2j)` divided by `4^j`,
and the `ln 2` terms are halved. A power of two commutes with rounding while no value is
subnormal, so each operation rounds to the reference's value times that power of two, and the
result has the reference's bits with two multiplications fewer. On a device the f32 division is
the fast path of the IEEE division without its range check and slow path, which only inputs
outside `[1, 4)` or near the float range's ends need. `tests/test_cuda.cu` compares the f32 map
with tandem-c's arithmetic for all 2^24 Float32 draws, on the device and on the host. On the A100
the f32 fill went from 1023 to 1150 GiB/s and the f64 fill from 909 to 935 to 931 to 955. The
f64 division keeps its check: the same steps with `rcp.approx.ftz.f64` as the seed differed from
`div.rn.f64` in 0.2 % of 2^34 test inputs.

Every fill reaches the A100's 250 W power cap when it runs for long. A bench that times the
fastest of a few calls after a shared warm-up then measures the power the previous rows left.
The f32 exponential row read 1022 GiB/s after the f64 Box-Muller rows, 1315 after the ziggurat
rows that replaced them in commit 0ff5f18, and 1140 alone, with the same machine code. So
`tools/bench` runs each fill for two seconds before it times that fill, and takes the median.
The f32 exponential row then read 1019 to 1036 alone and in the table, before the folded
arithmetic above.

At the power cap the SM clock depends on the work per byte. The uniform tile fills run at 1170
to 1215 MHz and the direct ones at 1260, bound by memory. The f32 normal and the exponentials run
at 1035 to 1050 MHz and the f64 normal at about 1100, bound by their arithmetic at that clock.
The f64 exponential spends about 22 f64 operations per element, eight of them in the division,
on 32 f64 lanes per SM. The f64 normal's table pass runs at 1270 GiB/s, and its misses kernel
takes 0.3 ms of the 1.9 ms of a 2^28 fill. Each miss seeds two child keys and its fallback chunk,
three runs of F, before its wedge test. A table in shared memory, the misses kernel at four
blocks per SM, and the fallback's `sub` key computed once on the host gained nothing there.
