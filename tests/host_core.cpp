// core.hpp on a host compiler in C++17, the standard its other consumers (Kokkos, Fortran, torch)
// build with: the scalar generator, the normals and the exponentials agree with the C library and
// the fixtures bit for bit, and the f32 Box-Muller block with libm to a tolerance.
// Run: make host
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>
#include "tandem/core.hpp"
extern "C" {
#include "tandem.h"
}
#include "cross_fill_below.h"
#include "cross_fill_exponential.h"
#include "cross_fill_normal.h"
#include "tests/cross_normal.h" // tandem-c's, through -I$(TANDEM_C)

static int bad;

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

// tandem-c's tests/test_normal_bits.c from the scalar draws, which equal the fills: the f64 normals
// of five starts, the bytes of its tools/dump_normals.c, and the f32 normals of the same starts,
// whose blocks run the explicit fused multiply-adds, so every compiler and target agrees.
constexpr uint64_t NORMAL_F64_HASH = 0xa61cfa844c85f7c1ull, NORMAL_F32_HASH = 0xaa1ea656ce73a4fbull;
constexpr uint64_t NORMAL_STARTS[] = {0, 1, 77, 12345, 1u << 30};

static uint64_t normal_bits_f64(FILE *dump) {
    std::vector<double> z(1000000);
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t s : NORMAL_STARTS) {
        tandem::Rng g(2026, 7, 0);
        g.set_position(s);
        for (auto &x : z) x = g.normal();
        h = fnv(h, z.data(), z.size() * 8);
        if (dump) std::fwrite(z.data(), 8, z.size(), dump);
    }
    return h;
}

// An odd count, so the last element is the scalar cos half and both the vector body and the
// one-pair path take part.
static uint64_t normal_bits_f32() {
    const size_t pairs = 1000000;
    std::vector<float> u(2 * (pairs - 1)), z(2 * pairs - 1);
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t s : NORMAL_STARTS) {
        tandem::Rng g(2026, 7, 0);
        g.set_position(s);
        for (auto &x : u) x = g.frand();
        tandem::normal_block_f32(u.data(), z.data(), pairs - 1);
        z.back() = g.normalf();
        h = fnv(h, z.data(), z.size() * 4);
    }
    return h;
}

// tandem-c's tests/test_exponential_bits.c from the scalar draws, which equal the fills.
constexpr uint64_t EXPONENTIAL_BITS_HASH = 0x47f8f98297d94ee2ull;

static uint64_t exponential_bits() {
    const size_t n = 1000000;
    const uint64_t starts[] = {0, 1, 77, 12345, 1u << 30};
    std::vector<double> e(n);
    std::vector<float> ef(n);
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t s : starts) {
        tandem::Rng g(2026, 7, 0);
        g.set_position(s);
        for (auto &x : e) x = g.exponential();
        h = fnv(h, e.data(), n * sizeof e[0]);
        for (auto &x : ef) x = g.exponentialf();
        h = fnv(h, ef.data(), n * sizeof ef[0]);
    }
    return h;
}

int main(int argc, char **argv) {
    // --dump writes the bytes of tandem-c's tools/dump_normals, for cmp against it.
    if (argc > 1 && std::strcmp(argv[1], "--dump") == 0) return normal_bits_f64(stdout), 0;
    static_assert(std::is_trivially_copyable<tandem::Rng>::value, "copy");
    tandem::Rng r(42, 0, 32);
    tandem_rng c = tandem_from_key(r.key().w, 0, 32);
    for (int i = 0; i < 1000; i++) {
        bad += r.urand(1000u) != tandem_u32_below(&c, 1000u);
        bad += r.urand64(3000000000000ull) != tandem_u64_below(&c, 3000000000000ull);
        bad += r.normal() != tandem_normal_f64(&c);
        bad += r.normalf() != tandem_normal_f32(&c);
        bad += r.exponential() != tandem_exponential_f64(&c);
        bad += r.exponentialf() != tandem_exponential_f32(&c);
        auto p = r.normal2(); // two f64 draws
        bad += p.z0 != tandem_normal_f64(&c);
        bad += p.z1 != tandem_normal_f64(&c);
        float fz[2];
        tandem_normal2_f32(&c, fz);
        auto pf = r.normalf2();
        bad += pf.z0 != fz[0] || pf.z1 != fz[1];
        bad += r.bit() != tandem_next_bool(&c);
    }
    tandem::Rng kids[3], a = r;
    r.fork(kids, 3);
    tandem_rng ck[3];
    tandem_fork(&c, ck, 3);
    for (int i = 0; i < 3; i++) bad += kids[i].urand64() != tandem_next_u64(&ck[i]);
    bad += r.position() != tandem_position(&c);
    bad += !(a == a);

    // The f32 block routine against libm over the whole range of a, including a near 0 and 1.
    const size_t m = 200000;
    std::vector<float> uf(2 * m), zf(2 * m);
    tandem::Rng g(7, 0, 32);
    for (size_t i = 0; i < 2 * m; i++) uf[i] = g.frand();
    uf[0] = 0.0f, uf[1] = 0.0f, uf[2] = 1.0f - 0x1p-24f, uf[3] = 0.5f;
    tandem::normal_block_f32(uf.data(), zf.data(), m);
    float worst_ulp = 0;
    for (size_t j = 0; j < m; j++) {
        double rf = std::sqrt(-2.0 * std::log(1.0 - (double)uf[2 * j]));
        double ang = 6.283185307179586 * (double)uf[2 * j + 1];
        double f0 = rf * std::cos(ang), f1 = rf * std::sin(ang);
        float ulp = 0x1p-23f;
        float d0 = (float)(std::fabs(zf[2 * j] - f0) / (rf + 1e-30) / ulp), d1 = (float)(std::fabs(zf[2 * j + 1] - f1) / (rf + 1e-30) / ulp);
        worst_ulp = std::fmax(worst_ulp, std::fmax(d0, d1));
    }
    bad += !(worst_ulp <= 8.0f);
    std::printf("f32 block vs libm: %.3g ulps of r\n", (double)worst_ulp);

    // The f64 normals of a Python implementation written from the text of spec Appendix A, as
    // tandem-c's tests/test_normal_bits.c hashes them: 2e5 from key {1, 2, 3, 4}, K = 32, at bits 0
    // and 2373.
    {
        const struct {
            uint64_t start, hash, end;
        } ref[] = {{0, 0x0c4059ed409d578dull, 12800000}, {2373, 0x30ce40c86b295193ull, 12802432}};
        std::vector<double> z(200000);
        for (const auto &f : ref) {
            tandem::Rng q = tandem::Rng::from_key(tandem::Key{{1, 2, 3, 4}}, f.start, 32);
            for (auto &x : z) x = q.normal();
            bad += fnv(0xcbf29ce484222325ull, z.data(), z.size() * 8) != f.hash || q.position() != f.end;
        }
    }

    // The fixtures come from this host code, printed to round trip, so they match exactly.
    tandem::Rng root(42, 0, 32);
    for (const auto &f : CROSS_NORMAL64) {
        tandem::Rng q = root;
        q.set_position(f.pos);
        double seq[64];
        for (unsigned i = 0; i < f.n; i++) seq[i] = q.normal();
        bad += std::memcmp(seq, f.out, f.n * 8) != 0;
    }
    // The f64 rows are tandem-c's tests/cross_normal.h rows, byte for byte, end positions too.
    static_assert(sizeof CROSS_NORMAL / sizeof CROSS_NORMAL[0] == sizeof CROSS_NORMAL64 / sizeof CROSS_NORMAL64[0], "rows");
    for (size_t i = 0; i < sizeof CROSS_NORMAL / sizeof CROSS_NORMAL[0]; i++) {
        const auto &f = CROSS_NORMAL64[i];
        bad += f.pos != CROSS_NORMAL[i].start || f.n != CROSS_NORMAL_COUNT;
        bad += std::memcmp(f.out, CROSS_NORMAL[i].want, sizeof CROSS_NORMAL[i].want) != 0;
        bad += tandem::align_pos(f.pos, 64) + 64 * f.n != CROSS_NORMAL[i].end_pos;
    }
    for (const auto &f : CROSS_NORMAL32) {
        tandem::Rng q = root;
        q.set_position(f.pos);
        for (unsigned i = 0; i < f.n; i++) {
            auto pr = tandem::box_muller2_f32(q.at_frand(2 * (i / 2)), q.at_frand(2 * (i / 2) + 1));
            float v = i % 2 ? pr.z1 : pr.z0;
            bad += v != f.out[i];
        }
    }
    for (const auto &f : CROSS_EXP64) {
        tandem::Rng q = root;
        q.set_position(f.pos);
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, f.pos, 32);
        double seq[64], fill[64];
        for (unsigned i = 0; i < f.n; i++) seq[i] = q.exponential();
        tandem_fill_exponential_f64(&g, fill, f.n);
        bad += std::memcmp(seq, f.out, f.n * 8) != 0 || std::memcmp(fill, f.out, f.n * 8) != 0;
        bad += q.position() != tandem_position(&g);
    }
    for (const auto &f : CROSS_EXP32) {
        tandem::Rng q = root;
        q.set_position(f.pos);
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, f.pos, 32);
        float seq[64], fill[64];
        for (unsigned i = 0; i < f.n; i++) seq[i] = q.exponentialf();
        tandem_fill_exponential_f32(&g, fill, f.n);
        bad += std::memcmp(seq, f.out, f.n * 4) != 0 || std::memcmp(fill, f.out, f.n * 4) != 0;
        bad += q.position() != tandem_position(&g);
    }
    // The bounded fill fixtures, with rejections keyed by the global draw index at nonzero
    // starts, are the C library's fills.
    auto below32_ok = [](uint64_t start, uint32_t range, const uint32_t *want) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, start, 32);
        uint32_t out[64];
        tandem_fill_u32_below(&g, out, 64, range);
        return std::memcmp(out, want, sizeof out) == 0;
    };
    auto below64_ok = [](uint64_t start, uint64_t range, const uint64_t *want) {
        tandem_rng g = tandem_from_key(CROSS_FILL_KEY, start, 32);
        uint64_t out[64];
        tandem_fill_u64_below(&g, out, 64, range);
        return std::memcmp(out, want, sizeof out) == 0;
    };
    for (const auto &f : CROSS_BELOW32) bad += !below32_ok(0, f.range, f.out);
    for (const auto &f : CROSS_BELOW64) bad += !below64_ok(0, f.range, f.out);
    for (const auto &f : CROSS_BELOW32_AT) bad += !below32_ok(f.start, f.range, f.out);
    for (const auto &f : CROSS_BELOW64_AT) bad += !below64_ok(f.start, f.range, f.out);

    uint64_t he = exponential_bits();
    bad += he != EXPONENTIAL_BITS_HASH;
    std::printf("exponential bits: hash %016llx, expected %016llx\n", (unsigned long long)he,
                (unsigned long long)EXPONENTIAL_BITS_HASH);

    uint64_t h = normal_bits_f64(nullptr), hf = normal_bits_f32();
    bad += h != NORMAL_F64_HASH || hf != NORMAL_F32_HASH;
    std::printf("normal bits: f64 %016llx, expected %016llx, f32 %016llx, expected %016llx\n",
                (unsigned long long)h, (unsigned long long)NORMAL_F64_HASH, (unsigned long long)hf,
                (unsigned long long)NORMAL_F32_HASH);
    std::printf("host core: %s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
