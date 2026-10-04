// core.hpp on a host compiler in C++17, the standard its other consumers (Kokkos, Fortran, torch)
// build with: the scalar generator and the host Box-Muller agree with the C library, bit for bit
// where both use the same polynomial code, and with libm and the fixtures to a tolerance.
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
#include "cross_fill_normal.h"

static int bad;

// tandem-c's tests/test_normal_bits.c: the same fills hashed the same way give the same value on
// every compiler and target, which the explicit fused multiply-adds make hold.
constexpr uint64_t NORMAL_BITS_HASH = 0x9414e1315e2653beull;

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

static void block(const double *u, double *z, size_t m) { tandem::normal_block_f64(u, z, m); }
static void block(const float *u, float *z, size_t m) { tandem::normal_block_f32(u, z, m); }

// tandem-c's tools/dump_normals.c: an odd count, so the last element is the scalar cos half and
// both the vector body and the one-pair path take part.
template <class T, class Uniform, class Normal>
static void normal_fill(tandem::Rng &g, std::vector<T> &u, std::vector<T> &z, Uniform uniform,
                        Normal normal) {
    for (auto &x : u) x = uniform(g);
    block(u.data(), z.data(), u.size() / 2);
    z.back() = normal(g);
}

static uint64_t normal_bits(FILE *dump) {
    const size_t pairs = 1000000;
    const uint64_t starts[] = {0, 1, 77, 12345, 1u << 30};
    std::vector<double> u(2 * (pairs - 1)), z(2 * pairs - 1);
    std::vector<float> uf(2 * (pairs - 1)), zf(2 * pairs - 1);
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t s : starts) {
        tandem::Rng g(2026, 7, 0);
        g.set_position(s);
        normal_fill(g, u, z, [](tandem::Rng &r) { return r.drand(); },
                    [](tandem::Rng &r) { return r.normal(); });
        h = fnv(h, z.data(), z.size() * sizeof z[0]);
        if (dump) std::fwrite(z.data(), sizeof z[0], z.size(), dump);
        normal_fill(g, uf, zf, [](tandem::Rng &r) { return r.frand(); },
                    [](tandem::Rng &r) { return r.normalf(); });
        h = fnv(h, zf.data(), zf.size() * sizeof zf[0]);
        if (dump) std::fwrite(zf.data(), sizeof zf[0], zf.size(), dump);
    }
    return h;
}

int main(int argc, char **argv) {
    // --dump writes the bytes of tandem-c's tools/dump_normals, for cmp against it.
    if (argc > 1 && std::strcmp(argv[1], "--dump") == 0) return normal_bits(stdout), 0;
    static_assert(std::is_trivially_copyable<tandem::Rng>::value, "copy");
    tandem::Rng r(42, 0, 32);
    tandem_rng c = tandem_from_key(r.key().w, 0, 32);
    for (int i = 0; i < 1000; i++) {
        bad += r.urand(1000u) != tandem_u32_below(&c, 1000u);
        bad += r.urand64(3000000000000ull) != tandem_u64_below(&c, 3000000000000ull);
        bad += r.normal() != tandem_normal_f64(&c);
        bad += r.normalf() != tandem_normal_f32(&c);
        double cz[2];
        tandem_normal2_f64(&c, cz);
        auto p = r.normal2();
        bad += p.z0 != cz[0] || p.z1 != cz[1];
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

    // The block routines against libm over the whole range of a, including a near 0 and 1.
    const size_t m = 200000;
    std::vector<double> u(2 * m), z(2 * m);
    std::vector<float> uf(2 * m), zf(2 * m);
    tandem::Rng g(7, 0, 32);
    for (size_t i = 0; i < 2 * m; i++) u[i] = g.drand(), uf[i] = g.frand();
    u[0] = 0.0, u[1] = 0.0, u[2] = 1.0 - 0x1p-53, u[3] = 0.5, uf[0] = 0.0f, uf[1] = 0.0f;
    uf[2] = 1.0f - 0x1p-24f, uf[3] = 0.5f;
    tandem::normal_block_f64(u.data(), z.data(), m);
    tandem::normal_block_f32(uf.data(), zf.data(), m);
    double worst = 0;
    float worst_ulp = 0;
    for (size_t j = 0; j < m; j++) {
        double r1 = std::sqrt(-2.0 * std::log(1.0 - u[2 * j]));
        double c0 = r1 * std::cos(6.283185307179586 * u[2 * j + 1]);
        double c1 = r1 * std::sin(6.283185307179586 * u[2 * j + 1]);
        worst = std::fmax(worst, std::fmax(std::fabs(z[2 * j] - c0) / (r1 + 1e-300), std::fabs(z[2 * j + 1] - c1) / (r1 + 1e-300)));
        double rf = std::sqrt(-2.0 * std::log(1.0 - (double)uf[2 * j]));
        double ang = 6.283185307179586 * (double)uf[2 * j + 1];
        double f0 = rf * std::cos(ang), f1 = rf * std::sin(ang);
        float ulp = 0x1p-23f;
        float d0 = (float)(std::fabs(zf[2 * j] - f0) / (rf + 1e-30) / ulp), d1 = (float)(std::fabs(zf[2 * j + 1] - f1) / (rf + 1e-30) / ulp);
        worst_ulp = std::fmax(worst_ulp, std::fmax(d0, d1));
    }
    bad += !(worst <= 1e-12);
    bad += !(worst_ulp <= 8.0f);
    std::printf("block vs libm: f64 %.3g of r, f32 %.3g ulps of r\n", worst, (double)worst_ulp);

    // The fixtures come from this host code, printed to round trip, so they match exactly.
    tandem::Rng root(42, 0, 32);
    for (const auto &f : CROSS_NORMAL64) {
        tandem::Rng q = root;
        q.set_position(f.pos);
        for (unsigned i = 0; i < f.n; i++) {
            auto pr = tandem::box_muller2(q.at_drand(2 * (i / 2)), q.at_drand(2 * (i / 2) + 1));
            double v = i % 2 ? pr.z1 : pr.z0;
            bad += v != f.out[i];
        }
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

    uint64_t h = normal_bits(nullptr);
    bad += h != NORMAL_BITS_HASH;
    std::printf("normal bits: hash %016llx, expected %016llx\n", (unsigned long long)h,
                (unsigned long long)NORMAL_BITS_HASH);
    std::printf("host core: %s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
