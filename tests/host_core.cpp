// core.hpp on a host compiler in C++17, the standard its other consumers (Kokkos, Fortran, torch)
// build with: the scalar generator and the host Box-Muller agree with the C library, bit for bit
// where both use the same polynomial code, and with libm and the fixtures to a tolerance.
// Run: make host
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>
#include "tandem/core.hpp"
extern "C" {
#include "tandem.h"
}
#include "cross_fill_normal.h"

static int bad;

int main() {
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

    // The fixtures, regenerated from libm before this code existed.
    tandem::Rng root(42, 0, 32);
    for (const auto &f : CROSS_NORMAL64) {
        tandem::Rng q = root;
        q.set_position(f.pos);
        for (unsigned i = 0; i < f.n; i++) {
            auto pr = tandem::box_muller2(q.at_drand(2 * (i / 2)), q.at_drand(2 * (i / 2) + 1));
            double v = i % 2 ? pr.z1 : pr.z0;
            bad += !(std::fabs(v - f.out[i]) <= 1e-12 * (1.0 + std::fabs(f.out[i])));
        }
    }
    for (const auto &f : CROSS_NORMAL32) {
        tandem::Rng q = root;
        q.set_position(f.pos);
        for (unsigned i = 0; i < f.n; i++) {
            auto pr = tandem::box_muller2_f32(q.at_frand(2 * (i / 2)), q.at_frand(2 * (i / 2) + 1));
            float v = i % 2 ? pr.z1 : pr.z0;
            bad += !(std::fabs(v - f.out[i]) <= 8 * 0x1p-23f * std::fabs(f.out[i]) + 1e-6f);
        }
    }
    std::printf("host core: %s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
