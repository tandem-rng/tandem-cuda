// core.hpp on a host compiler in C++17, the standard its other consumers (Kokkos, Fortran, torch)
// build with: the scalar generator, the bounded draws, the normals, the exponentials and weighted
// choice match the spec's conformance cases and the C library bit for bit, and the f32 Box-Muller
// block matches libm to a tolerance. The fills live on the device, so here the host computes each
// fill element from core.hpp's per-element maps or from the scalar draws that equal it.
// Run: make host
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include "tandem/core.hpp"
extern "C" {
#include "tandem.h"
}
#include "conformance.h"

static int bad;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            bad++;                                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                            \
        }                                                                                          \
    } while (0)

static tandem::Rng at(const conf::Case &c) {
    return tandem::Rng::from_key(tandem::Key{{c.key[0], c.key[1], c.key[2], c.key[3]}}, c.start, c.K);
}

// Scalar bounded draws: values and end position, which pins the draws a rejection discards.
static void test_below(const char *dir) {
    for (const auto &c : conf::cases(dir, "below.json")) {
        tandem::Rng q = at(c);
        bool ok = true;
        for (size_t i = 0; i < c.n; i++)
            ok &= (c.kind == "below_u32" ? q.urand((uint32_t)c.range) : q.urand64(c.range)) == c.values[i];
        CHECK(ok && q.position() == c.end);
    }
}

// Element i of a bounded fill is below_u32 or below_u64 of draw i with the global draw index
// g = align(start, w) / w + i, which keys a rejected draw's fallback. Every fixture, the cases with
// rejections included. Width from range: 64 draws with range 1000 from the 32-bit or 64-bit
// interface give CROSS_BELOW32[3] or CROSS_BELOW64[3], and range 0 returns 0 and takes one draw.
static void test_fill_below(const char *dir) {
    auto cs = conf::cases(dir, "fill_below.json");
    for (const auto &c : cs) {
        if (c.n == 0) continue; // empty fills are device fills, see test_cuda.cu
        const unsigned w = c.kind == "fill_below_u32" ? 32 : 64;
        tandem::Rng q = at(c);
        uint64_t g0 = tandem::align_pos(c.start, w) / w;
        bool ok = true;
        for (size_t i = 0; i < c.n; i++) {
            uint64_t v = w == 32 ? tandem::below_u32(q.at_urand(i), (uint32_t)c.range, c.key, c.K, g0 + i)
                                 : tandem::below_u64(q.at_urand64(i), c.range, c.key, c.K, g0 + i);
            ok &= v == c.values[i];
        }
        CHECK(ok);
        if (c.has_end) CHECK(tandem::align_pos(c.start, w) + w * c.n == c.end);
    }
    const auto &b32 = conf::find(cs, "CROSS_BELOW32[3]"), &b64 = conf::find(cs, "CROSS_BELOW64[3]");
    tandem::Rng q = at(b32), r = at(b64);
    for (size_t i = 0; i < 64; i++) {
        CHECK(q.urand(1000u) == b32.values[i]);
        CHECK(r.urand64(1000ull) == b64.values[i]);
    }
    uint64_t p = q.position();
    CHECK(q.urand(0u) == 0 && q.position() == p + 32);
    CHECK(q.urand64(0ull) == 0 && q.position() == p + 128);
    // The fallback is keyed by g, so a fill one draw later is the same fill shifted by one.
    const char *pairs[][2] = {{"CROSS_BELOW32_AT[4]", "CROSS_BELOW32[4]"},
                              {"CROSS_BELOW64_AT[6]", "CROSS_BELOW64[6]"}};
    for (const auto &pr : pairs) {
        const auto &a = conf::find(cs, pr[0]), &z = conf::find(cs, pr[1]);
        CHECK(a.rejected > 0 && std::equal(a.values.begin(), a.values.end() - 1, z.values.begin() + 1));
    }
}

// Float64 normals: n scalar draws equal each fill, end included, which with the fallback keyed by
// the global draw index covers the misses of CROSS_NORMAL[3] to [5]. Float32 normals: pairs of
// normalf2, cos half first, the last odd element the cos half of normalf, which consumes both
// draws. The host block is tandem-c's arithmetic, so it matches the fixtures bit for bit.
static void test_normal(const char *dir) {
    auto cs = conf::cases(dir, "normal.json");
    for (const auto &c : cs) {
        if (c.n == 0) continue;
        tandem::Rng q = at(c);
        bool ok = true;
        if (c.kind == "fill_normal_f64") {
            for (size_t i = 0; i < c.n; i++) {
                double z = q.normal();
                ok &= std::memcmp(&z, &c.values[i], 8) == 0;
            }
            CHECK(q.position() == tandem::align_pos(c.start, 64) + 64 * c.n);
        } else {
            for (size_t i = 0; i < c.n; i += 2) {
                if (i + 1 == c.n) {
                    ok &= q.normalf() == c.f32(i);
                } else {
                    auto p = q.normalf2();
                    ok &= p.z0 == c.f32(i) && p.z1 == c.f32(i + 1);
                }
            }
            CHECK(q.position() == tandem::align_pos(c.start, 32) + 64 * ((c.n + 1) / 2));
        }
        CHECK(ok);
        if (c.has_end) CHECK(q.position() == c.end);
    }
    // A start one draw (one pair) later shifts the output by one element (one pair), and the f32
    // fill of CROSS_NORMALF from start 1 begins with CROSS_NORMAL32[1] from start 32.
    const auto &n0 = conf::find(cs, "CROSS_NORMAL[0]"), &n1 = conf::find(cs, "CROSS_NORMAL[1]");
    CHECK(std::equal(n1.values.begin(), n1.values.end() - 1, n0.values.begin() + 1));
    const auto &f0 = conf::find(cs, "CROSS_NORMAL32[0]"), &f2 = conf::find(cs, "CROSS_NORMAL32[2]");
    CHECK(std::equal(f2.values.begin(), f2.values.end() - 2, f0.values.begin() + 2));
    const auto &nf = conf::find(cs, "CROSS_NORMALF"), &f1 = conf::find(cs, "CROSS_NORMAL32[1]");
    CHECK(nf.end == 4128 && std::equal(f1.values.begin(), f1.values.end(), nf.values.begin()));
    CHECK(conf::find(cs, "CROSS_NORMAL32[0]").n == 33);
}

// Exponentials: n scalar draws equal each fill, end included, bit for bit in both widths.
static void test_exponential(const char *dir) {
    for (const auto &c : conf::cases(dir, "exponential.json")) {
        if (c.n == 0) continue;
        tandem::Rng q = at(c);
        bool ok = true;
        for (size_t i = 0; i < c.n; i++) {
            if (c.kind == "fill_exponential_f64") {
                double e = q.exponential();
                ok &= std::memcmp(&e, &c.values[i], 8) == 0;
            } else {
                ok &= q.exponentialf() == c.f32(i);
            }
        }
        CHECK(ok && q.position() == c.end);
    }
}

// Weighted choice: the table of every vectors.json case, capacity, cut and alias, and the scalar
// draws of every case, which equal element i of the fill and consume 64 bits each. m = 1 always
// gives 0. The table of choice_build is tandem_choice_build's for random weights that span zeros,
// subnormals and DBL_MAX. Invalid weights build no table.
static void test_choice(const char *dir) {
    auto cs = conf::cases(dir, "choice.json");
    for (const auto &c : cs) {
        size_t m = c.weights.size();
        std::vector<uint64_t> cut(m);
        std::vector<uint32_t> alias(m);
        tandem::ChoiceTable t;
        CHECK(tandem::choice_build(t, c.weights.data(), m, cut.data(), alias.data()));
        if (c.n == 0) continue; // the empty fill is a device fill
        CHECK(t.capacity == c.capacity);
        if (!c.cut.empty()) CHECK(cut == c.cut && alias == c.alias);
        tandem::Rng q = at(c);
        bool ok = true;
        for (size_t i = 0; i < c.n; i++) ok &= q.choice(t) == c.values[i] && q.position() == tandem::align_pos(c.start, 64) + 64 * (i + 1);
        CHECK(ok);
        if (c.has_end) CHECK(q.position() == c.end);
        if (m == 1) for (uint64_t v : c.values) CHECK(v == 0);
    }
    const auto &a = conf::find(cs, "CROSS_CHOICE[1]"), &z = conf::find(cs, "CROSS_CHOICE[0]");
    CHECK(std::equal(a.values.begin(), a.values.end() - 1, z.values.begin() + 1));

    std::mt19937_64 gen(31337);
    for (int trial = 0; trial < 400; trial++) {
        size_t m = 1 + gen() % (trial < 300 ? 40 : 5000);
        std::vector<double> w(m);
        for (auto &x : w) {
            switch (gen() % 6) {
            case 0: x = 0; break;
            case 1: x = std::ldexp((double)(gen() >> 11), -1074 - 53 + (int)(gen() % 60)); break;
            case 2: x = std::ldexp((double)(gen() >> 11), 1023 - 53 + (int)(gen() % 2)); break;
            default: x = (double)(gen() >> 11) * 0x1p-53 * std::ldexp(1.0, (int)(gen() % 40) - 20);
            }
        }
        w[gen() % m] = 1;
        std::vector<uint64_t> cut(m), ccut(m);
        std::vector<uint32_t> alias(m), calias(m);
        tandem::ChoiceTable t;
        tandem_choice_table ct;
        bool ok = tandem::choice_build(t, w.data(), m, cut.data(), alias.data());
        CHECK(ok && tandem_choice_build(&ct, w.data(), m, ccut.data(), calias.data()));
        if (!ok) continue;
        CHECK(t.capacity == ct.capacity && cut == ccut && alias == calias);
        tandem::Rng q = tandem::Rng::from_key(tandem::Key{{1, 2, 3, (uint32_t)trial}}, gen() % 5000, 32);
        tandem_rng c = tandem_from_key(q.key().w, q.position(), 32);
        bool same = true;
        for (int i = 0; i < 100; i++) same &= q.choice(t) == tandem_choice(&c, &ct);
        CHECK(same);
    }
    const double nan = std::nan(""), inf = HUGE_VAL;
    const double invalid[][2] = {{1, -1}, {1, nan}, {1, inf}, {0, 0}, {-0.0, 0}};
    uint64_t cut[2];
    uint32_t alias[2];
    tandem::ChoiceTable t;
    for (const auto &w : invalid) CHECK(!tandem::choice_build(t, w, 2, cut, alias));
    CHECK(!tandem::choice_build(t, invalid[0], 0, cut, alias));
}

// The uniform streams of hashes.json from scalar draws, which cross blocks, rows and chunks, and
// random access, which must equal the sequential draws across those boundaries.
static void test_streams(const conf::Json &hashes) {
    struct S {
        const char *file;
        unsigned w;
    } streams[] = {{"k1234_K32_u32.bin", 32}, {"k1234_K8_u32.bin", 32}, {"k1234_K32_u64.bin", 64},
                   {"seed42_K32_f64.bin", 64}, {"seed42_K32_f32.bin", 32}, {"seed42_K32_bool.bin", 1}};
    for (const auto &s : streams) {
        const conf::Json &j = conf::stream(hashes, s.file);
        tandem::Key key;
        for (int w = 0; w < 4; w++) key.w[w] = (uint32_t)j["key"].a[w].hex();
        tandem::Rng q = tandem::Rng::from_key(key, j["start"].u(), (uint32_t)j["K"].u());
        const size_t n = (size_t)j["n"].u();
        const std::string type = j["type"].s;
        std::vector<unsigned char> bytes;
        bool random_access = true;
        for (size_t i = 0; i < n; i++) {
            unsigned char b[8];
            if (type == "UInt32") {
                uint32_t v = q.urand();
                random_access &= v == tandem::Rng::from_key(key, 0, q.chunk_length()).at_urand(i);
                std::memcpy(b, &v, 4);
            } else if (type == "UInt64") {
                uint64_t v = q.urand64();
                random_access &= v == tandem::Rng::from_key(key, 0, q.chunk_length()).at_urand64(i);
                std::memcpy(b, &v, 8);
            } else if (type == "Float64") {
                double v = q.drand();
                random_access &= v == tandem::Rng::from_key(key, 0, q.chunk_length()).at_drand(i);
                std::memcpy(b, &v, 8);
            } else if (type == "Float32") {
                float v = q.frand();
                random_access &= v == tandem::Rng::from_key(key, 0, q.chunk_length()).at_frand(i);
                std::memcpy(b, &v, 4);
            } else {
                b[0] = q.bit();
            }
            bytes.insert(bytes.end(), b, b + s.w / 8 + (s.w == 1));
        }
        CHECK(bytes.size() == j["bytes"].u());
        CHECK(conf::sha256(bytes.data(), bytes.size()) == j["sha256"].s);
        CHECK(random_access);
    }
}

// The long derived outputs of hashes.json: each start's fills run in order on one generator, here
// as the scalar draws that equal them, and the bytes of all starts hash to `fnv1a`. --dump writes
// the bytes of the first dump, tandem-c's tools/dump_normals, for cmp against it.
static void test_dumps(const conf::Json &hashes, FILE *dump) {
    for (const conf::Json &d : hashes["dumps"].a) {
        tandem::Key key;
        for (int w = 0; w < 4; w++) key.w[w] = (uint32_t)d["key"].a[w].hex();
        uint64_t h = conf::FNV_BASIS, bytes = 0, end = 0;
        for (const conf::Json &s : d["starts"].a) {
            tandem::Rng g = tandem::Rng::from_key(key, s.u(), (uint32_t)d["K"].u());
            for (const conf::Json &f : d["draws"].a) {
                const size_t n = (size_t)f["n"].u();
                const std::string kind = f["kind"].s;
                std::vector<unsigned char> out;
                if (kind == "fill_normal_f64" || kind == "fill_exponential_f64") {
                    std::vector<double> z(n);
                    for (auto &x : z) x = kind == "fill_normal_f64" ? g.normal() : g.exponential();
                    out.assign(reinterpret_cast<unsigned char *>(z.data()), reinterpret_cast<unsigned char *>(z.data() + n));
                } else if (kind == "fill_exponential_f32") {
                    std::vector<float> z(n);
                    for (auto &x : z) x = g.exponentialf();
                    out.assign(reinterpret_cast<unsigned char *>(z.data()), reinterpret_cast<unsigned char *>(z.data() + n));
                } else {
                    // The vector block for the whole pairs, and normalf for an odd last element.
                    std::vector<float> u(2 * (n / 2)), z(n);
                    for (auto &x : u) x = g.frand();
                    tandem::normal_block_f32(u.data(), z.data(), n / 2);
                    if (n % 2) z.back() = g.normalf();
                    out.assign(reinterpret_cast<unsigned char *>(z.data()), reinterpret_cast<unsigned char *>(z.data() + n));
                }
                h = conf::fnv1a(h, out.data(), out.size());
                bytes += out.size();
                if (dump && &d == &hashes["dumps"].a[0]) std::fwrite(out.data(), 1, out.size(), dump);
            }
            end = g.position();
        }
        if (dump) return;
        CHECK(bytes == d["bytes"].u());
        CHECK(h == d["fnv1a"].hex());
        if (const conf::Json *e = d.find("end")) CHECK(end == e->u());
        std::printf("%s: fnv1a %016llx\n", d["id"].s.c_str(), (unsigned long long)h);
    }
}

// Start positions below 2^63 are accepted, 2^63 and above rejected without a change. A UInt64 draw
// at 2^63 - 1 aligns to 2^63 and ends at 2^63 + 64.
static void test_position_bounds() {
    const uint64_t top = (uint64_t)1 << 63;
    tandem::Rng q(42, 0, 32), before = q;
    CHECK(q.set_position(top - 1) && q.position() == top - 1);
    tandem::Rng r = tandem::Rng::from_key(q.key(), top - 64, 32);
    CHECK(q.urand64() == r.at_urand64(1) && q.position() == top + 64);
    q = before;
    CHECK(!q.set_position(top) && !q.set_position(~(uint64_t)0) && q == before);
    CHECK(tandem::Rng::from_key(q.key(), top - 1, 32).position() == top - 1);
    CHECK(tandem::Rng::from_key(q.key(), top, 32).position() == 0);
    CHECK(tandem::Rng::from_key(q.key(), ~(uint64_t)0, 32).position() == 0);
}

int main(int argc, char **argv) {
    const bool dump = argc > 1 && std::strcmp(argv[1], "--dump") == 0;
    const char *dir = argc > 1 + dump ? argv[1 + dump] : "tests/conformance";
    const conf::Json hashes = conf::load(dir, "hashes.json");
    if (dump) return test_dumps(hashes, stdout), 0;
    static_assert(std::is_trivially_copyable<tandem::Rng>::value, "copy");

    // Every draw of one mixed sequence against the C library.
    tandem::Rng r(42, 0, 32);
    tandem_rng c = tandem_from_key(r.key().w, 0, 32);
    int diff = 0;
    for (int i = 0; i < 1000; i++) {
        diff += r.urand(1000u) != tandem_u32_below(&c, 1000u);
        diff += r.urand64(3000000000000ull) != tandem_u64_below(&c, 3000000000000ull);
        diff += r.normal() != tandem_normal_f64(&c);
        diff += r.normalf() != tandem_normal_f32(&c);
        diff += r.exponential() != tandem_exponential_f64(&c);
        diff += r.exponentialf() != tandem_exponential_f32(&c);
        auto p = r.normal2(); // two f64 draws
        diff += p.z0 != tandem_normal_f64(&c);
        diff += p.z1 != tandem_normal_f64(&c);
        float fz[2];
        tandem_normal2_f32(&c, fz);
        auto pf = r.normalf2();
        diff += pf.z0 != fz[0] || pf.z1 != fz[1];
        diff += r.bit() != tandem_next_bool(&c);
    }
    CHECK(diff == 0);
    tandem::Rng kids[3], a = r;
    r.fork(kids, 3);
    tandem_rng ck[3];
    tandem_fork(&c, ck, 3);
    for (int i = 0; i < 3; i++) CHECK(kids[i].urand64() == tandem_next_u64(&ck[i]));
    CHECK(r.position() == tandem_position(&c));
    CHECK(a == a);

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
    CHECK(worst_ulp <= 8.0f);
    std::printf("f32 block vs libm: %.3g ulps of r\n", (double)worst_ulp);

    test_below(dir);
    test_fill_below(dir);
    test_normal(dir);
    test_exponential(dir);
    test_choice(dir);
    test_streams(hashes);
    test_position_bounds();
    test_dumps(hashes, nullptr);
    std::printf("host core: %s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
