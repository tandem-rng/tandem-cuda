// core.hpp on a host compiler in C++17, the standard its other consumers (Kokkos, Fortran, torch)
// build with: the scalar generator agrees with the C library. Run: make host
#include <cmath>
#include <cstdio>
#include <type_traits>
#include "tandem/core.hpp"
extern "C" {
#include "tandem.h"
}
int main() {
    static_assert(std::is_trivially_copyable<tandem::Rng>::value, "copy");
    tandem::Rng r(42, 0, 32);
    tandem_rng c = tandem_from_key(r.key().w, 0, 32);
    int bad = 0;
    for (int i = 0; i < 1000; i++) {
        bad += r.urand(1000u) != tandem_u32_below(&c, 1000u);
        bad += r.urand64(3000000000000ull) != tandem_u64_below(&c, 3000000000000ull);
        {
            double a = r.normal(), b = tandem_normal_f64(&c); // sin and cos may differ in the last bit
            bad += !(std::fabs(a - b) <= 1e-14 * (1.0 + std::fabs(b)));
        }
        bad += r.bit() != tandem_next_bool(&c);
    }
    tandem::Rng kids[3], a = r;
    r.fork(kids, 3);
    tandem_rng ck[3];
    tandem_fork(&c, ck, 3);
    for (int i = 0; i < 3; i++) bad += kids[i].urand64() != tandem_next_u64(&ck[i]);
    bad += r.position() != tandem_position(&c);
    bad += !(a == a) ;
    std::printf("host core: %s\n", bad ? "FAIL" : "ok");
    return bad;
}
