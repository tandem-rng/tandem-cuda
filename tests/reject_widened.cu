// Must not compile: `make reject` checks that the widened tile store's static_assert fires. A
// 16-bit draw stored as a 32-bit element would take that store and leave half the block unwritten.
// With -DCONTROL the kind has 32-bit draws into 64-bit elements, which compiles.
#include "../tandem.cuh"

namespace tandem::detail {
struct widened {};
template <> struct elem<widened> {
#ifdef CONTROL
    using out_t = uint64_t;
    static constexpr unsigned bits = 32;
#else
    using out_t = uint32_t;
    static constexpr unsigned bits = 16;
#endif
    __device__ static out_t make(const uint32_t w[4], unsigned i, uint64_t, const Ctx &) {
        return w[i * bits / 32] >> (i * bits % 32);
    }
};
} // namespace tandem::detail

uint64_t probe(const uint32_t key[4], tandem::detail::elem<tandem::detail::widened>::out_t *out) {
    return tandem::detail::fill<tandem::detail::widened>(key, 0, 32, out, 64, 0);
}
