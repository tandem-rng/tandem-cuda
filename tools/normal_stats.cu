// Statistics of 10^8 Float64 normals from the device fill: moments 1 to 6, the Kolmogorov-Smirnov
// and Anderson-Darling tests against the standard normal, and the counts beyond 3, 3.5, 4, 4.5 and
// 5. A moment or a count passes at |z| < 4, a test at p > 0.001. Once per change of the normal
// code, not in CI. Build and run on a GPU host: make stats
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "../tandem.cuh"

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess) {                                                                    \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);    \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static double cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// The KS p-value from the Kolmogorov distribution, with Stephens' finite-n scaling.
static double ks_p(double d, double n) {
    double l = (std::sqrt(n) + 0.12 + 0.11 / std::sqrt(n)) * d, p = 0;
    for (int k = 1; k < 100; k++) p += 2 * ((k & 1) ? 1 : -1) * std::exp(-2.0 * k * k * l * l);
    return std::fmin(1.0, std::fmax(0.0, p));
}

// Marsaglia and Marsaglia (2004), the limiting distribution of A^2, which n = 10^8 reaches.
static double ad_p(double z) {
    double c = z < 2 ? std::exp(-1.2337141 / z) / std::sqrt(z) *
                           (2.00012 + (.247105 - (.0649821 - (.0347962 - (.011672 - .00168691 * z) * z) * z) * z) * z)
                     : std::exp(-std::exp(1.0776 - (2.30695 - (.43424 - (.082433 - (.008056 - .0003146 * z) * z) * z) * z) * z));
    return 1.0 - c;
}

int main() {
    const size_t n = 100000000;
    double *d;
    CUDA_CHECK(cudaMalloc(&d, n * 8));
    tandem::generator g = tandem::generator::seed(42);
    g.fill_normal_f64(d, n);
    std::vector<double> x(n);
    CUDA_CHECK(cudaMemcpy(x.data(), d, n * 8, cudaMemcpyDeviceToHost));
    int bad = 0;

    // E X^k = 0, 1, 0, 3, 0, 15 and E X^2k = 1, 3, 15, 105, 945, 10395. Kahan sums.
    const double mu[13] = {1, 0, 1, 0, 3, 0, 15, 0, 105, 0, 945, 0, 10395};
    double sum[7] = {}, comp[7] = {};
    for (double v : x) {
        double p = 1;
        for (int k = 1; k <= 6; k++) {
            p *= v;
            double y = p - comp[k], t = sum[k] + y;
            comp[k] = (t - sum[k]) - y;
            sum[k] = t;
        }
    }
    for (int k = 1; k <= 6; k++) {
        double m = sum[k] / n, z = (m - mu[k]) / std::sqrt((mu[2 * k] - mu[k] * mu[k]) / n);
        std::printf("moment %d: %.6f (exact %g), z = %+.2f\n", k, m, mu[k], z);
        bad += !(std::fabs(z) < 4);
    }
    for (double t : {3.0, 3.5, 4.0, 4.5, 5.0}) {
        size_t c = 0;
        for (double v : x) c += std::fabs(v) > t;
        double p = std::erfc(t / std::sqrt(2.0)), z = (c - n * p) / std::sqrt(n * p * (1 - p));
        std::printf("|x| > %.1f: %zu, expected %.1f, z = %+.2f\n", t, c, n * p, z);
        bad += !(std::fabs(z) < 4);
    }

    thrust::sort(thrust::device_pointer_cast(d), thrust::device_pointer_cast(d) + n);
    CUDA_CHECK(cudaMemcpy(x.data(), d, n * 8, cudaMemcpyDeviceToHost));
    double dmax = 0, a = 0;
    for (size_t i = 0; i < n; i++) {
        double f = cdf(x[i]);
        dmax = std::fmax(dmax, std::fmax((i + 1.0) / n - f, f - (double)i / n));
        // ln F(x_i) + ln(1 - F(x_(n+1-i))), the second through erfc to keep the far tail.
        double lo = std::log(f), hi = std::log(0.5 * std::erfc(x[n - 1 - i] / std::sqrt(2.0)));
        a += (2.0 * i + 1.0) * (lo + hi);
    }
    double a2 = -(double)n - a / n, pks = ks_p(dmax, n), pad = ad_p(a2);
    std::printf("KS D = %.3g, p = %.3g\nAD A^2 = %.3f, p = %.3g\n", dmax, pks, a2, pad);
    bad += !(pks > 0.001) + !(pad > 0.001);
    std::printf("normal stats: %s\n", bad ? "FAIL" : "ok");
    CUDA_CHECK(cudaFree(d));
    return bad != 0;
}
