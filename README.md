<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .cu"></p>

# tandem-cuda

[![CI](https://github.com/tandem-rng/tandem-cuda/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-cuda/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Header-only CUDA implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a
noncryptographic pseudorandom number generator. It produces the stream the specification
defines, bit for bit, and fills A100 memory at about 1390 GiB/s.

Copy `tandem.cuh` and `include/tandem/`, or put them on the include path. `tandem.cuh` builds
as C++23 with CUDA 13.4 and clang 20, and `include/tandem/core.hpp` stays valid C++17.

```sh
nvcc -I<tandem-cuda>/include ...
make test TANDEM_C=../tandem-c      # or: pixi install && pixi run test
```

```cpp
#include "tandem.cuh"

tandem::generator g = tandem::generator::seed(42);   // keeps the stream position for you
g.fill_f64(x, n);                                    // the spec's Float64 fill into device memory
g.fill_u32_below(6, die, n);                         // 6-sided dice, Lemire's method

__global__ void kernel(uint32_t k0, uint32_t k1, uint32_t k2, uint32_t k3, float *out) {
    const uint32_t key[4] = {k0, k1, k2, k3};
    tandem::device_rng rng = tandem::device_rng::from_key(key, 0, 32).split(blockIdx.x * blockDim.x + threadIdx.x);
    out[threadIdx.x] = rng.normalf();                // Box-Muller in float
}
```

See [API](docs/api.md) for every fill, `device_rng` and the Thrust adapters, and
[design](docs/design.md), [tests](docs/tests.md) and [speed](docs/speed.md) for the rest.

Portions of the code were generated with the assistance of LLMs.

[Documentation](docs/index.md) · [Apache 2.0 license](LICENSE)
