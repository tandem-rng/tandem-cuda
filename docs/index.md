# tandem-cuda

Header-only CUDA implementation of Tandem8x32. It produces the stream of the
[specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) bit for bit, and fills
A100 memory at about 1390 GiB/s.

- [API](api.md): the fills, the host `generator`, `device_rng`, the Thrust adapters and parallel use.
- [Design](design.md): the contract of the bounded, normal and exponential fills.
- [Tests](tests.md): what the suite checks and what CI runs.
- [Speed](speed.md): A100 figures against cuRAND.

## Install

The library is headers, so copy `tandem.cuh` and `include/tandem/` or put them on the include path. The
`packaging/` directory holds a Spack recipe (`spack/package.py`) and a conda-forge style recipe
(`conda/recipe.yaml`) that install both. Neither is submitted to Spack or conda-forge yet, and both
build from the `main` branch.

Put `include/` on the include path, for example `nvcc -I<tandem-cuda>/include`.

### Toolchains

`pixi.toml` provides conda-forge environments for hosts without a system install.
The default is CUDA 13.4 (nvcc 13.4.92) with clang 20 as the host compiler and `-std=c++23`, with
no warnings under `-Wall -Wextra`. `pixi run -e gcc` builds the same with gcc 14 as host compiler
(`make HOSTCXX=g++ HOSTCC=gcc`). `pixi run -e cuda12` is CUDA 12.8 (nvcc 12.8.93) with clang 19 and
`-std=c++20`, because nvcc 12.8 stops at C++20. CI builds all three nvcc environments.

Clang can also compile the CUDA sources itself (`clang++ -x cuda --cuda-gpu-arch=sm_80`), as Kokkos
builds do. `make clangcuda` does that for the tests in the `cuda12` environment, clang 19 with
CUDA 12.8 and C++20. It compiles without warnings, and on the A100 it passes the same suite with
the same f32 normal deviations as nvcc, so `__sincosf` behaves alike under both. CI compiles it.
Clang 20 does not build against the CUDA 13 headers yet, so that pairing is not offered. The GPU host batserv01 has
NVIDIA driver 570.124, which supports CUDA 12.8 at most, so its test suite and the speeds in [speed](speed.md) run
in the `cuda12` environment. A CUDA 13 binary needs driver 580 or newer. The GPU suite passed there
with CUDA 12.8, clang 19 and C++20; the CUDA 13 environments are compiled but not run.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
