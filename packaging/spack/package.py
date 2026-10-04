# Recipe for a Spack package repository. It is not submitted to spack-packages yet.
from spack_repo.builtin.build_systems.generic import Package

from spack.package import *


class TandemCuda(Package):
    """Header-only Tandem8x32 random number generator for CUDA devices and hosts."""

    homepage = "https://github.com/tandem-rng/tandem-cuda"
    git = "https://github.com/tandem-rng/tandem-cuda.git"

    license("Apache-2.0")

    # No releases exist. A release adds
    # version("X.Y.Z", sha256="...", url="https://github.com/tandem-rng/tandem-cuda/archive/refs/tags/vX.Y.Z.tar.gz")
    version("main", branch="main")

    # tandem.cuh includes cuda_runtime.h. include/tandem/core.hpp alone is plain C++17.
    # The user's nvcc compiles the headers, so the package builds nothing and needs no variant.
    depends_on("cuda@12.8:", type=("build", "run"))

    def install(self, spec, prefix):
        install_tree("include", prefix.include)
        install("tandem.cuh", prefix.include)
