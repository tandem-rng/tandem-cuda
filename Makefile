NVCC ?= nvcc
CC ?= cc
ARCH ?= -arch=sm_80
CUDASTD ?= c++23
NVCCFLAGS ?= -std=$(CUDASTD) -O3 $(ARCH) -Xcompiler -Wall,-Wextra
TANDEM_C ?= ../tandem-c
SPEC_VECTORS ?= ../tandem-spec/vectors.json
HEADERS := tandem.cuh include/tandem/core.hpp

# clang is the host compiler, gcc is a compatibility build: make HOSTCXX=g++ HOSTCC=gcc.
HOSTCXX ?= clang++
HOSTCC ?= clang
NVCCFLAGS += -ccbin $(HOSTCXX)
# conda activation prepends its own -ccbin, which nvcc would warn about.
export NVCC_PREPEND_FLAGS :=
CC = $(HOSTCC)
CXX_HOST = $(HOSTCXX)

# Binaries find the CUDA libraries that belong to the nvcc that built them, also inside a
# conda environment. The environment's LDFLAGS are gcc flags that nvcc rejects, so they
# are not used.
NVCC_LIB := $(dir $(realpath $(shell command -v $(NVCC))))../lib
LINK ?= -Xlinker -rpath,$(NVCC_LIB)

.PHONY: build test bench clangcuda host vectors cross clean

build: tests/test_cuda tools/bench

tandem_c.o: $(TANDEM_C)/tandem.c $(TANDEM_C)/tandem.h
	$(CC) -std=c99 -O2 -c -o $@ $<

tests/test_cuda: tests/test_cuda.cu tests/vectors.h tests/cross_fill_below.h tests/cross_fill_normal.h $(HEADERS) tandem_c.o
	$(NVCC) $(NVCCFLAGS) -Iinclude -I$(TANDEM_C) -o $@ tests/test_cuda.cu tandem_c.o $(LINK)

tools/bench: tools/bench.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -Iinclude -o $@ tools/bench.cu -lcurand $(LINK)

test: tests/test_cuda
	./tests/test_cuda tests/data

bench: tools/bench
	./tools/bench

# clang's own CUDA frontend, as Kokkos and others build. CUDA_PATH is the toolkit prefix, the
# conda environment's by default. Only the tests are built, with the same sources as nvcc.
CUDA_PATH ?= $(CONDA_PREFIX)
CLANGCUDA_STD ?= c++20
tests/test_clangcuda: tests/test_cuda.cu tests/vectors.h tests/cross_fill_below.h tests/cross_fill_normal.h $(HEADERS) tandem_c.o
	$(HOSTCXX) --cuda-path=$(CUDA_PATH) -Wno-unknown-cuda-version -isystem $(CUDA_PATH)/targets/x86_64-linux/include \
	  --cuda-gpu-arch=sm_80 -std=$(CLANGCUDA_STD) -O3 -Wall -Wextra -Iinclude -I$(TANDEM_C) -o $@ \
	  -x cuda tests/test_cuda.cu -x none tandem_c.o -L$(CUDA_PATH)/lib -L$(CUDA_PATH)/targets/x86_64-linux/lib \
	  -lcudart -Wl,-rpath,$(CUDA_PATH)/lib -Wl,-rpath,$(CUDA_PATH)/targets/x86_64-linux/lib

clangcuda: tests/test_clangcuda
	./tests/test_clangcuda tests/data

# core.hpp must stay valid C++17 for its other consumers, so this build pins the standard.
tests/host_core: tests/host_core.cpp include/tandem/core.hpp tandem_c.o
	$(CXX_HOST) -std=c++17 -O2 -Wall -Wextra -Iinclude -I$(TANDEM_C) -o $@ tests/host_core.cpp tandem_c.o

host: tests/host_core
	./tests/host_core

# Regenerate the vector header from a checkout of https://github.com/tandem-rng/spec.
vectors:
	python3 tools/gen_vectors.py $(SPEC_VECTORS) > tests/vectors.h

# Regenerate the bounded-fill fixtures from core.hpp alone, on any host compiler.
cross:
	$(CXX_HOST) -std=c++17 -O1 -Iinclude -o tools/gen_cross_fill_below tools/gen_cross_fill_below.cpp
	./tools/gen_cross_fill_below > tests/cross_fill_below.h
	$(CXX_HOST) -std=c++17 -O1 -Iinclude -o tools/gen_cross_fill_normal tools/gen_cross_fill_normal.cpp
	./tools/gen_cross_fill_normal > tests/cross_fill_normal.h

clean:
	rm -f tandem_c.o tests/test_clangcuda tests/test_cuda tests/host_core tools/bench tools/gen_cross_fill_below tools/gen_cross_fill_normal
