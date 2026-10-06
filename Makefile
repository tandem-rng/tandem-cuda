NVCC ?= nvcc
CC ?= cc
ARCH ?= -arch=sm_80
CUDASTD ?= c++23
NVCCFLAGS ?= -std=$(CUDASTD) -O3 $(ARCH) -Xcompiler -Wall,-Wextra
TANDEM_C ?= ../tandem-c
SPEC_VECTORS ?= ../tandem-spec/vectors.json
SPEC_TABLES ?= ../tandem-spec/tables/normal_f64_zig1024.json
HEADERS := tandem.cuh include/tandem/core.hpp include/tandem/normal_tables.hpp

# clang is the host compiler, gcc is a compatibility build: make HOSTCXX=g++ HOSTCC=gcc.
HOSTCXX ?= clang++
HOSTCC ?= clang
NVCCFLAGS += -ccbin $(HOSTCXX)
# conda activation prepends its own -ccbin, which nvcc would warn about.
export NVCC_PREPEND_FLAGS :=
CC = $(HOSTCC)
CXX_HOST = $(HOSTCXX)
# The host normal loops use explicit fused multiply-adds, which x86 needs flags to compile to
# one vectorizable instruction. Without them std::fma is a slow library call with the same bits.
FMAFLAGS := $(if $(filter x86_64 amd64,$(shell uname -m)),-mavx2 -mfma,)

# Binaries find the CUDA libraries that belong to the nvcc that built them, also inside a
# conda environment. The environment's LDFLAGS are gcc flags that nvcc rejects, so they
# are not used.
NVCC_LIB := $(dir $(realpath $(shell command -v $(NVCC))))../lib
LINK ?= -Xlinker -rpath,$(NVCC_LIB)

.PHONY: build test reject thrust hostvec bench stats clangcuda host vectors tables cross clean

build: tests/test_cuda tests/test_thrust tools/bench

tandem_c.o: $(TANDEM_C)/tandem.c $(TANDEM_C)/tandem.h
	$(CC) -std=c99 -O2 -ffp-contract=off $(FMAFLAGS) -c -o $@ $<

tests/test_cuda: tests/test_cuda.cu tests/vectors.h tests/cross_fill_below.h tests/cross_fill_normal.h tests/cross_fill_exponential.h $(TANDEM_C)/tests/cross_normal.h $(TANDEM_C)/tests/cross_exponential.h $(HEADERS) tandem_c.o
	$(NVCC) $(NVCCFLAGS) -Iinclude -I$(TANDEM_C) -o $@ tests/test_cuda.cu tandem_c.o $(LINK)

tests/test_thrust: tests/test_thrust.cu tandem_thrust.cuh $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -Iinclude -o $@ tests/test_thrust.cu $(LINK)

thrust: tests/test_thrust
	./tests/test_thrust tests/data

tools/bench: tools/bench.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -Iinclude -o $@ tools/bench.cu -lcurand $(LINK)

test: tests/test_cuda tests/test_thrust
	./tests/test_cuda tests/data
	./tests/test_thrust tests/data

# A negative compile test: the probe must fail on the widened store's static_assert alone, and
# its CONTROL variant must compile.
reject: tests/reject_widened.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -Iinclude -DCONTROL -c -o /dev/null tests/reject_widened.cu
	$(NVCC) $(NVCCFLAGS) -Iinclude -c -o /dev/null tests/reject_widened.cu 2>&1 | \
	  grep -q "the widened tile store needs 32-bit draws"

bench: tools/bench
	./tools/bench

tools/normal_stats: tools/normal_stats.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -Iinclude -o $@ tools/normal_stats.cu $(LINK)

stats: tools/normal_stats
	./tools/normal_stats

# clang's own CUDA frontend, as Kokkos and others build. CUDA_PATH is the toolkit prefix, the
# conda environment's by default. Only the tests are built, with the same sources as nvcc.
CUDA_PATH ?= $(CONDA_PREFIX)
CLANGCUDA_STD ?= c++20
tests/test_clangcuda: tests/test_cuda.cu tests/vectors.h tests/cross_fill_below.h tests/cross_fill_normal.h tests/cross_fill_exponential.h $(TANDEM_C)/tests/cross_normal.h $(TANDEM_C)/tests/cross_exponential.h $(HEADERS) tandem_c.o
	$(HOSTCXX) --cuda-path=$(CUDA_PATH) -Wno-unknown-cuda-version -isystem $(CUDA_PATH)/targets/x86_64-linux/include \
	  --cuda-gpu-arch=sm_80 -std=$(CLANGCUDA_STD) -O3 -Wall -Wextra -Iinclude -I$(TANDEM_C) -o $@ \
	  -x cuda tests/test_cuda.cu -x none tandem_c.o -L$(CUDA_PATH)/lib -L$(CUDA_PATH)/targets/x86_64-linux/lib \
	  -lcudart -Wl,-rpath,$(CUDA_PATH)/lib -Wl,-rpath,$(CUDA_PATH)/targets/x86_64-linux/lib

clangcuda: tests/test_clangcuda
	./tests/test_clangcuda tests/data

# core.hpp must stay valid C++17 for its other consumers, so this build pins the standard. The
# test checks that the normals hash to tandem-c's tests/test_normal_bits.c value.
tests/host_core: tests/host_core.cpp $(HEADERS) tests/cross_fill_below.h tests/cross_fill_normal.h tests/cross_fill_exponential.h $(TANDEM_C)/tests/cross_choice.h tandem_c.o
	$(CXX_HOST) -std=c++17 -O2 $(FMAFLAGS) -Wall -Wextra -Iinclude -I$(TANDEM_C) -o $@ tests/host_core.cpp tandem_c.o

host: tests/host_core
	./tests/host_core

# The host Float32 Box-Muller block must vectorize under clang, which is what makes it fast.
hostvec:
	$(CXX_HOST) -std=c++17 -O2 $(FMAFLAGS) -Iinclude -I$(TANDEM_C) -Rpass=loop-vectorize -c -o /dev/null tests/host_core.cpp 2>&1 \
	  | grep "core.hpp" | grep -c "vectorized loop" | awk '{ if ($$1 < 1) { print "normal block not vectorized"; exit 1 } else print "normal block vectorized" }'

# Regenerate the vector header and the ziggurat tables from a checkout of
# https://github.com/tandem-rng/spec.
vectors:
	python3 tools/gen_vectors.py $(SPEC_VECTORS) > tests/vectors.h

tables:
	python3 tools/gen_normal_tables.py $(SPEC_TABLES) > include/tandem/normal_tables.hpp

# Regenerate the fill fixtures from core.hpp alone, on any host compiler.
cross:
	$(CXX_HOST) -std=c++17 -O1 -Iinclude -o tools/gen_cross_fill_below tools/gen_cross_fill_below.cpp
	./tools/gen_cross_fill_below > tests/cross_fill_below.h
	$(CXX_HOST) -std=c++17 -O1 -Iinclude -o tools/gen_cross_fill_normal tools/gen_cross_fill_normal.cpp
	./tools/gen_cross_fill_normal > tests/cross_fill_normal.h
	$(CXX_HOST) -std=c++17 -O1 -Iinclude -o tools/gen_cross_fill_exponential tools/gen_cross_fill_exponential.cpp
	./tools/gen_cross_fill_exponential > tests/cross_fill_exponential.h

clean:
	rm -f tandem_c.o tests/test_clangcuda tests/test_thrust tests/test_cuda tests/host_core tools/bench tools/normal_stats tools/gen_cross_fill_below tools/gen_cross_fill_normal tools/gen_cross_fill_exponential
