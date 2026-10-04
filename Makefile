NVCC ?= nvcc
CC ?= cc
ARCH ?= -arch=sm_80
NVCCFLAGS ?= -std=c++17 -O3 $(ARCH) -Xcompiler -Wall,-Wextra
TANDEM_C ?= ../tandem-c
SPEC_VECTORS ?= ../tandem-spec/vectors.json
CXX_HOST ?= c++
HEADERS := tandem.cuh include/tandem/core.hpp

# nvcc needs a host compiler it knows; a conda toolchain names one through CXX.
ifdef CXX
NVCCFLAGS += -ccbin $(CXX)
endif
# Binaries find the CUDA libraries that belong to the nvcc that built them, also inside a
# conda environment. The environment's LDFLAGS are gcc flags that nvcc rejects, so they
# are not used.
NVCC_LIB := $(dir $(realpath $(shell command -v $(NVCC))))../lib
LINK ?= -Xlinker -rpath,$(NVCC_LIB)

.PHONY: build test bench vectors cross clean

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
	rm -f tandem_c.o tests/test_cuda tools/bench tools/gen_cross_fill_below tools/gen_cross_fill_normal
