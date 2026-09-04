# ==========================================================================
#  The GPU Matrix Package
# ==========================================================================
#
#   make          build libmatrixcpp_gpu.a
#   make test     correctness, every result checked against basic/
#   make bench    timings against basic/ and CuPy
#   make clean
#
# Only backend.cu needs nvcc. Everything else -- this library's public headers
# and any program using them -- is ordinary C++17, which is the whole point of
# splitting the package this way.

CUDA    ?= /usr/local/cuda
NVCC    := $(CUDA)/bin/nvcc
ARCH    ?= sm_120

ROOT    := ..
LIB     := libmatrixcpp_gpu.a

# -O3 -march=native everywhere, to match what basic/ is measured with; a
# benchmark that builds the two sides with different flags measures the flags.
CXXFLAGS := -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -I$(ROOT)
NVCCFLAGS := -std=c++17 -O3 -arch=$(ARCH) -Xcompiler -fPIC -Xcompiler -Wall

LDFLAGS := -L. -lmatrixcpp_gpu -L$(CUDA)/lib64 \
           -lcudart -lcublas -lcusolver -lcurand -lcufft -fopenmp

.PHONY: all test bench demo clean run-test run-bench run-demo

all: $(LIB)

$(LIB): src/backend.cu detail/backend.hpp
	$(NVCC) $(NVCCFLAGS) -I$(ROOT) -c src/backend.cu -o backend.o
	ar rcs $@ backend.o
	@rm -f backend.o

test: test/correctness
bench: test/benchmark
demo: demo/wigner

test/correctness: test/correctness.cpp $(LIB) gpu_matrix.hpp device.hpp MatrixGpu.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

test/benchmark: test/benchmark.cpp $(LIB) gpu_matrix.hpp device.hpp MatrixGpu.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

demo/wigner: demo/wigner.cpp $(LIB) gpu_matrix.hpp device.hpp MatrixGpu.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run-test: test/correctness
	@LD_LIBRARY_PATH=$(CUDA)/lib64:$$LD_LIBRARY_PATH ./test/correctness

run-bench: test/benchmark
	@LD_LIBRARY_PATH=$(CUDA)/lib64:$$LD_LIBRARY_PATH ./test/benchmark

run-demo: demo/wigner
	@LD_LIBRARY_PATH=$(CUDA)/lib64:$$LD_LIBRARY_PATH ./demo/wigner

clean:
	rm -f $(LIB) backend.o test/correctness test/benchmark demo/wigner
