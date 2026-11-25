# Makefile — CS3005 PDC Project #15: Parallel Ray Tracing of Large Scenes
# Builds CPU targets unconditionally and CUDA target conditionally when nvcc is present.

CXX      ?= g++
MPICXX   ?= mpic++
NO_CUDA  ?= 0
NATIVE   ?= 0

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Isrc/common
ifeq ($(NATIVE),1)
CXXFLAGS += -march=native
endif

# Auto-detect zlib for fast PNG compression (falls back to pure-C++ stored PNG if absent)
HAS_ZLIB := $(wildcard /usr/include/zlib.h)
ifneq ($(HAS_ZLIB),)
LDLIBS := -lz
else
CXXFLAGS += -DRT_NO_ZLIB
LDLIBS :=
endif

BIN_DIR     := bin
COMMON_HDRS := $(wildcard src/common/*.h)

SEQ_BIN    := $(BIN_DIR)/raytracer_seq
OMP_BIN    := $(BIN_DIR)/raytracer_omp
MPI_BIN    := $(BIN_DIR)/raytracer_mpi
HYBRID_BIN := $(BIN_DIR)/raytracer_hybrid
CUDA_BIN   := $(BIN_DIR)/raytracer_cuda

# Auto-detect nvcc unless NO_CUDA=1
NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
ifneq ($(wildcard /usr/local/cuda/bin/nvcc),)
NVCC := /usr/local/cuda/bin/nvcc
endif
endif

# Prefer g++-13 for nvcc host compiler if installed (e.g., alongside GCC 15 on Ubuntu 26.04)
NVCC_CCBIN := $(shell command -v g++-13 2>/dev/null)

# Auto-detect GPU compute capability (default sm_75 if nvidia-smi unavailable)
CUDA_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.\r ')
ifeq ($(CUDA_CAP),)
CUDA_CAP := 75
endif
CUDA_ARCH := -arch=sm_$(CUDA_CAP)
NVCCFLAGS := -O3 -std=c++17 -Isrc/common $(CUDA_ARCH) --fmad=false -Xcompiler -Wall
ifneq ($(NVCC_CCBIN),)
NVCCFLAGS += -ccbin $(NVCC_CCBIN)
endif
ifeq ($(HAS_ZLIB),)
NVCCFLAGS += -DRT_NO_ZLIB
endif

CUDA_ENABLED := 0
ifneq ($(NO_CUDA),1)
ifneq ($(NVCC),)
CUDA_ENABLED := 1
endif
endif

.PHONY: all cpu seq omp mpi hybrid cuda info clean summary

ifeq ($(CUDA_ENABLED),1)
all: cpu cuda summary
else
all: cpu summary
endif

cpu: seq omp mpi hybrid

seq: $(SEQ_BIN)
omp: $(OMP_BIN)
mpi: $(MPI_BIN)
hybrid: $(HYBRID_BIN)

ifeq ($(CUDA_ENABLED),1)
cuda: $(CUDA_BIN)
else
cuda:
	@echo "[CUDA] Skipped (NO_CUDA=$(NO_CUDA), nvcc='$(NVCC)')."
endif

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(SEQ_BIN): src/seq_main.cpp $(COMMON_HDRS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

$(OMP_BIN): src/omp_main.cpp $(COMMON_HDRS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@ $(LDLIBS)

$(MPI_BIN): src/mpi_main.cpp $(COMMON_HDRS) | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

$(HYBRID_BIN): src/hybrid_main.cpp $(COMMON_HDRS) | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -fopenmp $< -o $@ $(LDLIBS)

$(CUDA_BIN): src/cuda_main.cu $(COMMON_HDRS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

summary:
	@echo "============================================================"
	@echo " Build complete in $(BIN_DIR)/"
	@echo "  - raytracer_seq    : built"
	@echo "  - raytracer_omp    : built"
	@echo "  - raytracer_mpi    : built"
	@echo "  - raytracer_hybrid : built"
ifeq ($(CUDA_ENABLED),1)
	@echo "  - raytracer_cuda   : built ($(CUDA_ARCH))"
else
	@echo "  - raytracer_cuda   : skipped (no nvcc or NO_CUDA=1)"
endif
	@echo "============================================================"

info:
	@echo "CXX          = $(CXX) ($(shell command -v $(CXX) 2>/dev/null || echo missing))"
	@echo "MPICXX       = $(MPICXX) ($(shell command -v $(MPICXX) 2>/dev/null || echo missing))"
	@echo "NVCC         = $(if $(NVCC),$(NVCC),not found)"
	@echo "NO_CUDA      = $(NO_CUDA)"
	@echo "CUDA_ENABLED = $(CUDA_ENABLED)"
	@echo "CUDA_ARCH    = $(CUDA_ARCH)"
	@echo "CXXFLAGS     = $(CXXFLAGS)"
	@echo "CPU Threads  = $(shell nproc 2>/dev/null || echo unknown)"

clean:
	rm -rf $(BIN_DIR)
