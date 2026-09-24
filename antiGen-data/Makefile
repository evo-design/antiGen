# Build antiGen-data. Uses nvcc for a GPU-capable build when it is on PATH; otherwise
# falls back to a plain C++ (g++) CPU-only build. Both produce identical results — the
# CUDA kernels are compiled in only under -DUSE_CUDA, and the tool selects GPU vs CPU at
# runtime (see --cpu / cudaGetDeviceCount). Override the CPU compiler: make CPUCXX=clang++
CXXFLAGS = -std=c++17 -O2
INCLUDES = -Iinclude
BUILDDIR = build
CPUCXX  ?= g++

OBJ = $(BUILDDIR)/main.o $(BUILDDIR)/read_fasta.o $(BUILDDIR)/cache.o $(BUILDDIR)/attach.o

NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
  CC    = $(CPUCXX)
  XFLAG = -x c++
  DEFS  =
  $(info antiGen-data: nvcc not found -> CPU-only build ($(CPUCXX)))
else
  CC    = nvcc
  XFLAG = -x cu
  DEFS  = -DUSE_CUDA
  $(info antiGen-data: nvcc found -> CUDA build)
endif

antiGen-data: $(OBJ)
	$(CC) $(CXXFLAGS) $(INCLUDES) $(DEFS) -o $@ $(OBJ)

# Rebuild every object when any header changes. types.h defines the Node layout that all
# four translation units share, so a stale object file there is a silent ABI mismatch
# (a segfault at run time) rather than a compile error.
$(OBJ): $(wildcard include/*.h)

$(BUILDDIR)/%.o: src/%.cpp | $(BUILDDIR)
	$(CC) $(CXXFLAGS) $(INCLUDES) $(DEFS) $(XFLAG) -c $< -o $@

$(BUILDDIR)/%.o: src/%.cu | $(BUILDDIR)
	$(CC) $(CXXFLAGS) $(INCLUDES) $(DEFS) $(XFLAG) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) antiGen-data
