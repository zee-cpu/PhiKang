# ============================================================================
# PhiKang Makefile
# Kangaroo ECDLP Solver with GLV Endomorphism for secp256k1
#
# Usage:
#   make                    — build with GPU support
#   make gpu=0              — build CPU only
#   make SYMMETRY=1         — enable symmetry optimization
#   make DEBUG=1            — enable GPU kernel debug checks
#   make clean              — remove build artifacts
#
# GPU targets:
#   RTX 4090  — sm_89 (Ada Lovelace)
#   RTX 5090  — sm_92 (Blackwell consumer)
#   GB300     — sm_100 (Blackwell Ultra — uncomment when hardware arrives)
# ============================================================================

# ----------------------------------------------------------------------------
# Compiler and tools
# ----------------------------------------------------------------------------

CXX    = g++
NVCC   = nvcc
AR     = ar

# ----------------------------------------------------------------------------
# Directories
# ----------------------------------------------------------------------------

SRCDIR  = .
GPUDIR  = GPU
SECPDIR = SECPK1
OBJDIR  = obj

# ----------------------------------------------------------------------------
# GPU support (default: enabled)
# ----------------------------------------------------------------------------

gpu ?= 1

# ----------------------------------------------------------------------------
# CUDA architecture flags
#
# sm_89  = RTX 4090 (Ada Lovelace)
# sm_92  = RTX 5090 (Blackwell consumer)
# sm_100 = GB300 NVL72 (Blackwell Ultra) — uncomment when hardware arrives
#
# We compile for both 4090 and 5090 by default so the binary runs on either.
# Add sm_100 to the list when GB300 arrives and tensor core code is active.
# ----------------------------------------------------------------------------

CUDA_ARCH  = -gencode arch=compute_89,code=sm_89
CUDA_ARCH += -gencode arch=compute_92,code=sm_92
# CUDA_ARCH += -gencode arch=compute_100,code=sm_100  # TODO_TENSOR: GB300

# ----------------------------------------------------------------------------
# Symmetry flag
# ----------------------------------------------------------------------------

ifdef SYMMETRY
SYM_FLAG = -DUSE_SYMMETRY
else
SYM_FLAG =
endif

# ----------------------------------------------------------------------------
# Debug flag
# ----------------------------------------------------------------------------

ifdef DEBUG
DBG_FLAG = -DGPU_CHECK
else
DBG_FLAG =
endif

# ----------------------------------------------------------------------------
# C++ flags
# ----------------------------------------------------------------------------

CXXFLAGS  = -std=c++17
CXXFLAGS += -O3
CXXFLAGS += -march=native
CXXFLAGS += -Wall
CXXFLAGS += -Wno-unused-result
CXXFLAGS += -I.
CXXFLAGS += -I$(GPUDIR)
CXXFLAGS += -I$(SECPDIR)
CXXFLAGS += $(SYM_FLAG)
CXXFLAGS += $(DBG_FLAG)

ifeq ($(gpu), 1)
CXXFLAGS += -DWITHGPU
endif

# ----------------------------------------------------------------------------
# NVCC flags
# ----------------------------------------------------------------------------

NVCCFLAGS  = -std=c++17
NVCCFLAGS += -O3
NVCCFLAGS += $(CUDA_ARCH)
NVCCFLAGS += -Xcompiler -O3
NVCCFLAGS += -Xcompiler -march=native
NVCCFLAGS += -Xcompiler -Wall
NVCCFLAGS += -Xcompiler -Wno-unused-result
NVCCFLAGS += --use_fast_math
NVCCFLAGS += --extra-device-vectorization
NVCCFLAGS += -I.
NVCCFLAGS += -I$(GPUDIR)
NVCCFLAGS += -I$(SECPDIR)
NVCCFLAGS += $(SYM_FLAG)
NVCCFLAGS += $(DBG_FLAG)

ifeq ($(gpu), 1)
NVCCFLAGS += -DWITHGPU
endif

# ----------------------------------------------------------------------------
# Linker flags
# ----------------------------------------------------------------------------

LDFLAGS  = -lpthread
LDFLAGS += -lm

ifeq ($(gpu), 1)
LDFLAGS += -lcuda
LDFLAGS += -lcudart
endif

# ----------------------------------------------------------------------------
# Source files
# ----------------------------------------------------------------------------

# CPU sources compiled with g++
CPU_SRCS  = main.cpp
CPU_SRCS += Kangaroo.cpp
CPU_SRCS += KangarooWork.cpp
CPU_SRCS += KangarooNet.cpp
CPU_SRCS += Timer.cpp
CPU_SRCS += HashTable.cpp
CPU_SRCS += $(SECPDIR)/Int.cpp
CPU_SRCS += $(SECPDIR)/IntMod.cpp
CPU_SRCS += $(SECPDIR)/Random.cpp
CPU_SRCS += $(SECPDIR)/Point.cpp
CPU_SRCS += $(SECPDIR)/SECP256k1.cpp
CPU_SRCS += $(SECPDIR)/IntGroup.cpp

# GPU sources compiled with nvcc
GPU_SRCS  = $(GPUDIR)/GPUEngine.cu

# ----------------------------------------------------------------------------
# Object files
# ----------------------------------------------------------------------------

CPU_OBJS = $(patsubst %.cpp, $(OBJDIR)/%.o, $(CPU_SRCS))
GPU_OBJS = $(patsubst %.cu,  $(OBJDIR)/%.o, $(GPU_SRCS))

ifeq ($(gpu), 1)
ALL_OBJS = $(CPU_OBJS) $(GPU_OBJS)
else
ALL_OBJS = $(CPU_OBJS)
endif

# ----------------------------------------------------------------------------
# Output binary
# ----------------------------------------------------------------------------

TARGET = PhiKang

# ----------------------------------------------------------------------------
# Default target
# ----------------------------------------------------------------------------

all: banner dirs $(TARGET)

banner:
	@echo "============================================"
	@echo " PhiKang v3.0 — Building"
	@echo " GPU support : $(gpu)"
ifdef SYMMETRY
	@echo " Symmetry    : YES"
else
	@echo " Symmetry    : NO"
endif
ifdef DEBUG
	@echo " Debug       : YES"
else
	@echo " Debug       : NO"
endif
	@echo "============================================"

# ----------------------------------------------------------------------------
# Create object directories
# ----------------------------------------------------------------------------

dirs:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(OBJDIR)/$(GPUDIR)
	@mkdir -p $(OBJDIR)/$(SECPDIR)

# ----------------------------------------------------------------------------
# Link
# ----------------------------------------------------------------------------

$(TARGET): $(ALL_OBJS)
	@echo "[LD] $@"
	@$(CXX) $(ALL_OBJS) -o $@ $(LDFLAGS)
	@echo "[OK] Built: $@"

# ----------------------------------------------------------------------------
# Compile CPU sources
# ----------------------------------------------------------------------------

$(OBJDIR)/%.o: %.cpp
	@echo "[CC] $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# ----------------------------------------------------------------------------
# Compile GPU sources
# ----------------------------------------------------------------------------

$(OBJDIR)/$(GPUDIR)/%.o: $(GPUDIR)/%.cu
	@echo "[NVCC] $<"
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ----------------------------------------------------------------------------
# Clean
# ----------------------------------------------------------------------------

clean:
	@echo "[CLEAN]"
	@rm -rf $(OBJDIR) $(TARGET)
	@echo "[OK] Clean done"

# ----------------------------------------------------------------------------
# Install (copy binary to /usr/local/bin)
# ----------------------------------------------------------------------------

install: $(TARGET)
	@echo "[INSTALL] /usr/local/bin/$(TARGET)"
	@cp $(TARGET) /usr/local/bin/$(TARGET)
	@chmod 755 /usr/local/bin/$(TARGET)

# ----------------------------------------------------------------------------
# Uninstall
# ----------------------------------------------------------------------------

uninstall:
	@rm -f /usr/local/bin/$(TARGET)
	@echo "[UNINSTALL] done"

# ----------------------------------------------------------------------------
# Info — print CUDA device info
# ----------------------------------------------------------------------------

info: $(TARGET)
	@./$(TARGET) -l

# ----------------------------------------------------------------------------
# Check — run GPU kernel self-test (requires DEBUG=1 build)
# ----------------------------------------------------------------------------

check: $(TARGET)
	@./$(TARGET) -check

.PHONY: all banner dirs clean install uninstall info check
