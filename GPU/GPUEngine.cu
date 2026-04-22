#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include "GLVMath.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <stdint.h>
#include <string.h>

#include "../Timer.h"
#include "GPUMath.h"
#include "GPUCompute.h"

using namespace std;

// ============================================================================
// Kernel forward declaration
// Defined in GPUCompute.h / GPUCompute.cu
// ============================================================================

__global__ void comp_kangaroos(
    uint64_t *kangaroos,
    uint32_t  maxFound,
    uint32_t *found,
    uint64_t *dpMask,
    AIMetrics *metrics
);

// ============================================================================
// SM to core count mapping
// Updated to include Blackwell (SM 9.0) for GB300
// ============================================================================

static int ConvertSMVer2Cores(int major, int minor) {

    typedef struct { int SM; int Cores; } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x20,  32 }, // Fermi  2.0
        { 0x21,  48 }, // Fermi  2.1
        { 0x30, 192 }, // Kepler 3.0
        { 0x32, 192 }, // Kepler 3.2
        { 0x35, 192 }, // Kepler 3.5
        { 0x37, 192 }, // Kepler 3.7
        { 0x50, 128 }, // Maxwell 5.0
        { 0x52, 128 }, // Maxwell 5.2
        { 0x53, 128 }, // Maxwell 5.3
        { 0x60,  64 }, // Pascal  6.0
        { 0x61, 128 }, // Pascal  6.1
        { 0x62, 128 }, // Pascal  6.2
        { 0x70,  64 }, // Volta   7.0
        { 0x72,  64 }, // Volta   7.2
        { 0x75,  64 }, // Turing  7.5
        { 0x80,  64 }, // Ampere  8.0
        { 0x86, 128 }, // Ampere  8.6  (RTX 3090)
        { 0x87, 128 }, // Ampere  8.7
        { 0x89, 128 }, // Ada     8.9  (RTX 4090)
        { 0x90, 128 }, // Hopper  9.0  (H100)
        { 0x92, 128 }, // Blackwell 9.2 (RTX 5090)
        { 0x100,128 }, // Blackwell Ultra (GB300)
        { -1,    -1 }
    };

    for (int i = 0; nGpuArchCoresPerSM[i].SM != -1; i++) {
        if (nGpuArchCoresPerSM[i].SM == ((major << 4) + minor))
            return nGpuArchCoresPerSM[i].Cores;
    }

    // Unknown architecture — return conservative estimate
    printf("[!] Warning: Unknown SM version %d.%d, assuming 128 cores/SM\n",
           major, minor);
    return 128;
}

// ============================================================================
// GetGridSize — compute optimal grid dimensions for a given GPU
// ============================================================================

bool GPUEngine::GetGridSize(int gpuId, int *x, int *y) {

    if (*x > 0 && *y > 0)
        return true; // User specified, use as-is

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("GPUEngine::GetGridSize: %s\n", cudaGetErrorString(err));
        return false;
    }
    if (deviceCount == 0) {
        printf("GPUEngine::GetGridSize: No CUDA devices found\n");
        return false;
    }
    if (gpuId >= deviceCount) {
        printf("GPUEngine::GetGridSize: Invalid gpuId %d\n", gpuId);
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuId);

    if (*x <= 0) *x = 2 * prop.multiProcessorCount;
    if (*y <= 0) {
        int cores = ConvertSMVer2Cores(prop.major, prop.minor);
        *y = (cores > 0) ? 2 * cores : 128;
    }

    return true;
}

// ============================================================================
// ComputeOccupancy — query CUDA occupancy API for optimal launch config
//
// Determines the thread block size that maximises SM occupancy for
// comp_kangaroos kernel on the current GPU.
// Stores result in nbThreadPerGroup and optimalBlocks.
// ============================================================================

void GPUEngine::ComputeOccupancy() {

    int minGridSize  = 0;
    int blockSize    = 0;

    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void *)comp_kangaroos,
        0,    // dynamic shared memory bytes
        0     // block size limit (0 = no limit)
    );

    if (err != cudaSuccess) {
        printf("GPUEngine: OccupancyMaxPotentialBlockSize: %s\n",
               cudaGetErrorString(err));
        // Fall back to nbThreadPerGroup as set by constructor
        return;
    }

    // blockSize from occupancy API is the optimal threads per block
    // Clamp to a multiple of warp size (32) and to our group size
    int warpSize = 32;
    blockSize = (blockSize / warpSize) * warpSize;
    if (blockSize < warpSize) blockSize = warpSize;

    nbThreadPerGroup = blockSize;

    // Compute number of blocks to fill the device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuId);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        (void *)comp_kangaroos,
        nbThreadPerGroup,
        0
    );

    optimalBlocks = maxBlocksPerSM * prop.multiProcessorCount;

    // nbThread must be consistent: blocks * threadsPerBlock
    nbThread = optimalBlocks * nbThreadPerGroup;

    printf("[+] Occupancy API: %d threads/block, %d blocks, %d total threads\n",
           nbThreadPerGroup, optimalBlocks, nbThread);
}

// ============================================================================
// Constructor
// ============================================================================

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId,
                     uint32_t maxFound) {

    this->gpuId           = gpuId;
    this->nbThreadPerGroup= nbThreadPerGroup;
    this->maxFound        = maxFound;
    this->initialised     = false;
    this->lostWarning     = false;
    this->optimalBlocks   = nbThreadGroup;
    this->nbThread        = nbThreadGroup * nbThreadPerGroup;

    memset(&metrics, 0, sizeof(AIMetrics));

    // Null all pointers before any early return
    inputKangaroo       = NULL;
    inputKangarooPinned = NULL;
    outputItem          = NULL;
    outputItemPinned    = NULL;
    jumpPinned          = NULL;
    dpMask              = NULL;
    metricsDevice       = NULL;

    wildOffset.SetInt32(0);

    cudaError_t err;

    // Select device
    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("GPUEngine: cudaGetDeviceCount: %s\n", cudaGetErrorString(err));
        return;
    }
    if (deviceCount == 0) {
        printf("GPUEngine: No CUDA devices available\n");
        return;
    }

    err = cudaSetDevice(gpuId);
    if (err != cudaSuccess) {
        printf("GPUEngine: cudaSetDevice(%d): %s\n", gpuId,
               cudaGetErrorString(err));
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuId);

    char tmp[512];
    sprintf(tmp, "GPU #%d %s (%dx%d cores, SM %d.%d, %.0f MB) Grid(%dx%d)",
            gpuId, prop.name,
            prop.multiProcessorCount,
            ConvertSMVer2Cores(prop.major, prop.minor),
            prop.major, prop.minor,
            (double)prop.totalGlobalMem / 1048576.0,
            nbThread / nbThreadPerGroup,
            nbThreadPerGroup);
    deviceName = string(tmp);

    // Prefer L1 cache over shared memory for our access pattern
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (err != cudaSuccess) {
        printf("GPUEngine: SetCacheConfig: %s\n", cudaGetErrorString(err));
        return;
    }

    // -----------------------------------------------------------------------
    // Run occupancy API to get optimal launch parameters
    // This overrides nbThreadGroup/nbThreadPerGroup passed in by caller
    // with values tuned for the specific GPU being used
    // -----------------------------------------------------------------------
    ComputeOccupancy();

    // -----------------------------------------------------------------------
    // Allocate GPU memory
    // -----------------------------------------------------------------------

    // Distinguished point mask (32 bytes = 256 bits)
    err = cudaMalloc((void **)&dpMask, 32);
    if (err != cudaSuccess) {
        printf("GPUEngine: Alloc dpMask: %s\n", cudaGetErrorString(err));
        return;
    }

    // Kangaroo state buffer
    // Bug fix: KSIZE is now 12 (without sym) or 13 (with sym)
    // = nbThread * GPU_GRP_SIZE * KSIZE * 8 bytes
    kangarooSize = (uint32_t)((uint64_t)nbThread *
                              GPU_GRP_SIZE * KSIZE * 8);
    err = cudaMalloc((void **)&inputKangaroo, kangarooSize);
    if (err != cudaSuccess) {
        printf("GPUEngine: Alloc kangaroo buffer (%u MB): %s\n",
               kangarooSize / (1024*1024), cudaGetErrorString(err));
        return;
    }

    // Pinned kangaroo staging buffer (one block at a time)
    kangarooSizePinned = (uint32_t)((uint64_t)nbThreadPerGroup *
                                    GPU_GRP_SIZE * KSIZE * 8);
    err = cudaHostAlloc(&inputKangarooPinned, kangarooSizePinned,
                        cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Alloc pinned kangaroo: %s\n",
               cudaGetErrorString(err));
        return;
    }

    // Output buffer
    // Bug fix: ITEM_SIZE is now 72 bytes (was 56)
    outputSize = maxFound * ITEM_SIZE + 4;
    err = cudaMalloc((void **)&outputItem, outputSize);
    if (err != cudaSuccess) {
        printf("GPUEngine: Alloc output buffer: %s\n",
               cudaGetErrorString(err));
        return;
    }
    err = cudaHostAlloc(&outputItemPinned, outputSize, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: Alloc pinned output: %s\n",
               cudaGetErrorString(err));
        return;
    }

    // Jump table staging buffer
    // Sized for 3 tables: jD, jPx, jPy, jPhiPx, jD1, jD2
    // Each table: NB_JUMP * 4 * 8 bytes
    jumpSize = NB_JUMP * 4 * 8; // bytes for one table component
    err = cudaHostAlloc(&jumpPinned, jumpSize,
                        cudaHostAllocMapped | cudaHostAllocWriteCombined);
    if (err != cudaSuccess) {
        printf("GPUEngine: Alloc jump pinned: %s\n", cudaGetErrorString(err));
        return;
    }

    // AI metrics buffer (device side)
    err = cudaMalloc((void **)&metricsDevice, sizeof(AIMetrics));
    if (err != cudaSuccess) {
        printf("GPUEngine: Alloc metrics: %s\n", cudaGetErrorString(err));
        // Non-fatal — metrics just won't be populated
        metricsDevice = NULL;
    } else {
        cudaMemset(metricsDevice, 0, sizeof(AIMetrics));
    }

    initialised = true;
}

// ============================================================================
// Destructor
// ============================================================================

GPUEngine::~GPUEngine() {
    if (dpMask)              cudaFree(dpMask);
    if (inputKangaroo)       cudaFree(inputKangaroo);
    if (outputItem)          cudaFree(outputItem);
    if (metricsDevice)       cudaFree(metricsDevice);
    if (inputKangarooPinned) cudaFreeHost(inputKangarooPinned);
    if (outputItemPinned)    cudaFreeHost(outputItemPinned);
    if (jumpPinned)          cudaFreeHost(jumpPinned);
}

// ============================================================================
// GetMemory — return total GPU memory allocated in bytes
// ============================================================================

int GPUEngine::GetMemory() const {
    return (int)(kangarooSize + outputSize + jumpSize * 6);
}

// ============================================================================
// SetWildOffset
// ============================================================================

void GPUEngine::SetWildOffset(Int *offset) {
    wildOffset.Set(offset);
}

// ============================================================================
// SetParams — upload standard jump table to constant memory
// Called with dpMask, distances, jump point X and Y coordinates
// ============================================================================

void GPUEngine::SetParams(Int *dpMaskIn, Int *distance, Int *px, Int *py) {

    cudaError_t err;

    // Upload dpMask (256-bit = 32 bytes)
    uint64_t hostMask[4];
    hostMask[0] = dpMaskIn->bits64[0];
    hostMask[1] = dpMaskIn->bits64[1];
    hostMask[2] = dpMaskIn->bits64[2];
    hostMask[3] = dpMaskIn->bits64[3];
    cudaMemcpy(this->dpMask, hostMask, 32, cudaMemcpyHostToDevice);

    // Upload jump distances jD to constant memory
    for (int i = 0; i < NB_JUMP; i++)
        memcpy(jumpPinned + 4*i, distance[i].bits64, 32);
    cudaMemcpyToSymbol(jD, jumpPinned, jumpSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetParams jD: %s\n", cudaGetErrorString(err));
        return;
    }

    // Upload jump point X coords jPx
    for (int i = 0; i < NB_JUMP; i++)
        memcpy(jumpPinned + 4*i, px[i].bits64, 32);
    cudaMemcpyToSymbol(jPx, jumpPinned, jumpSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetParams jPx: %s\n", cudaGetErrorString(err));
        return;
    }

    // Upload jump point Y coords jPy
    for (int i = 0; i < NB_JUMP; i++)
        memcpy(jumpPinned + 4*i, py[i].bits64, 32);
    cudaMemcpyToSymbol(jPy, jumpPinned, jumpSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetParams jPy: %s\n", cudaGetErrorString(err));
        return;
    }
}

// ============================================================================
// SetGLVParams — upload GLV extended jump table to constant memory
//
// Must be called after SetParams()
// phiPx[i] = beta * jPx[i] mod p  (precomputed on host via ApplyPhiHost)
// d1[i], d2[i] = GLV decomposition of jD[i]  (via GLVDecomposeScalar)
// ============================================================================

void GPUEngine::SetGLVParams(Int *phiPx, Int *d1, Int *d2) {

    cudaError_t err;

    // Upload phi(J[i]).x = beta * J[i].x mod p
    for (int i = 0; i < NB_JUMP; i++)
        memcpy(jumpPinned + 4*i, phiPx[i].bits64, 32);
    cudaMemcpyToSymbol(jPhiPx, jumpPinned, jumpSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetGLVParams jPhiPx: %s\n", cudaGetErrorString(err));
        return;
    }

    // Upload jD1 (k1 component of each jump distance)
    for (int i = 0; i < NB_JUMP; i++)
        memcpy(jumpPinned + 4*i, d1[i].bits64, 32);
    cudaMemcpyToSymbol(jD1, jumpPinned, jumpSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetGLVParams jD1: %s\n", cudaGetErrorString(err));
        return;
    }

    // Upload jD2 (k2 component of each jump distance)
    for (int i = 0; i < NB_JUMP; i++)
        memcpy(jumpPinned + 4*i, d2[i].bits64, 32);
    cudaMemcpyToSymbol(jD2, jumpPinned, jumpSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: SetGLVParams jD2: %s\n", cudaGetErrorString(err));
        return;
    }
}

// ============================================================================
// SetKangaroos — upload all kangaroo state to GPU
//
// Bug fix: now stores all 4 distance limbs (bits64[0..3])
// Original only stored bits64[0..1] causing distance corruption on resume
//
// Memory layout per kangaroo (KSIZE words, each 8 bytes):
//   Words 0-3:   px  (4 limbs)
//   Words 4-7:   py  (4 limbs)
//   Words 8-11:  dist (4 limbs — was 2, BUG FIXED)
//   Word  12:    lastJump (USE_SYMMETRY only)
// ============================================================================

void GPUEngine::SetKangaroos(Int *px, Int *py, Int *d) {

    int gSize      = KSIZE * GPU_GRP_SIZE;
    int strideSize = nbThreadPerGroup * KSIZE;
    int nbBlock    = nbThread / nbThreadPerGroup;
    int blockSize  = nbThreadPerGroup * gSize;
    int idx        = 0;

    for (int b = 0; b < nbBlock; b++) {

        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            for (int t = 0; t < nbThreadPerGroup; t++) {

                // px — 4 limbs
                Int tpx = px[idx];
                inputKangarooPinned[g * strideSize + t + 0 * nbThreadPerGroup]
                    = tpx.bits64[0];
                inputKangarooPinned[g * strideSize + t + 1 * nbThreadPerGroup]
                    = tpx.bits64[1];
                inputKangarooPinned[g * strideSize + t + 2 * nbThreadPerGroup]
                    = tpx.bits64[2];
                inputKangarooPinned[g * strideSize + t + 3 * nbThreadPerGroup]
                    = tpx.bits64[3];

                // py — 4 limbs
                Int tpy = py[idx];
                inputKangarooPinned[g * strideSize + t + 4 * nbThreadPerGroup]
                    = tpy.bits64[0];
                inputKangarooPinned[g * strideSize + t + 5 * nbThreadPerGroup]
                    = tpy.bits64[1];
                inputKangarooPinned[g * strideSize + t + 6 * nbThreadPerGroup]
                    = tpy.bits64[2];
                inputKangarooPinned[g * strideSize + t + 7 * nbThreadPerGroup]
                    = tpy.bits64[3];

                // dist — 4 limbs (bug fix: was only 2 limbs in original)
                Int dOff;
                dOff.Set(&d[idx]);
                if (idx % 2 == WILD)
                    dOff.ModAddK1order(&wildOffset);

                inputKangarooPinned[g * strideSize + t + 8  * nbThreadPerGroup]
                    = dOff.bits64[0];
                inputKangarooPinned[g * strideSize + t + 9  * nbThreadPerGroup]
                    = dOff.bits64[1];
                inputKangarooPinned[g * strideSize + t + 10 * nbThreadPerGroup]
                    = dOff.bits64[2];
                inputKangarooPinned[g * strideSize + t + 11 * nbThreadPerGroup]
                    = dOff.bits64[3];

#ifdef USE_SYMMETRY
                // lastJump initialised to NB_JUMP (sentinel: no last jump)
                inputKangarooPinned[g * strideSize + t + 12 * nbThreadPerGroup]
                    = (uint64_t)NB_JUMP;
#endif
                idx++;
            }
        }

        uint32_t offset = b * blockSize;
        cudaMemcpy(inputKangaroo + offset, inputKangarooPinned,
                   kangarooSizePinned, cudaMemcpyHostToDevice);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("GPUEngine: SetKangaroos: %s\n", cudaGetErrorString(err));
}

// ============================================================================
// GetKangaroos — retrieve kangaroo state from GPU to host
//
// Bug fix: now reads all 4 distance limbs (bits64[0..3])
// Original only read bits64[0..1] — upper 128 bits always zero after resume
// This was the primary cause of workfile continuation being broken
// ============================================================================

void GPUEngine::GetKangaroos(Int *px, Int *py, Int *d) {

    if (inputKangarooPinned == NULL) {
        printf("GPUEngine: GetKangaroos: pinned memory freed\n");
        return;
    }

    int gSize      = KSIZE * GPU_GRP_SIZE;
    int strideSize = nbThreadPerGroup * KSIZE;
    int nbBlock    = nbThread / nbThreadPerGroup;
    int blockSize  = nbThreadPerGroup * gSize;
    int idx        = 0;

    for (int b = 0; b < nbBlock; b++) {

        uint32_t offset = b * blockSize;
        cudaMemcpy(inputKangarooPinned, inputKangaroo + offset,
                   kangarooSizePinned, cudaMemcpyDeviceToHost);

        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            for (int t = 0; t < nbThreadPerGroup; t++) {

                // px — 4 limbs
                px[idx].bits64[0] = inputKangarooPinned[
                    g * strideSize + t + 0 * nbThreadPerGroup];
                px[idx].bits64[1] = inputKangarooPinned[
                    g * strideSize + t + 1 * nbThreadPerGroup];
                px[idx].bits64[2] = inputKangarooPinned[
                    g * strideSize + t + 2 * nbThreadPerGroup];
                px[idx].bits64[3] = inputKangarooPinned[
                    g * strideSize + t + 3 * nbThreadPerGroup];
                px[idx].bits64[4] = 0;

                // py — 4 limbs
                py[idx].bits64[0] = inputKangarooPinned[
                    g * strideSize + t + 4 * nbThreadPerGroup];
                py[idx].bits64[1] = inputKangarooPinned[
                    g * strideSize + t + 5 * nbThreadPerGroup];
                py[idx].bits64[2] = inputKangarooPinned[
                    g * strideSize + t + 6 * nbThreadPerGroup];
                py[idx].bits64[3] = inputKangarooPinned[
                    g * strideSize + t + 7 * nbThreadPerGroup];
                py[idx].bits64[4] = 0;

                // dist — 4 limbs (bug fix: was only bits64[0..1])
                Int dOff;
                dOff.SetInt32(0);
                dOff.bits64[0] = inputKangarooPinned[
                    g * strideSize + t + 8  * nbThreadPerGroup];
                dOff.bits64[1] = inputKangarooPinned[
                    g * strideSize + t + 9  * nbThreadPerGroup];
                dOff.bits64[2] = inputKangarooPinned[
                    g * strideSize + t + 10 * nbThreadPerGroup];
                dOff.bits64[3] = inputKangarooPinned[
                    g * strideSize + t + 11 * nbThreadPerGroup];
                dOff.bits64[4] = 0;

                if (idx % 2 == WILD)
                    dOff.ModSubK1order(&wildOffset);

                d[idx].Set(&dOff);
                idx++;
            }
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("GPUEngine: GetKangaroos: %s\n", cudaGetErrorString(err));
}

// ============================================================================
// SetKangaroo — reset a single kangaroo (called on same-herd collision)
// ============================================================================

void GPUEngine::SetKangaroo(uint64_t kIdx, Int *px, Int *py, Int *d) {

    int gSize      = KSIZE * GPU_GRP_SIZE;
    int strideSize = nbThreadPerGroup * KSIZE;
    int blockSize  = nbThreadPerGroup * gSize;

    uint64_t t = kIdx % nbThreadPerGroup;
    uint64_t g = (kIdx / nbThreadPerGroup) % GPU_GRP_SIZE;
    uint64_t b = kIdx / ((uint64_t)nbThreadPerGroup * GPU_GRP_SIZE);

    uint64_t base = b * blockSize + g * strideSize + t;

    // Helper lambda to write one uint64 word to GPU
    auto WriteWord = [&](uint64_t wordOffset, uint64_t value) {
        inputKangarooPinned[0] = value;
        cudaMemcpy(inputKangaroo + base + wordOffset * nbThreadPerGroup,
                   inputKangarooPinned, 8, cudaMemcpyHostToDevice);
    };

    // px — 4 limbs
    WriteWord(0, px->bits64[0]);
    WriteWord(1, px->bits64[1]);
    WriteWord(2, px->bits64[2]);
    WriteWord(3, px->bits64[3]);

    // py — 4 limbs
    WriteWord(4, py->bits64[0]);
    WriteWord(5, py->bits64[1]);
    WriteWord(6, py->bits64[2]);
    WriteWord(7, py->bits64[3]);

    // dist — 4 limbs (bug fix: was only 2 limbs)
    Int dOff;
    dOff.Set(d);
    if (kIdx % 2 == WILD)
        dOff.ModAddK1order(&wildOffset);

    WriteWord(8,  dOff.bits64[0]);
    WriteWord(9,  dOff.bits64[1]);
    WriteWord(10, dOff.bits64[2]);
    WriteWord(11, dOff.bits64[3]);

#ifdef USE_SYMMETRY
    WriteWord(12, (uint64_t)NB_JUMP);
#endif
}

// ============================================================================
// callKernel — launch comp_kangaroos and return immediately (async)
// ============================================================================

bool GPUEngine::callKernel() {

    // Reset found counter
    cudaMemset(outputItem, 0, 4);

    // Reset metrics on device
    if (metricsDevice)
        cudaMemset(metricsDevice, 0, sizeof(AIMetrics));

    // Launch with occupancy-tuned grid
    comp_kangaroos<<<optimalBlocks, nbThreadPerGroup>>>(
        inputKangaroo,
        maxFound,
        outputItem,
        dpMask,
        metricsDevice
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel launch: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

// ============================================================================
// callKernelAndWait — launch and synchronise (used for debugging)
// ============================================================================

bool GPUEngine::callKernelAndWait() {

    if (!callKernel()) return false;

    cudaMemcpy(outputItemPinned, outputItem, outputSize,
               cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: callKernelAndWait sync: %s\n",
               cudaGetErrorString(err));
        return false;
    }

    return true;
}

// ============================================================================
// Launch — collect results from previous kernel, launch next kernel
//
// Bug fix: ITEM unpacking now reads full 256-bit distance (4 limbs)
// Original only read 2 limbs, silently truncating the upper 128 bits
// ============================================================================

bool GPUEngine::Launch(vector<ITEM> &hashFound, bool spinWait) {

    hashFound.clear();

    if (spinWait) {

        cudaMemcpy(outputItemPinned, outputItem, outputSize,
                   cudaMemcpyDeviceToHost);

    } else {

        // Async copy — poll event to free CPU
        cudaEvent_t evt;
        cudaEventCreate(&evt);
        cudaMemcpyAsync(outputItemPinned, outputItem, 4,
                        cudaMemcpyDeviceToHost, 0);
        cudaEventRecord(evt, 0);
        while (cudaEventQuery(evt) == cudaErrorNotReady)
            Timer::SleepMillis(1);
        cudaEventDestroy(evt);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return false;

    // Pull metrics from device
    if (metricsDevice) {
        cudaMemcpy(&metrics, metricsDevice, sizeof(AIMetrics),
                   cudaMemcpyDeviceToHost);
        metrics.kernelLaunches++;
    }

    // Process found items
    uint32_t nbFound = outputItemPinned[0];
    if (nbFound > maxFound) {
        if (!lostWarning) {
            printf("\n[!] Warning: %d items lost — reduce threads (-g) "
                   "or increase DP (-d)\n", nbFound - maxFound);
            lostWarning = true;
        }
        nbFound = maxFound;
    }

    if (nbFound > 0) {

        // Full copy of found items
        cudaMemcpy(outputItemPinned, outputItem,
                   nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost);

        for (uint32_t i = 0; i < nbFound; i++) {

            // Output buffer layout (ITEM_SIZE32 = 18 uint32 words):
            // Word 0:    count (skipped, at outputItemPinned[0])
            // Words 1-8: x (8 x uint32 = 256 bits)
            // Words 9-16: dist (8 x uint32 = 256 bits, was words 9-12 = bug)
            // Words 17-18: kIdx (2 x uint32 = 64 bits)

            uint32_t *itemPtr = outputItemPinned + (i * ITEM_SIZE32 + 1);
            ITEM it;

            // x — 256 bits
            uint64_t *xPtr = (uint64_t *)itemPtr;
            it.x.bits64[0] = xPtr[0];
            it.x.bits64[1] = xPtr[1];
            it.x.bits64[2] = xPtr[2];
            it.x.bits64[3] = xPtr[3];
            it.x.bits64[4] = 0;

            // dist — full 256 bits (bug fix: was only 2 limbs)
            uint64_t *dPtr = (uint64_t *)(itemPtr + 8);
            it.dist.bits64[0] = dPtr[0];
            it.dist.bits64[1] = dPtr[1];
            it.dist.bits64[2] = dPtr[2];
            it.dist.bits64[3] = dPtr[3];
            it.dist.bits64[4] = 0;

            // kIdx — 64 bits
            it.kIdx = *((uint64_t *)(itemPtr + 16));

            // Undo wild offset
            if (it.kIdx % 2 == WILD)
                it.dist.ModSubK1order(&wildOffset);

            hashFound.push_back(it);
        }
    }

    // Launch next kernel immediately
    return callKernel();
}

// ============================================================================
// Static utilities
// ============================================================================

void *GPUEngine::AllocatePinnedMemory(size_t size) {
    void *buff;
    cudaError_t err = cudaHostAlloc(&buff, size, cudaHostAllocPortable);
    if (err != cudaSuccess) {
        printf("GPUEngine: AllocatePinnedMemory: %s\n",
               cudaGetErrorString(err));
        return NULL;
    }
    return buff;
}

void GPUEngine::FreePinnedMemory(void *buff) {
    cudaFreeHost(buff);
}

void GPUEngine::PrintCudaInfo() {

    const char *modes[] = {
        "Multiple host threads",
        "Only one host thread",
        "No host thread",
        "Multiple process threads",
        "Unknown", NULL
    };

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("GPUEngine: cudaGetDeviceCount: %s\n", cudaGetErrorString(err));
        return;
    }
    if (deviceCount == 0) {
        printf("GPUEngine: No CUDA devices\n");
        return;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU #%d %s (%dx%d cores) SM %d.%d (%.1f MB) [%s]\n",
               i, prop.name,
               prop.multiProcessorCount,
               ConvertSMVer2Cores(prop.major, prop.minor),
               prop.major, prop.minor,
               (double)prop.totalGlobalMem / 1048576.0,
               modes[prop.computeMode]);
    }
}
