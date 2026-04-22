#ifndef GPUENGINEH
#define GPUENGINEH

// ============================================================================
// PhiKang — GPU Engine Interface
//
// Changes vs original:
//   - KSIZE fixed (10->12, 11->13) via Constants.h
//   - ITEM_SIZE fixed (56->72): full 256-bit distance storage
//   - GLV jump table support: jPhiPx, jD1, jD2
//   - SetGLVParams() added for GLV constant memory upload
//   - CUDA occupancy API for runtime-optimal launch configuration
//   - AI/Tensor hooks marked for GB300 phase
// ============================================================================

#include <vector>
#include <string>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include “../Constants.h”
#include “../SECPK1/SECP256k1.h”

// ––––––––––––––––––––––––––––––––––––––
// ITEM — distinguished point output structure
//
// Bug fix: original stored only 2 distance limbs (128-bit)
// Full 256-bit distance required for correct collision resolution
//
// Memory layout (72 bytes total):
//   Bytes  0-31 : x     (256-bit point X coordinate, 4 x uint64)
//   Bytes 32-63 : dist  (256-bit distance, 4 x uint64 — was 2 x uint64)
//   Bytes 64-71 : kIdx  (kangaroo index, encodes TAME/WILD via kIdx % 2)
// ––––––––––––––––––––––––––––––––––––––

typedef struct {
Int      x;       // Distinguished point X coordinate
Int      dist;    // Full 256-bit walk distance (bug fixed)
uint64_t kIdx;    // Kangaroo index (kIdx % 2 == WILD or TAME)
} ITEM;

// ––––––––––––––––––––––––––––––––––––––
// AIMetrics — performance metrics exported each kernel launch
// Populated by GPU kernel, read by host for monitoring
// Reserved for GB300 AI orchestration phase — zero cost when disabled
// ––––––––––––––––––––––––––––––––––––––

typedef struct {
uint64_t totalSteps;          // Cumulative walk steps
uint64_t dpFound;             // Distinguished points found
uint64_t collisions;          // Total collision events
uint64_t sameHerdCollisions;  // Collisions within same herd (wasted)
float    dpRate;              // DPs per million steps
float    avgJumpIdx;          // Average jump index (detects bias)
uint32_t activeKangaroos;     // Non-stalled kangaroo count
uint32_t kernelLaunches;      // Kernel launch counter
} AIMetrics;

// ––––––––––––––––––––––––––––––––––––––
// GPUEngine
// ––––––––––––––––––––––––––––––––––––––

class GPUEngine {

public:

```
// Construction / destruction
GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId,
          uint32_t maxFound);
~GPUEngine();

// Standard jump table parameters (distances + point coordinates)
void SetParams(Int *dpMask, Int *distance, Int *px, Int *py);

// GLV extended parameters
// Must be called after SetParams()
// Uploads precomputed phi(J[i]) and decomposed distances jD1, jD2
void SetGLVParams(Int *phiPx, Int *d1, Int *d2);

// Kangaroo state management
void SetKangaroos(Int *px, Int *py, Int *d);
void GetKangaroos(Int *px, Int *py, Int *d);
void SetKangaroo(uint64_t kIdx, Int *px, Int *py, Int *d);

// Wild kangaroo offset (applied to distances on WILD type)
void SetWildOffset(Int *offset);

// Kernel launch
bool Launch(std::vector<ITEM> &found, bool spinWait = false);
bool callKernel();
bool callKernelAndWait();

// Metrics (populated when USE_AI_MONITOR defined, zeroed otherwise)
AIMetrics GetMetrics() const { return metrics; }
void      ResetMetrics()     { memset(&metrics, 0, sizeof(AIMetrics)); }

// Utility
int  GetNbThread()  const { return nbThread; }
int  GetGroupSize() const { return GPU_GRP_SIZE; }
int  GetMemory()    const;
bool IsInitialised()const { return initialised; }

std::string deviceName;

// Static utilities
static void  *AllocatePinnedMemory(size_t size);
static void   FreePinnedMemory(void *buff);
static void   PrintCudaInfo();
static bool   GetGridSize(int gpuId, int *x, int *y);
```

private:

```
// --------------------------------------------------------------------
// Occupancy-based launch configuration
// Queries CUDA occupancy API at construction time for this GPU
// Stores optimal blocks and threads for comp_kangaroos kernel
// --------------------------------------------------------------------
void ComputeOccupancy();

// --------------------------------------------------------------------
// Internal state
// --------------------------------------------------------------------

Int      wildOffset;
int      gpuId;
int      nbThread;
int      nbThreadPerGroup;   // threads per block (from occupancy API)
int      optimalBlocks;      // blocks (from occupancy API)
bool     initialised;
bool     lostWarning;
uint32_t maxFound;

// Memory sizes
uint32_t outputSize;          // bytes for output buffer
uint32_t kangarooSize;        // bytes for kangaroo state (GPU)
uint32_t kangarooSizePinned;  // bytes for one block pinned transfer
uint32_t jumpSize;            // bytes for one jump table component

// GPU memory
uint64_t *inputKangaroo;      // GPU kangaroo state (device)
uint64_t *dpMask;             // GPU distinguished point mask
uint32_t *outputItem;         // GPU output buffer

// Pinned (host-mapped) memory for fast transfers
uint64_t *inputKangarooPinned;
uint32_t *outputItemPinned;
uint64_t *jumpPinned;         // Staging buffer for jump table uploads

// AI metrics buffer (device)
AIMetrics *metricsDevice;
AIMetrics  metrics;           // Host-side copy updated each Launch()
```

};

#endif // GPUENGINEH
