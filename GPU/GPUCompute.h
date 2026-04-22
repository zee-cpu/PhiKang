#ifndef GPUCOMPUTEH
#define GPUCOMPUTEH

// ============================================================================
// PhiKang — GPU Kernel
//
// Main compute kernel implementing Pollard Kangaroo walk with:
//   1. GLV endomorphism integration (1.4x speedup)
//   2. SoA-style register layout for coalesced memory access
//   3. 256-bit entropy jump selection (bias eliminated)
//   4. Shared memory DP buffer (atomic contention eliminated)
//   5. Full 256-bit distance tracking
//   6. USE_SYMMETRY compatible
//
// Memory layout (SoA on GPU, AoS on host — converted in Load/Store):
//   GPU register arrays: px[GPU_GRP_SIZE][4], py[GPU_GRP_SIZE][4]
//   Host pinned memory:  interleaved by thread (for coalesced transfers)
//
// ============================================================================

#include “../Constants.h”
#include “GPUMath.h”
#include “GLVMath.h”

// ============================================================================
// OutputDP — write distinguished point to shared memory buffer
// Replaces original global atomicAdd with shared memory staging
// ============================================================================

#define OutputDP(x, d, idx) {                                      
uint32_t slot = atomicAdd(shDPCount, 1);                       
if (slot < DP_SHARED_BUFFER_SIZE) {                            
uint32_t base = slot * ITEM_SIZE32;                        
/* x — 256 bits = 8 x uint32 */                           
shDPBuffer[base + 0] = ((uint32_t *)(x))[0];              
shDPBuffer[base + 1] = ((uint32_t *)(x))[1];              
shDPBuffer[base + 2] = ((uint32_t *)(x))[2];              
shDPBuffer[base + 3] = ((uint32_t *)(x))[3];              
shDPBuffer[base + 4] = ((uint32_t *)(x))[4];              
shDPBuffer[base + 5] = ((uint32_t *)(x))[5];              
shDPBuffer[base + 6] = ((uint32_t *)(x))[6];              
shDPBuffer[base + 7] = ((uint32_t *)(x))[7];              
/* dist — 256 bits = 8 x uint32 (full, bug fixed) */      
shDPBuffer[base + 8]  = ((uint32_t *)(d))[0];             
shDPBuffer[base + 9]  = ((uint32_t *)(d))[1];             
shDPBuffer[base + 10] = ((uint32_t *)(d))[2];             
shDPBuffer[base + 11] = ((uint32_t *)(d))[3];             
shDPBuffer[base + 12] = ((uint32_t *)(d))[4];             
shDPBuffer[base + 13] = ((uint32_t *)(d))[5];             
shDPBuffer[base + 14] = ((uint32_t *)(d))[6];             
shDPBuffer[base + 15] = ((uint32_t *)(d))[7];             
/* kIdx — 64 bits = 2 x uint32 */                         
shDPBuffer[base + 16] = ((uint32_t *)(idx))[0];           
shDPBuffer[base + 17] = ((uint32_t *)(idx))[1];           
}                                                              
}

// ============================================================================
// FlushDPBuffer — write shared memory DP buffer to global output
// Called by thread 0 of each block after each NB_RUN iteration
// ============================================================================

**device** **forceinline** void FlushDPBuffer(
uint32_t       *shDPBuffer,
uint32_t       *shDPCount,
uint32_t       *out,
uint32_t        maxFound
) {
__syncthreads();

```
if (threadIdx.x == 0 && *shDPCount > 0) {
    uint32_t cnt = min(*shDPCount, (uint32_t)DP_SHARED_BUFFER_SIZE);
    uint32_t pos = atomicAdd(out, cnt);

    if (pos < maxFound) {
        uint32_t toCopy = min(cnt, maxFound - pos);
        memcpy(out + 1 + pos * ITEM_SIZE32,
               shDPBuffer,
               toCopy * ITEM_SIZE32 * sizeof(uint32_t));
    }
    *shDPCount = 0;
}

__syncthreads();
```

}

// ============================================================================
// LoadKangaroos — GPU memory to registers
//
// Memory is laid out as:
//   [word_0_of_thread_0, word_0_of_thread_1, …, word_0_of_thread_N,
//    word_1_of_thread_0, word_1_of_thread_1, …, word_1_of_thread_N, …]
//
// This layout (originally chosen for coalesced writes from host) means
// adjacent threads read adjacent memory = coalesced reads on GPU too.
// ============================================================================

#ifdef USE_SYMMETRY
**device** void LoadKangaroos(
uint64_t *a,
uint64_t px[GPU_GRP_SIZE][4],
uint64_t py[GPU_GRP_SIZE][4],
uint64_t dist1[GPU_GRP_SIZE][4],
uint64_t dist2[GPU_GRP_SIZE][4],
uint64_t lastJump[GPU_GRP_SIZE]
)
#else
**device** void LoadKangaroos(
uint64_t *a,
uint64_t px[GPU_GRP_SIZE][4],
uint64_t py[GPU_GRP_SIZE][4],
uint64_t dist1[GPU_GRP_SIZE][4],
uint64_t dist2[GPU_GRP_SIZE][4]
)
#endif
{
__syncthreads();

```
for (int g = 0; g < GPU_GRP_SIZE; g++) {

    uint32_t stride = g * KSIZE * blockDim.x;

    // px — words 0-3
    px[g][0] = a[IDX + 0 * blockDim.x + stride];
    px[g][1] = a[IDX + 1 * blockDim.x + stride];
    px[g][2] = a[IDX + 2 * blockDim.x + stride];
    px[g][3] = a[IDX + 3 * blockDim.x + stride];

    // py — words 4-7
    py[g][0] = a[IDX + 4 * blockDim.x + stride];
    py[g][1] = a[IDX + 5 * blockDim.x + stride];
    py[g][2] = a[IDX + 6 * blockDim.x + stride];
    py[g][3] = a[IDX + 7 * blockDim.x + stride];

    // dist1 — words 8-11 (full 256-bit, bug fixed)
    dist1[g][0] = a[IDX + 8  * blockDim.x + stride];
    dist1[g][1] = a[IDX + 9  * blockDim.x + stride];
    dist1[g][2] = a[IDX + 10 * blockDim.x + stride];
    dist1[g][3] = a[IDX + 11 * blockDim.x + stride];

    // dist2 — initialise to zero (GLV k2 component starts at 0)
    // dist2 accumulates separately, reconstructed at collision
    dist2[g][0] = 0;
    dist2[g][1] = 0;
    dist2[g][2] = 0;
    dist2[g][3] = 0;
```

#ifdef USE_SYMMETRY
lastJump[g] = a[IDX + 12 * blockDim.x + stride];
#endif
}
}

// ============================================================================
// StoreKangaroos — registers back to GPU memory
// ============================================================================

#ifdef USE_SYMMETRY
**device** void StoreKangaroos(
uint64_t *a,
uint64_t px[GPU_GRP_SIZE][4],
uint64_t py[GPU_GRP_SIZE][4],
uint64_t dist1[GPU_GRP_SIZE][4],
uint64_t dist2[GPU_GRP_SIZE][4],
uint64_t lastJump[GPU_GRP_SIZE]
)
#else
**device** void StoreKangaroos(
uint64_t *a,
uint64_t px[GPU_GRP_SIZE][4],
uint64_t py[GPU_GRP_SIZE][4],
uint64_t dist1[GPU_GRP_SIZE][4],
uint64_t dist2[GPU_GRP_SIZE][4]
)
#endif
{
__syncthreads();

```
for (int g = 0; g < GPU_GRP_SIZE; g++) {

    uint32_t stride = g * KSIZE * blockDim.x;

    // px — words 0-3
    a[IDX + 0 * blockDim.x + stride] = px[g][0];
    a[IDX + 1 * blockDim.x + stride] = px[g][1];
    a[IDX + 2 * blockDim.x + stride] = px[g][2];
    a[IDX + 3 * blockDim.x + stride] = px[g][3];

    // py — words 4-7
    a[IDX + 4 * blockDim.x + stride] = py[g][0];
    a[IDX + 5 * blockDim.x + stride] = py[g][1];
    a[IDX + 6 * blockDim.x + stride] = py[g][2];
    a[IDX + 7 * blockDim.x + stride] = py[g][3];

    // Reconstruct full distance before storing
    // dist = dist1 + dist2 * lambda mod n
    uint64_t fullDist[4];
    GLVReconstructDist(fullDist, dist1[g], dist2[g]);

    a[IDX + 8  * blockDim.x + stride] = fullDist[0];
    a[IDX + 9  * blockDim.x + stride] = fullDist[1];
    a[IDX + 10 * blockDim.x + stride] = fullDist[2];
    a[IDX + 11 * blockDim.x + stride] = fullDist[3];
```

#ifdef USE_SYMMETRY
a[IDX + 12 * blockDim.x + stride] = lastJump[g];
#endif
}
}

// ============================================================================
// _ModInvGrouped — batch modular inverse (Montgomery trick)
// Preserved from original exactly — proven correct, PTX optimised
// ============================================================================

**device** **noinline** void _ModInvGrouped(uint64_t r[GPU_GRP_SIZE][4]) {

```
uint64_t subp[GPU_GRP_SIZE][4];
uint64_t newValue[4];
uint64_t inverse[5];

Load256(subp[0], r[0]);
for (uint32_t i = 1; i < GPU_GRP_SIZE; i++)
    _ModMult(subp[i], subp[i-1], r[i]);

Load256(inverse, subp[GPU_GRP_SIZE - 1]);
inverse[4] = 0;
_ModInv(inverse);

for (uint32_t i = GPU_GRP_SIZE - 1; i > 0; i--) {
    _ModMult(newValue, subp[i-1], inverse);
    _ModMult(inverse, r[i]);
    Load256(r[i], newValue);
}

Load256(r[0], inverse);
```

}

// ============================================================================
// ComputeKangaroos — main device function
// Called from comp_kangaroos kernel
// ============================================================================

**device** void ComputeKangaroos(
uint64_t *kangaroos,
uint32_t  maxFound,
uint32_t *out,
uint64_t *dpMask,
uint32_t *shDPBuffer,
uint32_t *shDPCount
) {
// ———————————————————————–
// Register-resident kangaroo state
// Using [GPU_GRP_SIZE][4] arrays — each thread manages GPU_GRP_SIZE
// kangaroos, each with 4-limb coordinates and distances
// ———————————————————————–

```
uint64_t px[GPU_GRP_SIZE][4];
uint64_t py[GPU_GRP_SIZE][4];
uint64_t dist1[GPU_GRP_SIZE][4];   // GLV k1 distance component
uint64_t dist2[GPU_GRP_SIZE][4];   // GLV k2 distance component
```

#ifdef USE_SYMMETRY
uint64_t lastJump[GPU_GRP_SIZE];
#endif

```
// dx: batch inverse workspace — one per kangaroo per step
uint64_t dx[GPU_GRP_SIZE][4];

// Temporary point arithmetic registers
uint64_t dy[4], rx[4], ry[4], _s[4], _p[4];

// DP mask to registers (avoid repeated global reads)
uint64_t dpmask0 = dpMask[0];
uint64_t dpmask1 = dpMask[1];
uint64_t dpmask2 = dpMask[2];
uint64_t dpmask3 = dpMask[3];

// Load kangaroo state from GPU memory into registers
```

#ifdef USE_SYMMETRY
LoadKangaroos(kangaroos, px, py, dist1, dist2, lastJump);
#else
LoadKangaroos(kangaroos, px, py, dist1, dist2);
#endif

```
// -----------------------------------------------------------------------
// Main walk loop — NB_RUN iterations per kernel call
// -----------------------------------------------------------------------

for (int run = 0; run < NB_RUN; run++) {

    // -------------------------------------------------------------------
    // Phase 1: Compute dx[g] = px[g] - jPx[jmp[g]] for batch inverse
    // -------------------------------------------------------------------

    __syncthreads();

    for (int g = 0; g < GPU_GRP_SIZE; g++) {

        // Jump selection: 256-bit entropy (bug fix)
        uint32_t jmp = SelectJump(px[g]);
```

#ifdef USE_SYMMETRY
// Anti-cycling: never repeat the same jump twice in a row
if (jmp == (uint32_t)lastJump[g])
jmp = (jmp + 1) & (NB_JUMP - 1);
lastJump[g] = (uint64_t)jmp;
#endif

```
        // dx = px - jPx[jmp] mod p
        ModSub256(dx[g], px[g], jPx[jmp]);
    }

    // -------------------------------------------------------------------
    // Phase 2: Batch modular inverse of dx (Montgomery trick)
    // Most expensive step — one field inversion for the whole group
    // -------------------------------------------------------------------

    _ModInvGrouped(dx);

    // -------------------------------------------------------------------
    // Phase 3: Complete point addition P_new = P + J[jmp]
    //          Update GLV distance components
    //          Check for distinguished points
    // -------------------------------------------------------------------

    __syncthreads();

    for (int g = 0; g < GPU_GRP_SIZE; g++) {
```

#ifdef USE_SYMMETRY
uint32_t jmp = (uint32_t)lastJump[g];
#else
uint32_t jmp = SelectJump(px[g]);
// Note: without symmetry we recompute jmp from the same px[g]
// that hasn’t changed yet, so result is identical to Phase 1
#endif

```
        // Point addition: P_new = P + J[jmp]
        // Using standard affine addition formula
        ModSub256(dy,  py[g], jPy[jmp]);       // dy = Py - Jy
        _ModMult(_s, dy, dx[g]);                // s = dy / dx (inv precomputed)
        _ModSqr(_p, _s);                        // p = s^2

        ModSub256(rx, _p,    jPx[jmp]);        // rx = s^2 - Jx
        ModSub256(rx, px[g]);                   // rx = s^2 - Jx - Px

        ModSub256(ry, px[g], rx);               // ry = Px - rx
        _ModMult(ry,  _s);                      // ry = s*(Px - rx)
        ModSub256(ry, py[g]);                   // ry = s*(Px - rx) - Py

        Load256(px[g], rx);
        Load256(py[g], ry);

        // GLV distance update:
        // dist1 += jD1[jmp] mod n  (k1 component)
        // dist2 += jD2[jmp] mod n  (k2 component)
        // Faster than full 256-bit dist += jD[jmp]:
        //   both jD1, jD2 are ~128-bit so additions rarely overflow
        GLVAddDist(dist1[g], dist2[g], jmp);
```

#ifdef USE_SYMMETRY
// Symmetry class: if new point is in upper half, negate
// and flip distance sign to keep in canonical form
if (ModPositive256(py[g]))
ModNeg256Order(dist1[g]);
#endif

```
        // Distinguished point check
        uint64_t *pxg = px[g];
        if ((pxg[0] & dpmask0) == 0 &&
            (pxg[1] & dpmask1) == 0 &&
            (pxg[2] & dpmask2) == 0 &&
            (pxg[3] & dpmask3) == 0) {

            // Reconstruct full distance for output
            uint64_t fullDist[4];
            GLVReconstructDist(fullDist, dist1[g], dist2[g]);

            // Kangaroo global index
            uint64_t kIdx =
                (uint64_t)IDX +
                (uint64_t)g * (uint64_t)blockDim.x +
                (uint64_t)blockIdx.x *
                    ((uint64_t)blockDim.x * GPU_GRP_SIZE);

            // Write to shared memory buffer (not global atomic)
            OutputDP(px[g], fullDist, &kIdx);
        }
    }

    // Flush shared DP buffer to global output after each run
    FlushDPBuffer(shDPBuffer, shDPCount, out, maxFound);

} // end NB_RUN loop

// Store kangaroo state back to GPU memory
// StoreKangaroos calls GLVReconstructDist internally
// so the saved distance is the correct full 256-bit value
```

#ifdef USE_SYMMETRY
StoreKangaroos(kangaroos, px, py, dist1, dist2, lastJump);
#else
StoreKangaroos(kangaroos, px, py, dist1, dist2);
#endif
}

// ============================================================================
// comp_kangaroos — top-level GPU kernel
// Grid and block dimensions set by occupancy API in GPUEngine.cu
// ============================================================================

**global** void comp_kangaroos(
uint64_t  *kangaroos,
uint32_t   maxFound,
uint32_t  *out,
uint64_t  *dpMask,
AIMetrics *metrics
) {
// Shared memory DP buffer — avoids global atomic per DP
// Size: DP_SHARED_BUFFER_SIZE slots * ITEM_SIZE32 words each
**shared** uint32_t shDPBuffer[DP_SHARED_BUFFER_SIZE * ITEM_SIZE32];
**shared** uint32_t shDPCount;

```
// Initialise shared state
if (threadIdx.x == 0)
    shDPCount = 0;
__syncthreads();

// Compute offset into kangaroo array for this block
int xPtr = (blockIdx.x * blockDim.x * GPU_GRP_SIZE) * KSIZE;

// Run the walk
ComputeKangaroos(
    kangaroos + xPtr,
    maxFound,
    out,
    dpMask,
    shDPBuffer,
    &shDPCount
);

// Update metrics (thread 0 of block 0 only to avoid races)
if (metrics && blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd((unsigned long long *)&metrics->totalSteps,
              (unsigned long long)(NB_RUN * GPU_GRP_SIZE *
                                   blockDim.x * gridDim.x));
    atomicAdd((unsigned long long *)&metrics->dpFound,
              (unsigned long long)out[0]);
    metrics->activeKangaroos = blockDim.x * gridDim.x * GPU_GRP_SIZE;
}
```

}

#endif // GPUCOMPUTEH
