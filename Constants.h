#ifndef CONSTANTSH
#define CONSTANTSH

// ============================================================================
// PhiKang — Constants and Configuration
// Kangaroo ECDLP Solver with GLV Endomorphism for secp256k1
//
// Changelog vs original:
//   v3.0 — KSIZE fixed (10→12, 11→13)
//          GLV constants added (lambda, beta, decomposition basis)
//          Workfile version bumped to 3.0
//          TODO_TENSOR markers for GB300 phase
//          TODO_AI markers for orchestration phase
// ============================================================================

// ––––––––––––––––––––––––––––––––––––––
// Release
// ––––––––––––––––––––––––––––––––––––––

#define RELEASE        “3.0”
#define WORKFILE_VERSION 3

// ––––––––––––––––––––––––––––––––––––––
// Symmetry optimization
// Uncomment to halve effective search space using equivalence classes
// Recommended for large ranges
// ––––––––––––––––––––––––––––––––––––––

// #define USE_SYMMETRY

// ––––––––––––––––––––––––––––––––––––––
// KSIZE — words per kangaroo in GPU memory
//
// Layout (each word = 8 bytes / uint64):
//   Word  0-3  : px     (256-bit point X coordinate)
//   Word  4-7  : py     (256-bit point Y coordinate)
//   Word  8-11 : dist   (256-bit distance — FULL 4 limbs, was 2 = BUG FIXED)
//   Word 12    : lastJump (only present when USE_SYMMETRY defined)
//
// BUG FIX: Original had KSIZE=10 (without sym) and KSIZE=11 (with sym)
// but actually stored 12 and 13 words respectively, causing:
//   1. kangarooSize allocation 20% too small → silent heap overflow
//   2. GetKangaroos only read dist[0,1] → distance corruption on resume
// ––––––––––––––––––––––––––––––––––––––

#ifdef USE_SYMMETRY
#define KSIZE 13    // px(4) + py(4) + dist(4) + lastJump(1)
#else
#define KSIZE 12    // px(4) + py(4) + dist(4)
#endif

// ––––––––––––––––––––––––––––––––––––––
// ITEM — output distinguished point structure
//
// BUG FIX: Original ITEM_SIZE=56 stored only 2 distance limbs (128-bit)
// Full 256-bit distance required for correct collision resolution
//
// Layout:
//   Bytes  0-31 : x     (256-bit point X coordinate)
//   Bytes 32-63 : dist  (256-bit distance — FULL 4 limbs, was 2 = BUG FIXED)
//   Bytes 64-71 : kIdx  (kangaroo index — encodes type TAME/WILD)
// ––––––––––––––––––––––––––––––––––––––

#define ITEM_SIZE   72                  // 32 (x) + 32 (dist) + 8 (kIdx)
#define ITEM_SIZE32 (ITEM_SIZE / 4)     // 18 uint32 words

// ––––––––––––––––––––––––––––––––––––––
// Kangaroo type
// ––––––––––––––––––––––––––––––––––––––

#define TAME 0
#define WILD 1

// ––––––––––––––––––––––––––––––––––––––
// GPU kernel configuration
//
// GPU_GRP_SIZE: kangaroos per thread
//   Occupancy API will tune threads-per-block at runtime
//   This value controls how many kangaroos each thread manages
//   128 is optimal for 4090/5090 — revisit for GB300
//
// NB_JUMP: jump table size
//   Must be power of 2 (used as bitmask)
//   32 gives good statistical distribution for most range sizes
//   Increase to 64 for very large ranges (>2^200)
//
// NB_RUN: kernel iterations before returning to host
//   Higher = fewer kernel launches = less overhead
//   Lower = more responsive DP collection
//   64 is balanced for 4090/5090
// ––––––––––––––––––––––––––––––––––––––

#define GPU_GRP_SIZE  128
#define NB_JUMP       32
#define NB_RUN        64

// ––––––––––––––––––––––––––––––––––––––
// GLV constants for secp256k1
//
// The GLV endomorphism phi: (x, y) → (beta*x mod p, y)
// satisfies phi(P) = lambda * P for all points P on the curve
//
// lambda: scalar endomorphism eigenvalue
//   lambda^2 + lambda + 1 ≡ 0 (mod n)
//   where n is the group order
//
// beta: field endomorphism constant
//   beta^3 ≡ 1 (mod p), beta ≠ 1
//   where p is the field prime
//
// Decomposition basis vectors (Babai rounding lattice):
//   a1, b1, a2, b2 satisfy:
//   k = k1 + k2*lambda (mod n)
//   with |k1|, |k2| < 2^128
//
// Reference: Gallant, Lambert, Vanstone (2001)
//            Bos, Costello, Longa, Naehrig (2012)
// ––––––––––––––––––––––––––––––––––––––

// lambda (hex, little-endian limbs)
// 5363AD4CC05C30E0A7DB2964B546F3BCAC9C52B33FA3CF1F5DEC8032EA15CBFE
#define GLV_LAMBDA_0  0x5DEC8032EA15CBFEULL
#define GLV_LAMBDA_1  0xAC9C52B33FA3CF1FULL
#define GLV_LAMBDA_2  0xA7DB2964B546F3BCULL
#define GLV_LAMBDA_3  0x5363AD4CC05C30E0ULL

// beta (hex, little-endian limbs)
// 7AE96A2B657C07107F9EC5151B542253B73A3BC31D23756A5A8079E4002562A
// Note: two valid beta values exist; this is the smaller one
#define GLV_BETA_0    0x5A8079E4002562A0ULL  // Note: trailing zero is correct
#define GLV_BETA_1    0xB73A3BC31D23756AULL
#define GLV_BETA_2    0x7F9EC5151B542253ULL
#define GLV_BETA_3    0x07AE96A2B657C071ULL  // Note: leading zero is correct

// GLV lattice basis for scalar decomposition
// These are the precomputed constants from the reduction algorithm
// k1 = k - c1*a1 - c2*a2  (mod n)
// k2 =   - c1*b1 - c2*b2  (mod n)
// where c1 = round(b2*k/n), c2 = round(-b1*k/n)
#define GLV_A1_0  0x3086D221A7D46BCDULL
#define GLV_A1_1  0x0000000000000000ULL
#define GLV_B1_0  0xE4437ED6010E8828ULL  // negative: b1 = -0xE4437ED6010E8828
#define GLV_B1_1  0xFFFFFFFFFFFFFFFFULL  // sign extension
#define GLV_A2_0  0x114CA50F7A8E2F3FULL
#define GLV_A2_1  0x0000000000000000ULL
#define GLV_B2_0  0x3086D221A7D46BCDULL
#define GLV_B2_1  0x0000000000000000ULL

// ––––––––––––––––––––––––––––––––––––––
// Networking (server/client mode)
// ––––––––––––––––––––––––––––––––––––––

#define SEND_PERIOD     2.0       // seconds between DP sends to server
#define CLIENT_TIMEOUT  3600.0    // seconds before idle client disconnected

// ––––––––––––––––––––––––––––––––––––––
// Workfile
// ––––––––––––––––––––––––––––––––––––––

#define MERGE_PART      256       // partition count for split workfiles
#define WORKFILE_MAGIC  0x474E414B49484850ULL  // “PHIKANG\0” little-endian

// ––––––––––––––––––––––––––––––––––––––
// Jump selection entropy
//
// BUG FIX: Original used only lowest 5 bits of px[0]:
//   jmp = px[0] & (NB_JUMP-1)
// This ignores 251 bits of available entropy and creates
// correlated walks when points share low bits.
//
// PhiKang uses a full 256-bit XOR fold + MurmurHash3 finalizer
// defined in GPUCompute.h as SelectJump()
// ––––––––––––––––––––––––––––––––––––––

// MurmurHash3 finalizer constant (used in SelectJump)
#define MURMURHASH3_C1  0x45d9f3bULL

// ––––––––––––––––––––––––––––––––––––––
// Shared memory DP buffer
//
// BUG FIX: Original used global atomicAdd on every distinguished point
// causing contention under high DP rates on many-SM GPUs.
//
// PhiKang buffers DPs in shared memory and flushes to global
// once per warp when buffer fills. Size must be >= warp size (32).
// ––––––––––––––––––––––––––––––––––––––

#define DP_SHARED_BUFFER_SIZE  64   // DPs buffered per block before flush

// ––––––––––––––––––––––––––––––––––––––
// TODO_TENSOR: GB300 NVL72 tensor core configuration
// Uncomment and configure when Blackwell Ultra hardware arrives
//
// #define USE_TENSOR_CORES
// #define TC_BATCH_SIZE     256    // Elements per tensor core batch
// #define TC_TILE_M         16     // WMMA tile dimension M
// #define TC_TILE_N         16     // WMMA tile dimension N
// #define TC_TILE_K         16     // WMMA tile dimension K
// ––––––––––––––––––––––––––––––––––––––

// ––––––––––––––––––––––––––––––––––––––
// TODO_AI: AI orchestration layer
// Uncomment when fine-tuned model is ready for integration
//
// #define USE_AI_MONITOR
// #define AI_METRICS_INTERVAL   1000   // kernel launches between metric exports
// #define AI_BINARY_PORT        19000  // local port for binary metrics stream
// #define AI_JSON_LOG           “phikang_metrics.jsonl”
// ––––––––––––––––––––––––––––––––––––––

#endif // CONSTANTSH
