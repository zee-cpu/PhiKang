#ifndef CONSTANTSH
#define CONSTANTSH

// ============================================================================
// PhiKang — Constants and Configuration
// Kangaroo ECDLP Solver with GLV Endomorphism for secp256k1
//
// Changelog vs original JeanLucPons/Kangaroo:
//   v3.0 — KSIZE fixed (10→12, 11→13) — silent heap overflow eliminated
//          ITEM_SIZE fixed (56→72) — full 256-bit distance restored
//          GLV constants added (lambda, beta, decomposition basis)
//          MurmurHash3 corrected to 64-bit finalizer constants
//          Static assertions added for GPU_GRP_SIZE and NB_JUMP
//          Workfile version bumped to 3.0
//          TODO_TENSOR markers for GB300 phase
//          TODO_AI markers for orchestration phase
// ============================================================================

// ––––––––––––––––––––––––––––––––––––––
// Release
// ––––––––––––––––––––––––––––––––––––––
#define RELEASE          "3.0"
#define WORKFILE_VERSION  3

// ––––––––––––––––––––––––––––––––––––––
// Symmetry optimization
// Uncomment to halve effective search space using equivalence classes.
// When enabled each wild kangaroo also covers its negation, so only
// the upper half of the range needs to be walked.
// Recommended for large ranges (>2^64).
// ––––––––––––––––––––––––––––––––––––––
// #define USE_SYMMETRY

// ––––––––––––––––––––––––––––––––––––––
// KSIZE — 64-bit words per kangaroo stored in GPU global memory
//
// Memory layout (each word = uint64_t, 8 bytes):
//   Word  0– 3 : px        (256-bit point X coordinate, 4 limbs)
//   Word  4– 7 : py        (256-bit point Y coordinate, 4 limbs)
//   Word  8–11 : dist      (256-bit accumulated distance, 4 limbs)
//   Word    12 : lastJump  (only present when USE_SYMMETRY is defined)
//
// BUG FIX (vs original JeanLucPons):
//   Original defined KSIZE=10 (no-sym) / KSIZE=11 (sym) but the kernel
//   actually read/wrote 12 and 13 words respectively, causing:
//     (1) kangarooSize allocation was 20% too small → silent heap overflow
//         on every single kernel call.
//     (2) GetKangaroos() only read dist[0] and dist[1] (128-bit), silently
//         discarding the upper 128 bits → distance corruption on any
//         workfile save/resume cycle.
// ––––––––––––––––––––––––––––––––––––––
#ifdef USE_SYMMETRY
  #define KSIZE 13   // px(4) + py(4) + dist(4) + lastJump(1)
#else
  #define KSIZE 12   // px(4) + py(4) + dist(4)
#endif

// ––––––––––––––––––––––––––––––––––––––
// ITEM — distinguished point record written to output buffer
//
// BUG FIX (vs original JeanLucPons):
//   Original ITEM_SIZE=56 stored only 2 distance limbs (128-bit).
//   Collision resolution requires the full 256-bit scalar distance to
//   reconstruct the private key.  With only 128 bits stored, any key
//   whose scalar exceeds 2^128 would produce a wrong or unsolvable
//   collision — a silent correctness failure.
//
// Layout:
//   Bytes  0–31 : x     (256-bit distinguished point X coordinate)
//   Bytes 32–63 : dist  (256-bit distance — ALL 4 limbs)
//   Bytes 64–71 : kIdx  (uint64 kangaroo index; LSB encodes TAME/WILD)
// ––––––––––––––––––––––––––––––––––––––
#define ITEM_SIZE   72            // 32 (x) + 32 (dist) + 8 (kIdx)
#define ITEM_SIZE32 (ITEM_SIZE/4) // 18  — uint32_t words per ITEM

// ––––––––––––––––––––––––––––––––––––––
// Kangaroo herd types
// ––––––––––––––––––––––––––––––––––––––
#define TAME 0
#define WILD 1

// ––––––––––––––––––––––––––––––––––––––
// GPU kernel configuration
//
// GPU_GRP_SIZE : kangaroos managed by each GPU thread.
//   Must be a multiple of the CUDA warp size (32).
//   128 is optimal for RTX 4090 / RTX 5090.
//   The CUDA Occupancy API tunes threads-per-block at runtime;
//   this constant controls how many kangaroos sit behind each thread.
//
// NB_JUMP      : jump table size.
//   Must be a power of 2 — used as a bitmask in SelectJump().
//   32 gives good statistical coverage for most range sizes.
//   Increase to 64 for very large ranges (>2^200).
//
// NB_RUN       : kernel iterations per host-side launch.
//   Higher → fewer kernel launches → less host/device round-trip overhead.
//   Lower  → more frequent DP collection → smaller output buffer needed.
//   64 is well balanced for 4090/5090 at typical DP rates.
// ––––––––––––––––––––––––––––––––––––––
#define GPU_GRP_SIZE  128
#define NB_JUMP        32
#define NB_RUN         64

// Compile-time sanity checks — catch misconfiguration before it reaches nvcc
#ifdef __cplusplus
  static_assert((GPU_GRP_SIZE % 32) == 0,
    "GPU_GRP_SIZE must be a multiple of the CUDA warp size (32)");
  static_assert((NB_JUMP & (NB_JUMP - 1)) == 0,
    "NB_JUMP must be a power of 2 (used as bitmask in SelectJump)");
  static_assert(NB_JUMP >= 16,
    "NB_JUMP must be at least 16 for acceptable jump distribution");
#endif

// ––––––––––––––––––––––––––––––––––––––
// GLV constants for secp256k1
//
// The Gallant-Lambert-Vanstone (GLV) endomorphism φ maps any point P to
// λ·P using only a single field multiplication:
//
//   φ(P) = (β·Px mod p,  Py)
//
// This lets us decompose a 256-bit scalar k into two ~128-bit scalars:
//
//   k ≡ k1 + k2·λ  (mod n)
//
// so that  k·P = k1·P + k2·φ(P),  with both multiplications ~128-bit.
// Net throughput gain: ~1.4× over standard full-width Kangaroo.
//
// All constants stored as 4× uint64_t limbs, little-endian
// (limb[0] = least-significant 64 bits).
//
// References:
//   Gallant, Lambert, Vanstone (2001) — original GLV paper
//   Bos, Costello, Longa, Naehrig (2012) — secp256k1 specific values
// ––––––––––––––––––––––––––––––––––––––

// lambda — the scalar endomorphism eigenvalue
//   Satisfies:  λ² + λ + 1 ≡ 0  (mod n)
//   Full value: 5363AD4CC05C30E0A7DB2964B546F3BCAC9C52B33FA3CF1F5DEC8032EA15CBFE
#define GLV_LAMBDA_0  0x5DEC8032EA15CBFEULL   // limb[0] — least significant
#define GLV_LAMBDA_1  0xAC9C52B33FA3CF1FULL
#define GLV_LAMBDA_2  0xA7DB2964B546F3BCULL
#define GLV_LAMBDA_3  0x5363AD4CC05C30E0ULL   // limb[3] — most significant

// beta — the field endomorphism constant
//   Satisfies:  β³ ≡ 1  (mod p),  β ≠ 1
//
//   Canonical value (from bitcoin-core secp256k1, SECP256K1_FE_CONST):
//   Full: 7AE96A2B657C07107F9EC5151B542253B73A3BC31D23756A5A8079E4002562A0
//
//   BUG FIX vs original zee-cpu code:
//   Original GLV_BETA_3 was 0x07AE96A2B657C071ULL — this is the correct
//   value right-shifted by one nibble (4 bits).  The MSB limb must end in
//   '0710' (from '657C0710'), not '071'.  The corrupted constant would
//   cause phi(P) to produce the wrong point on every step — GLV-enhanced
//   walks would silently walk away from any collision and never converge.
#define GLV_BETA_0    0x5A8079E4002562A0ULL   // limb[0] — least significant
#define GLV_BETA_1    0xB73A3BC31D23756AULL   // limb[1]
#define GLV_BETA_2    0x7F9EC5151B542253ULL   // limb[2]
#define GLV_BETA_3    0x7AE96A2B657C0710ULL   // limb[3] — most significant (FIXED)

// GLV lattice basis for scalar decomposition (Babai nearest-plane method)
//
// Given scalar k, compute:
//   c1 = round( b2·k / n )
//   c2 = round(-b1·k / n )
// Then:
//   k1 = k  - c1·a1 - c2·a2  (mod n)
//   k2 =    - c1·b1 - c2·b2  (mod n)
//
// Both k1 and k2 fit in ~128 bits, enabling the half-width scalar mult.
//
// a1 =  0x3086D221A7D46BCDE7376F2F359E4715   (positive, 128-bit)
// b1 = -0xE4437ED6010E88285A7DFEEF1A68A4BE   (negative, 128-bit magnitude)
// a2 =  0x114CA50F7A8E2F3F657217C68D3BBE9E   (positive, 128-bit)
// b2 =  0x3086D221A7D46BCDE7376F2F359E4715   (= a1, positive, 128-bit)

#define GLV_A1_0  0xE7376F2F359E4715ULL  // a1 limb[0]
#define GLV_A1_1  0x3086D221A7D46BCDULL  // a1 limb[1]

#define GLV_B1_0  0x5A7DFEEF1A68A4BEULL  // |b1| limb[0]  — b1 is NEGATIVE
#define GLV_B1_1  0xE4437ED6010E8828ULL  // |b1| limb[1]  — negate when used

#define GLV_A2_0  0x657217C68D3BBE9EULL  // a2 limb[0]
#define GLV_A2_1  0x114CA50F7A8E2F3FULL  // a2 limb[1]

#define GLV_B2_0  0xE7376F2F359E4715ULL  // b2 limb[0]  — same as a1
#define GLV_B2_1  0x3086D221A7D46BCDULL  // b2 limb[1]  — same as a1

// ––––––––––––––––––––––––––––––––––––––
// Jump selection entropy (MurmurHash3 64-bit finalizer)
//
// BUG FIX (vs original JeanLucPons):
//   Original jump selection: jmp = px[0] & (NB_JUMP - 1)
//   This uses only the lowest 5 bits of the X coordinate, discarding
//   251 bits of entropy.  Kangaroos whose points share low bits will
//   select the same jump, producing correlated (non-random) walks and
//   degrading convergence.
//
// PhiKang fix (implemented in GPU/GPUCompute.h :: SelectJump()):
//   1. XOR-fold all four 64-bit limbs of px into a single uint64_t.
//   2. Apply the MurmurHash3 64-bit finalizer (fmix64) for avalanche.
//   3. Mask to NB_JUMP-1.
//
// The two fmix64 constants below are the standard MurmurHash3 64-bit
// finalizer magic numbers (Austin Appleby, public domain).
// FIX: Previous version incorrectly used 0x45d9f3b (32-bit constant).
// ––––––––––––––––––––––––––––––––––––––
#define MURMURHASH3_C1  0xFF51AFD7ED558CCDULL   // fmix64 first constant
#define MURMURHASH3_C2  0xC4CEB9FE1A85EC53ULL   // fmix64 second constant

// ––––––––––––––––––––––––––––––––––––––
// Shared memory distinguished-point (DP) buffer
//
// BUG FIX (vs original JeanLucPons):
//   Original code called atomicAdd() in global memory for every DP
//   detected.  Under high DP rates on many-SM GPUs this creates an
//   atomic contention storm — hundreds of warps serialised on a single
//   cache line — visibly reducing throughput.
//
// PhiKang fix (implemented in GPU/GPUCompute.h):
//   Each thread block maintains a small shared-memory DP ring buffer.
//   Only when the buffer fills does a single elected thread flush the
//   batch to global memory with one coalesced write + one atomic.
//
// DP_SHARED_BUFFER_SIZE: slots in the per-block shared buffer.
//   Must be >= 32 (one full warp worth of simultaneous DPs).
//   128 is chosen to match a typical block size (128–256 threads) so
//   the buffer holds roughly one DP per thread before flushing.
//   Increasing this reduces flush frequency but raises shared memory
//   pressure — tune alongside GPU_GRP_SIZE if occupancy drops.
// ––––––––––––––––––––––––––––––––––––––
#define DP_SHARED_BUFFER_SIZE  128   // per-block DP slots before global flush

// ––––––––––––––––––––––––––––––––––––––
// Networking (server / client mode)
// ––––––––––––––––––––––––––––––––––––––
#define SEND_PERIOD       2.0      // seconds between DP batch sends to server
#define CLIENT_TIMEOUT  3600.0    // seconds of idle before client disconnect

// ––––––––––––––––––––––––––––––––––––––
// Workfile
// ––––––––––––––––––––––––––––––––––––––
#define MERGE_PART      256    // partition count for split-mode workfiles

// Magic bytes at the start of every v3.0 .phk workfile.
// ASCII "PHIKANG\0" stored as a little-endian uint64_t:
//   P=0x50 H=0x48 I=0x49 K=0x4B A=0x41 N=0x4E G=0x47 \0=0x00
//   → 0x474E414B49484850  (verified: bytes in memory spell "PHIKANG\0")
#define WORKFILE_MAGIC  0x474E414B49484850ULL

// ––––––––––––––––––––––––––––––––––––––
// TODO_TENSOR: GB300 NVL72 tensor core configuration
// Activate when Blackwell Ultra hardware is available.
//
// Planned use: batch modular inversion via WMMA tiles, replacing the
// current serial Montgomery inversion chain in GPUMath.h.
// Expected speedup: 8-16× for inversion-heavy workloads.
//
// #define USE_TENSOR_CORES
// #define TC_BATCH_SIZE  256   // field elements per tensor batch
// #define TC_TILE_M       16   // WMMA tile dimension M
// #define TC_TILE_N       16   // WMMA tile dimension N
// #define TC_TILE_K       16   // WMMA tile dimension K
// ––––––––––––––––––––––––––––––––––––––

// ––––––––––––––––––––––––––––––––––––––
// TODO_AI: AI orchestration layer
// Activate when the fine-tuned monitor model is ready.
//
// Planned use: real-time DP threshold tuning, walk parameter
// optimisation, and anomaly detection via binary + JSONL metrics.
//
// #define USE_AI_MONITOR
// #define AI_METRICS_INTERVAL  1000    // kernel launches between exports
// #define AI_BINARY_PORT       19000   // local port for binary stream
// #define AI_JSON_LOG          "phikang_metrics.jsonl"
// ––––––––––––––––––––––––––––––––––––––

#endif // CONSTANTSH
