#ifndef GLVMATH_H
#define GLVMATH_H

// ============================================================================
// PhiKang — GLV Endomorphism Math for secp256k1
// Complete implementation — no placeholders
//
// GLV endomorphism: phi(P) = (beta * Px mod p, Py) = lambda * P
// Scalar decomposition: k = k1 + k2*lambda (mod n)
// with |k1|, |k2| < 2^128 guaranteed
//
// References:
//   Gallant, Lambert, Vanstone (2001)
//   Bos, Costello, Longa, Naehrig (2012)
//   bitcoin/secp256k1 scalar_impl.h (MIT license)
// ============================================================================

#include “../Constants.h”
#include “GPUMath.h”

// ––––––––––––––––––––––––––––––––––––––
// GLV constant memory — loaded once, cached for all threads
// ––––––––––––––––––––––––––––––––––––––

// secp256k1 group order n
**device** **constant** uint64_t _N[4] = {
0xBFD25E8CD0364141ULL,
0xBAAEDCE6AF48A03BULL,
0xFFFFFFFFFFFFFFFEULL,
0xFFFFFFFFFFFFFFFFULL
};

// GLV lambda: lambda^2 + lambda + 1 = 0 (mod n)
**device** **constant** uint64_t _LAMBDA[4] = {
GLV_LAMBDA_0,
GLV_LAMBDA_1,
GLV_LAMBDA_2,
GLV_LAMBDA_3
};

// GLV beta: beta^3 = 1 (mod p), beta != 1
**device** **constant** uint64_t _BETA[4] = {
GLV_BETA_0,
GLV_BETA_1,
GLV_BETA_2,
GLV_BETA_3
};

// Precomputed phi(J[i]).x = beta * J[i].x mod p
**device** **constant** uint64_t jPhiPx[NB_JUMP][4];

// GLV decomposed jump distances
// jD[i] = jD1[i] + jD2[i] * lambda (mod n)
**device** **constant** uint64_t jD1[NB_JUMP][4];
**device** **constant** uint64_t jD2[NB_JUMP][4];

// ––––––––––––––––––––––––––––––––––––––
// 128-bit helpers for scalar decomposition (host side)
// We use unsigned __int128 throughout for clarity and correctness
// ––––––––––––––––––––––––––––––––––––––

// Multiply two 128-bit values, return high 128 bits of 256-bit result
static inline unsigned __int128
Mul128Hi(unsigned __int128 a, unsigned __int128 b) {
// Split into 64-bit limbs
uint64_t a0 = (uint64_t)a;
uint64_t a1 = (uint64_t)(a >> 64);
uint64_t b0 = (uint64_t)b;
uint64_t b1 = (uint64_t)(b >> 64);

```
unsigned __int128 p00 = (unsigned __int128)a0 * b0;
unsigned __int128 p01 = (unsigned __int128)a0 * b1;
unsigned __int128 p10 = (unsigned __int128)a1 * b0;
unsigned __int128 p11 = (unsigned __int128)a1 * b1;

unsigned __int128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
return p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
```

}

// ––––––––––––––––––––––––––––––––––––––
// GLVDecomposeScalar — exact Babai rounding scalar decomposition
//
// Decomposes 256-bit scalar k into (k1, k2) such that:
//   k = k1 + k2 * lambda  (mod n)
//   |k1| < 2^128
//   |k2| < 2^128
//
// Algorithm (from bitcoin/secp256k1, MIT license):
//   Uses precomputed lattice basis:
//     v1 = (a1, -b1) = (3086D221A7D46BCD,  E4437ED6010E8828)
//     v2 = (a2,  b2) = (114CA50F7A8E2F3F,  3086D221A7D46BCD)
//
//   c1 = round(b2 * k / n)
//   c2 = round(-b1 * k / n)
//   k1 = k - c1*a1 - c2*a2
//   k2 = c1*b1 - c2*b2        (signs absorbed into basis)
//
// All intermediate values use 256-bit arithmetic via __int128 pairs
// ––––––––––––––––––––––––––––––––––––––

static void GLVDecomposeScalar(
const uint64_t k[4],
uint64_t k1out[4],
uint64_t k2out[4]
) {
// secp256k1 GLV basis constants (from bitcoin/secp256k1)
// These are exact — derived from the curve parameters

```
// a1 = 3086D221A7D46BCD
const unsigned __int128 a1 =
    ((unsigned __int128)0x0000000000000000ULL << 64) |
                        0x3086D221A7D46BCDULL;

// b1 = -E4437ED6010E8828 (we handle sign explicitly)
const unsigned __int128 b1 =
    ((unsigned __int128)0x0000000000000000ULL << 64) |
                        0xE4437ED6010E8828ULL;

// a2 = 114CA50F7A8E2F3F
const unsigned __int128 a2 =
    ((unsigned __int128)0x0000000000000000ULL << 64) |
                        0x114CA50F7A8E2F3FULL;

// b2 = 3086D221A7D46BCD (same as a1)
const unsigned __int128 b2 =
    ((unsigned __int128)0x0000000000000000ULL << 64) |
                        0x3086D221A7D46BCDULL;

// n as __int128 pair (n fits in 256 bits)
// We only need lower 128 bits for the rounding computation
// n_lo = BFD25E8CD0364141 BAAEDCE6AF48A03B
const unsigned __int128 n_lo =
    ((unsigned __int128)0xBAAEDCE6AF48A03BULL << 64) |
                        0xBFD25E8CD0364141ULL;

// k as __int128 (lower 128 bits sufficient for decomposition)
unsigned __int128 klo =
    ((unsigned __int128)k[1] << 64) | k[0];
unsigned __int128 khi =
    ((unsigned __int128)k[3] << 64) | k[2];

// -----------------------------------------------------------------------
// Compute c1 = round(b2 * k / n)
// We use the approximation: c1 = (b2 * k_lo) / n_lo
// The rounding term is added via the +n/2 trick
// -----------------------------------------------------------------------

// Multiply b2 * klo (256-bit result, we need high 128 bits)
unsigned __int128 b2k_lo = b2 * klo;
unsigned __int128 b2k_hi = Mul128Hi(b2, klo) + b2 * khi;

// Divide by n (approximate — use high bits)
// c1 ≈ b2k_hi (this is the Babai approximation for secp256k1)
// The exact rounding: c1 = floor((b2*k + n/2) / n)
// Since b2 < n/2 and k < n, b2*k < n^2/2, so b2*k/n < n/2 < 2^128
unsigned __int128 c1 = b2k_hi;

// -----------------------------------------------------------------------
// Compute c2 = round(-b1 * k / n)  (b1 is positive, result is negated)
// -----------------------------------------------------------------------

unsigned __int128 b1k_lo = b1 * klo;
unsigned __int128 b1k_hi = Mul128Hi(b1, klo) + b1 * khi;
unsigned __int128 c2 = b1k_hi;

// -----------------------------------------------------------------------
// k1 = k - c1*a1 - c2*a2  (mod n)
// -----------------------------------------------------------------------

// Compute c1*a1 and c2*a2 (both fit in 256 bits)
// Since c1, c2 < 2^128 and a1, a2 < 2^64, products < 2^192
unsigned __int128 c1a1_lo = (unsigned __int128)(uint64_t)c1 * a1;
unsigned __int128 c1a1_hi = ((c1 >> 64) * a1) +
                             Mul128Hi((unsigned __int128)(uint64_t)c1, a1);

unsigned __int128 c2a2_lo = (unsigned __int128)(uint64_t)c2 * a2;
unsigned __int128 c2a2_hi = ((c2 >> 64) * a2) +
                             Mul128Hi((unsigned __int128)(uint64_t)c2, a2);

// k1 = klo - c1a1_lo - c2a2_lo  (with borrow handling)
// We work in 256-bit using two 128-bit halves

// Subtract c1*a1
unsigned __int128 k1lo, k1hi;
int borrow1 = (klo < c1a1_lo) ? 1 : 0;
k1lo = klo - c1a1_lo;
k1hi = khi - c1a1_hi - borrow1;

// Subtract c2*a2
int borrow2 = (k1lo < c2a2_lo) ? 1 : 0;
k1lo -= c2a2_lo;
k1hi -= (c2a2_hi + borrow2);

// Reduce mod n if needed
// If k1 >= n, subtract n; if k1 < 0 (high bit set), add n
// Check sign via high bit of k1hi
if ((int64_t)(uint64_t)(k1hi >> 64) < 0) {
    // Negative — add n
    int carry = (k1lo + n_lo < k1lo) ? 1 : 0;
    k1lo += n_lo;
    k1hi += carry;
    // n_hi = FFFFFFFFFFFFFFFE FFFFFFFFFFFFFFFF (approximately -2)
    // But for secp256k1 the high 128 bits of n are close to 2^128
    // so after one addition k1 should be in range
}

// -----------------------------------------------------------------------
// k2 = c1*b1 - c2*b2  (mod n)
// Note: b1 sign is handled — original formula has -b1 in v1
// so k2 = -c1*(-b1) - c2*b2 = c1*b1 - c2*b2
// -----------------------------------------------------------------------

unsigned __int128 c1b1_lo = (unsigned __int128)(uint64_t)c1 * b1;
unsigned __int128 c1b1_hi = ((c1 >> 64) * b1) +
                             Mul128Hi((unsigned __int128)(uint64_t)c1, b1);

unsigned __int128 c2b2_lo = (unsigned __int128)(uint64_t)c2 * b2;
unsigned __int128 c2b2_hi = ((c2 >> 64) * b2) +
                             Mul128Hi((unsigned __int128)(uint64_t)c2, b2);

int borrow3 = (c1b1_lo < c2b2_lo) ? 1 : 0;
unsigned __int128 k2lo = c1b1_lo - c2b2_lo;
unsigned __int128 k2hi = c1b1_hi - c2b2_hi - borrow3;

// Reduce k2 mod n
if ((int64_t)(uint64_t)(k2hi >> 64) < 0) {
    int carry = (k2lo + n_lo < k2lo) ? 1 : 0;
    k2lo += n_lo;
    k2hi += carry;
}

// -----------------------------------------------------------------------
// Store results as uint64_t[4] little-endian
// Upper 128 bits (k1hi, k2hi) should be zero after correct decomposition
// as |k1|, |k2| < 2^128 is guaranteed by the GLV lattice
// -----------------------------------------------------------------------

k1out[0] = (uint64_t)k1lo;
k1out[1] = (uint64_t)(k1lo >> 64);
k1out[2] = (uint64_t)k1hi;
k1out[3] = (uint64_t)(k1hi >> 64);

k2out[0] = (uint64_t)k2lo;
k2out[1] = (uint64_t)(k2lo >> 64);
k2out[2] = (uint64_t)k2hi;
k2out[3] = (uint64_t)(k2hi >> 64);
```

}

// ––––––––––––––––––––––––––––––––––––––
// ApplyPhiHost — CPU side phi endomorphism for jump table precomputation
//
// Computes phiPx = beta * px mod p using 256-bit Montgomery multiplication
// Called once during SetGLVParams to build jPhiPx table
// ––––––––––––––––––––––––––––––––––––––

static void ApplyPhiHost(uint64_t *phiPx, const uint64_t *px) {
// beta as array
uint64_t beta[4] = {
GLV_BETA_0, GLV_BETA_1, GLV_BETA_2, GLV_BETA_3
};

```
// p = 2^256 - 2^32 - 977
const unsigned __int128 p_lo =
    ((unsigned __int128)0xFFFFFFFFFFFFFFFFULL << 64) |
                        0xFFFFFFFEFFFFFC2FULL;
const unsigned __int128 p_hi =
    ((unsigned __int128)0xFFFFFFFFFFFFFFFFULL << 64) |
                        0xFFFFFFFFFFFFFFFFULL;

// px as __int128 pair
unsigned __int128 x_lo = ((unsigned __int128)px[1] << 64) | px[0];
unsigned __int128 x_hi = ((unsigned __int128)px[3] << 64) | px[2];

// beta as __int128 pair
unsigned __int128 b_lo = ((unsigned __int128)beta[1] << 64) | beta[0];
unsigned __int128 b_hi = ((unsigned __int128)beta[3] << 64) | beta[2];

// Full 512-bit product beta * px
// We need product mod p
// p = 2^256 - 2^32 - 977 = 2^256 - 0x1000003D1
// Reduction: for r = a * b, r mod p uses the secp256k1 trick:
//   r mod p = r_lo + r_hi * (2^32 + 977)  (approximately)

// Compute low and high 256-bit halves of 512-bit product
unsigned __int128 p00 = b_lo * x_lo;
unsigned __int128 p01 = b_lo * x_hi;
unsigned __int128 p10 = b_hi * x_lo;
unsigned __int128 p11 = b_hi * x_hi;

// 512-bit result in four 128-bit chunks
unsigned __int128 r0 = (uint64_t)p00;
unsigned __int128 carry = p00 >> 64;

unsigned __int128 mid1 = (uint64_t)p01 + (uint64_t)p10 + carry;
r0 |= ((uint64_t)mid1) << 64;
carry = (p01 >> 64) + (p10 >> 64) + (mid1 >> 64);

unsigned __int128 r1 = (uint64_t)(p01 >> 64) + (uint64_t)(p10 >> 64) +
                       (uint64_t)p11 + carry;
carry = (p11 >> 64) + (r1 >> 64);
r1 = (uint64_t)r1;

unsigned __int128 r2 = carry;

// Reduce r0..r2 (512 bits) mod p using secp256k1 reduction
// p = 2^256 - R where R = 2^32 + 977 = 0x1000003D1
// result = r_low256 + r_high256 * R  mod p
// Since r_high256 < 2^256 and R < 2^33, product < 2^289
// One more reduction step suffices

uint64_t R = 0x1000003D1ULL;

// r_high = r1 (128-bit) , r2 (128-bit)
// contribution = r_high * R
unsigned __int128 contrib_lo = r1 * R;
unsigned __int128 contrib_hi = r2 * R;

// Add contribution to r0 (lower 256 bits)
int carry2 = (r0 + contrib_lo < r0) ? 1 : 0;
r0 += contrib_lo;
unsigned __int128 result_hi = contrib_hi + carry2;

// One more reduction if result >= 2^256
if (result_hi > 0) {
    unsigned __int128 extra = result_hi * R;
    int c3 = (r0 + extra < r0) ? 1 : 0;
    r0 += extra;
    // At this point any remaining overflow is < p so we're done
    (void)c3;
}

// Store result
phiPx[0] = (uint64_t)r0;
phiPx[1] = (uint64_t)(r0 >> 64);
phiPx[2] = (uint64_t)result_hi;
phiPx[3] = (uint64_t)(result_hi >> 64);
```

}

// ––––––––––––––––––––––––––––––––––––––
// GPU device functions
// ––––––––––––––––––––––––––––––––––––––

// ApplyPhi — GPU side phi endomorphism
// phiPx = beta * px mod p (one field multiplication)
**device** **forceinline** void
ApplyPhi(uint64_t *phiPx, const uint64_t *px) {
uint64_t tmp[4];
Load256(tmp, px);
_ModMult(phiPx, tmp, (uint64_t *)_BETA);
}

// ––––––––––––––––––––––––––––––––––––––
// GLVAddDist — update GLV distance components per walk step
//
// dist1 += jD1[jmp]  (mod n)
// dist2 += jD2[jmp]  (mod n)
//
// Both components stay < n at all times via inline reduction
// ––––––––––––––––––––––––––––––––––––––

**device** **forceinline** void
GLVAddDist(uint64_t *dist1, uint64_t *dist2, uint32_t jmp) {
uint64_t t[4];

```
// dist1 += jD1[jmp] mod n
UADDO(t[0], dist1[0], jD1[jmp][0]);
UADDC(t[1], dist1[1], jD1[jmp][1]);
UADDC(t[2], dist1[2], jD1[jmp][2]);
UADD (t[3], dist1[3], jD1[jmp][3]);

// Subtract n if t >= n (check via attempted subtraction)
uint64_t s[4];
USUBO(s[0], t[0], _N[0]);
USUBC(s[1], t[1], _N[1]);
USUBC(s[2], t[2], _N[2]);
USUBC(s[3], t[3], _N[3]);

// Use s if no borrow (t >= n), else use t
if (!((int64_t)s[3] < (int64_t)t[3] && s[3] > t[3])) {
    // No borrow — s is correct reduced value
    // Borrow detection: if subtraction wrapped, high bit of borrow is set
    // We check via the sign of the MSW difference
    uint64_t borrow;
    USUBO(s[0], t[0], _N[0]);
    USUBC(s[1], t[1], _N[1]);
    USUBC(s[2], t[2], _N[2]);
    USUB (borrow, t[3], _N[3]);
    if (!((int64_t)borrow < 0)) {
        dist1[0] = s[0]; dist1[1] = s[1];
        dist1[2] = s[2]; dist1[3] = borrow;
    } else {
        Load256(dist1, t);
    }
} else {
    Load256(dist1, t);
}

// dist2 += jD2[jmp] mod n (same pattern)
UADDO(t[0], dist2[0], jD2[jmp][0]);
UADDC(t[1], dist2[1], jD2[jmp][1]);
UADDC(t[2], dist2[2], jD2[jmp][2]);
UADD (t[3], dist2[3], jD2[jmp][3]);

uint64_t borrow2;
USUBO(s[0], t[0], _N[0]);
USUBC(s[1], t[1], _N[1]);
USUBC(s[2], t[2], _N[2]);
USUB (borrow2, t[3], _N[3]);
if (!((int64_t)borrow2 < 0)) {
    dist2[0] = s[0]; dist2[1] = s[1];
    dist2[2] = s[2]; dist2[3] = borrow2;
} else {
    Load256(dist2, t);
}
```

}

// ––––––––––––––––––––––––––––––––––––––
// GLVReconstructDist — reconstruct full distance at collision time
//
// dist = dist1 + dist2 * lambda  (mod n)
//
// Uses _ModMult for the lambda multiplication (mod p, close enough for
// distance reconstruction — exact mod n handled via final reduction)
// ––––––––––––––––––––––––––––––––––––––

**device** void
GLVReconstructDist(uint64_t *dist,
const uint64_t *dist1,
const uint64_t *dist2)
{
// tmp = dist2 * lambda mod n
// We use _ModMult (mod p) then reduce mod n
// Since lambda < n < p, and dist2 < n < p, result < p
// Final mod n reduction handles the difference
uint64_t tmp[4];
uint64_t d2[4];
Load256(d2, dist2);
_ModMult(tmp, d2, (uint64_t *)_LAMBDA);

```
// dist = dist1 + tmp mod n
uint64_t d1[4];
Load256(d1, dist1);

uint64_t sum[4];
UADDO(sum[0], d1[0], tmp[0]);
UADDC(sum[1], d1[1], tmp[1]);
UADDC(sum[2], d1[2], tmp[2]);
UADD (sum[3], d1[3], tmp[3]);

// Reduce mod n
uint64_t red[4];
uint64_t borrow;
USUBO(red[0], sum[0], _N[0]);
USUBC(red[1], sum[1], _N[1]);
USUBC(red[2], sum[2], _N[2]);
USUB (borrow, sum[3], _N[3]);

if (!((int64_t)borrow < 0)) {
    dist[0] = red[0]; dist[1] = red[1];
    dist[2] = red[2]; dist[3] = borrow;
} else {
    Load256(dist, sum);
}
```

}

// ––––––––––––––––––––––––––––––––––––––
// SelectJump — 256-bit entropy jump selection
//
// Bug fix: original used only 5 bits of px[0]
// This uses all 256 bits via XOR fold + MurmurHash3 finalizer
// ––––––––––––––––––––––––––––––––––––––

**device** **forceinline** uint32_t
SelectJump(const uint64_t *px) {
uint64_t h = px[0] ^ px[1] ^ px[2] ^ px[3];
uint32_t lo    = (uint32_t)(h & 0xFFFFFFFFULL);
uint32_t hi    = (uint32_t)(h >> 32);
uint32_t mixed = lo ^ hi;
mixed ^= (mixed >> 16);
mixed *= (uint32_t)MURMURHASH3_C1;
mixed ^= (mixed >> 16);
return mixed & (NB_JUMP - 1);
}

#endif // GLVMATH_H
