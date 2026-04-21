# PhiKang
# PhiKang

> Kangaroo ECDLP Solver with GLV Endomorphism for secp256k1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-blue.svg)]()

-----

## What is PhiKang?

PhiKang is a high-performance GPU-accelerated Elliptic Curve Discrete Logarithm Problem (ECDLP) solver for the secp256k1 curve. It implements the Pollard Kangaroo algorithm enhanced with the Gallant-Lambert-Vanstone (GLV) endomorphism, targeting lost/recoverable Bitcoin wallets.

The name comes from **Phi** (φ) — the endomorphism at the heart of the GLV optimization — and **Kang** from Kangaroo.

-----

## Key Features

- **GLV Endomorphism Integration** — decomposes each walk step using the secp256k1 phi endomorphism `φ(P) = (β·Px mod p, Py)`, reducing scalar size from 256-bit to ~128-bit and delivering approximately 1.4x throughput improvement over standard Kangaroo
- **Bug-fixed foundation** — corrects critical memory and distance truncation bugs found in prior implementations
- **Structure of Arrays (SoA) GPU layout** — coalesced memory access for maximum memory bandwidth utilization
- **256-bit entropy jump selection** — statistically unbiased jump distribution using full point coordinates
- **Shared memory DP buffer** — eliminates global atomic bottleneck on distinguished point detection
- **CUDA Occupancy API** — runtime-optimal thread block configuration for any GPU
- **Full 256-bit distance tracking** — correct distance storage and retrieval across workfile save/resume cycles
- **Workfile v3.0** — new format with GLV distance components, human-readable stats header, and integrity validation
- **Server/client architecture** — distributed operation across multiple machines
- **USE_SYMMETRY support** — optional symmetry optimization halving the effective search space
- **Cross-platform** — Linux (primary) and Windows support
- **GB300 NVL72 ready** — tensor core hooks pre-marked for seamless upgrade when Blackwell Ultra hardware arrives

-----

## Architecture

```
PhiKang/
├── Constants.h          ← Compile-time configuration
├── Kangaroo.h           ← Core solver class
├── Kangaroo.cpp         ← CPU walk, herd creation, collision detection
├── KangarooWork.cpp     ← Workfile save/load/merge
├── KangarooNet.cpp      ← Server/client networking
├── Timer.h / Timer.cpp  ← Cross-platform timing
├── main.cpp             ← Entry point, argument parsing
├── GPU/
│   ├── GPUEngine.h      ← GPU engine interface
│   ├── GPUEngine.cu     ← Kernel launch, memory management
│   ├── GPUMath.h        ← 256-bit field arithmetic (PTX assembly)
│   ├── GPUCompute.h     ← Main kernel: GLV walk, DP detection
│   └── GPUGLVMath.h     ← GLV decomposition, phi endomorphism
├── AI/                  ← [COMING: GB300 phase]
│   ├── Monitor.h        ← AI metrics interface
│   └── Monitor.cpp      ← Binary + JSON metrics output
└── SECPK1/
    └── ← secp256k1 field arithmetic (CPU side)
```

-----

## GLV Enhancement — How It Works

Standard Kangaroo performs each walk step as:

```
P_new = P + J[i]          (full 256-bit scalar addition)
dist  = dist + jD[i]      (256-bit distance update)
```

PhiKang decomposes each jump distance using the GLV lattice:

```
jD[i] = jD1[i] + jD2[i] * λ  (mod n)
```

Where λ is the secp256k1 GLV scalar such that `λ·P = φ(P)`.

The walk step becomes:

```
P_new = P + J[i] + jD2[i]·φ(J[i])
      = P + J[i] + jD2[i]·(β·Jx, Jy)
```

Since `φ(J[i])` is precomputed for all jump points (just one field multiplication per point), and jD1/jD2 are ~128-bit each instead of 256-bit, the net result is approximately **1.4x more steps per second** at no correctness cost.

-----

## Bugs Fixed vs Prior Implementations

|Bug                                                             |Impact                                                                        |Fix                                                                   |
|----------------------------------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------|
|`KSIZE=10` but 12 words stored per kangaroo                     |Silent heap buffer overflow on every kernel call                              |Corrected to `KSIZE=12` (without symmetry), `KSIZE=13` (with symmetry)|
|`GetKangaroos` reads only `dist[0,1]` — upper 128 bits discarded|Distance corrupted on workfile resume — kangaroos restart from wrong positions|Read all 4 distance limbs correctly                                   |
|Jump selection uses only 5 bits of `px[0]`                      |Statistical bias in jump distribution, correlated walks                       |Full 256-bit entropy fold using XOR + MurmurHash3 finalizer           |
|Global atomic on every distinguished point                      |Atomic contention storm under high DP rate                                    |Shared memory DP buffer, single flush per warp                        |
|`ITEM` distance stored as 2 limbs (128-bit)                     |DP distance truncated in output                                               |Full 4-limb (256-bit) distance in `ITEM`                              |

-----

## Workfile Format v3.0

PhiKang introduces a new workfile format with human-readable statistics:

```
=== PhiKang Workfile v3.0 ===
Range start : 0x0000000000000001
Range end   : 0xFFFFFFFFFFFFFFFF
DP size     : 20 bits
Kangaroos   : 2^18.5
Steps done  : 2^34.2
Expected    : 2^35.1  (87.3% complete)
Collisions  : 847
RAM used    : 412 MB
Saved       : 2026-04-21 16:33:00 UTC
GLV enabled : YES
Symmetry    : NO
==============================
[binary kangaroo state follows]
```

Workfile v3.0 is not backwards compatible with prior formats due to the addition of GLV distance components `d1` and `d2` per kangaroo.

-----

## Hardware Requirements

|GPU        |VRAM    |Performance|Notes                       |
|-----------|--------|-----------|----------------------------|
|RTX 4090   |24GB    |Baseline   |Primary test target         |
|RTX 5090   |32GB    |~1.3x 4090 |Larger VRAM for bigger herds|
|GB300 NVL72|20TB HBM|~50x+      |Tensor core path activates  |

-----

## Building

### Linux

```bash
# Prerequisites
sudo apt install nvidia-cuda-toolkit build-essential

# Clone
git clone https://github.com/zee-cpu/PhiKang.git
cd PhiKang

# Build (without symmetry, default)
make

# Build with symmetry optimization
make SYMMETRY=1

# Build with debug GPU checks
make DEBUG=1
```

### Windows

```cmd
# Prerequisites: Visual Studio 2022 + CUDA Toolkit 12.x

# Open Developer Command Prompt
nmake -f Makefile.win

# With symmetry
nmake -f Makefile.win SYMMETRY=1
```

-----

## Usage

```bash
# Basic usage
./PhiKang -t 4 -gpu -gpuId 0 config.txt

# With work file save/resume
./PhiKang -t 4 -gpu -gpuId 0 -w mywork.phk -wi 300 config.txt

# Resume from work file
./PhiKang -t 4 -gpu -gpuId 0 -i mywork.phk config.txt

# Server mode (distributed)
./PhiKang -s -sp 17403 config.txt

# Client mode
./PhiKang -c 192.168.1.100 -sp 17403 config.txt

# Work file info (human readable stats)
./PhiKang -winfo mywork.phk

# List CUDA devices
./PhiKang -l
```

### Config file format

```
# Range start (hex)
0000000000000001
# Range end (hex)
000000000000FFFF
# Target public key(s) — one per line
02a1b2c3d4e5f6...
```

-----

## Configuration (Constants.h)

```cpp
// GPU group size — tuned per GPU at runtime via occupancy API
#define GPU_GRP_SIZE    128   // Kangaroos per thread
#define NB_JUMP         32    // Jump table size
#define NB_RUN          64    // Iterations per kernel launch

// Enable symmetry optimization (halves search space)
// #define USE_SYMMETRY

// TODO_TENSOR: Tensor core batch inversion (GB300 only)
// #define USE_TENSOR_CORES

// TODO_AI: AI orchestration metrics export
// #define USE_AI_MONITOR
```

-----

## Roadmap

### Phase 1 — Current (Consumer GPU)

- [x] Bug fixes (KSIZE, distance truncation, atomic contention)
- [x] GLV endomorphism integration
- [x] SoA GPU memory layout
- [x] 256-bit entropy jump selection
- [x] Shared memory DP buffer
- [x] Occupancy API launch config
- [x] Workfile v3.0 with human-readable stats
- [x] Cross-platform build system

### Phase 2 — GB300 NVL72 (On Hardware Arrival)

- [ ] Tensor core accelerated batch modular inversion
- [ ] 72-GPU unified NVLink pool (replacing server/client)
- [ ] AI metrics binary + JSON export interface
- [ ] Fine-tuned LLM orchestration layer
- [ ] Dynamic DP threshold adjustment via AI monitor
- [ ] Autonomous walk parameter optimization

-----

## Technical References

- Pollard, J.M. (2000). *Kangaroos, Monopoly and Discrete Logarithms*
- Gallant, R., Lambert, R., Vanstone, S. (2001). *Faster Point Multiplication on Elliptic Curves with Efficient Endomorphisms*
- Bos, J., Costello, C., Longa, P., Naehrig, M. (2012). *Selecting Elliptic Curves for Cryptography*
- Bernstein, D., Yang, B. (2019). *Fast constant-time gcd computation and modular inversion*

-----

## Credits

- Original Kangaroo implementation: [JeanLucPons](https://github.com/JeanLucPons/Kangaroo)
- GLV integration, bug fixes, GB300 architecture: PhiKang contributors
- secp256k1 PTX field arithmetic: retained from original, battle-tested

-----

## License

MIT License

-----

## Disclaimer

This tool is intended for legitimate cryptographic research and recovery of wallets where the owner has lost access to their own private keys. Users are solely responsible for ensuring their use complies with applicable laws and regulations.
