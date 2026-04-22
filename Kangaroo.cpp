#include <string.h>
#include <fstream>
#include “Kangaroo.h”
#include “SECPK1/IntGroup.h”
#include “Timer.h”
#include “GPU/GLVMath.h”
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <memory>
#ifndef WIN64
#include <pthread.h>
#endif

using namespace std;

#define safe_delete_array(x) if (x) { delete[] x; x = NULL; }

// ============================================================================
// Constructor
// ============================================================================

Kangaroo::Kangaroo(
Secp256K1  *secp,
int32_t     initDPSize,
bool        useGpu,
string     &workFile,
string     &iWorkFile,
uint32_t    savePeriod,
bool        saveKangaroo,
bool        saveKangarooByServer,
double      maxStep,
int         wtimeout,
int         port,
int         ntimeout,
string      serverIp,
string      outputFile,
bool        splitWorkfile
) {
this->secp                     = secp;
this->initDPSize               = initDPSize;
this->useGpu                   = useGpu;
this->offsetCount              = 0;
this->offsetTime               = 0.0;
this->workFile                 = workFile;
this->saveWorkPeriod           = savePeriod;
this->inputFile                = iWorkFile;
this->nbLoadedWalk             = 0;
this->clientMode               = serverIp.length() > 0;
this->saveKangarooByServer     = this->clientMode && saveKangarooByServer;
this->saveKangaroo             = saveKangaroo || this->saveKangarooByServer;
this->fRead                    = NULL;
this->maxStep                  = maxStep;
this->wtimeout                 = wtimeout;
this->port                     = port;
this->ntimeout                 = ntimeout;
this->serverIp                 = serverIp;
this->outputFile               = outputFile;
this->hostInfo                 = NULL;
this->endOfSearch              = false;
this->saveRequest              = false;
this->connectedClient          = 0;
this->totalRW                  = 0;
this->collisionInSameHerd      = 0;
this->keyIdx                   = 0;
this->splitWorkfile            = splitWorkfile;
this->pid                      = Timer::getPID();
this->CPU_GRP_SIZE             = 1024;

```
memset(counters, 0, sizeof(counters));
```

#ifdef WIN64
ghMutex   = CreateMutex(NULL, FALSE, NULL);
saveMutex = CreateMutex(NULL, FALSE, NULL);
#else
pthread_mutex_init(&ghMutex,   NULL);
pthread_mutex_init(&saveMutex, NULL);
signal(SIGPIPE, SIG_IGN);
#endif
}

// ============================================================================
// Destructor
// ============================================================================

Kangaroo::~Kangaroo() {
#ifdef WIN64
CloseHandle(ghMutex);
CloseHandle(saveMutex);
#else
pthread_mutex_destroy(&ghMutex);
pthread_mutex_destroy(&saveMutex);
#endif
}

// ============================================================================
// ParseConfigFile
// ============================================================================

bool Kangaroo::ParseConfigFile(string &fileName) {

```
if (clientMode) return true;

FILE *fp = fopen(fileName.c_str(), "rb");
if (fp == NULL) {
    ::printf("[+] Error: Cannot open %s %s\n",
             fileName.c_str(), strerror(errno));
    return false;
}
fclose(fp);

vector<string> lines;
string line;
ifstream inFile(fileName);
while (getline(inFile, line)) {
    int l = (int)line.length() - 1;
    while (l >= 0 && isspace(line.at(l))) { line.pop_back(); l--; }
    if (line.length() > 0) lines.push_back(line);
}

if (lines.size() < 3) {
    ::printf("[+] Error: %s — not enough arguments (need start, end, key)\n",
             fileName.c_str());
    return false;
}

rangeStart.SetBase16((char *)lines[0].c_str());
rangeEnd.SetBase16((char *)lines[1].c_str());

for (int i = 2; i < (int)lines.size(); i++) {
    Point p;
    bool  isCompressed;
    if (!secp->ParsePublicKeyHex(lines[i], p, isCompressed)) {
        ::printf("[+] %s, error line %d: %s\n",
                 fileName.c_str(), i, lines[i].c_str());
        return false;
    }
    keysToSearch.push_back(p);
}

::printf("[+] Range start : %s\n", rangeStart.GetBase16().c_str());
::printf("[+] Range end   : %s\n", rangeEnd.GetBase16().c_str());
::printf("[+] Keys        : %d\n", (int)keysToSearch.size());

return true;
```

}

// ============================================================================
// Distinguished point helpers
// ============================================================================

bool Kangaroo::IsDP(Int *x) {
return ((x->bits64[3] & dMask.i64[3]) == 0) &&
((x->bits64[2] & dMask.i64[2]) == 0) &&
((x->bits64[1] & dMask.i64[1]) == 0) &&
((x->bits64[0] & dMask.i64[0]) == 0);
}

void Kangaroo::SetDP(int size) {
dpSize = size;
dMask.i64[0] = 0;
dMask.i64[1] = 0;
dMask.i64[2] = 0;
dMask.i64[3] = 0;
if (dpSize > 0) {
if (dpSize > 256) dpSize = 256;
for (int i = 0; i < size; i += 64) {
int end = (i + 64 > size) ? (size - 1) % 64 : 63;
uint64_t mask = ((1ULL << end) - 1) << 1 | 1ULL;
dMask.i64[(int)(i / 64)] = mask;
}
}
::printf(”[+] DP size: %d [0x%016” PRIx64 “%016” PRIx64
“%016” PRIx64 “%016” PRIx64 “]\n”,
dpSize,
dMask.i64[3], dMask.i64[2],
dMask.i64[1], dMask.i64[0]);
}

// ============================================================================
// Output — write found key to file or stdout
// ============================================================================

bool Kangaroo::Output(Int *pk, char sInfo, int sType) {

```
FILE *f        = stdout;
bool  needClose = false;

if (outputFile.length() > 0) {
    f = fopen(outputFile.c_str(), "a");
    if (f == NULL) {
        printf("[+] Cannot open %s for writing\n", outputFile.c_str());
        f = stdout;
    } else {
        needClose = true;
    }
}

if (!needClose) ::printf("\n");

Point PR = secp->ComputePublicKey(pk);

::fprintf(f, "Key#%2d [%d%c] Pub: 0x%s\n",
          keyIdx, sType, sInfo,
          secp->GetPublicKeyHex(true, keysToSearch[keyIdx]).c_str());

if (PR.equals(keysToSearch[keyIdx])) {
    ::fprintf(f, "Priv: 0x%s\n", pk->GetBase16().c_str());
} else {
    ::fprintf(f, "Failed!\n");
    if (needClose) fclose(f);
    return false;
}

if (needClose) fclose(f);
return true;
```

}

// ============================================================================
// CheckKey — test candidate private key against target
// ============================================================================

bool Kangaroo::CheckKey(Int d1, Int d2, uint8_t type) {

```
if (type & 0x1) d1.ModNegK1order();
if (type & 0x2) d2.ModNegK1order();

Int pk(&d1);
pk.ModAddK1order(&d2);

Point P = secp->ComputePublicKey(&pk);

if (P.equals(keyToSearch)) {
```

#ifdef USE_SYMMETRY
pk.ModAddK1order(&rangeWidthDiv2);
#endif
pk.ModAddK1order(&rangeStart);
return Output(&pk, ‘N’, type);
}

```
if (P.equals(keyToSearchNeg)) {
    pk.ModNegK1order();
```

#ifdef USE_SYMMETRY
pk.ModAddK1order(&rangeWidthDiv2);
#endif
pk.ModAddK1order(&rangeStart);
return Output(&pk, ‘S’, type);
}

```
return false;
```

}

// ============================================================================
// CollisionCheck — resolve tame/wild collision
// ============================================================================

bool Kangaroo::CollisionCheck(Int *d1, uint32_t type1,
Int *d2, uint32_t type2) {
if (type1 == type2)
return false;

```
Int Td, Wd;
if (type1 == TAME) { Td.Set(d1); Wd.Set(d2); }
else               { Td.Set(d2); Wd.Set(d1); }

endOfSearch = CheckKey(Td, Wd, 0) ||
              CheckKey(Td, Wd, 1) ||
              CheckKey(Td, Wd, 2) ||
              CheckKey(Td, Wd, 3);

return endOfSearch;
```

}

// ============================================================================
// AddToTable — add distinguished point to hash table
// ============================================================================

bool Kangaroo::AddToTable(Int *pos, Int *dist, uint32_t kType) {
int status = hashTable.Add(pos, dist, kType);
if (status == ADD_COLLISION)
return CollisionCheck(&hashTable.kDist, hashTable.kType, dist, kType);
return status == ADD_OK;
}

bool Kangaroo::AddToTable(int256_t *x, int256_t *d, uint32_t kType) {
int status = hashTable.Add(x, d, kType);
if (status == ADD_COLLISION) {
Int dist;
HashTable::toInt(d, &dist);
return CollisionCheck(&hashTable.kDist, hashTable.kType, &dist, kType);
}
return status == ADD_OK;
}

// ============================================================================
// CreateHerd — initialise a batch of kangaroos with random starting positions
// ============================================================================

void Kangaroo::CreateHerd(int nbKangaroo, Int *px, Int *py, Int *d,
int firstType, bool lock) {

```
vector<Int>   pk;
vector<Point> S, Sp;
pk.reserve(nbKangaroo);
S.reserve(nbKangaroo);
Sp.reserve(nbKangaroo);
Point Z; Z.Clear();

if (lock) LOCK(ghMutex);

for (int j = 0; j < nbKangaroo; j++) {
```

#ifdef USE_SYMMETRY
d[j].Rand(rangePower - 1);
if ((j + firstType) % 2 == WILD)
d[j].ModSubK1order(&rangeWidthDiv4);
#else
d[j].Rand(rangePower);
if ((j + firstType) % 2 == WILD)
d[j].ModSubK1order(&rangeWidthDiv2);
#endif
pk.push_back(d[j]);
}

```
if (lock) UNLOCK(ghMutex);

S = secp->ComputePublicKeys(pk);

for (int j = 0; j < nbKangaroo; j++) {
    if ((j + firstType) % 2 == TAME)
        Sp.push_back(Z);
    else
        Sp.push_back(keyToSearch);
}

S = secp->AddDirect(Sp, S);

for (int j = 0; j < nbKangaroo; j++) {
    px[j].Set(&S[j].x);
    py[j].Set(&S[j].y);
```

#ifdef USE_SYMMETRY
if (py[j].ModPositiveK1()) d[j].ModNegK1order();
#endif
}
}

// ============================================================================
// CreateJumpTable — generate random jump distances and points
// ============================================================================

void Kangaroo::CreateJumpTable() {

#ifdef USE_SYMMETRY
int jumpBit = rangePower / 2;
#else
int jumpBit = rangePower / 2 + 1;
#endif

```
if (jumpBit > 256) jumpBit = 256;

// Constant seed for workfile compatibility
rseed(0x600DCAFE);

int    maxRetry = 100;
bool   ok       = false;
double distAvg;
double maxAvg = pow(2.0, (double)jumpBit - 0.95);
double minAvg = pow(2.0, (double)jumpBit - 1.05);

while (!ok && maxRetry > 0) {
    Int totalDist;
    totalDist.SetInt32(0);
    for (int i = 0; i < NB_JUMP; i++) {
        jumpDistance[i].Rand(jumpBit);
        if (jumpDistance[i].IsZero()) jumpDistance[i].SetInt32(1);
        totalDist.Add(&jumpDistance[i]);
    }
    distAvg = totalDist.ToDouble() / (double)NB_JUMP;
    ok = (distAvg > minAvg && distAvg < maxAvg);
    maxRetry--;
}

for (int i = 0; i < NB_JUMP; i++) {
    Point J = secp->ComputePublicKey(&jumpDistance[i]);
    jumpPointx[i].Set(&J.x);
    jumpPointy[i].Set(&J.y);
}

::printf("[+] Jump avg distance: 2^%.2f\n", log2(distAvg));

// Restore random seed for runtime use
unsigned long seed = Timer::getSeed32();
rseed(seed);
```

}

// ============================================================================
// CreateGLVJumpTable — precompute GLV extended jump table
//
// For each jump point J[i] with distance jD[i]:
//   1. jPhiPxArr[i] = beta * J[i].x mod p   (phi endomorphism X coord)
//      J[i].y unchanged — endomorphism preserves Y
//   2. Decompose jD[i] into (jD1Arr[i], jD2Arr[i])
//      such that jD[i] = jD1Arr[i] + jD2Arr[i]*lambda (mod n)
//
// Results uploaded to GPU constant memory via GPUEngine::SetGLVParams()
// ============================================================================

void Kangaroo::CreateGLVJumpTable() {

```
::printf("[+] Precomputing GLV jump table...\n");

for (int i = 0; i < NB_JUMP; i++) {

    // Step 1: phi(J[i]).x = beta * J[i].x mod p
    // Using host-side ApplyPhiHost from GLVMath.h
    uint64_t phiX[4];
    ApplyPhiHost(phiX, jumpPointx[i].bits64);
    jPhiPxArr[i].bits64[0] = phiX[0];
    jPhiPxArr[i].bits64[1] = phiX[1];
    jPhiPxArr[i].bits64[2] = phiX[2];
    jPhiPxArr[i].bits64[3] = phiX[3];
    jPhiPxArr[i].bits64[4] = 0;

    // Step 2: GLV decompose jumpDistance[i] into (k1, k2)
    // jD[i] = jD1Arr[i] + jD2Arr[i] * lambda  (mod n)
    uint64_t k1[4], k2[4];
    GLVDecomposeScalar(jumpDistance[i].bits64, k1, k2);

    jD1Arr[i].bits64[0] = k1[0];
    jD1Arr[i].bits64[1] = k1[1];
    jD1Arr[i].bits64[2] = k1[2];
    jD1Arr[i].bits64[3] = k1[3];
    jD1Arr[i].bits64[4] = 0;

    jD2Arr[i].bits64[0] = k2[0];
    jD2Arr[i].bits64[1] = k2[1];
    jD2Arr[i].bits64[2] = k2[2];
    jD2Arr[i].bits64[3] = k2[3];
    jD2Arr[i].bits64[4] = 0;
}

::printf("[+] GLV jump table ready\n");
```

}

// ============================================================================
// InitRange — compute range derived values
// ============================================================================

void Kangaroo::InitRange() {
rangeWidth.Set(&rangeEnd);
rangeWidth.Sub(&rangeStart);
rangePower = rangeWidth.GetBitLength();
::printf(”[+] Range width: 2^%d\n”, rangePower);
rangeWidthDiv2.Set(&rangeWidth);
rangeWidthDiv2.ShiftR(1);
rangeWidthDiv4.Set(&rangeWidthDiv2);
rangeWidthDiv4.ShiftR(1);
rangeWidthDiv8.Set(&rangeWidthDiv4);
rangeWidthDiv8.ShiftR(1);
}

// ============================================================================
// InitSearchKey — adjust target key for range start offset
// ============================================================================

void Kangaroo::InitSearchKey() {
Int SP;
SP.Set(&rangeStart);
#ifdef USE_SYMMETRY
SP.ModAddK1order(&rangeWidthDiv2);
#endif
if (!SP.IsZero()) {
Point RS = secp->ComputePublicKey(&SP);
RS.y.ModNeg();
keyToSearch = secp->AddDirect(keysToSearch[keyIdx], RS);
} else {
keyToSearch = keysToSearch[keyIdx];
}
keyToSearchNeg = keyToSearch;
keyToSearchNeg.y.ModNeg();
}

// ============================================================================
// ComputeExpected — estimate operations and RAM for given DP size
// ============================================================================

void Kangaroo::ComputeExpected(double dp, double *op, double *ram,
double *overHead) {
#ifdef USE_SYMMETRY
double gainS = 1.0 / sqrt(2.0);
#else
double gainS = 1.0;
#endif

```
double k     = (double)totalRW;
double N     = pow(2.0, (double)rangePower);
double theta = pow(2.0, dp);
double Z0    = (2.0 * (2.0 - sqrt(2.0)) * gainS) * sqrt(M_PI);
double avgDP0 = Z0 * sqrt(N);

*op  = Z0 * pow(N * (k * theta + sqrt(N)), 1.0 / 3.0);
*ram = (double)sizeof(HASH_ENTRY) * (double)HASH_SIZE +
       (double)sizeof(ENTRY *) * (double)(HASH_SIZE * 4) +
       (double)(sizeof(ENTRY) + sizeof(ENTRY *)) * (*op / theta);
*ram /= (1024.0 * 1024.0);

if (overHead) *overHead = *op / avgDP0;
```

}

// ============================================================================
// SolveKeyCPU — CPU thread walk
// ============================================================================

void Kangaroo::SolveKeyCPU(TH_PARAM *ph) {

```
vector<ITEM> dps;
double       lastSent = 0;
int          thId     = ph->threadId;

ph->nbKangaroo = CPU_GRP_SIZE;
```

#ifdef USE_SYMMETRY
ph->symClass = new uint64_t[CPU_GRP_SIZE];
for (int i = 0; i < CPU_GRP_SIZE; i++) ph->symClass[i] = 0;
#endif

```
IntGroup *grp = new IntGroup(CPU_GRP_SIZE);
Int      *dx  = new Int[CPU_GRP_SIZE];

if (ph->px == NULL) {
    ph->px       = new Int[CPU_GRP_SIZE];
    ph->py       = new Int[CPU_GRP_SIZE];
    ph->distance = new Int[CPU_GRP_SIZE];
    CreateHerd(CPU_GRP_SIZE, ph->px, ph->py, ph->distance, TAME);
}

if (keyIdx == 0)
    ::printf("[+] CPU Thread %02d: %d kangaroos\n",
             ph->threadId, CPU_GRP_SIZE);

ph->hasStarted = true;

Int dy, rx, ry, _s, _p;

while (!endOfSearch) {

    // Compute dx for batch inverse
    for (int g = 0; g < CPU_GRP_SIZE; g++) {
```

#ifdef USE_SYMMETRY
uint64_t jmp = ph->px[g].bits64[0] % (NB_JUMP / 2) +
(NB_JUMP / 2) * ph->symClass[g];
#else
uint64_t jmp = ph->px[g].bits64[0] % NB_JUMP;
#endif
dx[g].ModSub(&ph->px[g], &jumpPointx[jmp]);
}

```
    grp->Set(dx);
    grp->ModInv();

    for (int g = 0; g < CPU_GRP_SIZE; g++) {
```

#ifdef USE_SYMMETRY
uint64_t jmp = ph->px[g].bits64[0] % (NB_JUMP / 2) +
(NB_JUMP / 2) * ph->symClass[g];
#else
uint64_t jmp = ph->px[g].bits64[0] % NB_JUMP;
#endif

```
        dy.ModSub(&ph->py[g], &jumpPointy[jmp]);
        _s.ModMulK1(&dy, &dx[g]);
        _p.ModSquareK1(&_s);

        rx.ModSub(&_p, &jumpPointx[jmp]);
        rx.ModSub(&ph->px[g]);

        ry.ModSub(&ph->px[g], &rx);
        ry.ModMulK1(&_s);
        ry.ModSub(&ph->py[g]);

        ph->distance[g].ModAddK1order(&jumpDistance[jmp]);
```

#ifdef USE_SYMMETRY
if (ry.ModPositiveK1()) {
ph->distance[g].ModNegK1order();
ph->symClass[g] = !ph->symClass[g];
}
#endif

```
        ph->px[g].Set(&rx);
        ph->py[g].Set(&ry);
    }

    if (clientMode) {
        for (int g = 0; g < CPU_GRP_SIZE; g++) {
            if (IsDP(&ph->px[g])) {
                ITEM it;
                it.x.Set(&ph->px[g]);
                it.dist.Set(&ph->distance[g]);
                it.kIdx = g;
                dps.push_back(it);
            }
        }
        double now = Timer::get_tick();
        if (now - lastSent > SEND_PERIOD) {
            LOCK(ghMutex);
            SendToServer(dps, ph->threadId, 0xFFFF);
            UNLOCK(ghMutex);
            lastSent = now;
        }
        if (!endOfSearch) counters[thId] += CPU_GRP_SIZE;
    } else {
        for (int g = 0; g < CPU_GRP_SIZE && !endOfSearch; g++) {
            if (IsDP(&ph->px[g])) {
                LOCK(ghMutex);
                if (!endOfSearch) {
                    if (!AddToTable(&ph->px[g], &ph->distance[g],
                                    g % 2)) {
                        CreateHerd(1, &ph->px[g], &ph->py[g],
                                   &ph->distance[g], g % 2, false);
                        collisionInSameHerd++;
                    }
                }
                UNLOCK(ghMutex);
            }
            if (!endOfSearch) counters[thId]++;
        }
    }

    if (saveRequest && !endOfSearch) {
        ph->isWaiting = true;
        LOCK(saveMutex);
        ph->isWaiting = false;
        UNLOCK(saveMutex);
    }
}

delete grp;
delete[] dx;
safe_delete_array(ph->px);
safe_delete_array(ph->py);
safe_delete_array(ph->distance);
```

#ifdef USE_SYMMETRY
safe_delete_array(ph->symClass);
#endif

```
ph->isRunning = false;
```

}

// ============================================================================
// SolveKeyGPU — GPU thread: upload state, run kernel, collect DPs
// ============================================================================

void Kangaroo::SolveKeyGPU(TH_PARAM *ph) {

```
double lastSent = 0;
int    thId     = ph->threadId;
```

#ifdef WITHGPU

```
vector<ITEM> dps;
vector<ITEM> gpuFound;

GPUEngine *gpu = new GPUEngine(
    ph->gridSizeX, ph->gridSizeY, ph->gpuId, 65536 * 2);

if (keyIdx == 0)
    ::printf("[+] GPU: %s (%.1f MB)\n",
             gpu->deviceName.c_str(),
             gpu->GetMemory() / 1048576.0);

double t0 = Timer::get_tick();

if (ph->px == NULL) {
    if (keyIdx == 0)
        ::printf("[+] GPU#%d: creating kangaroos...\n", ph->gpuId);
    uint64_t nbThread = gpu->GetNbThread();
    ph->px       = new Int[ph->nbKangaroo];
    ph->py       = new Int[ph->nbKangaroo];
    ph->distance = new Int[ph->nbKangaroo];
    for (uint64_t i = 0; i < nbThread; i++) {
        CreateHerd(GPU_GRP_SIZE,
                   &ph->px[i * GPU_GRP_SIZE],
                   &ph->py[i * GPU_GRP_SIZE],
                   &ph->distance[i * GPU_GRP_SIZE],
                   TAME);
    }
}
```

#ifdef USE_SYMMETRY
gpu->SetWildOffset(&rangeWidthDiv4);
#else
gpu->SetWildOffset(&rangeWidthDiv2);
#endif

```
// Upload standard jump table
Int dmaskInt;
HashTable::toInt(&dMask, &dmaskInt);
gpu->SetParams(&dmaskInt, jumpDistance, jumpPointx, jumpPointy);

// Upload GLV extended jump table
gpu->SetGLVParams(jPhiPxArr, jD1Arr, jD2Arr);

gpu->SetKangaroos(ph->px, ph->py, ph->distance);

if (workFile.length() == 0 || !saveKangaroo) {
    safe_delete_array(ph->px);
    safe_delete_array(ph->py);
    safe_delete_array(ph->distance);
}

gpu->callKernel();

double t1 = Timer::get_tick();

if (keyIdx == 0)
    ::printf("[+] GPU#%d: 2^%.2f kangaroos ready [%.1fs]\n",
             ph->gpuId, log2((double)ph->nbKangaroo), t1 - t0);

ph->hasStarted = true;

while (!endOfSearch) {

    gpu->Launch(gpuFound);
    counters[thId] += ph->nbKangaroo * NB_RUN;

    if (clientMode) {
        for (auto &item : gpuFound) dps.push_back(item);
        double now = Timer::get_tick();
        if (now - lastSent > SEND_PERIOD) {
            LOCK(ghMutex);
            SendToServer(dps, ph->threadId, ph->gpuId);
            UNLOCK(ghMutex);
            lastSent = now;
        }
    } else {
        if (!gpuFound.empty()) {
            LOCK(ghMutex);
            for (int g = 0;
                 !endOfSearch && g < (int)gpuFound.size(); g++) {

                uint32_t kType = (uint32_t)(gpuFound[g].kIdx % 2);

                if (!AddToTable(&gpuFound[g].x,
                                &gpuFound[g].dist,
                                kType)) {
                    Int px, py, d;
                    CreateHerd(1, &px, &py, &d, kType, false);
                    gpu->SetKangaroo(gpuFound[g].kIdx, &px, &py, &d);
                    collisionInSameHerd++;
                }
            }
            UNLOCK(ghMutex);
        }
    }

    if (saveRequest && !endOfSearch) {
        if (saveKangaroo) gpu->GetKangaroos(ph->px, ph->py, ph->distance);
        ph->isWaiting = true;
        LOCK(saveMutex);
        ph->isWaiting = false;
        UNLOCK(saveMutex);
    }
}

safe_delete_array(ph->px);
safe_delete_array(ph->py);
safe_delete_array(ph->distance);
delete gpu;
```

#else
ph->hasStarted = true;
#endif

```
ph->isRunning = false;
```

}

// ============================================================================
// Thread trampolines
// ============================================================================

#ifdef WIN64
DWORD WINAPI _SolveKeyCPU(LPVOID lpParam) {
#else
void *_SolveKeyCPU(void *lpParam) {
#endif
TH_PARAM *p = (TH_PARAM *)lpParam;
p->obj->SolveKeyCPU(p);
return 0;
}

#ifdef WIN64
DWORD WINAPI _SolveKeyGPU(LPVOID lpParam) {
#else
void *_SolveKeyGPU(void *lpParam) {
#endif
TH_PARAM *p = (TH_PARAM *)lpParam;
p->obj->SolveKeyGPU(p);
return 0;
}

// ============================================================================
// Run — main entry point
// ============================================================================

void Kangaroo::Run(int nbThread, vector<int> gpuId, vector<int> gridSize) {

```
double t0 = Timer::get_tick();

nbCPUThread = nbThread;
nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
totalRW     = 0;
```

#ifndef WITHGPU
if (nbGPUThread > 0) {
::printf(“GPU code not compiled, use -DWITHGPU\n”);
nbGPUThread = 0;
}
#endif

```
uint64_t totalThread = (uint64_t)nbCPUThread + (uint64_t)nbGPUThread;
if (totalThread == 0) {
    ::printf("No threads specified, exiting.\n");
    ::exit(0);
}

TH_PARAM     *params    = (TH_PARAM *)malloc(totalThread * sizeof(TH_PARAM));
THREAD_HANDLE *thHandles = (THREAD_HANDLE *)malloc(
    totalThread * sizeof(THREAD_HANDLE));

memset(params,    0, totalThread * sizeof(TH_PARAM));
memset(counters,  0, sizeof(counters));

::printf("[+] CPU threads: %d\n", nbCPUThread);
```

#ifdef WITHGPU
for (int i = 0; i < nbGPUThread; i++) {
int x = gridSize[2ULL * i];
int y = gridSize[2ULL * i + 1ULL];
if (!GPUEngine::GetGridSize(gpuId[i], &x, &y)) {
free(params); free(thHandles);
return;
}
params[nbCPUThread + i].gridSizeX  = x;
params[nbCPUThread + i].gridSizeY  = y;
params[nbCPUThread + i].nbKangaroo = (uint64_t)GPU_GRP_SIZE * x * y;
totalRW += params[nbCPUThread + i].nbKangaroo;
}
#endif

```
totalRW += nbCPUThread * (uint64_t)CPU_GRP_SIZE;

if (clientMode) {
    if (!GetConfigFromServer()) ::exit(0);
    if (workFile.length() > 0) saveKangaroo = true;
}

InitRange();

// Create standard jump table
CreateJumpTable();

// Create GLV extended jump table
CreateGLVJumpTable();

::printf("[+] Kangaroos: 2^%.2f\n", log2((double)totalRW));

if (!clientMode) {
    double dpOverHead;
    int suggestedDP = (int)((double)rangePower / 2.0 -
                            log2((double)totalRW));
    if (suggestedDP < 0) suggestedDP = 0;
    ComputeExpected((double)suggestedDP, &expectedNbOp,
                    &expectedMem, &dpOverHead);
    while (dpOverHead > 1.05 && suggestedDP > 0) {
        suggestedDP--;
        ComputeExpected((double)suggestedDP, &expectedNbOp,
                        &expectedMem, &dpOverHead);
    }
    if (initDPSize < 0) initDPSize = suggestedDP;
    ComputeExpected((double)initDPSize, &expectedNbOp, &expectedMem);
    if (nbLoadedWalk == 0)
        ::printf("[+] Suggested DP    : %d\n", suggestedDP);
    ::printf("[+] Expected ops    : 2^%.2f\n", log2(expectedNbOp));
    ::printf("[+] Expected RAM    : %.1f MB\n", expectedMem);
} else {
    keyIdx = 0;
    InitSearchKey();
}

SetDP(initDPSize);
FectchKangaroos(params);

for (keyIdx = 0; keyIdx < (int)keysToSearch.size(); keyIdx++) {

    InitSearchKey();
    endOfSearch       = false;
    collisionInSameHerd = 0;
    memset(counters, 0, sizeof(counters));

    // Launch CPU threads
    for (int i = 0; i < nbCPUThread; i++) {
        params[i].threadId  = i;
        params[i].isRunning = true;
        params[i].obj       = this;
        thHandles[i] = LaunchThread(_SolveKeyCPU, params + i);
    }
```

#ifdef WITHGPU
// Launch GPU threads
for (int i = 0; i < nbGPUThread; i++) {
int id = nbCPUThread + i;
params[id].threadId  = 0x80L + i;
params[id].isRunning = true;
params[id].gpuId     = gpuId[i];
params[id].obj       = this;
thHandles[id] = LaunchThread(_SolveKeyGPU, params + id);
}
#endif

```
    Process(params, "MK/s");
    JoinThreads(thHandles, nbCPUThread + nbGPUThread);
    FreeHandles(thHandles, nbCPUThread + nbGPUThread);
    hashTable.Reset();
}

double t1 = Timer::get_tick();
::printf("\n[+] Done: Total time %s\n",
         GetTimeStr(t1 - t0 + offsetTime).c_str());

free(params);
free(thHandles);
```

}
