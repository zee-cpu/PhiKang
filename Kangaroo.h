#ifndef KANGAOOTH
#define KANGAOOTH

// ============================================================================
// PhiKang — Kangaroo Solver Class
//
// Changes vs original:
//   - GLV jump table precomputation (jPhiPx, jD1, jD2)
//   - Workfile v3.0 with human-readable stats header
//   - ITEM uses full 256-bit distance (bug fix flows from GPUEngine.h)
//   - Server/client architecture preserved
//   - USE_SYMMETRY preserved
// ============================================================================

#include <string>
#include <vector>

#ifdef WIN64
#include <windows.h>
#else
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#endif

#include “SECPK1/SECP256k1.h”
#include “GPU/GPUEngine.h”
#include “HashTable.h”
#include “Timer.h”
#include “Constants.h”

// ––––––––––––––––––––––––––––––––––––––
// Platform threading abstractions
// ––––––––––––––––––––––––––––––––––––––

#ifdef WIN64
typedef HANDLE           THREAD_HANDLE;
typedef CRITICAL_SECTION MUTEX_TYPE;
#define LOCK(m)          EnterCriticalSection(&m)
#define UNLOCK(m)        LeaveCriticalSection(&m)
#define CREATEM(m)       InitializeCriticalSection(&m)
#define DESTROYM(m)      DeleteCriticalSection(&m)
#else
typedef pthread_t        THREAD_HANDLE;
typedef pthread_mutex_t  MUTEX_TYPE;
#define LOCK(m)          pthread_mutex_lock(&m)
#define UNLOCK(m)        pthread_mutex_unlock(&m)
#define CREATEM(m)       pthread_mutex_init(&m, NULL)
#define DESTROYM(m)      pthread_mutex_destroy(&m)
#endif

// ––––––––––––––––––––––––––––––––––––––
// Thread parameter block
// Passed to each CPU and GPU solver thread
// ––––––––––––––––––––––––––––––––––––––

typedef struct {
// Thread identity
int       threadId;
int       gpuId;
int       gridSizeX;
int       gridSizeY;

```
// Kangaroo state (heap allocated per thread)
Int      *px;
Int      *py;
Int      *distance;
uint64_t  nbKangaroo;

// Thread lifecycle
bool      isRunning;
bool      hasStarted;
bool      isWaiting;
```

#ifdef USE_SYMMETRY
uint64_t *symClass;
#endif

```
// Back-pointer to owning Kangaroo instance
class Kangaroo *obj;
```

} TH_PARAM;

// ––––––––––––––––––––––––––––––––––––––
// Workfile v3.0 header (human-readable + binary)
// Written at the start of every .phk workfile
// ––––––––––––––––––––––––––––––––––––––

typedef struct {
uint64_t magic;           // WORKFILE_MAGIC = “PHIKANG\0”
uint32_t version;         // WORKFILE_VERSION = 3
uint32_t dpSize;          // Distinguished point bit count
uint64_t rangeStart[4];   // Range start (256-bit)
uint64_t rangeEnd[4];     // Range end (256-bit)
uint64_t expectedOps;     // Expected operations (as double bits)
uint64_t stepsCompleted;  // Steps completed so far
uint64_t collisions;      // Total collisions
uint64_t sameHerd;        // Same-herd collisions
uint32_t nbKangaroo;      // Total kangaroo count
uint32_t glvEnabled;      // 1 if GLV was active during this run
uint32_t symmetryEnabled; // 1 if USE_SYMMETRY was active
uint32_t nbKeys;          // Number of target keys
// Followed by binary kangaroo state
} WorkfileHeader;

// ––––––––––––––––––––––––––––––––––––––
// Kangaroo — main solver class
// ––––––––––––––––––––––––––––––––––––––

class Kangaroo {

public:

```
// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

Kangaroo(
    Secp256K1  *secp,
    int32_t     initDPSize,
    bool        useGpu,
    std::string &workFile,
    std::string &iWorkFile,
    uint32_t    savePeriod,
    bool        saveKangaroo,
    bool        saveKangarooByServer,
    double      maxStep,
    int         wtimeout,
    int         port,
    int         ntimeout,
    std::string serverIp,
    std::string outputFile,
    bool        splitWorkfile
);

~Kangaroo();

// -----------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------

bool ParseConfigFile(std::string &fileName);
void InitRange();
void InitSearchKey();
void SetDP(int size);
bool IsDP(Int *x);

// -----------------------------------------------------------------------
// Jump table — standard + GLV
// -----------------------------------------------------------------------

void CreateJumpTable();

// Precompute GLV jump table components:
//   jPhiPxArr[i] = beta * jumpPointx[i] mod p   (phi endomorphism)
//   jD1Arr[i], jD2Arr[i] = GLV decomposition of jumpDistance[i]
// Called after CreateJumpTable(), before Run()
void CreateGLVJumpTable();

// -----------------------------------------------------------------------
// Herd management
// -----------------------------------------------------------------------

void CreateHerd(
    int   nbKangaroo,
    Int  *px,
    Int  *py,
    Int  *d,
    int   firstType,
    bool  lock = true
);

// -----------------------------------------------------------------------
// Solver entry points
// -----------------------------------------------------------------------

void Run(
    int nbThread,
    std::vector<int> gpuId,
    std::vector<int> gridSize
);

void SolveKeyCPU(TH_PARAM *ph);
void SolveKeyGPU(TH_PARAM *ph);

// -----------------------------------------------------------------------
// Workfile operations (v3.0)
// -----------------------------------------------------------------------

bool LoadWork(std::string &fileName);
bool SaveWork(std::string &fileName);
bool MergeWork(std::string &file1, std::string &file2,
               std::string &dest);
bool MergeDir(std::string &dir, std::string &dest);
void WorkInfo(std::string &fileName);    // Human-readable stats dump
bool CheckWorkFile(int nbThread, std::string &fileName);

static void CreateEmptyPartWork(std::string &fileName);

// -----------------------------------------------------------------------
// Server / client (preserved from original)
// -----------------------------------------------------------------------

void RunServer();
void RunClient();
bool GetConfigFromServer();
void SendToServer(std::vector<ITEM> &dps, int threadId, int gpuId);

// -----------------------------------------------------------------------
// Utility / debug
// -----------------------------------------------------------------------

void Check(std::vector<int> gpuId, std::vector<int> gridSize);
void ComputeExpected(double dp, double *op, double *ram,
                     double *overHead);

std::string GetTimeStr(double t);

// -----------------------------------------------------------------------
// Public data — search parameters
// -----------------------------------------------------------------------

Int    rangeStart;
Int    rangeEnd;
Int    rangeWidth;
Int    rangeWidthDiv2;
Int    rangeWidthDiv4;
Int    rangeWidthDiv8;
int    rangePower;

std::vector<Point> keysToSearch;
Point  keyToSearch;
Point  keyToSearchNeg;
int    keyIdx;

double expectedNbOp;
double expectedMem;
```

private:

```
// -----------------------------------------------------------------------
// Collision resolution
// -----------------------------------------------------------------------

bool Output(Int *pk, char sInfo, int sType);
bool CheckKey(Int d1, Int d2, uint8_t type);
bool CollisionCheck(Int *d1, uint32_t type1, Int *d2, uint32_t type2);
bool AddToTable(Int *pos, Int *dist, uint32_t kType);
bool AddToTable(int256_t *x, int256_t *d, uint32_t kType);

// -----------------------------------------------------------------------
// Network helpers
// -----------------------------------------------------------------------

bool ConnectToServer(int *sock);
void DisconnectFromServer(int sock);

// -----------------------------------------------------------------------
// Thread helpers
// -----------------------------------------------------------------------

THREAD_HANDLE LaunchThread(
```

#ifdef WIN64
LPTHREAD_START_ROUTINE func,
#else
void *(*func)(void *),
#endif
void *arg
);
void JoinThreads(THREAD_HANDLE *handles, int count);
void FreeHandles(THREAD_HANDLE *handles, int count);
void Process(TH_PARAM *params, const char *unit);
void FectchKangaroos(TH_PARAM *params);

```
uint64_t getCPUCount();
uint64_t getGPUCount();

// -----------------------------------------------------------------------
// Jump table storage
// Standard (original)
Int jumpDistance[NB_JUMP];
Int jumpPointx[NB_JUMP];
Int jumpPointy[NB_JUMP];

// GLV extended
Int jPhiPxArr[NB_JUMP];    // phi(J[i]).x = beta * J[i].x mod p
Int jD1Arr[NB_JUMP];       // GLV k1 component of jumpDistance[i]
Int jD2Arr[NB_JUMP];       // GLV k2 component of jumpDistance[i]

// -----------------------------------------------------------------------
// Runtime state
// -----------------------------------------------------------------------

Secp256K1  *secp;
HashTable   hashTable;

int         nbCPUThread;
int         nbGPUThread;
uint64_t    totalRW;           // total kangaroos (tame + wild)
uint64_t    collisionInSameHerd;

// Counters (one per thread)
uint64_t    counters[256];

// Work file state
std::string workFile;
std::string inputFile;
std::string outputFile;
uint32_t    saveWorkPeriod;    // seconds between auto-saves
bool        saveKangaroo;
bool        saveKangarooByServer;
bool        splitWorkfile;
uint64_t    nbLoadedWalk;
FILE       *fRead;

// Search control
bool        endOfSearch;
bool        saveRequest;
int32_t     initDPSize;
int         dpSize;
int256_t    dMask;
double      maxStep;
double      offsetTime;
uint64_t    offsetCount;

// GPU
bool        useGpu;

// Network
bool        clientMode;
bool        saveKangarooByServer_flag;
std::string serverIp;
int         port;
int         wtimeout;
int         ntimeout;
int         connectedClient;
void       *hostInfo;
uint32_t    pid;

// Symmetry (range subdivisions)
Int         rangeWidthDiv2Neg;

// Mutexes
```

#ifdef WIN64
HANDLE      ghMutex;
HANDLE      saveMutex;
#else
pthread_mutex_t ghMutex;
pthread_mutex_t saveMutex;
#endif

```
// CPU group size (tunable)
int         CPU_GRP_SIZE;
```

};

// ———————————————————————–
// Thread entry point trampolines (defined in Kangaroo.cpp)
// ———————————————————————–

#ifdef WIN64
DWORD WINAPI _SolveKeyCPU(LPVOID lpParam);
DWORD WINAPI _SolveKeyGPU(LPVOID lpParam);
#else
void *_SolveKeyCPU(void *lpParam);
void *_SolveKeyGPU(void *lpParam);
#endif

#endif // KANGAOOTH
