#ifndef KANGAROOH
#define KANGAROOH

// ============================================================================
// PhiKang — Kangaroo Solver Class Declaration
//
// Changes vs original JeanLucPons/Kangaroo:
//   - GLV jump table arrays (jPhiPxArr, jD1Arr, jD2Arr) + CreateGLVJumpTable()
//   - glvReady flag guards GLV arrays before use in Run()
//   - WorkfileHeader v3.0: packed struct, glvEnabled/symmetryEnabled fields,
//     clarified field types
//   - ITEM uses full 256-bit distance (4 limbs) — bug fix
//   - FetchKangaroos typo fixed (was FectchKangaroos)
//   - counters[] size documented
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

#include "SECPK1/SECP256k1.h"
#include "GPU/GPUEngine.h"
#include "HashTable.h"
#include "Timer.h"
#include "Constants.h"

// ––––––––––––––––––––––––––––––––––––––
// Platform threading abstractions
// ––––––––––––––––––––––––––––––––––––––
#ifdef WIN64
  typedef HANDLE          THREAD_HANDLE;
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
// One instance is heap-allocated per solver thread (CPU or GPU).
// The obj back-pointer lets thread functions call Kangaroo methods.
// ––––––––––––––––––––––––––––––––––––––
typedef struct {

  // Thread identity
  int threadId;
  int gpuId;
  int gridSizeX;
  int gridSizeY;

  // Kangaroo state arrays (heap-allocated, nbKangaroo elements each)
  Int      *px;           // X coordinates
  Int      *py;           // Y coordinates
  Int      *distance;     // 256-bit accumulated distances
  uint64_t  nbKangaroo;   // number of kangaroos owned by this thread

  // Thread lifecycle flags (written by thread, read by host)
  bool isRunning;
  bool hasStarted;
  bool isWaiting;   // true while thread blocks waiting for server data

#ifdef USE_SYMMETRY
  uint64_t *symClass;     // equivalence class tag per kangaroo
#endif

  // Back-pointer to the owning Kangaroo instance
  class Kangaroo *obj;

} TH_PARAM;

// ––––––––––––––––––––––––––––––––––––––
// WorkfileHeader — binary header for .phk workfiles (v3.0)
//
// Written at offset 0 of every workfile, followed immediately by the
// human-readable stats block (null-terminated ASCII), then the binary
// kangaroo state array.
//
// IMPORTANT: #pragma pack(1) is required.  Without it the compiler may
// insert padding between mixed uint32_t / uint64_t fields, making the
// binary layout differ across compilers and platforms — breaking
// workfile portability.
//
// Workfile v3.0 is NOT backwards-compatible with v2.x / v1.x files
// because each kangaroo now stores four distance limbs (d0–d3) instead
// of two, and GLV runs add d1_glv / d2_glv components.
// ––––––––––––––––––––––––––––––––––––––
#pragma pack(push, 1)
typedef struct {

  uint64_t magic;              // Must equal WORKFILE_MAGIC ("PHIKANG\0")
  uint32_t version;            // Must equal WORKFILE_VERSION (3)
  uint32_t dpSize;             // Distinguished point bit count used during run

  uint64_t rangeStart[4];      // Search range start (256-bit, 4 limbs LE)
  uint64_t rangeEnd[4];        // Search range end   (256-bit, 4 limbs LE)

  // Progress tracking
  // expectedOps is stored as a IEEE 754 double reinterpreted as uint64_t
  // (use memcpy to convert — avoids strict-aliasing UB).
  // Represents the expected number of group operations, e.g. 2^34.2.
  uint64_t expectedOpsRaw;     // reinterpret_cast<uint64_t>(double expectedOps)
  uint64_t stepsCompleted;     // actual group operations completed so far
  uint64_t collisions;         // total distinguished-point collisions found
  uint64_t collisionsSameHerd; // collisions discarded (same herd — useless)

  // Configuration snapshot (must match on resume, or DP overhead applies)
  uint32_t nbKangaroo;         // total kangaroo count (tame + wild)
  uint32_t glvEnabled;         // 1 if GLV was active during this run
  uint32_t symmetryEnabled;    // 1 if USE_SYMMETRY was active
  uint32_t nbKeys;             // number of target public keys

  // NOTE: No padding needed — fields above sum to exactly 128 bytes.
  // If you add fields here, add a corresponding _reserved[] shrink
  // and keep the static_assert below passing.

} WorkfileHeader;
#pragma pack(pop)

// Verify header is exactly 128 bytes so the binary state that follows
// starts on a cache-line boundary.  If this fires, adjust _reserved[].
#ifdef __cplusplus
  static_assert(sizeof(WorkfileHeader) == 128,
    "WorkfileHeader must be exactly 128 bytes — adjust _reserved[] padding");
#endif

// ––––––––––––––––––––––––––––––––––––––
// Kangaroo — main solver class
// ––––––––––––––––––––––––––––––––––––––
class Kangaroo {

public:

  // -----------------------------------------------------------------------
  // Construction / destruction
  // -----------------------------------------------------------------------
  Kangaroo(
    Secp256K1   *secp,
    int32_t      initDPSize,
    bool         useGpu,
    std::string &workFile,
    std::string &iWorkFile,
    uint32_t     savePeriod,
    bool         saveKangaroo,
    bool         saveKangarooByServer,
    double       maxStep,
    int          wtimeout,
    int          port,
    int          ntimeout,
    std::string  serverIp,
    std::string  outputFile,
    bool         splitWorkfile
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

  // Build the standard jump table: NB_JUMP random points and distances.
  // Always called first.  Also calls CreateGLVJumpTable() internally
  // to keep the two tables in sync — do not call CreateGLVJumpTable()
  // separately unless you know what you're doing.
  void CreateJumpTable();

  // Precompute the GLV extension of the jump table:
  //   jPhiPxArr[i] = beta * jumpPointx[i] mod p   (phi endomorphism)
  //   jD1Arr[i], jD2Arr[i] = GLV decomposition of jumpDistance[i]
  //
  // Sets glvReady = true on success.
  // Called automatically from CreateJumpTable(); safe to call directly
  // if the base jump table has been loaded from a workfile.
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

  // Top-level solver.  Launches CPU + GPU threads, monitors progress,
  // handles save/resume, terminates when key found or maxStep reached.
  // Precondition: CreateJumpTable() (and therefore CreateGLVJumpTable())
  // must have been called, and glvReady must be true.
  void Run(
    int               nbThread,
    std::vector<int>  gpuId,
    std::vector<int>  gridSize
  );

  // Thread worker functions — do not call directly; use Run().
  void SolveKeyCPU(TH_PARAM *ph);
  void SolveKeyGPU(TH_PARAM *ph);

  // -----------------------------------------------------------------------
  // Workfile operations (v3.0)
  // -----------------------------------------------------------------------
  bool LoadWork(std::string &fileName);
  bool SaveWork(std::string &fileName);
  bool MergeWork(std::string &file1, std::string &file2, std::string &dest);
  bool MergeDir(std::string &dir, std::string &dest);
  void WorkInfo(std::string &fileName);       // prints human-readable stats
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
  void ComputeExpected(double dp, double *op, double *ram, double *overHead);
  std::string GetTimeStr(double t);

  // -----------------------------------------------------------------------
  // Public data — search parameters (set by ParseConfigFile + InitRange)
  // -----------------------------------------------------------------------
  Int rangeStart;
  Int rangeEnd;
  Int rangeWidth;
  Int rangeWidthDiv2;
  Int rangeWidthDiv4;
  Int rangeWidthDiv8;
  int rangePower;               // floor(log2(rangeWidth))

  std::vector<Point> keysToSearch;
  Point keyToSearch;
  Point keyToSearchNeg;         // negation of keyToSearch (for USE_SYMMETRY)
  int   keyIdx;                 // index of key currently being solved

  double expectedNbOp;          // expected group operations (floating point)
  double expectedMem;           // expected RAM usage in MB

private:

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

  // Copies kangaroo state from GPU back to host TH_PARAM arrays.
  // NOTE: was mis-spelled "FectchKangaroos" in earlier drafts — fixed.
  void FetchKangaroos(TH_PARAM *params);

  uint64_t getCPUCount();
  uint64_t getGPUCount();

  // -----------------------------------------------------------------------
  // Jump table storage
  //
  // Standard table (computed by CreateJumpTable):
  //   jumpDistance[i]  — scalar distance added at jump i
  //   jumpPointx[i]    — X coordinate of jump point J[i]
  //   jumpPointy[i]    — Y coordinate of jump point J[i]
  //
  // GLV extension (computed by CreateGLVJumpTable, requires glvReady):
  //   jPhiPxArr[i]     — beta * J[i].x mod p  (phi endomorphism X)
  //   jD1Arr[i]        — GLV k1 component of jumpDistance[i]
  //   jD2Arr[i]        — GLV k2 component of jumpDistance[i]
  //
  // All arrays are NB_JUMP elements long (see Constants.h).
  // -----------------------------------------------------------------------
  Int jumpDistance[NB_JUMP];
  Int jumpPointx[NB_JUMP];
  Int jumpPointy[NB_JUMP];

  Int jPhiPxArr[NB_JUMP];   // precomputed phi(J[i]).x = beta * J[i].x mod p
  Int jD1Arr[NB_JUMP];      // GLV k1 component: jumpDistance[i] = k1 + k2*λ
  Int jD2Arr[NB_JUMP];      // GLV k2 component

  // Set to true by CreateGLVJumpTable(); checked by Run() before GPU launch.
  // Prevents accidental use of uninitialised GLV arrays.
  bool glvReady;

  // -----------------------------------------------------------------------
  // Runtime state
  // -----------------------------------------------------------------------
  Secp256K1  *secp;
  HashTable   hashTable;

  int      nbCPUThread;
  int      nbGPUThread;
  uint64_t totalRW;               // total kangaroo count (tame + wild)
  uint64_t collisionInSameHerd;   // same-herd collisions (diagnostic only)

  // Per-thread step counters.
  // Sized to 256 to cover any realistic CPU thread count + GPU thread
  // count combination without dynamic allocation.  Index [i] is owned
  // exclusively by thread i — no locking needed for individual reads.
  uint64_t counters[256];

  // -----------------------------------------------------------------------
  // Work file state
  // -----------------------------------------------------------------------
  std::string workFile;
  std::string inputFile;
  std::string outputFile;
  uint32_t    saveWorkPeriod;       // seconds between automatic saves
  bool        saveKangaroo;         // include kangaroo state in workfile
  bool        saveKangarooByServer; // server-side kangaroo save
  bool        splitWorkfile;        // write partitioned workfiles
  uint64_t    nbLoadedWalk;         // kangaroos loaded from input workfile
  FILE       *fRead;                // open handle during streaming load

  // -----------------------------------------------------------------------
  // Search control
  // -----------------------------------------------------------------------
  bool     endOfSearch;    // set true when key is found or maxStep reached
  bool     saveRequest;    // set true by SIGINT/SIGTERM handler
  int32_t  initDPSize;     // initial DP bit size (may be auto-tuned)
  int      dpSize;         // current DP bit size
  int256_t dMask;          // bitmask derived from dpSize for fast DP test
  double   maxStep;        // stop after this many group ops (0 = unlimited)
  double   offsetTime;     // wall-clock seconds already spent (from workfile)
  uint64_t offsetCount;    // group operations already completed (from workfile)

  // -----------------------------------------------------------------------
  // GPU
  // -----------------------------------------------------------------------
  bool useGpu;

  // -----------------------------------------------------------------------
  // Network
  // -----------------------------------------------------------------------
  bool        clientMode;
  std::string serverIp;
  int         port;
  int         wtimeout;    // workfile send timeout (seconds)
  int         ntimeout;    // network operation timeout (seconds)
  int         connectedClient;
  void       *hostInfo;    // platform DNS/socket resolve result
  uint32_t    pid;         // process ID (used in client handshake)

  // -----------------------------------------------------------------------
  // Symmetry helpers
  // -----------------------------------------------------------------------
  Int rangeWidthDiv2Neg;   // -(rangeWidth/2) — used for negation in USE_SYMMETRY

  // -----------------------------------------------------------------------
  // Mutexes
  // ghMutex:   protects hashTable and collision state
  // saveMutex: protects workfile write operations
  // -----------------------------------------------------------------------
#ifdef WIN64
  HANDLE ghMutex;
  HANDLE saveMutex;
#else
  pthread_mutex_t ghMutex;
  pthread_mutex_t saveMutex;
#endif

  // -----------------------------------------------------------------------
  // CPU group size (kangaroos per CPU thread batch)
  // Set at runtime; default derived from system thread count.
  // -----------------------------------------------------------------------
  int CPU_GRP_SIZE;

};

// ––––––––––––––––––––––––––––––––––––––
// Thread entry-point trampolines
// Defined in Kangaroo.cpp; bridge OS thread API to member functions.
// ––––––––––––––––––––––––––––––––––––––
#ifdef WIN64
  DWORD WINAPI _SolveKeyCPU(LPVOID lpParam);
  DWORD WINAPI _SolveKeyGPU(LPVOID lpParam);
#else
  void *_SolveKeyCPU(void *lpParam);
  void *_SolveKeyGPU(void *lpParam);
#endif

#endif // KANGAROOH
