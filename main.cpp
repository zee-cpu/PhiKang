#include <string.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>

#ifndef WIN64
#include <unistd.h>
#endif

#include “GPU/GPUEngine.h”
#include “Kangaroo.h”
#include “SECPK1/SECP256k1.h”
#include “Timer.h”

using namespace std;

// ============================================================================
// CHECKARG macro — fixed:
//   - No increment inside macro (call site owns the increment)
//   - exit(1) not exit(0) on error
//   - do/while(0) wrapper prevents dangling-else hazard
// ============================================================================

#define CHECKARG(opt, n) do {                                
if (a >= argc - 1) {                                       
::printf(”[!] “ opt “ missing argument #%d\n”, n);      
exit(1);                                                 
}                                                          
} while(0)

// ============================================================================
// Argument parsing helpers
// ============================================================================

static void printUsage() {
printf(”\n”);
printf(“PhiKang v” RELEASE “ — Kangaroo ECDLP Solver with GLV\n”);
printf(“secp256k1 | Pollard Kangaroo + GLV Endomorphism\n”);
printf(”\n”);
printf(“Usage:\n”);
printf(”  PhiKang [options] inFile\n”);
printf(”\n”);
printf(“Options:\n”);
printf(”  -t <n>                Number of CPU threads (default: all cores)\n”);
printf(”  -d <n>                Distinguished point bit size (default: auto)\n”);
printf(”  -gpu                  Enable GPU acceleration\n”);
printf(”  -gpuId <ids>          GPU IDs comma separated (default: 0)\n”);
printf(”  -g <gx,gy,…>        GPU kernel grid size per GPU\n”);
printf(”  -o <file>             Output file for found keys\n”);
printf(”  -w <file>             Work file — save progress\n”);
printf(”  -i <file>             Work file — load and resume\n”);
printf(”  -wi <n>               Work save interval seconds (default: 60)\n”);
printf(”  -ws                   Save kangaroo state in work file\n”);
printf(”  -wss                  Save kangaroo state via server\n”);
printf(”  -wsplit               Split server work file\n”);
printf(”  -wm <f1> <f2> <dest>  Merge two work files (all 3 required)\n”);
printf(”  -wmdir <dir> <dest>   Merge directory of work files\n”);
printf(”  -winfo <file>         Print human-readable work file statistics\n”);
printf(”  -wcheck <file>        Check work file integrity\n”);
printf(”  -wpartcreate <name>   Create empty partitioned work file\n”);
printf(”  -wt <ms>              Work save timeout ms (default: 3000)\n”);
printf(”  -m <n>                Max steps multiplier before giving up\n”);
printf(”  -s                    Server mode\n”);
printf(”  -c <ip>               Client mode — connect to server\n”);
printf(”  -sp <port>            Server port (default: 17403)\n”);
printf(”  -nt <ms>              Network timeout ms (default: 3000)\n”);
printf(”  -l                    List CUDA devices\n”);
printf(”  -check                Run GPU kernel self-test\n”);
printf(”  -v                    Print version and exit\n”);
printf(”  -h                    Print this help\n”);
printf(”\n”);
printf(“Config file format:\n”);
printf(”  Line 1: Range start (hex)\n”);
printf(”  Line 2: Range end   (hex)\n”);
printf(”  Line 3+: Target public key(s) compressed hex\n”);
printf(”\n”);
exit(0);
}

static int getInt(const string &name, char *v) {
try {
return stoi(string(v));
} catch (out_of_range &) {
printf(”[!] %s value out of range\n”, name.c_str());
exit(1);
} catch (invalid_argument &) {
printf(”[!] Invalid %s argument, integer expected\n”, name.c_str());
exit(1);
}
}

static double getDouble(const string &name, char *v) {
try {
return stod(string(v));
} catch (out_of_range &) {
printf(”[!] %s value out of range\n”, name.c_str());
exit(1);
} catch (invalid_argument &) {
printf(”[!] Invalid %s argument, number expected\n”, name.c_str());
exit(1);
}
}

static void getInts(const string &name, vector<int> &tokens,
const string &text, char sep) {
size_t start = 0, end = 0;
tokens.clear();
try {
while ((end = text.find(sep, start)) != string::npos) {
tokens.push_back(stoi(text.substr(start, end - start)));
start = end + 1;
}
tokens.push_back(stoi(text.substr(start)));
} catch (out_of_range &) {
printf(”[!] %s value out of range\n”, name.c_str());
exit(1);
} catch (invalid_argument &) {
printf(”[!] Invalid %s argument, integers expected\n”, name.c_str());
exit(1);
}
}

// ============================================================================
// Default parameters
// ============================================================================

static int         dp                   = -1;
static int         nbCPUThread          = 0;
static string      configFile           = “”;
static bool        checkFlag            = false;
static bool        gpuEnable            = false;
static vector<int> gpuId                = {0};
static vector<int> gridSize;
static string      workFile             = “”;
static string      checkWorkFile        = “”;
static string      iWorkFile            = “”;
static string      infoFile             = “”;
static uint32_t    savePeriod           = 60;
static bool        saveKangaroo         = false;
static bool        saveKangarooByServer = false;
static string      merge1               = “”;
static string      merge2               = “”;
static string      mergeDest            = “”;
static string      mergeDir             = “”;
static double      maxStep              = 0.0;
static int         wtimeout             = 3000;
static int         ntimeout             = 3000;
static int         port                 = 17403;
static bool        serverMode           = false;
static string      serverIP             = “”;
static string      outputFile           = “KEYFOUND.txt”;
static bool        splitWorkFile        = false;

// ============================================================================
// main
// ============================================================================

int main(int argc, char *argv[]) {

```
// Terminal detection — no ANSI escapes if output is redirected
```

#ifndef WIN64
bool isTTY = (isatty(STDOUT_FILENO) == 1);
#else
bool isTTY = true;
#endif

```
// Print banner without system("clear")
time_t     now = time(nullptr);
struct tm *tm  = localtime(&now);
char       timeBuf[64];
strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", tm);

if (isTTY) printf("\033[01;33m");
```

#ifdef USE_SYMMETRY
printf(”[+] PhiKang v” RELEASE
“ — Kangaroo + GLV (symmetry enabled)\n”);
#else
printf(”[+] PhiKang v” RELEASE “ — Kangaroo + GLV\n”);
#endif
printf(”[+] %s\n”, timeBuf);
if (isTTY) printf(”\033[0m”);

```
// Init timer and random seed
Timer::Init();
rseed(Timer::getSeed32());

// Init secp256k1
Secp256K1 *secp = new Secp256K1();
secp->Init();

// Default CPU thread count with fallback
nbCPUThread = Timer::getCoreNumber();
if (nbCPUThread <= 0) {
    printf("[!] Could not detect CPU core count, defaulting to 1\n");
    nbCPUThread = 1;
}

// -----------------------------------------------------------------------
// Argument parsing
// Rule: CHECKARG never increments a
//       Each branch does its own a++ after reading each value
// -----------------------------------------------------------------------

int a = 1;
while (a < argc) {

    if (strcmp(argv[a], "-t") == 0) {
        CHECKARG("-t", 1);
        a++;
        nbCPUThread = getInt("nbCPUThread", argv[a]);
        a++;

    } else if (strcmp(argv[a], "-d") == 0) {
        CHECKARG("-d", 1);
        a++;
        dp = getInt("dpSize", argv[a]);
        a++;

    } else if (strcmp(argv[a], "-h") == 0) {
        printUsage(); // exits internally

    } else if (strcmp(argv[a], "-v") == 0) {
        printf("PhiKang v" RELEASE "\n");
        delete secp;
        return 0;

    } else if (strcmp(argv[a], "-l") == 0) {
```

#ifdef WITHGPU
GPUEngine::PrintCudaInfo();
#else
printf(“GPU code not compiled. Rebuild with: make gpu=1\n”);
#endif
delete secp;
return 0;

```
    } else if (strcmp(argv[a], "-gpu") == 0) {
        gpuEnable = true;
        a++;

    } else if (strcmp(argv[a], "-gpuId") == 0) {
        CHECKARG("-gpuId", 1);
        a++;
        getInts("gpuId", gpuId, string(argv[a]), ',');
        a++;

    } else if (strcmp(argv[a], "-g") == 0) {
        CHECKARG("-g", 1);
        a++;
        getInts("gridSize", gridSize, string(argv[a]), ',');
        a++;

    } else if (strcmp(argv[a], "-o") == 0) {
        CHECKARG("-o", 1);
        a++;
        outputFile = string(argv[a]);
        a++;

    } else if (strcmp(argv[a], "-w") == 0) {
        CHECKARG("-w", 1);
        a++;
        workFile = string(argv[a]);
        a++;

    } else if (strcmp(argv[a], "-i") == 0) {
        CHECKARG("-i", 1);
        a++;
        iWorkFile = string(argv[a]);
        a++;

    } else if (strcmp(argv[a], "-wi") == 0) {
        CHECKARG("-wi", 1);
        a++;
        savePeriod = (uint32_t)getInt("savePeriod", argv[a]);
        a++;

    } else if (strcmp(argv[a], "-wt") == 0) {
        CHECKARG("-wt", 1);
        a++;
        wtimeout = getInt("wtimeout", argv[a]);
        a++;

    } else if (strcmp(argv[a], "-ws") == 0) {
        saveKangaroo = true;
        a++;

    } else if (strcmp(argv[a], "-wss") == 0) {
        saveKangarooByServer = true;
        a++;

    } else if (strcmp(argv[a], "-wsplit") == 0) {
        splitWorkFile = true;
        a++;

    } else if (strcmp(argv[a], "-wm") == 0) {
        // All three arguments are mandatory
        CHECKARG("-wm", 1);
        a++;
        merge1 = string(argv[a]);
        CHECKARG("-wm", 2);
        a++;
        merge2 = string(argv[a]);
        CHECKARG("-wm", 3);
        a++;
        mergeDest = string(argv[a]);
        a++;

    } else if (strcmp(argv[a], "-wmdir") == 0) {
        CHECKARG("-wmdir", 1);
        a++;
        mergeDir = string(argv[a]);
        CHECKARG("-wmdir", 2);
        a++;
        mergeDest = string(argv[a]);
        a++;

    } else if (strcmp(argv[a], "-winfo") == 0) {
        CHECKARG("-winfo", 1);
        a++;
        infoFile = string(argv[a]);
        a++;

    } else if (strcmp(argv[a], "-wcheck") == 0) {
        CHECKARG("-wcheck", 1);
        a++;
        checkWorkFile = string(argv[a]);
        a++;

    } else if (strcmp(argv[a], "-wpartcreate") == 0) {
        CHECKARG("-wpartcreate", 1);
        a++;
        string partName = string(argv[a]);
        a++;
        Kangaroo::CreateEmptyPartWork(partName);
        delete secp;
        return 0;

    } else if (strcmp(argv[a], "-m") == 0) {
        CHECKARG("-m", 1);
        a++;
        maxStep = getDouble("maxStep", argv[a]);
        a++;

    } else if (strcmp(argv[a], "-s") == 0) {
        if (serverIP.length() > 0) {
            printf("[!] -s and -c are incompatible\n");
            delete secp;
            return 1;
        }
        serverMode = true;
        a++;

    } else if (strcmp(argv[a], "-c") == 0) {
        if (serverMode) {
            printf("[!] -s and -c are incompatible\n");
            delete secp;
            return 1;
        }
        CHECKARG("-c", 1);
        a++;
        serverIP = string(argv[a]);
        a++;

    } else if (strcmp(argv[a], "-sp") == 0) {
        CHECKARG("-sp", 1);
        a++;
        port = getInt("serverPort", argv[a]);
        a++;

    } else if (strcmp(argv[a], "-nt") == 0) {
        CHECKARG("-nt", 1);
        a++;
        ntimeout = getInt("ntimeout", argv[a]);
        a++;

    } else if (strcmp(argv[a], "-check") == 0) {
        checkFlag = true;
        a++;

    } else if (a == argc - 1) {
        // Trailing positional argument — config file
        configFile = string(argv[a]);
        a++;

    } else {
        printf("[!] Unexpected argument: %s\n", argv[a]);
        printf("    Run PhiKang -h for usage\n");
        delete secp;
        return 1;
    }
}

// -----------------------------------------------------------------------
// Validate grid size vs GPU count
// -----------------------------------------------------------------------

if (gridSize.size() == 0) {
    for (size_t i = 0; i < gpuId.size(); i++) {
        gridSize.push_back(0);
        gridSize.push_back(0);
    }
} else if (gridSize.size() != gpuId.size() * 2) {
    printf("[!] gridSize (%zu entries) and gpuId (%zu GPUs) mismatch\n"
           "    Expected %zu grid entries (gx,gy per GPU)\n",
           gridSize.size(), gpuId.size(), gpuId.size() * 2);
    delete secp;
    return 1;
}

// -----------------------------------------------------------------------
// Construct solver
// -----------------------------------------------------------------------

Kangaroo *v = new Kangaroo(
    secp, dp, gpuEnable,
    workFile, iWorkFile,
    savePeriod, saveKangaroo, saveKangarooByServer,
    maxStep, wtimeout, port, ntimeout,
    serverIP, outputFile, splitWorkFile
);

// -----------------------------------------------------------------------
// Dispatch — every path cleans up before returning
// -----------------------------------------------------------------------

if (checkFlag) {
    v->Check(gpuId, gridSize);
    delete v; delete secp;
    return 0;
}

if (checkWorkFile.length() > 0) {
    bool ok = v->CheckWorkFile(nbCPUThread, checkWorkFile);
    delete v; delete secp;
    return ok ? 0 : 1;
}

if (infoFile.length() > 0) {
    v->WorkInfo(infoFile);
    delete v; delete secp;
    return 0;
}

if (mergeDir.length() > 0) {
    bool ok = v->MergeDir(mergeDir, mergeDest);
    delete v; delete secp;
    return ok ? 0 : 1;
}

if (merge1.length() > 0) {
    bool ok = v->MergeWork(merge1, merge2, mergeDest);
    delete v; delete secp;
    return ok ? 0 : 1;
}

if (iWorkFile.length() > 0) {
    if (!v->LoadWork(iWorkFile)) {
        delete v; delete secp;
        return 1;
    }
}

if (configFile.length() > 0) {
    if (!v->ParseConfigFile(configFile)) {
        delete v; delete secp;
        return 1;
    }
} else {
    if (serverIP.length() == 0 && !serverMode) {
        printf("[!] No input file specified\n");
        printf("    Run PhiKang -h for usage\n");
        delete v; delete secp;
        return 1;
    }
}

if (serverMode)
    v->RunServer();
else
    v->Run(nbCPUThread, gpuId, gridSize);

delete v;
delete secp;
return 0;
```

}