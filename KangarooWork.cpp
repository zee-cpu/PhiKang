#include <string.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>

#ifndef WIN64
#include <dirent.h>
#include <unistd.h>
#endif

#include “Kangaroo.h”
#include “Timer.h”
#include “HashTable.h”

using namespace std;

// ============================================================================
// Helper: write uint64 little-endian to file
// ============================================================================

static bool WriteU64(FILE *f, uint64_t v) {
return fwrite(&v, 8, 1, f) == 1;
}

static bool ReadU64(FILE *f, uint64_t *v) {
return fread(v, 8, 1, f) == 1;
}

static bool WriteU32(FILE *f, uint32_t v) {
return fwrite(&v, 4, 1, f) == 1;
}

static bool ReadU32(FILE *f, uint32_t *v) {
return fread(v, 4, 1, f) == 1;
}

static bool WriteInt(FILE *f, Int *v) {
for (int i = 0; i < 4; i++)
if (!WriteU64(f, v->bits64[i])) return false;
return true;
}

static bool ReadInt(FILE *f, Int *v) {
v->SetInt32(0);
for (int i = 0; i < 4; i++)
if (!ReadU64(f, &v->bits64[i])) return false;
v->bits64[4] = 0;
return true;
}

// ============================================================================
// Helper: format timestamp for workfile header
// ============================================================================

static string GetTimestamp() {
time_t    now = time(NULL);
struct tm *tm = gmtime(&now);
char       buf[64];
strftime(buf, sizeof(buf), “%Y-%m-%d %H:%M:%S UTC”, tm);
return string(buf);
}

// ============================================================================
// Helper: format number as 2^x string
// ============================================================================

static string FormatPow2(double v) {
if (v <= 0.0) return “0”;
char buf[32];
snprintf(buf, sizeof(buf), “2^%.2f”, log2(v));
return string(buf);
}

// ============================================================================
// WriteWorkfileHeader — write binary header + human-readable stats block
//
// Format:
//   [MAGIC 8 bytes]
//   [Binary WorkfileHeader struct]
//   [Human-readable text block, null-terminated]
//   [Binary kangaroo state follows]
// ============================================================================

static bool WriteWorkfileHeader(
FILE            *f,
WorkfileHeader  &hdr,
const string    &statsText
) {
// Magic
if (!WriteU64(f, hdr.magic)) return false;

```
// Binary header fields
if (!WriteU32(f, hdr.version))          return false;
if (!WriteU32(f, hdr.dpSize))           return false;
for (int i = 0; i < 4; i++)
    if (!WriteU64(f, hdr.rangeStart[i])) return false;
for (int i = 0; i < 4; i++)
    if (!WriteU64(f, hdr.rangeEnd[i]))   return false;
if (!WriteU64(f, hdr.expectedOps))      return false;
if (!WriteU64(f, hdr.stepsCompleted))   return false;
if (!WriteU64(f, hdr.collisions))       return false;
if (!WriteU64(f, hdr.sameHerd))         return false;
if (!WriteU32(f, hdr.nbKangaroo))       return false;
if (!WriteU32(f, hdr.glvEnabled))       return false;
if (!WriteU32(f, hdr.symmetryEnabled))  return false;
if (!WriteU32(f, hdr.nbKeys))           return false;

// Human-readable stats block length + content
uint32_t textLen = (uint32_t)statsText.length() + 1; // include null
if (!WriteU32(f, textLen)) return false;
if (fwrite(statsText.c_str(), 1, textLen, f) != textLen) return false;

return true;
```

}

// ============================================================================
// ReadWorkfileHeader — read and validate binary header
// Returns false if magic or version mismatch
// ============================================================================

static bool ReadWorkfileHeader(
FILE           *f,
WorkfileHeader &hdr,
string         &statsText
) {
if (!ReadU64(f, &hdr.magic)) return false;
if (hdr.magic != WORKFILE_MAGIC) {
printf(”[!] Invalid workfile magic (expected PhiKang v3.0 format)\n”);
return false;
}

```
if (!ReadU32(f, &hdr.version)) return false;
if (hdr.version != WORKFILE_VERSION) {
    printf("[!] Workfile version %d not supported (expected %d)\n",
           hdr.version, WORKFILE_VERSION);
    printf("[!] Cannot resume work from a different version\n");
    return false;
}

if (!ReadU32(f, &hdr.dpSize)) return false;
for (int i = 0; i < 4; i++)
    if (!ReadU64(f, &hdr.rangeStart[i])) return false;
for (int i = 0; i < 4; i++)
    if (!ReadU64(f, &hdr.rangeEnd[i])) return false;
if (!ReadU64(f, &hdr.expectedOps))     return false;
if (!ReadU64(f, &hdr.stepsCompleted))  return false;
if (!ReadU64(f, &hdr.collisions))      return false;
if (!ReadU64(f, &hdr.sameHerd))        return false;
if (!ReadU32(f, &hdr.nbKangaroo))      return false;
if (!ReadU32(f, &hdr.glvEnabled))      return false;
if (!ReadU32(f, &hdr.symmetryEnabled)) return false;
if (!ReadU32(f, &hdr.nbKeys))          return false;

// Read stats text
uint32_t textLen = 0;
if (!ReadU32(f, &textLen)) return false;
if (textLen > 0 && textLen < 65536) {
    char *buf = new char[textLen];
    if (fread(buf, 1, textLen, f) != textLen) {
        delete[] buf;
        return false;
    }
    statsText = string(buf);
    delete[] buf;
}

return true;
```

}

// ============================================================================
// BuildStatsText — generate the human-readable header block
// ============================================================================

static string BuildStatsText(
WorkfileHeader &hdr,
const string   &timestamp,
double          expectedOpsDouble,
double          stepsDouble
) {
ostringstream ss;

```
ss << "=== PhiKang Workfile v" << hdr.version << " ===\n";
ss << "Saved       : " << timestamp << "\n";

// Range
ss << "Range start : 0x";
for (int i = 3; i >= 0; i--)
    ss << hex << setw(16) << setfill('0') << hdr.rangeStart[i];
ss << "\n";
ss << "Range end   : 0x";
for (int i = 3; i >= 0; i--)
    ss << hex << setw(16) << setfill('0') << hdr.rangeEnd[i];
ss << "\n" << dec;

// Progress
ss << "DP size     : " << hdr.dpSize << " bits\n";
ss << "Kangaroos   : " << FormatPow2((double)hdr.nbKangaroo) << "\n";

if (expectedOpsDouble > 0.0) {
    double pct = (stepsDouble / expectedOpsDouble) * 100.0;
    if (pct > 100.0) pct = 100.0;
    ss << "Steps done  : " << FormatPow2(stepsDouble) << "\n";
    ss << "Expected    : " << FormatPow2(expectedOpsDouble)
       << "  (" << fixed << setprecision(1) << pct << "% complete)\n";
} else {
    ss << "Steps done  : " << FormatPow2(stepsDouble) << "\n";
    ss << "Expected    : unknown\n";
}

ss << "Collisions  : " << dec << hdr.collisions << "\n";
ss << "Same herd   : " << hdr.sameHerd << "\n";
ss << "Keys        : " << hdr.nbKeys << "\n";
ss << "GLV         : " << (hdr.glvEnabled ? "YES" : "NO") << "\n";
ss << "Symmetry    : " << (hdr.symmetryEnabled ? "YES" : "NO") << "\n";
ss << "==============================\n";

return ss.str();
```

}

// ============================================================================
// SaveWork — write kangaroo state to workfile v3.0
// ============================================================================

bool Kangaroo::SaveWork(string &fileName) {

```
FILE *f = fopen(fileName.c_str(), "wb");
if (f == NULL) {
    printf("[!] SaveWork: Cannot open %s: %s\n",
           fileName.c_str(), strerror(errno));
    return false;
}

// Gather current counts
uint64_t steps      = offsetCount;
uint64_t totalColls = collisionInSameHerd;

for (int i = 0; i < 256; i++) steps += counters[i];

// Build header
WorkfileHeader hdr;
memset(&hdr, 0, sizeof(hdr));
hdr.magic   = WORKFILE_MAGIC;
hdr.version = WORKFILE_VERSION;
hdr.dpSize  = (uint32_t)dpSize;

for (int i = 0; i < 4; i++) {
    hdr.rangeStart[i] = rangeStart.bits64[i];
    hdr.rangeEnd[i]   = rangeEnd.bits64[i];
}

hdr.expectedOps    = *((uint64_t *)&expectedNbOp);  // store as raw bits
hdr.stepsCompleted = steps;
hdr.collisions     = (uint64_t)hashTable.GetNbItem();
hdr.sameHerd       = totalColls;
hdr.nbKangaroo     = (uint32_t)totalRW;
hdr.glvEnabled     = 1;
```

#ifdef USE_SYMMETRY
hdr.symmetryEnabled = 1;
#else
hdr.symmetryEnabled = 0;
#endif
hdr.nbKeys = (uint32_t)keysToSearch.size();

```
string timestamp = GetTimestamp();
string statsText = BuildStatsText(
    hdr, timestamp, expectedNbOp, (double)steps);

if (!WriteWorkfileHeader(f, hdr, statsText)) {
    printf("[!] SaveWork: Failed to write header\n");
    fclose(f);
    return false;
}

// Write target keys
for (auto &key : keysToSearch) {
    string hexKey = secp->GetPublicKeyHex(true, key);
    uint32_t len  = (uint32_t)hexKey.length();
    if (!WriteU32(f, len)) goto writeError;
    if (fwrite(hexKey.c_str(), 1, len, f) != len) goto writeError;
}

// Write kangaroo count
if (!WriteU64(f, totalRW)) goto writeError;

// Write hash table entries (distinguished points)
{
    uint64_t nbItems = hashTable.GetNbItem();
    if (!WriteU64(f, nbItems)) goto writeError;

    ENTRY **entries = hashTable.GetEntries();
    uint64_t written = 0;
    for (uint64_t i = 0; i < HASH_SIZE && written < nbItems; i++) {
        ENTRY *e = entries[i];
        while (e != NULL) {
            // Write x (256-bit)
            for (int l = 0; l < 4; l++)
                if (!WriteU64(f, e->x.i64[l])) goto writeError;
            // Write dist (256-bit — full 4 limbs, bug fixed)
            for (int l = 0; l < 4; l++)
                if (!WriteU64(f, e->d.i64[l])) goto writeError;
            // Write type
            if (!WriteU32(f, e->kType)) goto writeError;
            written++;
            e = e->next;
        }
    }
}

fclose(f);
printf("[+] Work saved: %s (%llu kangaroos)\n",
       fileName.c_str(), (unsigned long long)totalRW);
return true;
```

writeError:
printf(”[!] SaveWork: Write error: %s\n”, strerror(errno));
fclose(f);
return false;
}

// ============================================================================
// LoadWork — load kangaroo state from workfile v3.0
// ============================================================================

bool Kangaroo::LoadWork(string &fileName) {

```
FILE *f = fopen(fileName.c_str(), "rb");
if (f == NULL) {
    printf("[!] LoadWork: Cannot open %s: %s\n",
           fileName.c_str(), strerror(errno));
    return false;
}

WorkfileHeader hdr;
string         statsText;

if (!ReadWorkfileHeader(f, hdr, statsText)) {
    fclose(f);
    return false;
}

// Print the stored stats so user can see where they left off
printf("\n%s\n", statsText.c_str());

// Restore range
rangeStart.SetInt32(0);
rangeEnd.SetInt32(0);
for (int i = 0; i < 4; i++) {
    rangeStart.bits64[i] = hdr.rangeStart[i];
    rangeEnd.bits64[i]   = hdr.rangeEnd[i];
}
rangeStart.bits64[4] = 0;
rangeEnd.bits64[4]   = 0;

// Restore DP size
initDPSize = (int32_t)hdr.dpSize;

// Restore expected ops
expectedNbOp = *((double *)&hdr.expectedOps);

// Restore offset counts for progress display
offsetCount = hdr.stepsCompleted;
offsetTime  = 0.0;

// Read target keys
keysToSearch.clear();
for (uint32_t k = 0; k < hdr.nbKeys; k++) {
    uint32_t len = 0;
    if (!ReadU32(f, &len)) goto readError;
    char *buf = new char[len + 1];
    buf[len]  = 0;
    if (fread(buf, 1, len, f) != len) {
        delete[] buf;
        goto readError;
    }
    string hexKey(buf);
    delete[] buf;
    Point p;
    bool  isCompressed;
    if (!secp->ParsePublicKeyHex(hexKey, p, isCompressed)) {
        printf("[!] LoadWork: Invalid public key in workfile\n");
        fclose(f);
        return false;
    }
    keysToSearch.push_back(p);
}

// Read kangaroo count
if (!ReadU64(f, &nbLoadedWalk)) goto readError;

// Read hash table entries
{
    uint64_t nbItems = 0;
    if (!ReadU64(f, &nbItems)) goto readError;

    hashTable.Reset();

    for (uint64_t i = 0; i < nbItems; i++) {
        int256_t x, d;
        uint32_t kType = 0;

        // Read x
        for (int l = 0; l < 4; l++)
            if (!ReadU64(f, (uint64_t *)&x.i64[l])) goto readError;

        // Read dist — full 4 limbs (bug fix: was only reading 2)
        for (int l = 0; l < 4; l++)
            if (!ReadU64(f, (uint64_t *)&d.i64[l])) goto readError;

        // Read type
        if (!ReadU32(f, &kType)) goto readError;

        Int pos, dist;
        HashTable::toInt(&x, &pos);
        HashTable::toInt(&d, &dist);
        hashTable.Add(&pos, &dist, kType);
    }

    printf("[+] Loaded %llu distinguished points from workfile\n",
           (unsigned long long)nbItems);
}

fclose(f);
return true;
```

readError:
printf(”[!] LoadWork: Read error: %s\n”, strerror(errno));
fclose(f);
return false;
}

// ============================================================================
// WorkInfo — print human-readable workfile statistics
// Called by: PhiKang -winfo mywork.phk
// ============================================================================

void Kangaroo::WorkInfo(string &fileName) {

```
FILE *f = fopen(fileName.c_str(), "rb");
if (f == NULL) {
    printf("[!] WorkInfo: Cannot open %s: %s\n",
           fileName.c_str(), strerror(errno));
    return;
}

WorkfileHeader hdr;
string         statsText;

if (!ReadWorkfileHeader(f, hdr, statsText)) {
    fclose(f);
    return;
}

fclose(f);

// Print the stored human-readable block
printf("\n%s\n", statsText.c_str());

// Additional computed stats
printf("File        : %s\n", fileName.c_str());

struct stat st;
if (stat(fileName.c_str(), &st) == 0) {
    double sizeMB = (double)st.st_size / (1024.0 * 1024.0);
    printf("File size   : %.1f MB\n", sizeMB);
}
```

}

// ============================================================================
// CheckWorkFile — verify workfile integrity
// ============================================================================

bool Kangaroo::CheckWorkFile(int nbThread, string &fileName) {

```
FILE *f = fopen(fileName.c_str(), "rb");
if (f == NULL) {
    printf("[!] CheckWorkFile: Cannot open %s\n", fileName.c_str());
    return false;
}

WorkfileHeader hdr;
string         statsText;

printf("[+] Checking %s...\n", fileName.c_str());

if (!ReadWorkfileHeader(f, hdr, statsText)) {
    fclose(f);
    printf("[!] Header invalid\n");
    return false;
}

printf("[+] Header OK\n%s\n", statsText.c_str());

// Verify we can read all hash table entries
uint32_t dummy32;
uint64_t dummy64;

// Skip keys
for (uint32_t k = 0; k < hdr.nbKeys; k++) {
    uint32_t len = 0;
    if (!ReadU32(f, &len)) { printf("[!] Key read error\n"); fclose(f); return false; }
    fseek(f, len, SEEK_CUR);
}

uint64_t nbKang = 0;
if (!ReadU64(f, &nbKang)) { printf("[!] Kangaroo count read error\n"); fclose(f); return false; }

uint64_t nbItems = 0;
if (!ReadU64(f, &nbItems)) { printf("[!] Item count read error\n"); fclose(f); return false; }

uint64_t read = 0;
bool     ok   = true;

for (uint64_t i = 0; i < nbItems && ok; i++) {
    // x (4 limbs) + dist (4 limbs) + type (1 uint32)
    for (int l = 0; l < 4 && ok; l++)
        if (!ReadU64(f, &dummy64)) ok = false;
    for (int l = 0; l < 4 && ok; l++)
        if (!ReadU64(f, &dummy64)) ok = false;
    if (ok && !ReadU32(f, &dummy32)) ok = false;
    if (ok) read++;
}

fclose(f);

if (!ok) {
    printf("[!] Integrity check FAILED at item %llu of %llu\n",
           (unsigned long long)read,
           (unsigned long long)nbItems);
    return false;
}

printf("[+] Integrity check PASSED (%llu items verified)\n",
       (unsigned long long)nbItems);
return true;
```

}

// ============================================================================
// MergeWork — merge two workfiles into a destination
// ============================================================================

bool Kangaroo::MergeWork(string &file1, string &file2, string &dest) {

```
FILE *f1 = fopen(file1.c_str(), "rb");
FILE *f2 = fopen(file2.c_str(), "rb");

if (!f1) { printf("[!] Cannot open %s\n", file1.c_str()); return false; }
if (!f2) { printf("[!] Cannot open %s\n", file2.c_str()); fclose(f1); return false; }

WorkfileHeader hdr1, hdr2;
string         stats1, stats2;

if (!ReadWorkfileHeader(f1, hdr1, stats1)) {
    printf("[!] Invalid header in %s\n", file1.c_str());
    fclose(f1); fclose(f2);
    return false;
}
if (!ReadWorkfileHeader(f2, hdr2, stats2)) {
    printf("[!] Invalid header in %s\n", file2.c_str());
    fclose(f1); fclose(f2);
    return false;
}

// Validate same range and key count
for (int i = 0; i < 4; i++) {
    if (hdr1.rangeStart[i] != hdr2.rangeStart[i] ||
        hdr1.rangeEnd[i]   != hdr2.rangeEnd[i]) {
        printf("[!] Range mismatch between workfiles\n");
        fclose(f1); fclose(f2);
        return false;
    }
}
if (hdr1.nbKeys != hdr2.nbKeys) {
    printf("[!] Key count mismatch between workfiles\n");
    fclose(f1); fclose(f2);
    return false;
}

// Load both hash tables
HashTable merged;

auto LoadTable = [&](FILE *f, WorkfileHeader &hdr) -> bool {
    // Skip keys
    for (uint32_t k = 0; k < hdr.nbKeys; k++) {
        uint32_t len = 0;
        if (!ReadU32(f, &len)) return false;
        fseek(f, len, SEEK_CUR);
    }
    uint64_t nbKang = 0;
    if (!ReadU64(f, &nbKang)) return false;
    uint64_t nbItems = 0;
    if (!ReadU64(f, &nbItems)) return false;

    for (uint64_t i = 0; i < nbItems; i++) {
        int256_t x, d;
        uint32_t kType = 0;
        for (int l = 0; l < 4; l++)
            if (!ReadU64(f, (uint64_t *)&x.i64[l])) return false;
        for (int l = 0; l < 4; l++)
            if (!ReadU64(f, (uint64_t *)&d.i64[l])) return false;
        if (!ReadU32(f, &kType)) return false;
        Int pos, dist;
        HashTable::toInt(&x, &pos);
        HashTable::toInt(&d, &dist);
        merged.Add(&pos, &dist, kType);
    }
    return true;
};

if (!LoadTable(f1, hdr1)) {
    printf("[!] Error reading %s\n", file1.c_str());
    fclose(f1); fclose(f2);
    return false;
}
if (!LoadTable(f2, hdr2)) {
    printf("[!] Error reading %s\n", file2.c_str());
    fclose(f1); fclose(f2);
    return false;
}

fclose(f1);
fclose(f2);

printf("[+] Merged %llu + %llu = %llu items\n",
       (unsigned long long)hdr1.collisions,
       (unsigned long long)hdr2.collisions,
       (unsigned long long)merged.GetNbItem());

// Write merged workfile
FILE *fOut = fopen(dest.c_str(), "wb");
if (!fOut) {
    printf("[!] Cannot open destination %s\n", dest.c_str());
    return false;
}

WorkfileHeader hdrOut = hdr1;
hdrOut.stepsCompleted = hdr1.stepsCompleted + hdr2.stepsCompleted;
hdrOut.collisions     = merged.GetNbItem();
hdrOut.sameHerd       = hdr1.sameHerd + hdr2.sameHerd;

string ts   = GetTimestamp();
double ops  = *((double *)&hdrOut.expectedOps);
string text = BuildStatsText(hdrOut, ts, ops,
                             (double)hdrOut.stepsCompleted);

if (!WriteWorkfileHeader(fOut, hdrOut, text)) {
    printf("[!] Write error on destination\n");
    fclose(fOut);
    return false;
}

// Write keys (from file1)
FILE *f1b = fopen(file1.c_str(), "rb");
if (f1b) {
    WorkfileHeader dummy; string ds;
    ReadWorkfileHeader(f1b, dummy, ds);
    for (uint32_t k = 0; k < hdr1.nbKeys; k++) {
        string hexKey = secp->GetPublicKeyHex(true, keysToSearch[k]);
        uint32_t len  = (uint32_t)hexKey.length();
        WriteU32(fOut, len);
        fwrite(hexKey.c_str(), 1, len, fOut);
    }
    fclose(f1b);
}

// Write kangaroo count + merged hash table
WriteU64(fOut, (uint64_t)hdrOut.nbKangaroo);
uint64_t nbItems = merged.GetNbItem();
WriteU64(fOut, nbItems);

ENTRY **entries = merged.GetEntries();
uint64_t written = 0;
for (uint64_t i = 0; i < HASH_SIZE && written < nbItems; i++) {
    ENTRY *e = entries[i];
    while (e != NULL) {
        for (int l = 0; l < 4; l++) WriteU64(fOut, e->x.i64[l]);
        for (int l = 0; l < 4; l++) WriteU64(fOut, e->d.i64[l]);
        WriteU32(fOut, e->kType);
        written++;
        e = e->next;
    }
}

fclose(fOut);
printf("[+] Merged workfile written: %s\n", dest.c_str());
return true;
```

}

// ============================================================================
// MergeDir — merge all workfiles in a directory
// ============================================================================

bool Kangaroo::MergeDir(string &dir, string &dest) {

#ifdef WIN64
WIN32_FIND_DATA ffd;
string pattern = dir + “\*.phk”;
HANDLE hFind   = FindFirstFile(pattern.c_str(), &ffd);
if (hFind == INVALID_HANDLE_VALUE) {
printf(”[!] MergeDir: No .phk files in %s\n”, dir.c_str());
return false;
}
vector<string> files;
do {
files.push_back(dir + “\” + string(ffd.cFileName));
} while (FindNextFile(hFind, &ffd));
FindClose(hFind);
#else
DIR *d = opendir(dir.c_str());
if (!d) {
printf(”[!] MergeDir: Cannot open directory %s\n”, dir.c_str());
return false;
}
vector<string> files;
struct dirent *entry;
while ((entry = readdir(d)) != NULL) {
string name(entry->d_name);
if (name.length() > 4 &&
name.substr(name.length() - 4) == “.phk”)
files.push_back(dir + “/” + name);
}
closedir(d);
#endif

```
if (files.empty()) {
    printf("[!] MergeDir: No .phk files found in %s\n", dir.c_str());
    return false;
}

sort(files.begin(), files.end());
printf("[+] MergeDir: Found %zu workfiles\n", files.size());

// Iteratively merge pairs
string current = files[0];
for (size_t i = 1; i < files.size(); i++) {
    string tmp = dest + ".tmp";
    if (!MergeWork(current, files[i], tmp)) {
        printf("[!] MergeDir: Failed merging %s with %s\n",
               current.c_str(), files[i].c_str());
        return false;
    }
    if (i > 1) remove(current.c_str()); // clean up intermediate
    current = tmp;
}

// Rename final result to dest
remove(dest.c_str());
rename(current.c_str(), dest.c_str());
printf("[+] MergeDir: Final workfile: %s\n", dest.c_str());
return true;
```

}

// ============================================================================
// CreateEmptyPartWork — create empty partitioned workfile directory
// ============================================================================

void Kangaroo::CreateEmptyPartWork(string &dirName) {

#ifdef WIN64
CreateDirectory(dirName.c_str(), NULL);
#else
mkdir(dirName.c_str(), 0755);
#endif

```
for (int i = 0; i < MERGE_PART; i++) {
    char name[512];
```

#ifdef WIN64
snprintf(name, sizeof(name), “%s\part%03d.phk”,
dirName.c_str(), i);
#else
snprintf(name, sizeof(name), “%s/part%03d.phk”,
dirName.c_str(), i);
#endif
FILE *f = fopen(name, “wb”);
if (f) fclose(f);
}

```
printf("[+] Created empty partitioned workfile: %s (%d parts)\n",
       dirName.c_str(), MERGE_PART);
```

}
