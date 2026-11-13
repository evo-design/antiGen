// Step 3's input: the open-data sequences that are eligible for placement.
//
// Selecting them means one pass over a 12 GB FASTA, so the result is written to
// an intermediate file and the placement engine reads that instead. The file is
// in a deterministic order -- date, then strain name -- and carries no seed:
// the within-day shuffle belongs to `place`, so the expensive pass does not have
// to be repeated to try another seed.
//
// Three filters, all from NOTES:
//   date      collection date strictly before the cutoff (cutoff 0: no filter)
//   coverage  at least minCoveredFrac of the nSites residues non-missing
//   clock     the divergence from the pinned root has to be plausible for the
//             collection date the record carries, to within a tolerance of
//             clockTolMonths. "Plausible" is measured off the data rather than
//             assumed -- see ClockEnvelope -- because no rate is constant over
//             the whole period: Omicron adds ~20 S substitutions in one month,
//             which no line through 2020 or through 2026 can bound without
//             either deleting it or admitting a 2022 sequence as a 2020 one.
//             useClock = false turns it off
#pragma once

#include "alphabet.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Dates are held as YYYYMMDD so that ordering is numeric and no float is
// involved in the comparison the cutoff makes.
//
// YYYY-MM-DD, with or without the zero padding on the month and the day: the
// dengue metadata carries fifteen rows written '2023-5-6' or '2024-05-2', and
// a length test rejecting those loses them for no reason at all. `imputeDay`,
// when non-zero, additionally accepts YYYY-MM and puts the record on that day
// of the month -- which the dengue arm needs and nothing else uses. A year on
// its own is never accepted: both the quarterly cutoff and the clock envelope
// work at the resolution of a month, so imputing a *day* cannot move a record
// across either and imputing a *month* could.
int32_t parseDateCodeFlex(const char* s, size_t len, int imputeDay);
int32_t parseDateCode(const char* s, size_t len);
double  decimalYear(int32_t dateCode);
// Midnight at the start of that day, rather than the middle of it. A date used
// as a threshold wants this: a sample collected on the day is then at or after
// it, and so is an inferred node placed anywhere within the day.
double  decimalYearStart(int32_t dateCode);

// The longest run of '-' still read as a deletion rather than as missing
// coverage. There is no published maximum plausible in-frame deletion, so this
// is an engineering threshold, not a convention: it is the smallest value that
// admits every adjacent-deletion run present on any of the Nextstrain backbones
// this tool reads (the longest is 10, on ncov's 2026-03-15 tree). See NOTES.
inline constexpr int kDefaultMaxGapRun = 10;

struct PrepareConfig {
    // One FASTA per gene, in the layout's order, and the site count each of
    // them must carry. A single gene streams; several are joined by strain
    // name, because the per-CDS translations are neither in the same order nor
    // over quite the same strains -- h3n2's HA1 has one record HA2 does not.
    std::vector<std::string> fasta;
    std::vector<int>         geneLength;
    std::string metadata;
    std::string rootSeq;      // pinned root, nSites characters
    int         nSites      = 0;
    int32_t     cutoff      = 20210101;  // keep collection dates strictly before this;
                                         // 0 keeps every dated sequence
    // A fraction of nSites rather than a count, so that it is the same criterion
    // on a 1,273-site spike and a 550-site HA. 0.9 of 1,273 rounds up to the
    // 1,146 this was written as.
    double      minCoveredFrac = 0.9;
    // Year-month collection dates: the day to put them on, or 0 to drop them.
    int         dateImputeDay  = 0;
    // --alphabet 22: whether a '-' in the queue is read as a deletion at all,
    // and the longest run of them that still is. See classifyGaps.
    bool        gapsAreStates  = false;
    int         maxGapRun      = kDefaultMaxGapRun;
    bool        useClock    = true;      // false drops the clock filter entirely
    // Write the kept records in the FASTA's own order instead of by date. For a
    // benchmark that must place in exactly the order another tool does; 'place'
    // needs --keep-order too, or it shuffles within each day.
    bool        keepOrder   = false;
    // The clock envelope. The quantiles are deliberately asymmetric: an
    // isolated month whose upper tail is contaminated is clipped by its own
    // future, since the envelope is made non-decreasing, so the upper edge can
    // sit near the maximum. The lower edge has no such protection -- a lineage
    // that lingers genuinely lowers it -- so it is a quantile that the
    // contamination cannot reach.
    int         clockTolMonths = 3;      // the date may be off by this much
    int         clockSlack     = 3;      // substitutions of slack on the upper edge
    double      clockHiQ       = 0.9999; // upper quantile of a month's divergence
    double      clockLoQ       = 0.001;  // lower quantile
    long long   clockMinN      = 2000;   // widen a thin month's window to at least this

    int minCovered() const;
};

// What divergence from the root is plausible at a given collection date,
// calibrated from the data itself: per calendar month, a low and a high
// quantile of the divergence over every coverage-passing record, with the high
// curve made non-decreasing.
//
// Non-decreasing is both true of the real envelope -- a lineage cannot be
// around before it arises, so the most diverged thing circulating only grows --
// and what makes the calibration robust: a month whose upper tail is nothing
// but mislabelled recent sequences is clipped by the months that follow it,
// which is how January 2020 is bounded at 9 rather than at its own 52.
//
// A record is kept when its divergence falls inside the band of *some* month
// within tolMonths of the date it carries, which is the plain statement of
// "this divergence is reasonable for this date". Because the high curve is
// non-decreasing that reduces to a per-month interval:
//
//     lo(t) = min over u >= t - tol of loQuantile(u)
//     hi(t) = hiCurve(t + tol) + slack
struct ClockEnvelope {
    int              firstMonth = 0;   // y*12 + (m-1) of the first month with data
    std::vector<int> hi;               // the non-decreasing upper curve, per month
    std::vector<int> loSuffixMin;      // suffix minima of the lower quantile
    int              tolMonths = 0;
    int              slack     = 0;

    bool empty() const { return hi.empty(); }
    // The interval a record collected in `month` (y*12 + (m-1)) must fall in.
    void bandAt(int month, int& lo, int& hiOut) const;
};

// Absolute month index, so that December does not wrap into the next year.
int monthIndex(int32_t dateCode);

// What classifyGaps did, over every record it was handed -- before the coverage
// test, so this counts the whole queue and not just the kept part of it.
struct GapCalls {
    long long keptRuns     = 0, keptSites     = 0;  // read as deletions
    long long terminalRuns = 0, terminalSites = 0;  // masked: touching a gene's end
    long long longRuns     = 0, longSites     = 0;  // masked: longer than maxGapRun
};

// Decide, in place and one gene at a time, which of a queue sequence's '-' are
// deletions and which are regions the submitter never sequenced. Nextclade
// writes both as '-': its own docs define an unsequenced region as "consecutive
// gap ('-') character ranges on the 5' and 3' ends", and only a fully-gapped
// codon '---' becomes '-' in the translation, so a truncated submission arrives
// looking exactly like a 400-residue deletion.
//
// The rule, which is the one the Nextstrain stack applies to its own
// alignments -- TreeTime's fill_overhangs, on by default, augur's
// --keep-overhangs to switch it off:
//
//   a run touching either end of the gene   -> missing coverage, rewritten to X
//   an internal run of at most maxRun       -> a deletion, left as '-'
//   a longer internal run                   -> missing coverage, rewritten to X
//
// Per gene and before the concatenation, or h3n2's HA1/HA2 seam would turn two
// terminal runs into one internal one. Rewriting to 'X' rather than dropping
// the record is what keeps the coverage filter meaning what it meant: with '-'
// a state it would otherwise stop counting against coverage, and a half-empty
// record would start passing.
void classifyGaps(char* seq, int len, int maxRun, GapCalls& g);

struct PrepareStats {
    long long metadataRows   = 0;
    long long dated          = 0;
    long long beforeCutoff   = 0;
    long long fastaRecords   = 0;
    long long matched        = 0;  // in-window strains found in the FASTA
    long long badLength      = 0;
    long long joined         = 0;  // multi-gene only: strains assembled from the files
    long long partial        = 0;  // of those, the ones some gene's file did not hold
    long long failCoverage   = 0;
    long long failClockHigh  = 0;  // too diverged for the date it carries
    long long failClockLow   = 0;  // too ancestral for it
    long long kept           = 0;
    long long missingTotal   = 0;  // missing sites summed over the kept records
    long long fullyCovered   = 0;
    GapCalls  gaps;                // --alphabet 22 only; all zero otherwise
};

// Sequences are concatenated into one blob rather than held as 351k separate
// strings; the records index into it.
struct PreparedSet {
    struct Record {
        std::string strain;
        int32_t     dateCode = 0;
        size_t      offset   = 0;  // into blob
    };

    int                 nSites = 0;
    // The alphabet the file was written under, from its #alphabet header. A file
    // written before that header existed carries no '-' or '*' state by
    // construction, so its absence means the 20-token alphabet.
    std::string         alphabet = kAlphabet20;
    std::string         blob;
    std::vector<Record> rec;

    size_t      size() const { return rec.size(); }
    const char* seq(size_t i) const { return blob.data() + rec[i].offset; }
};

// One streaming pass over the FASTA, after reading the metadata into a table of
// the strains inside the date window.
// `envelope`, when given, receives the calibration the run used, for reporting.
PreparedSet prepareSequences(const PrepareConfig& cfg, PrepareStats* stats,
                             ClockEnvelope* envelope = nullptr);

void writeClockEnvelope(const ClockEnvelope& env, const std::string& path);

void        writePrepared(const PreparedSet& set, const std::string& path);
PreparedSet readPrepared(const std::string& path);

// A (site, amino acid) count matrix over the sequences that got through the
// filters -- the queue, not the tree. Every record counts once, whether it was
// attached as a leaf or matched a node and was counted there, since what it
// summarises is the data that entered, not what the placement did with it.
//
// Only what the alphabet names is counted: a character outside it names no state
// here any more than it does anywhere else, so it increments nothing and a
// site's row sums to the number of sequences covering that site. That is '-',
// '*' and X at 20 tokens; at 22 only X is left out and the matrix is 22 columns
// wide.
struct ResidueCounts {
    int                    nSites    = 0;
    std::vector<long long> n;              // nSites * numStates(), site-major
    long long              sequences = 0;
    long long              residues  = 0;  // residues in the alphabet, counted

    void init(int sites);
    void add(const char* seq);  // one record, nSites characters

    long long at(int site1, int aa) const {
        return n[static_cast<size_t>(site1 - 1) * numStates() + static_cast<size_t>(aa)];
    }
};

void writeResidueCounts(const ResidueCounts& counts, const std::string& path);

// The same sequences again, as the alignment they came from: one `>strain` line
// and one nSites-character line each, which is the ingested FASTA's own layout.
// Nothing is folded here -- gaps, stops and X are written as they arrived --
// so this is the data the count matrix was built from, missing data included.
// The width is the tool's site count: a terminal stop is dropped at the parse
// and is not a site anywhere else either, so a column of it would not line up
// with the matrix.
void writeMsaRecord(std::FILE* f, const std::string& strain, const char* seq, int nSites);
void writeMsa(const PreparedSet& set, const std::string& path);
