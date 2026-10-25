// antiGen-data-revised: annotate the Nextstrain tree with parsimony-consistent
// S-protein allele sets, place new sequences on it, then resolve and collapse the
// result into a mutation tree of training examples. Nothing is collapsed and no
// node is committed to a genotype until that last step.
#include "alphabet.hpp"
#include "examples.hpp"
#include "hartigan.hpp"
#include "tree.hpp"
#include "placement.hpp"
#include "sequences.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

const char* kUsage =
    "usage: antigen <command> --tree <dataset.json> --root-seq <root-sequence.json> [options]\n"
    "\n"
    "commands:\n"
    "  tree       read the Nextstrain tree and report it, uncollapsed\n"
    "  sets       the above, plus Hartigan allele sets at every (node, site)\n"
    "  prepare    select the open-data sequences eligible for placement\n"
    "  place      append prepared sequences to the tree by parsimony placement\n"
    "\n"
    "options:\n"
    "  --tree <path>      Nextstrain v2 dataset JSON (tree, sets, place)\n"
    "  --root-seq <path>  root-sequence sidecar JSON (required)\n"
    "  --gene <key>       the CDS to read: the root-sequence key and the\n"
    "                     branch_attrs.mutations key, e.g. 'S' (default), 'E', 'G',\n"
    "                     'F', 'H'. nSites follows the sidecar entry's length\n"
    "  --out <path>       TSV out: the tree, the allele sets, or the prepared\n"
    "                     sequences\n"
    "  --out-tree <path>  for 'sets' and 'place', also write the tree here\n"
    "  --out-examples <p> for 'sets' and 'place', also write the examples file:\n"
    "                     one allele drawn per (node, site), one mutation per edge,\n"
    "                     positions 0-based\n"
    "  --fasta <path>     for 'prepare': the aligned spike FASTA\n"
    "  --metadata <path>  for 'prepare': strain/date TSV alongside it\n"
    "  --prepared <path>  for 'place': what 'prepare' wrote\n"
    "  --seed <n>         the within-day shuffle and tie-breaking for 'place', and\n"
    "                     the allele draw and mutation order for --out-examples\n"
    "  --tie-seed <n>     for 'place': seed the placement tie-break separately from\n"
    "                     the queue shuffle, to isolate the two sources of variation\n"
    "  --limit <n>        for 'place': stop after n sequences\n"
    "  --threads <n>      for 'place': cores to run the placement scan on, which\n"
    "                     is ~85% of the work. The tree written does not depend\n"
    "                     on the count: results are merged in tree order, not in\n"
    "                     completion order (default 1)\n"
    "  --date-cutoff <d>  for 'prepare': keep collection dates before YYYY-MM-DD,\n"
    "                     or 'none' to keep every dated sequence (default 2021-01-01)\n"
    "  --no-clock         for 'prepare': drop the clock filter entirely\n"
    "  --clock-tol <n>    for 'prepare': how many months the collection date may be\n"
    "                     off by before the divergence counts against it (default 3)\n"
    "  --clock-slack <n>  for 'prepare': substitutions of slack on the envelope's\n"
    "                     upper edge (default 3)\n"
    "  --out-clock <p>    for 'prepare': the calibrated envelope, month by month\n"
    "  --variant <name>   for 'sets' and 'place': the clade whose root to flag in the\n"
    "                     examples, e.g. 'BA.2'. Matched as the whole clade label or\n"
    "                     as '(name)' inside it; it must begin at exactly one node\n"
    "  --post-cutoff <d>  for 'sets' and 'place': a date, or a comma-separated list\n"
    "                     of them, the examples' POST_CUTOFF flags are taken\n"
    "                     against; repeatable. A record lists each one every\n"
    "                     observed date below it is at or after, as '<d>,True'.\n"
    "                     Default: none at all, and the field is written empty --\n"
    "                     which is what a train tree wants, since it stops before\n"
    "                     its own cutoff. Give the whole series here on the test\n"
    "                     tree, which holds every sequence and so can answer for\n"
    "                     every cutoff at once\n"
    "  --out-counts <p>   for 'prepare' and 'place': a (site x 20) count matrix over\n"
    "                     the sequences that got through the filters, one increment\n"
    "                     per standard residue. Missing data counts for nothing\n"
    "  --out-msa <path>   for 'prepare' and 'place': those same sequences as a FASTA,\n"
    "                     gaps and stops included -- the alignment the matrix counts\n"
    "  --out-matches <p>  for 'place': TSV of the sequences that matched a node's\n"
    "                     allele sets exactly and were counted there, not attached\n"
    "  --attach-matches   for 'place': attach those as leaves anyway, as the rule\n"
    "                     before 2026-10-22 did. Slower, and a much larger tree\n"
    "  --verify           for 'place': recompute the sets from scratch after every\n"
    "                     placement and compare (small --limit only)\n"
    "\n"
    "The terminal stop codon is dropped, leaving the 1,273 residues of S. Gaps,\n"
    "stop codons and X are missing data. Observations are the input tree's leaves\n"
    "and the placed sequences; the root is pinned to the sidecar; every other node\n"
    "is free. The mutation tree exists only in the examples file.\n";

struct Options {
    std::string command;
    std::string tree;
    std::string rootSeq;
    std::string gene = "S";  // the CDS to read: root-sequence key and branch_attrs.mutations key
    std::string out;
    std::string outTree;
    std::string outExamples;
    std::string fasta;
    std::string metadata;
    std::string prepared;
    std::string outMatches;
    std::string outCounts;
    std::string outMsa;
    bool        attachMatches = false;
    long long   dateCutoff    = 20210101;  // for 'prepare'; 0 means keep every date
    bool        noClock       = false;     // for 'prepare'
    long long   clockTol      = -1;        // for 'prepare'; < 0 means the default
    long long   clockSlack    = -1;        // for 'prepare'; < 0 means the default
    std::string outClock;                  // for 'prepare': dump the calibration
    std::string variant;                   // for 'sets'/'place': the clade to flag
    std::vector<std::string> postCutoffs;  // for 'sets'/'place': the POST_CUTOFF dates;
                                           // empty means the field is written empty
    uint64_t    seed   = 0;
    long long   tieSeed = -1;  // < 0 means "follow --seed"
    long long   limit  = -1;
    long long   threads = 1;   // for 'place': cores on the scan
    bool        verify = false;
};

int sitesOf(const std::string& rootS) {
    int n = static_cast<int>(rootS.size());
    if (n > 0 && rootS[n - 1] == '*') --n;
    return n;
}

// The variant root, resolved once from the input tree and reported. Throws out
// of findCladeRoot if the clade is absent or begins at more than one node, which
// is a stop rather than a choice: a clade with two entry points is not a variant
// root and nothing downstream could read it as one.
ExampleOptions exampleOptions(const Options& opt, const std::vector<RawNode>& raw) {
    ExampleOptions eo;
    eo.seed = opt.seed;
    for (const std::string& d : opt.postCutoffs) {
        const int32_t code = parseDateCode(d.c_str(), d.size());
        if (!code) throw std::runtime_error("--post-cutoff wants YYYY-MM-DD, got '" + d + "'");
        eo.cutoffs.push_back({d, decimalYearStart(code)});
    }
    // Ascending and deduplicated, which is what makes the flags a record carries
    // a prefix of this list rather than an arbitrary subset of it.
    std::sort(eo.cutoffs.begin(), eo.cutoffs.end(),
              [](const Cutoff& a, const Cutoff& b) { return a.when < b.when; });
    eo.cutoffs.erase(std::unique(eo.cutoffs.begin(), eo.cutoffs.end(),
                                 [](const Cutoff& a, const Cutoff& b) { return a.label == b.label; }),
                     eo.cutoffs.end());
    if (eo.cutoffs.empty()) {
        std::printf("post-cutoff  : none given -- POST_CUTOFF is written empty on every record\n");
    } else {
        std::printf("post-cutoff  : %zu cutoffs, %s (%.4f) .. %s (%.4f) -- each record lists\n"
                    "               the ones every observed date below it is at or after\n",
                    eo.cutoffs.size(), eo.cutoffs.front().label.c_str(), eo.cutoffs.front().when,
                    eo.cutoffs.back().label.c_str(), eo.cutoffs.back().when);
    }
    if (!opt.variant.empty()) {
        std::string      label;
        int              nMatched = 0;
        std::vector<int> entries;
        eo.variantRoot  = findCladeRoot(raw, opt.variant, &label, &nMatched, &entries);
        eo.variantLabel = label;
        std::printf("variant      : '%s' matches clade '%s' on %d nodes, which begin at\n"
                    "               exactly one of them: node %d (%s)\n",
                    opt.variant.c_str(), label.c_str(), nMatched, eo.variantRoot,
                    raw[eo.variantRoot].name.c_str());
    }
    return eo;
}

void reportExamples(const Tree& t, const AlleleSets& sets, const std::string& rootS,
                    const ExampleOptions& eo, const std::string& path) {
    const ExampleStats e = writeExamples(t, sets, rootS, eo, path);
    std::printf("examples     : %lld records, %lld mutations, one per edge\n", e.examples,
                e.mutations);
    std::printf("               %lld childless nodes left out, carrying nothing to predict\n",
                e.leafRecords);
    std::printf("               %lld resolved nodes (%lld edges collapsed to nothing),\n",
                e.resolvedNodes, e.collapsedEdges);
    std::printf("               %lld split from %lld multi-mutation edges (longest %d),\n",
                e.intermediates, e.splitEdges, e.maxEdgeMut);
    std::printf("               %lld (node, site) alleles drawn from more than one option\n",
                e.drawn);
    for (size_t k = 0; k < eo.cutoffs.size(); ++k) {
        std::printf("               %s: %lld records at or after it, %lld before it\n",
                    eo.cutoffs[k].label.c_str(), e.postCutoff[k],
                    e.examples - e.postCutoff[k]);
    }
    if (eo.variantRoot >= 0) {
        std::printf("               %lld inside the variant, %lld of them the variant root\n",
                    e.descended, e.variantRoots);
    }
    std::printf("wrote        : %s\n", path.c_str());
}

// The matrix is a summary of the queue, so what is worth printing is the two
// numbers that say the queue is the one it was meant to be: how many sequences
// went in, and how much of them was actually readable.
void reportCounts(const ResidueCounts& counts) {
    const double perSeq = counts.sequences ? static_cast<double>(counts.residues) /
                                                 static_cast<double>(counts.sequences)
                                           : 0.0;
    std::printf("counts       : %lld sequences, %lld standard residues (mean %.1f of %d per\n"
                "               sequence); gaps, stops and X count for nothing\n",
                counts.sequences, counts.residues, perSeq, counts.nSites);
}

double secondsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool needValue(int argc, char** argv, int i, const char* flag, std::string& dst) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", flag);
        return false;
    }
    dst = argv[i + 1];
    return true;
}

// The tree, as given. Input node ids are tree node ids; the only thing derived
// here is each leaf's sequence, from the mutation path down to it.
Tree loadTree(const Options& opt, const std::string& rootS, int nSites,
              std::vector<RawNode>* rawOut) {
    std::vector<RawNode> raw = parseNextstrainTree(opt.tree, opt.gene, nSites);

    size_t rawTips = 0, rawEdgesWithMut = 0, rawMutCount = 0, maxDeg = 0;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i].children.empty()) ++rawTips;
        maxDeg = std::max(maxDeg, raw[i].children.size());
        if (i > 0 && !raw[i].sMut.empty()) ++rawEdgesWithMut;
        if (i > 0) rawMutCount += raw[i].sMut.size();
    }
    std::printf("%-5s gene    : %zu aa in the sidecar, %d sites kept\n",
                opt.gene.c_str(), rootS.size(), nSites);
    std::printf("tree         : %zu nodes, %zu tips, %zu edges, largest multifurcation %zu\n",
                raw.size(), rawTips, raw.size() - 1, maxDeg);
    std::printf("               %zu edges carry S mutations (%zu mutations total); the other\n",
                rawEdgesWithMut, rawMutCount);
    std::printf("               %zu carry none and are kept -- nothing is collapsed here\n",
                raw.size() - 1 - rawEdgesWithMut);

    std::vector<std::string> bad;
    Tree t = buildTree(raw, rootS, nSites, &bad);

    size_t observed = 0, obsEntries = 0, missingObs = 0;
    for (int i = 0; i < t.size(); ++i) {
        if (t.observed(i)) ++observed;
        obsEntries += t.obs[i].size();
        for (const ObsAllele& o : t.obs[i]) if (isMissing(o.aa)) ++missingObs;
    }
    std::printf("observations : %zu nodes (%zu leaves plus the pinned root); %d free\n",
                observed, rawTips, t.size() - static_cast<int>(observed));
    std::printf("               %zu residues recorded against the root, %zu of them missing\n",
                obsEntries, missingObs);
    if (bad.empty()) {
        std::printf("               every branch's from-allele matches the path above it\n");
    } else {
        std::printf("               %zu from-allele mismatches, first few:\n", bad.size());
        for (size_t i = 0; i < bad.size() && i < 5; ++i)
            std::printf("                 %s\n", bad[i].c_str());
    }
    if (observed != rawTips + 1) throw std::runtime_error("tips were lost building the tree");
    if (rawOut) *rawOut = std::move(raw);
    return t;
}

int runTree(const Options& opt) {
    std::string rootS = parseRootSequenceS(opt.rootSeq, opt.gene);
    int nSites = sitesOf(rootS);

    Tree t = loadTree(opt, rootS, nSites, nullptr);

    std::unordered_set<std::string> distinct;
    std::string                     seq;
    for (int i = 0; i < t.size(); ++i) {
        if (!t.observed(i)) continue;
        seq.assign(rootS.begin(), rootS.begin() + nSites);
        for (const ObsAllele& o : t.obs[i]) seq[o.site - 1] = o.aa;
        distinct.insert(seq);
    }
    std::printf("sequences    : %zu distinct observed sequences\n", distinct.size());

    if (!opt.out.empty()) {
        writeTree(t, opt.out);
        std::printf("wrote        : %s\n", opt.out.c_str());
    }
    return 0;
}

int runSets(const Options& opt) {
    std::string rootS = parseRootSequenceS(opt.rootSeq, opt.gene);
    int nSites = sitesOf(rootS);

    std::vector<RawNode> raw;
    Tree t = loadTree(opt, rootS, nSites, &raw);

    size_t nsMut = 0, missingMut = 0;
    for (size_t i = 1; i < raw.size(); ++i) {
        nsMut += raw[i].sMut.size();
        for (const Mutation& m : raw[i].sMut) {
            if (isMissing(m.from) || isMissing(m.to)) ++missingMut;
        }
    }

    AlleleSets sets = computeAlleleSets(t, rootS, nSites);

    size_t active = 0;
    for (int s = 0; s < nSites; ++s) if (sets.siteActive[s]) ++active;
    size_t deltas = 0, nodesWithDelta = 0;
    for (int i = 0; i < t.size(); ++i) {
        deltas += sets.delta[i].size();
        if (!sets.delta[i].empty()) ++nodesWithDelta;
    }

    std::printf("allele sets  : %zu of %d sites carry more than one allele (the rest are\n",
                active, nSites);
    std::printf("               a single fixed state everywhere, at zero storage cost)\n");
    std::printf("               parsimony score %lld changes, against %zu Nextstrain\n",
                sets.parsimonyScore, nsMut);
    std::printf("               mutations on the same tree (%zu of them touch missing data,\n",
                missingMut);
    std::printf("               which is free here)\n");
    std::printf("               %zu set-deltas over %zu of %d nodes, %zu bytes of payload\n",
                deltas, nodesWithDelta, t.size(), deltas * sizeof(SetDelta));
    size_t v12s = 0, nodesWithV12 = 0;
    for (int i = 0; i < t.size(); ++i) {
        v12s += sets.v12[i].size();
        if (!sets.v12[i].empty()) ++nodesWithV12;
    }
    std::printf("               %zu V1/V2 entries over %zu nodes, %zu bytes of payload\n",
                v12s, nodesWithV12, v12s * sizeof(V12Delta));

    // Materialise every node to check the sets and count the ambiguity. The
    // observations must be inside their own sets; Nextstrain's genotype at a free
    // node is an inference this tree does not honour, so how often it lands inside
    // the set is a statistic, not a check.
    const std::vector<std::string> nsGeno = nextstrainGenotypes(raw, rootS, nSites);
    size_t ambigPairs = 0, ambigNodes = 0, violations = 0, nsOutside = 0, nsOutsideNodes = 0;
    size_t hist[kNumStates + 1] = {0};
    std::vector<std::string> examples;
    walkMaterialised(t, sets, [&](int node, const AaSet* cur) {
        size_t here = 0, outside = 0;
        for (int s = 0; s < nSites; ++s) {
            int n = setSize(cur[s]);
            ++hist[n];
            if (n > 1) {
                ++here;
                if (examples.size() < 6) {
                    examples.push_back("node " + std::to_string(node) + " site " +
                                       std::to_string(s + 1) + " = {" + setToString(cur[s]) +
                                       "}, Nextstrain says " + std::string(1, nsGeno[node][s]));
                }
            }
            if (t.observed(node)) {
                const char a = alleleAt(t, node, s + 1, rootS);
                if (!isMissing(a) && !(cur[s] & aaBit(a))) ++violations;
            }
            const char g = nsGeno[node][s];
            if (!isMissing(g) && !(cur[s] & aaBit(g))) ++outside;
        }
        ambigPairs += here;
        if (here) ++ambigNodes;
        nsOutside += outside;
        if (outside) ++nsOutsideNodes;
    });

    std::printf("ambiguity    : %zu of %lld (node, site) pairs admit more than one allele\n",
                ambigPairs, static_cast<long long>(t.size()) * nSites);
    std::printf("               spread over %zu of %d nodes\n", ambigNodes, t.size());
    std::printf("               set sizes:");
    for (int n = 1; n <= kNumStates; ++n) if (hist[n]) std::printf("  %d:%zu", n, hist[n]);
    std::printf("\n");
    for (const std::string& e : examples) std::printf("                 %s\n", e.c_str());
    std::printf("check        : %zu observations outside their own allele set\n", violations);
    std::printf("               %zu (node, site) pairs where Nextstrain's own reconstruction\n",
                nsOutside);
    std::printf("               is outside the set, over %zu nodes -- a statistic: the interior\n",
                nsOutsideNodes);
    std::printf("               is free here, so that reconstruction is not binding\n");

    if (!opt.outExamples.empty()) {
        reportExamples(t, sets, rootS, exampleOptions(opt, raw), opt.outExamples);
    }

    if (!opt.outTree.empty()) {
        writeTree(t, opt.outTree);
        std::printf("wrote        : %s\n", opt.outTree.c_str());
    }
    if (!opt.out.empty()) {
        writeAlleleSets(t, sets, opt.out);
        std::printf("wrote        : %s\n", opt.out.c_str());
    }
    return 0;
}


int runPrepare(const Options& opt) {
    const std::string rootS  = parseRootSequenceS(opt.rootSeq, opt.gene);
    PrepareConfig     cfg;
    cfg.fasta    = opt.fasta;
    cfg.metadata = opt.metadata;
    cfg.nSites   = sitesOf(rootS);
    cfg.rootSeq  = rootS.substr(0, cfg.nSites);
    // Coverage floor is 90% of the gene length, not a baked SC2 constant
    // (0.9 * 1273 = 1146 is preserved exactly for spike).
    cfg.minCovered = static_cast<int>(std::lround(0.9 * cfg.nSites));
    cfg.cutoff   = static_cast<int32_t>(opt.dateCutoff);
    cfg.useClock = !opt.noClock;
    if (opt.clockTol   >= 0) cfg.clockTolMonths = opt.clockTol;
    if (opt.clockSlack >= 0) cfg.clockSlack     = opt.clockSlack;

    const auto    t0 = std::chrono::steady_clock::now();
    PrepareStats  st;
    ClockEnvelope env;
    PreparedSet   set = prepareSequences(cfg, &st, &env);

    if (cfg.cutoff) {
        std::printf("metadata     : %lld rows, %lld dated, %lld before %d\n", st.metadataRows,
                    st.dated, st.beforeCutoff, cfg.cutoff);
    } else {
        std::printf("metadata     : %lld rows, %lld dated, no date cutoff\n", st.metadataRows,
                    st.dated);
    }
    std::printf("fasta        : %lld records, %lld of them dated, %lld in the date window\n",
                st.fastaRecords, st.matched, st.beforeCutoff);
    if (st.badLength) std::printf("               %lld of a length other than %d\n",
                                  st.badLength, cfg.nSites);
    std::printf("filters      : %lld fail coverage (<%d of %d sites)\n",
                st.failCoverage, cfg.minCovered, cfg.nSites);
    if (cfg.useClock) {
        std::printf("               %lld too diverged for their date, %lld too ancestral\n",
                    st.failClockHigh, st.failClockLow);
        std::printf("clock        : %d months of calibration, +-%d months of tolerance,\n"
                    "               %d substitutions of slack on the upper edge\n",
                    env.empty() ? 0 : static_cast<int>(env.hi.size()), cfg.clockTolMonths,
                    cfg.clockSlack);
    } else {
        std::printf("               the clock is off\n");
    }
    std::printf("attached     : %lld (%.2f%%)\n", st.kept,
                st.beforeCutoff
                    ? 100.0 * static_cast<double>(st.kept) / static_cast<double>(st.beforeCutoff)
                    : 0.0);
    if (!opt.outClock.empty() && !env.empty()) {
        writeClockEnvelope(env, opt.outClock);
        std::printf("wrote        : %s\n", opt.outClock.c_str());
    }
    if (st.kept) {
        std::printf("missing sites: mean %.2f, %lld of the kept are fully covered (%.1f%%)\n",
                    static_cast<double>(st.missingTotal) / static_cast<double>(st.kept),
                    st.fullyCovered,
                    100.0 * static_cast<double>(st.fullyCovered) / static_cast<double>(st.kept));
    }
    std::printf("read         : %.1f s\n", secondsSince(t0));

    if (!opt.outCounts.empty()) {
        // The kept set is already in memory, so this is a pass over it rather
        // than a second pass over the FASTA.
        ResidueCounts counts;
        counts.init(set.nSites);
        for (size_t i = 0; i < set.size(); ++i) counts.add(set.seq(i));
        reportCounts(counts);
        writeResidueCounts(counts, opt.outCounts);
        std::printf("wrote        : %s\n", opt.outCounts.c_str());
    }

    if (!opt.outMsa.empty()) {
        writeMsa(set, opt.outMsa);
        std::printf("msa          : %zu sequences x %d sites, as they arrived\n", set.size(),
                    set.nSites);
        std::printf("wrote        : %s\n", opt.outMsa.c_str());
    }

    if (!opt.out.empty()) {
        writePrepared(set, opt.out);
        std::printf("wrote        : %s\n", opt.out.c_str());
    }
    return 0;
}

// Everything known about one (node, site), from both the incremental state and
// the recompute, so a verification failure says what actually diverged.
void dumpSite(const Tree& t, const AlleleSets& a, const AlleleSets& b, int n, int site1,
              const std::string& rootS) {
    auto wOf = [&](const AlleleSets& st, int node) {
        AaSet w = st.rootSet[site1 - 1];
        std::vector<int> up;
        for (int x = node; x != -1; x = t.parent[x]) up.push_back(x);
        for (size_t i = up.size(); i-- > 0;) {
            for (const SetDelta& d : st.delta[up[i]]) if (d.site == site1) w = d.set;
        }
        return w;
    };
    auto v12Of = [&](const AlleleSets& st, int node) {
        for (const V12Delta& d : st.v12[node]) {
            if (d.site == site1) return std::string("{") + setToString(d.v1) + "}/{" +
                                        setToString(d.v2) + "}";
        }
        return std::string("{") + setToString(aaBit(alleleAt(t, node, site1, rootS))) +
               "}/{} (default)";
    };
    auto line = [&](int node) {
        std::printf("    node %7d  parent %7d  kids %2zu  %s %c  "
                    "W %-6s vs %-6s  V1/V2 %-28s vs %s\n",
                    node, t.parent[node], t.children[node].size(),
                    t.observed(node) ? "obs" : "free", alleleAt(t, node, site1, rootS),
                    setToString(wOf(a, node)).c_str(), setToString(wOf(b, node)).c_str(),
                    v12Of(a, node).c_str(), v12Of(b, node).c_str());
    };
    std::printf("  site %d, incremental vs recomputed:\n", site1);
    if (t.parent[n] >= 0) {
        if (t.parent[t.parent[n]] >= 0) line(t.parent[t.parent[n]]);
        line(t.parent[n]);
        for (int c : t.children[t.parent[n]]) if (c != n) line(c);
    }
    line(n);
    for (int c : t.children[n]) line(c);
}

// Every stored value, not just the W sets: v12 is what lets the next placement
// resume the upward pass, so a drift there is a real defect even when W agrees.
bool sameSets(const AlleleSets& a, const AlleleSets& b, std::string& why, int& badNode,
              int& badSite) {
    badNode = -1;
    badSite = -1;
    char buf[256];
    if (a.rootSet != b.rootSet) { why = "root sets differ"; return false; }
    if (a.parsimonyScore != b.parsimonyScore) {
        std::snprintf(buf, sizeof buf, "score %lld incremental vs %lld recomputed",
                      a.parsimonyScore, b.parsimonyScore);
        why = buf;
        return false;
    }
    if (a.delta.size() != b.delta.size()) { why = "node counts differ"; return false; }
    for (size_t i = 0; i < a.delta.size(); ++i) {
        if (a.delta[i].size() != b.delta[i].size()) {
            {
                const std::vector<SetDelta>& big =
                    a.delta[i].size() > b.delta[i].size() ? a.delta[i] : b.delta[i];
                const SetDelta& e = big[std::min(a.delta[i].size(), b.delta[i].size())];
                std::snprintf(buf, sizeof buf,
                              "node %zu: %zu deltas vs %zu; first extra at site %u = {%s} "
                              "(active %d/%d)",
                              i, a.delta[i].size(), b.delta[i].size(), e.site,
                              setToString(e.set).c_str(), (int)a.siteActive[e.site - 1],
                              (int)b.siteActive[e.site - 1]);
                badNode = static_cast<int>(i);
                badSite = e.site;
            }
            why = buf;
            return false;
        }
        for (size_t k = 0; k < a.delta[i].size(); ++k) {
            if (a.delta[i][k].site != b.delta[i][k].site ||
                a.delta[i][k].set != b.delta[i][k].set) {
                std::snprintf(buf, sizeof buf, "node %zu site %u: {%s} vs {%s}", i,
                              a.delta[i][k].site, setToString(a.delta[i][k].set).c_str(),
                              setToString(b.delta[i][k].set).c_str());
                badNode = static_cast<int>(i);
                badSite = a.delta[i][k].site;
                why = buf;
                return false;
            }
        }
    }
    for (size_t i = 0; i < a.v12.size(); ++i) {
        for (size_t k = 0; k < a.v12[i].size() && k < b.v12[i].size(); ++k) {
            if (a.v12[i][k].site != b.v12[i][k].site || a.v12[i][k].v1 != b.v12[i][k].v1 ||
                a.v12[i][k].v2 != b.v12[i][k].v2) {
                std::snprintf(buf, sizeof buf,
                              "node %zu site %u vs %u: V1={%s} V2={%s} vs V1={%s} V2={%s} (active %d/%d)",
                              i, a.v12[i][k].site, b.v12[i][k].site,
                              setToString(a.v12[i][k].v1).c_str(),
                              setToString(a.v12[i][k].v2).c_str(),
                              setToString(b.v12[i][k].v1).c_str(),
                              setToString(b.v12[i][k].v2).c_str(),
                              (int)a.siteActive[a.v12[i][k].site - 1],
                              (int)b.siteActive[b.v12[i][k].site - 1]);
                badNode = static_cast<int>(i);
                badSite = a.v12[i][k].site;
                why = buf;
                return false;
            }
        }
        if (a.v12[i].size() != b.v12[i].size()) {
            {
                const std::vector<V12Delta>& big =
                    a.v12[i].size() > b.v12[i].size() ? a.v12[i] : b.v12[i];
                const V12Delta& e = big[std::min(a.v12[i].size(), b.v12[i].size())];
                std::snprintf(buf, sizeof buf,
                              "node %zu: %zu V1/V2 entries vs %zu; first extra at site %u "
                              "V1={%s} V2={%s} (active %d/%d)",
                              i, a.v12[i].size(), b.v12[i].size(), e.site,
                              setToString(e.v1).c_str(), setToString(e.v2).c_str(),
                              (int)a.siteActive[e.site - 1], (int)b.siteActive[e.site - 1]);
                badNode = static_cast<int>(i);
                badSite = e.site;
            }
            why = buf;
            return false;
        }
    }
    return true;
}

int runPlace(const Options& opt) {
    const std::string rootS  = parseRootSequenceS(opt.rootSeq, opt.gene);
    const int         nSites = sitesOf(rootS);

    std::vector<RawNode> raw;
    Tree       t    = loadTree(opt, rootS, nSites, &raw);
    AlleleSets sets = computeAlleleSets(t, rootS, nSites);
    std::printf("allele sets  : parsimony score %lld over %d nodes\n", sets.parsimonyScore,
                t.size());

    PreparedSet queue = readPrepared(opt.prepared);
    if (queue.nSites != nSites) throw std::runtime_error("prepared sites do not match the tree");
    std::printf("prepared     : %zu sequences, %s .. %s\n", queue.size(),
                queue.size() ? std::to_string(queue.rec.front().dateCode).c_str() : "-",
                queue.size() ? std::to_string(queue.rec.back().dateCode).c_str() : "-");

    // Chronological, with the order inside one collection day shuffled: the file
    // is already sorted by date, so only the equal-date runs need permuting.
    std::vector<size_t> order(queue.size());
    std::iota(order.begin(), order.end(), size_t{0});
    {
        std::mt19937_64 rng(opt.seed);
        size_t i = 0;
        while (i < order.size()) {
            size_t j = i;
            while (j < order.size() && queue.rec[order[j]].dateCode == queue.rec[order[i]].dateCode)
                ++j;
            std::shuffle(order.begin() + i, order.begin() + j, rng);
            i = j;
        }
    }

    Placer     placer(t, sets, rootS,
                      opt.tieSeed < 0 ? opt.seed : static_cast<uint64_t>(opt.tieSeed),
                      !opt.attachMatches, static_cast<int>(opt.threads));
    if (opt.threads > 1) {
        std::printf("threads      : %lld on the placement scan; the written tree does not\n"
                    "               depend on how many\n", opt.threads);
    }

    // The sequences that are not attached are still provenance: where each one
    // matched is written as it goes, so nothing has to be held in memory.
    std::FILE* matchesFile = nullptr;
    if (!opt.outMatches.empty()) {
        matchesFile = std::fopen(opt.outMatches.c_str(), "wb");
        if (!matchesFile) throw std::runtime_error("cannot write " + opt.outMatches);
        std::fprintf(matchesFile, "#strain\tnode\n");
    }

    // The alignment is written as the queue is consumed, for the same reason:
    // it is the input to this run, record by record, and nothing has to be held.
    std::FILE* msaFile = nullptr;
    if (!opt.outMsa.empty()) {
        msaFile = std::fopen(opt.outMsa.c_str(), "wb");
        if (!msaFile) throw std::runtime_error("cannot write " + opt.outMsa);
    }

    // Counted over the queue as it is consumed, so --limit cuts the matrix at
    // the same place it cuts the run.
    ResidueCounts counts;
    counts.init(nSites);

    const auto t0      = std::chrono::steady_clock::now();
    long long  costSum = 0;
    size_t     done    = 0;
    const size_t total =
        opt.limit < 0 ? order.size() : std::min<size_t>(order.size(), static_cast<size_t>(opt.limit));

    for (size_t k = 0; k < total; ++k) {
        const size_t    i = order[k];
        if (!opt.outCounts.empty()) counts.add(queue.seq(i));
        if (msaFile) writeMsaRecord(msaFile, queue.rec[i].strain, queue.seq(i), nSites);
        const Placement p = placer.place(queue.seq(i), queue.rec[i].strain,
                                              decimalYear(queue.rec[i].dateCode));
        costSum += p.cost;
        ++done;
        if (matchesFile && p.skipped) {
            std::fprintf(matchesFile, "%s\t%d\n", queue.rec[i].strain.c_str(), p.matchedNode);
        }

        if (opt.verify) {
            AlleleSets  ref = computeAlleleSets(t, rootS, nSites);
            std::string why;
            int         badNode = -1, badSite = -1;
            if (!sameSets(sets, ref, why, badNode, badSite)) {
                std::printf("VERIFY FAILED after %zu placements (%s, cost %d, attached %d,%s): %s\n",
                            done, queue.rec[i].strain.c_str(), p.cost, p.attached,
                            p.split ? " split" : (p.skipped ? " skipped" : (p.exactMatch ? " exact" : "")),
                            why.c_str());
                if (badNode >= 0 && badSite > 0) dumpSite(t, sets, ref, badNode, badSite, rootS);
                return 1;
            }
        }
        if (!opt.verify && (done % 10000 == 0)) {
            std::printf("  %7zu placed, %d nodes, %.1f s\n", done, t.size(), secondsSince(t0));
            std::fflush(stdout);
        }
    }

    if (matchesFile) {
        std::fclose(matchesFile);
        std::printf("wrote        : %s\n", opt.outMatches.c_str());
    }
    if (msaFile) {
        const bool bad = std::ferror(msaFile) != 0;
        std::fclose(msaFile);
        if (bad) throw std::runtime_error("write failed: " + opt.outMsa);
        std::printf("wrote        : %s\n", opt.outMsa.c_str());
    }

    const Placer::Stats& st = placer.stats();
    std::printf("placed       : %lld sequences in %.1f s, tree now %d nodes\n", st.placements,
                secondsSince(t0), t.size());
    if (opt.attachMatches) {
        std::printf("               %lld split a branch, %lld hung straight off a node whose\n"
                    "               allele sets were already the sequence's own;\n",
                    st.splits, st.placements - st.splits);
    } else {
        std::printf("               %lld split a branch, %lld carried allele sets a node\n"
                    "               already had and were counted there, not attached;\n",
                    st.splits, st.skipped);
    }
    std::printf("               largest multifurcation now %lld\n", st.maxDegree);
    std::printf("               added %lld to the parsimony score (mean %.3f per sequence)\n",
                costSum, done ? static_cast<double>(costSum) / static_cast<double>(done) : 0.0);
    std::printf("scan         : %lld node visits (%.0f per placement)\n", st.scanned,
                done ? static_cast<double>(st.scanned) / static_cast<double>(done) : 0.0);
    std::printf("update       : %lld upward and %lld downward (node, site) steps, %lld set writes\n",
                st.upSteps, st.downSteps, st.setWrites);
    std::printf("               %lld ancestors actually moved, longest chain %lld\n",
                st.upHops, st.upHopsMax);
    const double pc = done ? 100.0 / static_cast<double>(done) : 0.0;
    std::printf("ties         : %lld placements matched an existing node exactly (%.2f%%)\n",
                st.exactHits, st.exactHits * pc);
    if (!opt.attachMatches) {
        std::printf("               %lld of those were settled by the match cache without a\n"
                    "               scan, over %lld probes (%.2f%% of probes hit)\n",
                    st.cacheHits, st.cacheProbes,
                    st.cacheProbes ? 100.0 * static_cast<double>(st.cacheHits) /
                                         static_cast<double>(st.cacheProbes) : 0.0);
    }
    // The draw statistics describe the placements that were actually drawn for:
    // a match is not a draw at all under the skip rule.
    const long long drawn = st.placements - (opt.attachMatches ? 0 : st.skipped);
    const double    dpc   = drawn ? 100.0 / static_cast<double>(drawn) : 0.0;
    std::printf("               of the %lld that did not match:\n", drawn);
    std::printf("               %lld had a single best candidate (%.2f%%)\n",
                st.unique, st.unique * dpc);
    std::printf("               %lld tied, but every candidate hangs the leaf in one place (%.2f%%)\n",
                st.tieOneAnchor, st.tieOneAnchor * dpc);
    std::printf("               %lld tied between different places (%.2f%%)\n",
                st.tieAnchors, st.tieAnchors * dpc);
    // The anchor is parent(v) for a tied branch, v itself for an exact match.
    std::printf("               draw size mean %.1f max %lld; distinct anchors mean %.3f max %lld\n",
                drawn ? static_cast<double>(st.candSum) / static_cast<double>(drawn) : 0.0,
                st.candMax,
                drawn ? static_cast<double>(st.anchorSum) / static_cast<double>(drawn) : 0.0,
                st.anchorMax);
    if (opt.verify) std::printf("verify       : every placement matches a full recompute\n");

    // Same materialised walk as 'sets', over the tree as it stands after placement.
    size_t ambigPairs = 0, ambigNodes = 0, ambigLabelled = 0;
    size_t hist[kNumStates + 1] = {0};
    std::vector<std::string> examples;
    walkMaterialised(t, sets, [&](int node, const AaSet* cur) {
        size_t here = 0;
        for (int s = 0; s < nSites; ++s) {
            int n = setSize(cur[s]);
            ++hist[n];
            if (n > 1) {
                ++here;
                if (examples.size() < 12)
                    examples.push_back("node " + std::to_string(node) + " site " +
                                       std::to_string(s + 1) + " = {" + setToString(cur[s]) + "}");
            }
        }
        ambigPairs += here;
        if (here) {
            ++ambigNodes;
            if (t.observed(node)) ++ambigLabelled;
        }
    });
    const long long pairs = static_cast<long long>(t.size()) * nSites;
    std::printf("ambiguity    : %zu of %lld (node, site) pairs admit more than one allele (%.6f%%)\n",
                ambigPairs, pairs,
                pairs ? 100.0 * static_cast<double>(ambigPairs) / static_cast<double>(pairs) : 0.0);
    std::printf("               spread over %zu of %d nodes (%zu of them observed)\n", ambigNodes,
                t.size(), ambigLabelled);
    std::printf("               set sizes:");
    for (int n = 1; n <= kNumStates; ++n) if (hist[n]) std::printf("  %d:%zu", n, hist[n]);
    std::printf("\n");
    for (const std::string& e : examples) std::printf("                 %s\n", e.c_str());

    if (!opt.outExamples.empty()) {
        reportExamples(t, sets, rootS, exampleOptions(opt, raw), opt.outExamples);
    }

    if (!opt.outCounts.empty()) {
        reportCounts(counts);
        writeResidueCounts(counts, opt.outCounts);
        std::printf("wrote        : %s\n", opt.outCounts.c_str());
    }

    if (!opt.outTree.empty()) {
        writeTree(t, opt.outTree);
        std::printf("wrote        : %s\n", opt.outTree.c_str());
    }
    if (!opt.out.empty()) {
        writeAlleleSets(t, sets, opt.out);
        std::printf("wrote        : %s\n", opt.out.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0) {
        std::fputs(kUsage, stdout);
        return argc < 2 ? 1 : 0;
    }

    Options opt;
    opt.command = argv[1];
    if (opt.command != "tree" && opt.command != "sets" && opt.command != "prepare" &&
        opt.command != "place") {
        std::fprintf(stderr, "error: unknown command '%s'\n\n%s", opt.command.c_str(), kUsage);
        return 1;
    }
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        bool ok = true;
        if      (a == "--tree")      ok = needValue(argc, argv, i++, "--tree", opt.tree);
        else if (a == "--root-seq")  ok = needValue(argc, argv, i++, "--root-seq", opt.rootSeq);
        else if (a == "--gene")      ok = needValue(argc, argv, i++, "--gene", opt.gene);
        else if (a == "--out")       ok = needValue(argc, argv, i++, "--out", opt.out);
        else if (a == "--out-tree")  ok = needValue(argc, argv, i++, "--out-tree", opt.outTree);
        else if (a == "--fasta")     ok = needValue(argc, argv, i++, "--fasta", opt.fasta);
        else if (a == "--metadata")  ok = needValue(argc, argv, i++, "--metadata", opt.metadata);
        else if (a == "--prepared")  ok = needValue(argc, argv, i++, "--prepared", opt.prepared);
        else if (a == "--out-examples")
            ok = needValue(argc, argv, i++, "--out-examples", opt.outExamples);
        else if (a == "--out-matches")
            ok = needValue(argc, argv, i++, "--out-matches", opt.outMatches);
        else if (a == "--attach-matches") opt.attachMatches = true;
        else if (a == "--no-clock")  opt.noClock = true;
        else if (a == "--threads") {
            std::string val;
            ok = needValue(argc, argv, i++, "--threads", val);
            if (ok) opt.threads = std::atoll(val.c_str());
            if (ok && opt.threads < 1) {
                std::fprintf(stderr, "error: --threads wants a positive count\n");
                return 1;
            }
        }
        else if (a == "--clock-tol") {
            std::string val;
            ok = needValue(argc, argv, i++, "--clock-tol", val);
            if (ok) opt.clockTol = std::atoll(val.c_str());
        }
        else if (a == "--clock-slack") {
            std::string val;
            ok = needValue(argc, argv, i++, "--clock-slack", val);
            if (ok) opt.clockSlack = std::atoll(val.c_str());
        }
        else if (a == "--out-clock") ok = needValue(argc, argv, i++, "--out-clock", opt.outClock);
        else if (a == "--out-counts")
            ok = needValue(argc, argv, i++, "--out-counts", opt.outCounts);
        else if (a == "--out-msa")
            ok = needValue(argc, argv, i++, "--out-msa", opt.outMsa);
        else if (a == "--variant")   ok = needValue(argc, argv, i++, "--variant", opt.variant);
        else if (a == "--post-cutoff") {
            std::string v;
            ok = needValue(argc, argv, i++, "--post-cutoff", v);
            // A comma-separated list, and repeatable: 24 quarterly cutoffs are
            // one `cut -f1 ... | paste -sd,` away, so the shell need not loop.
            for (size_t b = 0; ok && b <= v.size(); ) {
                const size_t e = std::min(v.find(',', b), v.size());
                if (e > b) opt.postCutoffs.push_back(v.substr(b, e - b));
                b = e + 1;
            }
        }
        else if (a == "--date-cutoff") {
            std::string val;
            ok = needValue(argc, argv, i++, "--date-cutoff", val);
            if (ok) {
                if (val == "none" || val == "None" || val == "0") opt.dateCutoff = 0;
                else {
                    const int32_t d = parseDateCode(val.c_str(), val.size());
                    if (!d) {
                        std::fprintf(stderr, "error: --date-cutoff wants YYYY-MM-DD or 'none'\n");
                        return 1;
                    }
                    opt.dateCutoff = d;
                }
            }
        }
        else if (a == "--verify")    opt.verify = true;
        else if (a == "--seed" || a == "--limit" || a == "--tie-seed") {
            std::string val;
            ok = needValue(argc, argv, i++, a.c_str(), val);
            if (ok) {
                if      (a == "--seed")     opt.seed = std::strtoull(val.c_str(), nullptr, 10);
                else if (a == "--tie-seed") opt.tieSeed = std::strtoll(val.c_str(), nullptr, 10);
                else                        opt.limit = std::strtoll(val.c_str(), nullptr, 10);
            }
        }
        else {
            std::fprintf(stderr, "error: unknown argument '%s'\n\n%s", a.c_str(), kUsage);
            return 1;
        }
        if (!ok) return 1;
    }
    if (opt.rootSeq.empty()) {
        std::fprintf(stderr, "error: --root-seq is required\n\n%s", kUsage);
        return 1;
    }
    if (opt.command != "prepare" && opt.tree.empty()) {
        std::fprintf(stderr, "error: --tree is required for '%s'\n\n%s", opt.command.c_str(),
                     kUsage);
        return 1;
    }
    if (opt.command == "prepare" && (opt.fasta.empty() || opt.metadata.empty())) {
        std::fprintf(stderr, "error: --fasta and --metadata are required for 'prepare'\n\n%s",
                     kUsage);
        return 1;
    }
    if (opt.command == "place" && opt.prepared.empty()) {
        std::fprintf(stderr, "error: --prepared is required for 'place'\n\n%s", kUsage);
        return 1;
    }
    // The matrix counts sequences, and 'tree' and 'sets' read none: silently
    // writing an all-zero one would be worse than saying so.
    if (!opt.outCounts.empty() && opt.command != "prepare" && opt.command != "place") {
        std::fprintf(stderr, "error: --out-counts is for 'prepare' and 'place', not '%s'\n\n%s",
                     opt.command.c_str(), kUsage);
        return 1;
    }
    // Only the placement scan is threaded, so asking for cores anywhere else is
    // a misunderstanding worth naming rather than ignoring.
    if (opt.threads > 1 && opt.command != "place") {
        std::fprintf(stderr, "error: --threads is for 'place', not '%s'\n\n%s",
                     opt.command.c_str(), kUsage);
        return 1;
    }
    if (!opt.outClock.empty() && opt.command != "prepare") {
        std::fprintf(stderr, "error: --out-clock is for 'prepare', not '%s'\n\n%s",
                     opt.command.c_str(), kUsage);
        return 1;
    }
    if (!opt.outMsa.empty() && opt.command != "prepare" && opt.command != "place") {
        std::fprintf(stderr, "error: --out-msa is for 'prepare' and 'place', not '%s'\n\n%s",
                     opt.command.c_str(), kUsage);
        return 1;
    }

    try {
        if (opt.command == "tree") return runTree(opt);
        if (opt.command == "sets")     return runSets(opt);
        if (opt.command == "prepare")  return runPrepare(opt);
        return runPlace(opt);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
