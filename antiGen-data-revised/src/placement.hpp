// Step 3: append new sequences to the tree by parsimony placement, keeping the
// Hartigan sets current after each one.
//
// Scoring reads straight off the sets step 2 already computed, and is exact.
// For an unweighted alphabet the score increase from hanging an observation x
// off node v is 0 at every site where x's allele is in W(v) and 1 at every other
// site -- never more, because the alternative to forcing v off its optimum is
// one mutation on the pendant branch. Inserting a new node on the branch p->v
// instead gives that node the free run of W(p) | W(v). So
//
//     cost(child of v)       = #{ s : x_s observed, x_s not in W(v) }
//     cost(new node on p->v) = #{ s : x_s observed, x_s not in W(p) | W(v) }
//
// The second is never worse than the first -- nor worse than a child of p -- so
// the minimum always sits on a branch, and a node placement can only ever join
// the tie set, never win outright.
//
// Where it goes is decided in two stages, not one draw over everything tied:
//
//  1. An **exact match** means the sequence is not attached at all. Alongside
//     the cost the scan carries
//
//         e(v) = #{ s : x_s observed, W(v) != {x_s} }
//
//     which is 0 exactly when a leaf hung under v would carry an empty
//     set-delta -- its allele sets would be v's, at every site, including the
//     ones x is missing at, where the leaf inherits v's set anyway. Such a leaf
//     is a no-op on the tree's parsimony: W(v) = {x_s} says every MPR already
//     puts x_s at v, so the leaf's pendant edge costs nothing in any of them,
//     the score does not move, and the MPR set at every node is exactly what it
//     was. So the sequence is counted against v and dropped. Since any matching
//     node does as well as any other, the scan stops at the first one it sees,
//     and a small cache of recently matched nodes is tried before the scan
//     starts at all. See NOTES, "Directive, 2026-10-22: a match is not attached".
//
//  2. Otherwise **only branches are considered**, since no node placement can
//     beat every branch out of its own neighbourhood. The draw is uniform over
//     the cheapest branches.
//
// Both quantities are maintained incrementally down a DFS: W changes only at the
// sites in delta[v], so per-node scan work is proportional to that node's deltas,
// and the union correction involves those same sites.
//
// The surgery, where there is one, is a pendant leaf under a new free node
// spliced into the chosen branch. Nothing is committed to a genotype and nothing
// is collapsed here -- two nodes with the same allele sets are fine, and step 4
// is where sameness is settled.
//
// The update afterwards is local because the upward pass reads only its children's
// V1, and a new child can only move V1 where its allele departs from the baseline
// the sets are stored against. It is no longer true that an observed ancestor
// absorbs the change within a hop or two: the interior of the tree holds no
// observations now. See NOTES, "Consequences to expect".
#pragma once

#include "alphabet.hpp"
#include "hartigan.hpp"
#include "tree.hpp"

#include <atomic>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

struct Placement {
    int  cost     = 0;   // increase in the tree's parsimony score
    int  attached = -1;  // the leaf the sequence became; -1 if it was skipped
    bool split    = false;  // a node was created on a branch to carry it
    bool exactMatch = false;  // a node's allele sets were already the sequence's own
    bool skipped    = false;  // and so the sequence was counted there, not attached
    int  matchedNode = -1;    // that node, for the provenance file
    // How much choice the scan had. `cands` is the size of the set the reservoir
    // sampler drew from: the exact matches if there were any, otherwise the
    // cheapest branches. `anchors` collapses that set by the node the sequence
    // hangs under -- parent(v) for a branch, v itself for an exact match --
    // because every branch out of one parent puts the new node in the same place
    // and differs only in which sibling is pulled down under it. anchors == 1
    // means the draw cannot move the sequence.
    int  cands   = 1;
    int  anchors = 1;
};

class Placer {
  public:
    struct Stats {
        long long placements = 0;
        long long scanned    = 0;  // nodes visited by the placement scan
        long long splits     = 0;
        long long upSteps    = 0;  // (node, site) pairs the upward pass recomputed
        long long downSteps  = 0;  // (node, site) pairs the downward pass recomputed
        long long setWrites  = 0;  // delta insertions, updates and removals
        long long upHops     = 0;  // ancestors the upward pass actually moved
        long long upHopsMax  = 0;  // the longest single chain it walked
        long long maxDegree  = 0;  // largest child count anywhere in the tree
        long long exactHits  = 0;  // placements settled by an exact allele-set match
        long long skipped    = 0;  // of those, the ones not attached (all of them,
                                   // unless the old attach-every-match rule is on)
        long long cacheProbes = 0; // match-cache tests, and how many hit, which is
        long long cacheHits  = 0;  // how often the scan was skipped outright
        // Tie-breaking, one count per placement by the shape of the set drawn from.
        long long unique      = 0;  // argmin held one candidate
        long long tieOneAnchor = 0; // several candidates, all under one parent
        long long tieAnchors  = 0;  // the draw was between different parents
        long long candSum     = 0;  // argmin candidates summed over placements
        long long candMax     = 0;
        long long anchorSum   = 0;
        long long anchorMax   = 0;
    };

    // skipMatches is the rule above: a sequence whose allele sets a node already
    // carries is counted there and dropped. False restores the older rule, which
    // drew uniformly over the matching nodes and hung a leaf off the one it drew.
    Placer(Tree& tree, AlleleSets& sets, const std::string& rootSeq, uint64_t seed,
           bool skipMatches = true, int threads = 1);

    // seq is nSites characters. Missing data constrains nothing. `date` is the
    // collection date as a decimal year; it is carried only for step 4, and a
    // skipped sequence leaves it behind on the node it matched.
    Placement place(const char* seq, const std::string& label, double date);

    const Stats& stats() const { return st_; }

  private:
    struct Cand {
        int  node  = 0;
        bool split = false;  // a new node on the branch above `node`, not a child of it
    };

    // The DFS stack, one frame per level. A member on the serial walk and a
    // per-worker buffer on the parallel one, so the hot path allocates nothing.
    struct ScanFrame {
        int    node;
        size_t nextChild;
        size_t mark;
        int    d;
        int    e;
    };

    // --- sparse lookups -------------------------------------------------
    const SetDelta* deltaAt(int n, int site1) const;
    void            setDelta(int n, int site1, AaSet value, AaSet parentValue);
    // The stored default is V1 = {alleleAt(n, site)}, V2 = {}; only a departure
    // from it is recorded.
    void            getV12(int n, int site1, AaSet& v1, AaSet& v2) const;
    void            setV12(int n, int site1, AaSet v1, AaSet v2);
    char            alleleOf(int n, int site1) const;
    AaSet           baseline(int n, int site1) const { return aaBit(alleleOf(n, site1)); }

    // W at a node of the materialised path, for one site.
    AaSet wAt(int n, int site1) const;

    // --- the scan --------------------------------------------------------
    //
    // The scan is ~85% of a placement, and it is a DFS over the whole tree, so
    // it is what gets the cores. The tree is cut into tasks -- subtrees small
    // enough to balance, with the nodes above them ("the spine") evaluated on
    // the way down -- and every task is independent, because the scan only
    // reads the sets.
    //
    // Every task and every spine node carries an **ordinal**, handed out in DFS
    // order, and results are merged by ordinal rather than by completion. That
    // is what makes the output independent of the thread count: the candidate
    // list comes out in exactly the DFS order a serial scan would have built it
    // in, and the draw over it is a single call on the one RNG. One thread and
    // sixty-four threads write the same tree, bit for bit.
    struct ScanTask {
        int  node;      // the subtree root, or the single node when selfOnly
        int  parent;    // where the worker has to materialise W
        int  d, e;      // the running counts at `parent`
        int  ord;       // position in DFS order
        bool selfOnly;  // a spine node: evaluate it, do not descend
    };
    // Aligned because neighbouring entries are written by different threads,
    // and two of them in one cache line is a silent halving of the speedup.
    struct alignas(64) TaskOut {
        int               bestCost  = 0;
        bool              any       = false;  // bestCost means nothing without this
        long long         scanned   = 0;
        std::vector<Cand> argmin;
        std::vector<int>  exacts;  // in DFS order; only the first, under skipMatches
    };
    // One of these per thread, kept across placements so the hot path allocates
    // nothing. `W` is resynced to the root's sets once per placement and then
    // only ever perturbed and unwound, so no task pays for a dense copy.
    struct alignas(64) Worker {
        std::vector<AaSet>                 W;
        std::vector<std::pair<int, AaSet>> undo;
        std::vector<ScanFrame>             stack;
        std::vector<int>                   path;
    };

    Cand scan(int& bestCost, int& cands, int& anchors, bool& exact);
    // Walks the spine, evaluating it, and leaves the subtrees below in tasks_.
    void buildScanTasks(int d0, int e0);
    // `firstExact` is the lowest ordinal known to hold a match. A worker whose
    // own ordinal is above it has nothing left to contribute -- the answer is
    // the lowest ordinal, not the first one finished -- so it stops. Checked
    // every so often rather than per node: it is a shared line.
    void runScanTask(const ScanTask& task, Worker& w, TaskOut& out,
                     const std::atomic<int>* firstExact) const;
    void refreshSubtreeSizes();
    void materialisePath(int node);
    // e(v) == 0 at one node, against the sequence in xbit_/xobs_. Materialises
    // v's path, so it costs one pass over the sites rather than one over the tree.
    bool matchesHere(int v);

    // --- surgery ---------------------------------------------------------
    // The whole of a skipped placement: count it against the node, remember the
    // node, and return.
    Placement skipAt(int node, double date);
    int  newLeaf(int parent, const char* seq, const std::string& label, double date);
    // W_ holds `w`; the sequence becomes a new observed leaf under it.
    int  attachLeaf(int w, const char* seq, const std::string& label, double date);
    // W_ holds `p`; a free node is spliced into p -> v and the leaf hangs off it.
    int  splitBranch(int p, int v, const char* seq, const std::string& label, double date);
    // delta[L] over every site, against the W_ buffer as it stands at L's parent.
    void writeLeafSets(int leaf, const char* seq);

    // --- keeping the sets current ----------------------------------------
    void recomputeV12(int n, int site1, AaSet& v1, AaSet& v2) const;
    void refresh(const std::vector<int>& sites, int start);
    void descend(int n, int site1, AaSet oldWp, AaSet newWp);
    static AaSet downward(AaSet wp, AaSet v1, AaSet v2) {
        return ((wp & ~v1) == 0) ? wp : (v1 | (wp & v2));
    }

    Tree&           t_;
    AlleleSets&     sets_;
    std::string     rootSeq_;
    int             nSites_;
    std::mt19937_64 rng_;
    bool            skipMatches_;
    Stats           st_;

    // The nodes the last few matches landed on, most recent first. Consecutive
    // sequences in a chronological queue are very often the same genotype, so
    // one test against the dense site buffer usually settles a placement without
    // the scan. A node is never removed from the tree, so a stale entry is only
    // ever a wasted test.
    std::vector<int> matchCache_;
    static constexpr size_t kMatchCache = 4;

    // Per-placement scratch, kept across calls so nothing is reallocated.
    std::vector<AaSet> xbit_;     // aaBit(seq[s]); kAllStates where missing, so a
                                  // missing site never fails the `W & x` test
    std::vector<uint8_t> xobs_;   // !isMissing(seq[s]); the sites the `W == x`
                                  // test is allowed to look at
    std::vector<AaSet> W_;        // dense sets along the current path
    std::vector<int>   path_;     // root .. attachment point
    std::vector<int>   chain_;    // the nodes the upward pass moved, deepest first
    std::vector<int>   touched_;  // the 1-based sites whose sets can move
    std::vector<std::pair<int, AaSet>> wUndo_;
    std::vector<ScanFrame>             sstack_;
    std::vector<Cand>                  argmin_;  // the current best-cost set

    int                   threads_ = 1;
    std::vector<ScanTask> tasks_;
    std::vector<TaskOut>  outs_;
    std::vector<Worker>   workers_;
    // The root's sets as the workers last saw them. Resyncing a worker by
    // copying the whole root buffer costs O(threads x sites) of *serial* work
    // per placement, which caps the speedup on its own; a placement moves a
    // handful of sites at the root, so the workers are handed the difference.
    std::vector<AaSet>                 rootPrev_;
    std::vector<std::pair<int, AaSet>> rootDirty_;
    // A balancing hint, not a fact: it is rebuilt when the tree has grown a few
    // percent, and a stale size only makes a task the wrong size.
    std::vector<int>      subtreeSize_;
    int                   sizeStamp_ = 0;
    uint64_t              scanEpoch_ = 0;
    std::vector<uint64_t>              anchorSeen_;  // stamp per node, for the distinct count
    uint64_t                           epoch_ = 0;

    struct DownFrame {
        int   node;
        AaSet oldWp;
        AaSet newWp;
    };
    std::vector<DownFrame> downStack_;
};
