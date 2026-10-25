#include "placement.hpp"

#include <atomic>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <stdexcept>

namespace {

// delta, v12 and obs are all kept ascending by `site`, so one node's list can be
// searched for a single site.
template <class Vec>
size_t lowerBoundSite(const Vec& v, int site) {
    size_t lo = 0, hi = v.size();
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (v[mid].site < site) lo = mid + 1;
        else                    hi = mid;
    }
    return lo;
}

void sortUnique(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

}  // namespace

Placer::Placer(Tree& tree, AlleleSets& sets, const std::string& rootSeq, uint64_t seed,
               bool skipMatches, int threads)
    : t_(tree), sets_(sets), rootSeq_(rootSeq), nSites_(sets.nSites), rng_(seed),
      skipMatches_(skipMatches), threads_(threads < 1 ? 1 : threads) {
    if (static_cast<int>(rootSeq_.size()) < nSites_) {
        throw std::runtime_error("root sequence is shorter than the site count");
    }
    rootSeq_.resize(nSites_);
    xbit_.assign(nSites_, kAllStates);
    xobs_.assign(nSites_, 0);
    W_.assign(nSites_, kAllStates);
    for (int v = 0; v < t_.size(); ++v) {
        st_.maxDegree = std::max(st_.maxDegree, static_cast<long long>(t_.children[v].size()));
    }
    workers_.resize(static_cast<size_t>(threads_));
    for (Worker& w : workers_) w.W.assign(nSites_, kAllStates);
    rootPrev_.assign(nSites_, kAllStates);  // matches the workers, so the first
                                            // placement syncs everything that moved
    refreshSubtreeSizes();
}

// Sizes drive nothing but the cut, so they are rebuilt on growth rather than
// maintained through every piece of surgery: O(N) every few percent is a
// couple of dozen operations per placement, and a stale size costs balance,
// never correctness.
void Placer::refreshSubtreeSizes() {
    const int N = t_.size();
    subtreeSize_.assign(static_cast<size_t>(N), 1);
    std::vector<int>& order = path_;
    order.clear();
    order.push_back(0);
    for (size_t i = 0; i < order.size(); ++i) {
        for (int ch : t_.children[order[i]]) order.push_back(ch);
    }
    for (size_t i = order.size(); i-- > 1;) {
        const int v = order[i];
        subtreeSize_[t_.parent[v]] += subtreeSize_[v];
    }
    sizeStamp_ = N;
    order.clear();
}

// One node, given W and the counts at its parent: apply its set-delta, and
// report what hanging the sequence on the branch above it would cost. `broke`
// is the sites this node's own delta pushed out of range, which the new node
// on the branch would get back for free out of W(p) | W(v).
namespace {
// How often a worker looks at the shared "someone found a match earlier than
// you" flag. Often enough that a doomed task dies quickly, rarely enough that
// the load is not in the inner loop.
constexpr int kExactPollEvery = 256;

inline void stepNode(const std::vector<SetDelta>& delta, const std::vector<AaSet>& xbit,
                     const std::vector<uint8_t>& xobs, std::vector<AaSet>& W,
                     std::vector<std::pair<int, AaSet>>& undo, int& d, int& e, int& broke) {
    broke = 0;
    for (const SetDelta& dd : delta) {
        const int   s   = dd.site - 1;
        const AaSet old = W[s];
        undo.emplace_back(s, old);
        W[s] = dd.set;
        const bool wasBad = (old & xbit[s]) == 0;
        const bool nowBad = (dd.set & xbit[s]) == 0;
        if (wasBad && !nowBad)      --d;
        else if (!wasBad && nowBad) { ++d; ++broke; }
        if (xobs[s]) {
            const bool wasNe = old != xbit[s];
            const bool nowNe = dd.set != xbit[s];
            if (wasNe && !nowNe)      --e;
            else if (!wasNe && nowNe) ++e;
        }
    }
}
}  // namespace

// A task: materialise W at the task's parent, then walk. Touches nothing the
// other workers touch -- the sets are read-only for the length of a scan.
void Placer::runScanTask(const ScanTask& task, Worker& w, TaskOut& out,
                         const std::atomic<int>* firstExact) const {
    const size_t base = w.undo.size();

    // Down the path to the parent. The worker's buffer already holds the root's
    // sets, so this is the only setup a task pays for, and it is why the tasks
    // are sized by the thread count rather than made as small as possible: the
    // cost of a cut is one root path, so cuts have to stay rare against nodes.
    w.path.clear();
    for (int n = task.parent; n > 0; n = t_.parent[n]) w.path.push_back(n);
    for (size_t i = w.path.size(); i-- > 0;) {
        for (const SetDelta& dd : sets_.delta[w.path[i]]) {
            w.undo.emplace_back(dd.site - 1, w.W[dd.site - 1]);
            w.W[dd.site - 1] = dd.set;
        }
    }

    auto consider = [&](int cost, int node) {
        if (!out.any || cost < out.bestCost) {
            out.any      = true;
            out.bestCost = cost;
            out.argmin.clear();
            out.argmin.push_back(Cand{node, true});
        } else if (cost == out.bestCost) {
            out.argmin.push_back(Cand{node, true});
        }
    };

    int d = task.d, e = task.e, broke = 0;
    stepNode(sets_.delta[task.node], xbit_, xobs_, w.W, w.undo, d, e, broke);
    consider(d - broke, task.node);
    ++out.scanned;
    if (e == 0) out.exacts.push_back(task.node);

    if (!task.selfOnly && !(skipMatches_ && !out.exacts.empty())) {
        w.stack.clear();
        w.stack.push_back({task.node, 0, w.undo.size(), d, e});
        int countdown = kExactPollEvery;
        while (!w.stack.empty()) {
            if (firstExact && --countdown <= 0) {
                countdown = kExactPollEvery;
                if (task.ord > firstExact->load(std::memory_order_relaxed)) break;
            }
            ScanFrame& f = w.stack.back();
            if (f.nextChild < t_.children[f.node].size()) {
                const int    ch   = t_.children[f.node][f.nextChild++];
                const size_t mark = w.undo.size();
                int          cd = f.d, ce = f.e, cbroke = 0;
                stepNode(sets_.delta[ch], xbit_, xobs_, w.W, w.undo, cd, ce, cbroke);
                consider(cd - cbroke, ch);
                ++out.scanned;
                if (ce == 0) {
                    out.exacts.push_back(ch);
                    if (skipMatches_) break;
                }
                w.stack.push_back({ch, 0, mark, cd, ce});
            } else {
                const size_t mark = f.mark;
                w.stack.pop_back();
                while (w.undo.size() > mark) {
                    w.W[w.undo.back().first] = w.undo.back().second;
                    w.undo.pop_back();
                }
            }
        }
    }

    while (w.undo.size() > base) {
        w.W[w.undo.back().first] = w.undo.back().second;
        w.undo.pop_back();
    }
}

// --- sparse lookups ---------------------------------------------------------

const SetDelta* Placer::deltaAt(int n, int site1) const {
    const std::vector<SetDelta>& v = sets_.delta[n];
    const size_t i = lowerBoundSite(v, site1);
    return (i < v.size() && v[i].site == site1) ? &v[i] : nullptr;
}

void Placer::setDelta(int n, int site1, AaSet value, AaSet parentValue) {
    std::vector<SetDelta>& v = sets_.delta[n];
    const size_t i   = lowerBoundSite(v, site1);
    const bool   has = i < v.size() && v[i].site == site1;
    if (value == parentValue) {
        if (has) { v.erase(v.begin() + i); ++st_.setWrites; }
        return;
    }
    if (has) {
        if (v[i].set != value) { v[i].set = value; ++st_.setWrites; }
    } else {
        v.insert(v.begin() + i, SetDelta{value, static_cast<uint16_t>(site1)});
        ++st_.setWrites;
    }
}

char Placer::alleleOf(int n, int site1) const {
    const std::vector<ObsAllele>& o = t_.obs[n];
    const size_t i = lowerBoundSite(o, site1);
    return (i < o.size() && o[i].site == site1) ? o[i].aa : rootSeq_[site1 - 1];
}

void Placer::getV12(int n, int site1, AaSet& v1, AaSet& v2) const {
    const std::vector<V12Delta>& v = sets_.v12[n];
    const size_t i = lowerBoundSite(v, site1);
    if (i < v.size() && v[i].site == site1) { v1 = v[i].v1; v2 = v[i].v2; return; }
    v1 = baseline(n, site1);
    v2 = 0;
}

void Placer::setV12(int n, int site1, AaSet v1, AaSet v2) {
    std::vector<V12Delta>& v = sets_.v12[n];
    const size_t i   = lowerBoundSite(v, site1);
    const bool   has = i < v.size() && v[i].site == site1;
    if (v1 == baseline(n, site1) && v2 == 0) {
        if (has) { v.erase(v.begin() + i); ++st_.setWrites; }
        return;
    }
    if (has) {
        if (v[i].v1 != v1 || v[i].v2 != v2) { v[i].v1 = v1; v[i].v2 = v2; ++st_.setWrites; }
    } else {
        v.insert(v.begin() + i, V12Delta{v1, v2, static_cast<uint16_t>(site1)});
        ++st_.setWrites;
    }
}

AaSet Placer::wAt(int n, int site1) const {
    AaSet w = sets_.rootSet[site1 - 1];
    for (size_t i = 0; i < path_.size(); ++i) {
        if (i) {
            const SetDelta* d = deltaAt(path_[i], site1);
            if (d) w = d->set;
        }
        if (path_[i] == n) break;
    }
    return w;
}

// --- the scan ---------------------------------------------------------------

// One DFS over the whole tree, carrying W in a dense buffer. Two quantities ride
// along, both incremental in this node's own deltas:
//
//     d(v) = #{ s : x_s observed, x_s not in W(v) }   -- cost of a child of v
//     e(v) = #{ s : x_s observed, W(v) != {x_s} }     -- distance from v's sets
//                                                        being the sequence's own
//
// The branch cost is d minus the sites this node's own deltas just broke, because
// those are exactly the sites where W(v) misses the allele but W(p) holds it.
//
// e == 0 is an exact match: a leaf under v would have W(L) == W(v) at every site,
// so the leaf would be a no-op on the sets and on the score. It implies d == 0 --
// a non-empty x_s outside W(v) also differs from it -- so it is always a
// zero-cost placement, and it wins outright over every branch. Once one is seen
// the branch bookkeeping is dropped: the answer is already known, and the argmin
// sets are large. Under the skip rule the scan returns there and then, since one
// matching node is as good as another when the sequence is not being attached.
// The spine: the nodes too big to be one task. They are evaluated here, on the
// way down, because the walk already has W in hand -- and they are few, since
// a node is only on the spine if its subtree is larger than one task.
void Placer::buildScanTasks(int d0, int e0) {
    const int N     = t_.size();
    // One cut costs one root path, so cuts are sized against the thread count
    // rather than made as small as possible: a few per thread is enough for
    // dynamic scheduling to balance, and the setup stays a rounding error.
    const int want  = threads_ * 4;
    const int limit = std::max(64, N / std::max(1, want));

    tasks_.clear();
    int ord = 0;

    std::vector<ScanFrame>& stack = sstack_;
    stack.clear();
    wUndo_.clear();
    stack.push_back({0, 0, 0, d0, e0});

    while (!stack.empty()) {
        ScanFrame& f = stack.back();
        if (f.nextChild < t_.children[f.node].size()) {
            const int    ch   = t_.children[f.node][f.nextChild++];
            const size_t mark = wUndo_.size();
            if (subtreeSize_[ch] <= limit) {
                // Small enough to hand over whole. The worker re-derives W from
                // the root, so nothing of this walk's state goes with it.
                tasks_.push_back({ch, f.node, f.d, f.e, ord++, false});
                continue;
            }
            // On the spine: evaluate it here and carry on down.
            int d = f.d, e = f.e, broke = 0;
            stepNode(sets_.delta[ch], xbit_, xobs_, W_, wUndo_, d, e, broke);
            tasks_.push_back({ch, f.node, f.d, f.e, ord++, true});
            stack.push_back({ch, 0, mark, d, e});
        } else {
            const size_t mark = f.mark;
            stack.pop_back();
            while (wUndo_.size() > mark) {
                W_[wUndo_.back().first] = wUndo_.back().second;
                wUndo_.pop_back();
            }
        }
    }
    // Grown, never shrunk, and the entries are reset rather than destroyed:
    // a TaskOut owns two vectors, and rebuilding them per placement was the
    // same mistake as allocating the DFS stack per placement.
    if (outs_.size() < static_cast<size_t>(ord)) outs_.resize(static_cast<size_t>(ord));
}

Placer::Cand Placer::scan(int& bestCost, int& cands, int& anchors, bool& exact) {
    const int N = t_.size();
    if (N > sizeStamp_ + 64 && N > sizeStamp_ * 101 / 100) refreshSubtreeSizes();
    // Nodes added since the last refresh have no entry yet, and buildScanTasks
    // indexes every node it meets. They are new, so 1 is close enough until the
    // next refresh -- but the slot has to exist.
    if (static_cast<int>(subtreeSize_.size()) < N) subtreeSize_.resize(static_cast<size_t>(N), 1);

    for (int s = 0; s < nSites_; ++s) W_[s] = sets_.rootSet[s];
    int d0 = 0, e0 = 0;
    for (int s = 0; s < nSites_; ++s) {
        if ((W_[s] & xbit_[s]) == 0) ++d0;
        if (xobs_[s] && W_[s] != xbit_[s]) ++e0;
    }

    ++st_.scanned;
    // Seeded at the cost of a child of the root, which is an upper bound on the
    // cheapest branch: splitting the branch to any child of the root costs no
    // more. So the best cost is attained by some real branch whenever the tree
    // has one.
    bestCost = d0;

    if (e0 == 0 && skipMatches_) {
        exact = true; bestCost = 0; cands = 1; anchors = 1;
        return Cand{0, false};
    }
    const bool rootExact = (e0 == 0);

    if (N <= 1) {
        exact = rootExact; cands = 1; anchors = 1;
        if (rootExact) bestCost = 0;
        return Cand{0, false};
    }

    // --- cut, walk, merge -------------------------------------------------
    buildScanTasks(d0, e0);
    const int nTasks = static_cast<int>(tasks_.size());

    // Under the skip rule the first match in DFS order settles the placement,
    // so a worker that is past the best ordinal found so far has nothing to
    // contribute and stops. The *answer* does not depend on how far anyone got
    // -- it is the minimum ordinal, not the first one finished.
    std::atomic<int> firstExact(nTasks);
    rootDirty_.clear();
    for (int s = 0; s < nSites_; ++s) {
        if (rootPrev_[s] != sets_.rootSet[s]) {
            rootDirty_.emplace_back(s, sets_.rootSet[s]);
            rootPrev_[s] = sets_.rootSet[s];
        }
    }
    for (Worker& w : workers_) {
        for (const std::pair<int, AaSet>& ch : rootDirty_) w.W[ch.first] = ch.second;
        w.undo.clear();
    }

    // Below a few thousand nodes a scan is shorter than the cost of starting a
    // parallel region, and early in a run most of them stop at the first match
    // anyway. The cut is the same either way, so this changes speed and not the
    // answer.
    const bool par = threads_ > 1 && nTasks > 1 && N >= 8192;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads_) if (par)
#endif
    for (int i = 0; i < nTasks; ++i) {
#ifdef _OPENMP
        Worker& w = workers_[static_cast<size_t>(omp_get_thread_num())];
#else
        Worker& w = workers_[0];
#endif
        TaskOut& o = outs_[static_cast<size_t>(i)];
        o.any = false;
        o.bestCost = 0;
        o.scanned = 0;
        o.argmin.clear();
        o.exacts.clear();
        if (skipMatches_ && i > firstExact.load(std::memory_order_relaxed)) continue;
        runScanTask(tasks_[static_cast<size_t>(i)], w, o, skipMatches_ ? &firstExact : nullptr);
        if (skipMatches_ && !o.exacts.empty()) {
            int cur = firstExact.load(std::memory_order_relaxed);
            while (i < cur && !firstExact.compare_exchange_weak(cur, i,
                                                                std::memory_order_relaxed)) {
            }
        }
    }

    // --- merge, strictly by ordinal --------------------------------------
    for (int i = 0; i < nTasks; ++i) st_.scanned += outs_[static_cast<size_t>(i)].scanned;

    if (rootExact) {  // the root is ordinal -1, ahead of everything
        exact = true; bestCost = 0;
        if (skipMatches_) { cands = 1; anchors = 1; return Cand{0, false}; }
    }
    if (skipMatches_) {
        for (int i = 0; i < nTasks; ++i) {
            if (!outs_[static_cast<size_t>(i)].exacts.empty()) {
                exact = true; bestCost = 0; cands = 1; anchors = 1;
                return Cand{outs_[static_cast<size_t>(i)].exacts.front(), false};
            }
        }
    } else {
        argmin_.clear();
        for (int i = 0; i < nTasks; ++i)
            for (int n : outs_[static_cast<size_t>(i)].exacts) argmin_.push_back(Cand{n, false});
        if (rootExact) argmin_.insert(argmin_.begin(), Cand{0, false});
        if (!argmin_.empty()) {
            exact    = true;
            bestCost = 0;
            cands    = static_cast<int>(argmin_.size());
            anchors  = cands;
            return argmin_[rng_() % argmin_.size()];
        }
    }

    exact = false;
    int best = 0;
    bool any = false;
    for (int i = 0; i < nTasks; ++i) {
        const TaskOut& o = outs_[static_cast<size_t>(i)];
        if (o.any && (!any || o.bestCost < best)) { best = o.bestCost; any = true; }
    }
    if (!any) {  // a lone root has no branch; hang the leaf off it
        cands = 1; anchors = 1; bestCost = d0;
        return Cand{0, false};
    }
    bestCost = best;

    // In DFS order, so the list -- and the draw over it -- is what a serial
    // scan would have produced, whatever the cut was.
    argmin_.clear();
    for (int i = 0; i < nTasks; ++i) {
        const TaskOut& o = outs_[static_cast<size_t>(i)];
        if (o.any && o.bestCost == best) argmin_.insert(argmin_.end(), o.argmin.begin(),
                                                        o.argmin.end());
    }

    // Distinct anchors: the parent of each tied branch, since that is where the
    // new node lands. Anchors recur all over the DFS order, so this is a stamped
    // pass, not runs.
    cands   = static_cast<int>(argmin_.size());
    anchors = 0;
    if (static_cast<int>(anchorSeen_.size()) < N) anchorSeen_.assign(N, 0);
    ++epoch_;
    for (const Cand& ca : argmin_) {
        const int a = t_.parent[ca.node];
        if (a >= 0 && anchorSeen_[a] != epoch_) { anchorSeen_[a] = epoch_; ++anchors; }
    }
    return argmin_[rng_() % argmin_.size()];
}

void Placer::materialisePath(int node) {
    path_.clear();
    for (int n = node; n != -1; n = t_.parent[n]) path_.push_back(n);
    std::reverse(path_.begin(), path_.end());

    for (int s = 0; s < nSites_; ++s) W_[s] = sets_.rootSet[s];
    for (size_t i = 1; i < path_.size(); ++i) {
        for (const SetDelta& d : sets_.delta[path_[i]]) W_[d.site - 1] = d.set;
    }
}

// One node's e(v), without the scan: materialise its path and compare W against
// the sequence at every site the sequence is observed at. Two passes over the
// sites, against a scan's pass over the whole tree.
bool Placer::matchesHere(int v) {
    ++st_.cacheProbes;
    materialisePath(v);
    for (int s = 0; s < nSites_; ++s) {
        if (xobs_[s] && W_[s] != xbit_[s]) return false;
    }
    ++st_.cacheHits;
    return true;
}

// --- surgery ----------------------------------------------------------------

// The sequence, as a new observed leaf: its label pins it, and its residues are
// stored as the diff against the root sequence.
int Placer::newLeaf(int parent, const char* seq, const std::string& label, double date) {
    const int L = t_.addNode(parent, "");  // srcName is for input nodes; the label is below
    sets_.delta.emplace_back();
    sets_.v12.emplace_back();
    t_.labels[L].push_back(label);
    t_.date[L] = date;
    std::vector<ObsAllele>& o = t_.obs[L];
    for (int s = 0; s < nSites_; ++s) {
        if (seq[s] != rootSeq_[s]) {
            o.push_back({static_cast<uint16_t>(s + 1), seq[s]});
            // A site that was a single fixed state everywhere becomes polymorphic
            // the moment an observation brings a second allele to it. No back-fill
            // is needed: the stored default is already exact at a monomorphic site.
            if (!isMissing(seq[s])) sets_.siteActive[s] = true;
        }
    }
    st_.maxDegree = std::max(st_.maxDegree, static_cast<long long>(t_.children[parent].size()));
    return L;
}

// A leaf is pinned to its own allele, so V1(L) = {x_s} and V2(L) = {}, and the
// downward rule gives W(L) = {x_s} wherever x is observed -- W(L) can only equal
// its parent's where x is missing there, or where the parent already held exactly
// that one state. One pass over every site, built straight in site order.
void Placer::writeLeafSets(int leaf, const char* seq) {
    std::vector<SetDelta>& d = sets_.delta[leaf];
    d.clear();
    for (int s = 0; s < nSites_; ++s) {
        if (isMissing(seq[s])) continue;  // V1 = all states, so W(L) = W(parent)
        if (W_[s] != xbit_[s]) d.push_back({xbit_[s], static_cast<uint16_t>(s + 1)});
    }
    st_.setWrites += static_cast<long long>(d.size());
}

// W_ holds node w. The sequence becomes a new leaf under it.
//
// Only V1 moves propagate upward, and a new child whose V1 is exactly the
// baseline the sets are stored against cannot move its parent's V1 at all: the
// baseline state's count rises by one and nothing else changes, so V1 and V2
// both hold. So the sites that can move are where the sequence departs from the
// root, where w carries an observation of its own, and where w's V1/V2 had
// already drifted off the default.
int Placer::attachLeaf(int w, const char* seq, const std::string& label, double date) {
    touched_.clear();
    for (int s = 0; s < nSites_; ++s) {
        if (seq[s] != rootSeq_[s]) touched_.push_back(s + 1);
    }
    for (const ObsAllele& o : t_.obs[w])   touched_.push_back(o.site);
    for (const V12Delta& d : sets_.v12[w]) touched_.push_back(d.site);
    sortUnique(touched_);

    const int L = newLeaf(w, seq, label, date);
    writeLeafSets(L, seq);

    // Where w is pinned -- observed and not missing there -- the upward pass
    // stops on its first comparison, so this is cheap rather than skipped: a
    // labelled node that is missing at a site is free there, exactly as in
    // computeAlleleSets, and that site does propagate.
    refresh(touched_, w);
    return L;
}

// A new free node u on the branch p -> v, with the sequence hanging off it as a
// leaf. W_ holds p. u commits to nothing: it is exactly the node Hartigan says
// can hold any of W(p) | W(v), and which of those it ends up holding is step 4's
// business, not this one's.
int Placer::splitBranch(int p, int v, const char* seq, const std::string& label, double date) {
    touched_.clear();
    for (int s = 0; s < nSites_; ++s) {
        if (seq[s] != rootSeq_[s]) touched_.push_back(s + 1);
    }
    for (const ObsAllele& o : t_.obs[v])     touched_.push_back(o.site);
    for (const V12Delta& d : sets_.v12[v])   touched_.push_back(d.site);
    // v's own deltas are stored against p and have to be rewritten against u,
    // even where nothing about v moved.
    for (const SetDelta& d : sets_.delta[v]) touched_.push_back(d.site);
    sortUnique(touched_);

    ++st_.splits;
    const int u = t_.addNode(p, "");
    sets_.delta.emplace_back();
    sets_.v12.emplace_back();
    {
        std::vector<int>& kids = t_.children[p];
        kids.erase(std::find(kids.begin(), kids.end(), v));
    }
    t_.parent[v] = u;
    t_.children[u].push_back(v);
    const int L = newLeaf(u, seq, label, date);

    // V1/V2 at u, then W at u and at v, all still against the unchanged W(p).
    // Anything above that moves is picked up by the two passes below, and the
    // values written here are what their pruning compares against. No upward
    // pass has run yet, so descend() has no path to be forced along.
    chain_.clear();
    for (int site1 : touched_) {
        const int s = site1 - 1;
        AaSet uv1, uv2;
        recomputeV12(u, site1, uv1, uv2);
        setV12(u, site1, uv1, uv2);

        const AaSet newWu = downward(W_[s], uv1, uv2);
        setDelta(u, site1, newWu, W_[s]);

        // v keeps its own V1/V2; only its parent changed.
        descend(v, site1, W_[s], newWu);
        W_[s] = newWu;  // W_ now holds u, which is what the leaf's sets need
    }
    writeLeafSets(L, seq);

    refresh(touched_, p);
    return L;
}

// --- keeping the sets current ------------------------------------------------

// The same rule computeAlleleSets uses, read off one site at a time.
void Placer::recomputeV12(int n, int site1, AaSet& v1, AaSet& v2) const {
    const AaSet allowed = t_.observed(n) ? baseline(n, site1) : kAllStates;
    const std::vector<int>& kids = t_.children[n];
    if (allowed != kAllStates || kids.empty()) {
        v1 = allowed;
        v2 = 0;
        return;
    }
    int count[kNumStates] = {0};
    for (int c : kids) {
        AaSet cv1, cv2;
        getV12(c, site1, cv1, cv2);
        for (AaSet bits = cv1; bits; bits &= bits - 1) ++count[__builtin_ctz(bits)];
    }
    int best = 0;
    for (int a = 0; a < kNumStates; ++a) best = std::max(best, count[a]);
    AaSet s1 = 0, s2 = 0;
    for (int a = 0; a < kNumStates; ++a) {
        if (count[a] == best)          s1 |= AaSet(1) << a;
        else if (count[a] == best - 1) s2 |= AaSet(1) << a;
    }
    v1 = s1;
    v2 = s2;
}

// One site, both passes.
//
// Upward: stop at the first ancestor whose V1 is unchanged. If V1 holds but V2
// moved, store it and stop -- the parent reads only V1.
//
// Downward: from the highest node the upward pass touched. Off the path it
// walked, V1 and V2 are unchanged, so an unchanged W there prunes the subtree.
// On that path they are NOT unchanged, and W can move below a node whose own W
// held still -- a node whose V2 moved while its V1 did not is exactly that case.
// So the descent is forced along the path and prunes only once it steps off it.
// A node's own delta may still need rewriting under a prune, since deltas are
// stored against the parent and the parent's W may have moved.
void Placer::refresh(const std::vector<int>& sites, int start) {
    for (int site1 : sites) {
        chain_.clear();
        int n = start;
        while (n != -1) {
            AaSet nv1, nv2, ov1, ov2;
            recomputeV12(n, site1, nv1, nv2);
            getV12(n, site1, ov1, ov2);
            ++st_.upSteps;
            if (nv1 == ov1 && nv2 == ov2) break;
            setV12(n, site1, nv1, nv2);
            chain_.push_back(n);
            if (nv1 == ov1) break;
            n = t_.parent[n];
        }
        if (chain_.empty()) continue;  // nothing moved; the new nodes are already written
        st_.upHops += static_cast<long long>(chain_.size());
        st_.upHopsMax = std::max(st_.upHopsMax, static_cast<long long>(chain_.size()));

        const int h  = chain_.back();
        const int hp = t_.parent[h];
        if (hp < 0) {
            descend(0, site1, 0, 0);
        } else {
            const AaSet wp = wAt(hp, site1);
            descend(h, site1, wp, wp);
        }
    }
}

void Placer::descend(int n, int site1, AaSet oldWp, AaSet newWp) {
    downStack_.clear();
    downStack_.push_back({n, oldWp, newWp});
    while (!downStack_.empty()) {
        const DownFrame f = downStack_.back();
        downStack_.pop_back();
        ++st_.downSteps;

        AaSet v1, v2;
        getV12(f.node, site1, v1, v2);

        const bool   isRoot = (f.node == 0);
        const AaSet  newW   = isRoot ? v1 : downward(f.newWp, v1, v2);
        const SetDelta* d   = isRoot ? nullptr : deltaAt(f.node, site1);
        const AaSet  oldW   = isRoot ? sets_.rootSet[site1 - 1] : (d ? d->set : f.oldWp);

        if (isRoot) sets_.rootSet[site1 - 1] = newW;
        else        setDelta(f.node, site1, newW, f.newWp);

        const bool onChain =
            std::find(chain_.begin(), chain_.end(), f.node) != chain_.end();
        if (newW == oldW && !onChain) continue;
        for (int c : t_.children[f.node]) downStack_.push_back({c, oldW, newW});
    }
}

// --- the whole placement -----------------------------------------------------

// A match is recorded against the node and the sequence is dropped: nothing in
// the tree, the sets or the score moves, so there is nothing else to do.
Placement Placer::skipAt(int node, double date) {
    ++st_.exactHits;
    ++st_.skipped;
    if (node < static_cast<int>(t_.matched.size())) {
        ++t_.matched[node];
        // The leaf is not built, so its date has nowhere else to live -- and it
        // is the earliest such date that step 4's rule needs, since a leaf can
        // only ever have dated this node no later than itself.
        if (dated(date) && (!dated(t_.matchedDate[node]) || date < t_.matchedDate[node])) {
            t_.matchedDate[node] = date;
        }
    }

    auto it = std::find(matchCache_.begin(), matchCache_.end(), node);
    if (it != matchCache_.end()) matchCache_.erase(it);
    matchCache_.insert(matchCache_.begin(), node);
    if (matchCache_.size() > kMatchCache) matchCache_.pop_back();

    Placement out;
    out.cost        = 0;
    out.exactMatch  = true;
    out.skipped     = true;
    out.matchedNode = node;
    return out;
}

Placement Placer::place(const char* seq, const std::string& label, double date) {
    ++st_.placements;
    for (int s = 0; s < nSites_; ++s) {
        xbit_[s] = aaBit(seq[s]);
        xobs_[s] = isMissing(seq[s]) ? 0 : 1;
    }

    // The nodes the recent matches landed on, tested first. Any matching node
    // settles the placement, so a hit here is the same answer the scan would
    // give, found in two passes over the sites instead of one over the tree.
    if (skipMatches_) {
        for (size_t k = 0; k < matchCache_.size(); ++k) {
            if (matchesHere(matchCache_[k])) return skipAt(matchCache_[k], date);
        }
    }

    int        bestCost = 0;
    int        cands = 0, anchors = 0;
    bool       exact  = false;
    const Cand c      = scan(bestCost, cands, anchors, exact);

    if (exact && skipMatches_) return skipAt(c.node, date);

    Placement out;
    out.cost       = bestCost;
    out.cands      = cands;
    out.anchors    = anchors;
    out.exactMatch = exact;
    if (exact) { ++st_.exactHits; out.matchedNode = c.node; }
    st_.candSum   += cands;
    st_.anchorSum += anchors;
    if (cands > st_.candMax)     st_.candMax = cands;
    if (anchors > st_.anchorMax) st_.anchorMax = anchors;
    if (cands == 1)       ++st_.unique;
    else if (anchors > 1) ++st_.tieAnchors;
    else                  ++st_.tieOneAnchor;

    if (!c.split) {
        materialisePath(c.node);
        out.attached = attachLeaf(c.node, seq, label, date);
    } else {
        const int v = c.node;
        const int p = t_.parent[v];
        materialisePath(p);
        out.attached = splitBranch(p, v, seq, label, date);
        out.split    = true;
    }
    sets_.parsimonyScore += out.cost;
    return out;
}
