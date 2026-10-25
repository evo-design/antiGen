#include "examples.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

namespace {

// The resolved tree: one allele per node per site, every edge carrying at least
// one mutation. Mutations are (site, new allele), the site a 0-based index into
// the sequence -- which is what the written file carries -- and the from-allele
// whatever the parent holds, which the writing pass has in hand anyway.
using ResMut = std::pair<uint16_t, char>;

struct Resolved {
    std::vector<int>                 parent;
    std::vector<std::vector<int>>    children;
    std::vector<std::vector<ResMut>> mut;   // on the branch above, shuffled
    std::vector<int>                 repr;  // the tree node this resolved node came
                                            // from: the topmost of a collapsed run,
                                            // and where its flags are read from
};

// The earlier of two dates, either of which may be kNoDate, which is not a date
// at all rather than an early one.
double earlier(double a, double b) {
    if (!dated(a)) return b;
    if (!dated(b)) return a;
    return a < b ? a : b;
}

char pickFrom(AaSet s, std::mt19937_64& rng) {
    const int n = setSize(s);
    if (n == 0) throw std::runtime_error("empty allele set");
    int k = n == 1 ? 0 : static_cast<int>(rng() % static_cast<uint64_t>(n));
    for (AaSet bits = s; bits; bits &= bits - 1) {
        if (k-- == 0) return kAlphabet[__builtin_ctz(bits)];
    }
    throw std::runtime_error("unreachable");
}

}  // namespace

ExampleStats writeExamples(const Tree& t, const AlleleSets& sets, const std::string& rootSeq,
                           const ExampleOptions& opt, const std::string& path) {
    ExampleStats st;
    st.postCutoff.assign(opt.cutoffs.size(), 0);
    const int N      = t.size();
    const int nSites = sets.nSites;
    if (N == 0) return st;

    // Not the placement seed's own stream: the two shuffles are unrelated.
    std::mt19937_64 rng(opt.seed * 6364136223846793005ULL + 1442695040888963407ULL);

    // ---- the per-node flags, on the tree as it stands ---------------------
    //
    // POST_CUTOFF is True of a cutoff exactly when every leaf below the node is
    // at or after it. Only observations count -- the leaves of the tree as it
    // stands, which are the input tree's tips and the placed sequences, plus
    // the sequences that matched a node and were never attached, whose date has
    // nowhere else to live and is evidence that this genotype existed then. An
    // interior num_date is an inference about a node, not an observation of
    // one, and is not read at all.
    //
    // So minLeafDated[v] is the earliest observed date anywhere below v -- one
    // number per node, from which the flag at *every* cutoff follows, which is
    // why labelling one tree against 24 of them costs nothing beyond the lines
    // written. It is monotone down the tree by construction -- a parent's
    // minimum is no later than any child's -- so at no cutoff does a record
    // before it hang under one at or after it.
    //
    // Children can carry a lower index than their parent: a placement splices a
    // new node *above* an existing one, which on the base tree plus 20,000
    // placements is 8,348 of them. The fold therefore needs a real post-order
    // rather than a walk down the index range.
    std::vector<int> order;
    order.reserve(static_cast<size_t>(N));
    {
        std::vector<int> stack{0};
        while (!stack.empty()) {
            const int v = stack.back();
            stack.pop_back();
            order.push_back(v);
            for (int c : t.children[v]) stack.push_back(c);
        }
    }
    std::vector<double> minLeafDated(N, kNoDate);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int v = *it;
        double    m = t.children[v].empty() ? t.date[v] : kNoDate;
        m = earlier(m, t.matchedDate[v]);
        for (int c : t.children[v]) m = earlier(m, minLeafDated[c]);
        minLeafDated[v] = m;
    }
    // A node and an inferred ancestor of it -- an intermediate on its own
    // incoming edge -- stand above the same leaves, so they get the same date,
    // and so the same flags at every cutoff.

    // Descent is a tree relation, not a label -- but the relation that counts is
    // the one in the *resolved* tree, which the collapse can change. If a
    // placement splices a node into the edge into the variant root and the
    // mutations all land above the splice, the variant root's own edge carries
    // nothing and the two merge; the merged node holds the variant root's
    // genotype, because that is what collapsing means, and anything else hanging
    // off the splice is then genuinely below that genotype. So the walk is left
    // until the resolved tree exists -- see below.

    // ---- a. resolve ------------------------------------------------------
    //
    // One DFS carrying W and the drawn sequence, both dense, both undone on the
    // way back up. A child's draw can differ from its parent's only where W moved
    // (a set-delta) or where the parent itself had a choice, so those sites are
    // the only ones examined. No genotype buffer: the V1/V2 default is a lookup
    // against the node's own observation, and no node is committed to anything.
    std::vector<AaSet> cur(sets.rootSet);
    std::vector<char>  seq(nSites);
    std::vector<int>   ambig;  // sites where the current node's W holds more than one allele

    for (int s = 0; s < nSites; ++s) {
        if (setSize(cur[s]) > 1) {
            ++st.drawn;
            ambig.push_back(s);
        }
        seq[s] = pickFrom(cur[s], rng);
    }

    Resolved R;
    R.parent.push_back(-1);
    R.children.emplace_back();
    R.mut.emplace_back();
    R.repr.push_back(0);

    std::vector<std::pair<int, AaSet>> setUndo;
    std::vector<std::pair<int, char>>  seqUndo;

    struct Frame {
        int              node;
        int              eid;  // resolved id; a collapsed node shares its parent's
        size_t           nextChild;
        size_t           setMark, seqMark;
        std::vector<int> ambigAtEntry;  // the parent's, restored on the way out
    };
    std::vector<Frame> stack;
    stack.push_back({0, 0, 0, 0, 0, {}});

    // Which resolved node each tree node ended up in; a collapsed node shares
    // its parent's. This is what locates the variant root after the collapse.
    std::vector<int> eidOf(N, -1);
    eidOf[0] = 0;

    std::vector<int>    cand, dsites, na;
    std::vector<ResMut> muts;

    while (!stack.empty()) {
        Frame&       top   = stack.back();
        const int    node  = top.node;
        const int    eid   = top.eid;
        const size_t which = top.nextChild;
        if (which >= t.children[node].size()) {
            const Frame done = stack.back();
            stack.pop_back();
            while (setUndo.size() > done.setMark) {
                cur[setUndo.back().first] = setUndo.back().second;
                setUndo.pop_back();
            }
            while (seqUndo.size() > done.seqMark) {
                seq[seqUndo.back().first] = seqUndo.back().second;
                seqUndo.pop_back();
            }
            if (!stack.empty()) ambig = done.ambigAtEntry;
            continue;
        }
        ++top.nextChild;
        const int c = t.children[node][which];

        Frame f;
        f.node         = c;
        f.nextChild    = 0;
        f.setMark      = setUndo.size();
        f.seqMark      = seqUndo.size();
        f.ambigAtEntry = ambig;

        dsites.clear();
        for (const SetDelta& d : sets.delta[c]) dsites.push_back(d.site - 1);
        cand = dsites;
        cand.insert(cand.end(), ambig.begin(), ambig.end());
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

        for (const SetDelta& d : sets.delta[c]) {
            setUndo.emplace_back(d.site - 1, cur[d.site - 1]);
            cur[d.site - 1] = d.set;
        }
        na.clear();
        for (int s : ambig)
            if (!std::binary_search(dsites.begin(), dsites.end(), s)) na.push_back(s);
        for (int s : dsites)
            if (setSize(cur[s]) > 1) na.push_back(s);
        std::sort(na.begin(), na.end());
        ambig.swap(na);

        muts.clear();
        for (int s : cand) {
            const char  a    = seq[s];
            const AaSet bitA = aaBit(a);
            AaSet       v1 = aaBit(alleleAt(t, c, s + 1, rootSeq)), v2 = 0;
            const std::vector<V12Delta>& vv = sets.v12[c];
            auto it = std::lower_bound(vv.begin(), vv.end(), s + 1,
                                       [](const V12Delta& d, int site) { return d.site < site; });
            if (it != vv.end() && it->site == s + 1) {
                v1 = it->v1;
                v2 = it->v2;
            }
            // The parent's draw is one state, so this is the downward rule with
            // W(p) replaced by that state: the result is what a most-parsimonious
            // reconstruction can hold at c given what p was given.
            const AaSet opts = (v1 & bitA) ? bitA : (v1 | (v2 & bitA));
            if (opts & ~cur[s]) {
                throw std::runtime_error("resolved option outside the allele set at node " +
                                         std::to_string(c) + " site " + std::to_string(s + 1));
            }
            if (setSize(opts) > 1) ++st.drawn;
            const char x = pickFrom(opts, rng);
            if (x != a) {
                seqUndo.emplace_back(s, a);
                seq[s] = x;
                muts.push_back({static_cast<uint16_t>(s), x});
            }
        }

        if (muts.empty()) {
            // The resolved sequence is the parent's: an edge the input tree
            // carried no mutation on, a placement that matched what was already
            // there, or an edge whose every mutation touched missing data.
            // Collapse it rather than emit an edge carrying nothing.
            ++st.collapsedEdges;
            f.eid = eid;
        } else {
            const int id = static_cast<int>(R.parent.size());
            R.parent.push_back(eid);
            R.children.emplace_back();
            R.mut.push_back(muts);
            R.repr.push_back(c);
            R.children[eid].push_back(id);
            st.mutations += static_cast<long long>(muts.size());
            f.eid = id;
        }
        eidOf[c] = f.eid;
        stack.push_back(std::move(f));
    }

    // A jointly parsimonious assignment changes state exactly as often as the
    // parsimony score says. Any site the diff above failed to examine would show
    // up here as a shortfall. This is checked before the merge below, which is
    // what makes it a check on the resolution: merging removes edges, so the
    // merged tree is deliberately shorter than the score.
    if (st.mutations != sets.parsimonyScore) {
        throw std::runtime_error("resolved tree carries " + std::to_string(st.mutations) +
                                 " mutations against a parsimony score of " +
                                 std::to_string(sets.parsimonyScore));
    }

    // ---- a'. merge siblings that resolved to the same genotype -----------
    //
    // ---- a''. no sibling merge -------------------------------------------
    //
    // Two children of one node can resolve to the same sequence -- because one was
    // missing where the other was not, or because both are placements of the same
    // genotype -- and they are deliberately left as two records. Merging them was
    // the one thing that made the written mutation count fall short of the
    // parsimony score, by exactly one per merge, because a redundant twin's edge
    // still carries a real mutation. Keeping them makes
    //
    //     mutations written == parsimonyScore
    //
    // true by construction rather than nearly true, at the price of a node listing
    // the same (pos, newAA) twice under MUTATIONS_TO_CHILDREN and two identical
    // records below it. The legacy file carries the same duplication -- 68 sibling
    // duplicates in 347 records -- so this is the convention, not a departure from
    // it. See NOTES, "Duplicate siblings are kept".
    std::vector<int> live(R.parent.size());
    for (size_t i = 0; i < live.size(); ++i) live[i] = static_cast<int>(i);

    // The flags. The earliest observed date below is carried over by each node's
    // representative, and the cutoff lines are derived from it at write time;
    // the variant ones are a walk of the resolved tree.
    const size_t        RN = R.parent.size();
    std::vector<char>   rDesc(RN), rIsRoot(RN);
    std::vector<double> rMin(RN);
    for (size_t i = 0; i < RN; ++i) rMin[i] = minLeafDated[R.repr[i]];

    // The variant root is the topmost resolved node inside the variant: the one
    // the tree's variant root collapsed into, which is itself when its own edge
    // survived. It is unique by construction -- a tree node lands in exactly one
    // resolved node -- where "the resolved node whose representative *is* the
    // tree's variant root" was not: a collapse leaves no such node at all.
    if (opt.variantRoot >= 0) {
        const int vr = eidOf[opt.variantRoot];
        if (vr < 0) {
            throw std::runtime_error("the variant root (node " +
                                     std::to_string(opt.variantRoot) +
                                     ") was never reached by the resolution walk");
        }
        rIsRoot[vr] = 1;
        std::vector<int> stack{vr};
        while (!stack.empty()) {
            const int i = stack.back();
            stack.pop_back();
            rDesc[i] = 1;
            for (int c : R.children[i]) stack.push_back(c);
        }
    }

    st.resolvedNodes = static_cast<long long>(live.size());
    st.mutations     = 0;
    for (int r : live) {
        const size_t k = R.mut[r].size();
        st.mutations += static_cast<long long>(k);
        st.maxEdgeMut = std::max(st.maxEdgeMut, static_cast<int>(k));
        if (k > 1) {
            ++st.splitEdges;
            st.intermediates += static_cast<long long>(k) - 1;
        }
        std::shuffle(R.mut[r].begin(), R.mut[r].end(), rng);
    }

    // ---- b/c/d. split, and write one example per node --------------------
    std::FILE* out = std::fopen(path.c_str(), "wb");
    if (!out) throw std::runtime_error("cannot write " + path);
    std::vector<char> iobuf(1 << 20);
    std::setvbuf(out, iobuf.data(), _IOFBF, iobuf.size());

    std::string rec;
    auto writeRecord = [&](long long id, long long parentId, const ResMut* fromParent, char fromAa,
                           const std::vector<ResMut>& toChildren, double minDate, bool desc,
                           bool isVariantRoot) {
        rec.clear();
        rec += "EXAMPLE_ID: ";
        rec += std::to_string(id);
        rec += "\nPARENT_EXAMPLE_ID: ";
        rec += parentId < 0 ? "None" : std::to_string(parentId);
        rec += "\nMUTATION_FROM_PARENT:\n";
        if (fromParent) {
            rec += fromAa;
            rec += ',';
            rec += std::to_string(fromParent->first);
            rec += ',';
            rec += fromParent->second;
        } else {
            rec += "None";
        }
        rec += "\nPOST_CUTOFF:\n";
        // Ascending, and the predicate is minDate >= cutoff, so the cutoffs that
        // hold are a prefix: the first miss ends the list.
        if (dated(minDate)) {
            for (size_t k = 0; k < opt.cutoffs.size(); ++k) {
                if (minDate < opt.cutoffs[k].when) break;
                rec += opt.cutoffs[k].label;
                rec += ",True\n";
                ++st.postCutoff[k];
            }
        }
        rec += "IS_DESCENDED_FROM_VARIANT:\n";
        if (desc) { rec += opt.variantLabel; rec += ",True\n"; }
        rec += "IS_VARIANT_ROOT:\n";
        if (isVariantRoot) { rec += opt.variantLabel; rec += ",True\n"; }
        rec.append(seq.begin(), seq.end());
        rec += "\nMUTATIONS_TO_CHILDREN:\n";
        for (const ResMut& m : toChildren) {
            rec += std::to_string(m.first);
            rec += ',';
            rec += m.second;
            rec += '\n';
        }
        rec += '\n';
        if (std::fwrite(rec.data(), 1, rec.size(), out) != rec.size()) {
            throw std::runtime_error("short write to " + path);
        }
        ++st.examples;
        st.descended    += desc ? 1 : 0;
        st.variantRoots += isVariantRoot ? 1 : 0;
    };

    // The mutation each child edge leads with, which is what this node's record
    // lists -- after the split a child edge is only its first mutation.
    std::vector<ResMut> heads;
    auto headsOf = [&](int r) {
        heads.clear();
        for (int g : R.children[r]) heads.push_back(R.mut[g].front());
        return heads;
    };

    struct WFrame {
        int    node;
        long long exampleId;
        size_t nextChild;
        size_t seqMark;
    };
    std::vector<WFrame> wstack;
    seqUndo.clear();
    long long nextId = 0;

    // A node with no children carries an empty MUTATIONS_TO_CHILDREN: there is
    // nothing to predict from it, so it is not written, and it takes no id with
    // it. Nothing becomes unrecoverable -- its sequence is its parent's with the
    // mutation the parent already lists applied. The root goes the same way if it
    // has no children, which only happens on a tree with no mutations at all.
    const std::vector<ResMut> rootHeads = headsOf(0);
    long long rootId = -1;
    if (rootHeads.empty()) {
        ++st.leafRecords;
    } else {
        rootId = nextId++;
        writeRecord(rootId, -1, nullptr, 0, rootHeads, rMin[0], rDesc[0], rIsRoot[0]);
    }
    wstack.push_back({0, rootId, 0, 0});

    while (!wstack.empty()) {
        WFrame&      wtop  = wstack.back();
        const int    r     = wtop.node;
        const size_t which = wtop.nextChild;
        if (which >= R.children[r].size()) {
            const WFrame done = wstack.back();
            wstack.pop_back();
            while (seqUndo.size() > done.seqMark) {
                seq[seqUndo.back().first] = seqUndo.back().second;
                seqUndo.pop_back();
            }
            continue;
        }
        ++wtop.nextChild;
        const int                   c    = R.children[r][which];
        const std::vector<ResMut>&  ms   = R.mut[c];
        const size_t                mark = seqUndo.size();
        long long                   parentId = wtop.exampleId;
        long long                   myId     = -1;

        for (size_t i = 0; i < ms.size(); ++i) {
            const int  s    = ms[i].first;
            const char from = seq[s];
            seqUndo.emplace_back(s, from);
            seq[s] = ms[i].second;

            if (i + 1 < ms.size()) {
                // An intermediate node: degree 2, so exactly one child mutation,
                // so never childless and always written.
                myId = nextId++;
                const std::vector<ResMut> one{ms[i + 1]};
                // An inferred ancestor of c on c's own edge: it stands above
                // everything c stands above, so it carries c's own cutoff flags,
                // and it is inside the variant only if the edge started inside it.
                writeRecord(myId, parentId, &ms[i], from, one, rMin[c], rDesc[r], false);
                parentId = myId;
            } else {
                const std::vector<ResMut> h = headsOf(c);
                if (h.empty()) {
                    ++st.leafRecords;
                    myId = -1;  // nothing hangs below it, so no record refers to it
                } else {
                    myId = nextId++;
                    writeRecord(myId, parentId, &ms[i], from, h, rMin[c], rDesc[c],
                                rIsRoot[c]);
                }
            }
        }
        wstack.push_back({c, myId, 0, mark});
    }

    std::fclose(out);

    // The resolved variant root is unique, but a childless node is not written,
    // so the record can still fail to exist. It would take a variant whose root
    // genotype has nothing below it, which is not a variant -- assert rather
    // than assume.
    if (opt.variantRoot >= 0 && st.variantRoots != 1) {
        throw std::runtime_error(
            "the variant root is on " + std::to_string(st.variantRoots) +
            " records, not one: its resolved node has no children, so it is a childless "
            "node and childless nodes are not written");
    }
    return st;
}
