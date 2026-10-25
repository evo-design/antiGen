#include "hartigan.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace {

// Parents always precede their children, so one forward scan propagates down
// the tree and one backward scan propagates up it.
std::vector<int> preorderOf(const Tree& t) {
    std::vector<int> order;
    if (t.size() == 0) return order;
    order.reserve(t.size());
    std::vector<int> stack{0};
    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        order.push_back(v);
        for (int c : t.children[v]) stack.push_back(c);
    }
    return order;
}

}  // namespace

char alleleAt(const Tree& t, int node, int site1, const std::string& rootSeq) {
    const std::vector<ObsAllele>& o = t.obs[node];
    size_t lo = 0, hi = o.size();
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (o[mid].site < site1) lo = mid + 1;
        else                     hi = mid;
    }
    return (lo < o.size() && o[lo].site == site1) ? o[lo].aa : rootSeq[site1 - 1];
}

AlleleSets computeAlleleSets(const Tree& t, const std::string& rootSeq, int nSites) {
    AlleleSets out;
    const int N = t.size();
    out.nSites = nSites;
    out.rootSet.assign(nSites, kAllStates);
    out.delta.assign(N > 0 ? N : 0, {});
    out.v12.assign(N > 0 ? N : 0, {});
    out.siteActive.assign(nSites, false);
    if (N == 0) return out;
    if (static_cast<int>(rootSeq.size()) < nSites) {
        throw std::runtime_error("root sequence is shorter than the site count");
    }
    for (int s = 0; s < nSites; ++s) out.rootSet[s] = aaBit(rootSeq[s]);

    // Which observations depart from the root at each site. An observed node with
    // no entry here carries the root's residue; a free node has no entry anywhere.
    std::vector<std::vector<std::pair<int, char>>> siteObs(nSites);
    for (int v = 0; v < N; ++v) {
        for (const ObsAllele& o : t.obs[v]) {
            if (o.site >= 1 && o.site <= nSites) siteObs[o.site - 1].emplace_back(v, o.aa);
        }
    }

    // Whether more than one real allele is observed at a site. A monomorphic site
    // generates no set-deltas, but it is still solved: missing data can put V1/V2
    // off their default there, and step 3 resumes the upward pass from them.
    for (int s = 0; s < nSites; ++s) {
        AaSet seen = isMissing(rootSeq[s]) ? 0 : aaBit(rootSeq[s]);
        for (const auto& [v, aa] : siteObs[s]) {
            (void)v;
            if (!isMissing(aa)) seen |= aaBit(aa);
        }
        out.siteActive[s] = setSize(seen) > 1;
    }

    const std::vector<int> pre = preorderOf(t);
    std::vector<char>  isObs(N);
    for (int v = 0; v < N; ++v) isObs[v] = t.observed(v) ? 1 : 0;
    std::vector<char>  ovr(N, 0);  // this site's observation, where it leaves the root
    std::vector<AaSet> v1(N), v2(N), w(N);
    std::vector<int>   gmin(N);    // subtree cost given the node takes a V1 state

    for (int s = 0; s < nSites; ++s) {
        for (const auto& [v, aa] : siteObs[s]) ovr[v] = aa;
        const char  rootAa  = rootSeq[s];
        const AaSet rootBit = aaBit(rootAa);

        // The allele a node is stored against: its observation where it has one,
        // the root's residue otherwise. A free node is confined by nothing.
        auto defaultBit = [&](int v) { return ovr[v] ? aaBit(ovr[v]) : rootBit; };

        // Upward pass. v1 holds the states of minimum subtree cost, v2 those one
        // step worse; a node pinned by an observation admits only that state.
        for (size_t i = pre.size(); i-- > 0;) {
            const int   v       = pre[i];
            const AaSet allowed = isObs[v] ? defaultBit(v) : kAllStates;
            const std::vector<int>& kids = t.children[v];
            if (kids.empty()) {
                v1[v] = allowed;
                v2[v] = 0;
                gmin[v] = 0;
                continue;
            }
            int count[kNumStates] = {0};
            int below = 0;
            for (int c : kids) {
                below += gmin[c];
                for (AaSet bits = v1[c]; bits; bits &= bits - 1) ++count[__builtin_ctz(bits)];
            }
            const int k = static_cast<int>(kids.size());
            if (allowed != kAllStates) {
                v1[v] = allowed;
                v2[v] = 0;
                gmin[v] = below + k - count[__builtin_ctz(allowed)];
            } else {
                int best = 0;
                for (int a = 0; a < kNumStates; ++a) best = std::max(best, count[a]);
                AaSet s1 = 0, s2 = 0;
                for (int a = 0; a < kNumStates; ++a) {
                    if (count[a] == best)          s1 |= AaSet(1) << a;
                    else if (count[a] == best - 1) s2 |= AaSet(1) << a;
                }
                v1[v] = s1;
                v2[v] = s2;
                gmin[v] = below + k - best;
            }
        }

        // Record V1/V2 wherever the node's own allele does not already imply them.
        for (int v = 0; v < N; ++v) {
            if (v1[v] != defaultBit(v) || v2[v] != 0) {
                out.v12[v].push_back({v1[v], v2[v], static_cast<uint16_t>(s + 1)});
            }
        }

        // Downward pass. A node can hold any state its parent holds that costs it
        // at most one extra step; otherwise it is confined to its own optimum.
        w[0] = v1[0];
        out.rootSet[s] = w[0];
        for (size_t i = 1; i < pre.size(); ++i) {
            int v = pre[i];
            AaSet wp = w[t.parent[v]];
            w[v] = ((wp & ~v1[v]) == 0) ? wp : (v1[v] | (wp & v2[v]));
            if (w[v] != wp) out.delta[v].push_back({w[v], static_cast<uint16_t>(s + 1)});
        }
        out.parsimonyScore += gmin[0];

        for (const auto& [v, aa] : siteObs[s]) { (void)aa; ovr[v] = 0; }
    }
    return out;
}

void walkMaterialised(const Tree& t, const AlleleSets& sets,
                      const std::function<void(int, const AaSet*)>& visit) {
    const int N = t.size();
    if (N == 0) return;

    std::vector<AaSet> cur(sets.rootSet);
    std::vector<std::pair<int, AaSet>> setUndo;

    struct Frame {
        int    node;
        size_t nextChild;
        size_t setMark;
    };
    std::vector<Frame> stack;

    visit(0, cur.data());
    stack.push_back({0, 0, 0});

    while (!stack.empty()) {
        const int node = stack.back().node;
        const size_t which = stack.back().nextChild;
        if (which < t.children[node].size()) {
            ++stack.back().nextChild;
            const int c = t.children[node][which];
            const size_t setMark = setUndo.size();
            for (const SetDelta& d : sets.delta[c]) {
                setUndo.emplace_back(d.site - 1, cur[d.site - 1]);
                cur[d.site - 1] = d.set;
            }
            visit(c, cur.data());
            stack.push_back({c, 0, setMark});
        } else {
            const Frame done = stack.back();
            stack.pop_back();
            while (setUndo.size() > done.setMark) {
                cur[setUndo.back().first] = setUndo.back().second;
                setUndo.pop_back();
            }
        }
    }
}

void writeAlleleSets(const Tree& t, const AlleleSets& sets, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);

    std::string root;
    for (int s = 0; s < sets.nSites; ++s) {
        std::string letters = setToString(sets.rootSet[s]);
        root.push_back(letters.size() == 1 ? letters[0] : '?');
    }
    std::fprintf(f, "#sites\t%d\n", sets.nSites);
    std::fprintf(f, "#root\t%s\n", root.c_str());
    std::fprintf(f, "#node\tparent\tn_delta\tdeltas\n");

    for (int i = 0; i < t.size(); ++i) {
        std::string deltas;
        for (size_t k = 0; k < sets.delta[i].size(); ++k) {
            if (k) deltas += ',';
            deltas += std::to_string(sets.delta[i][k].site);
            deltas += ':';
            deltas += setToString(sets.delta[i][k].set);
        }
        std::fprintf(f, "%d\t%d\t%zu\t%s\n", i, t.parent[i], sets.delta[i].size(),
                     deltas.empty() ? "." : deltas.c_str());
    }
    std::fclose(f);
}
