// Hartigan (1973): the set of alleles at every (node, site) that appears in at
// least one most-parsimonious reconstruction.
//
// Observations are the sample sequences: the input tree's leaves, every placed
// sequence, and the root, which is pinned to the sidecar. Those nodes are pinned
// wherever their sequence is not missing. Every other node is free and is never
// committed to a genotype -- nodes with identical allele sets are expected and
// cost nothing.
//
// Sets are stored as a per-node list of the sites where the node's set differs
// from its parent's, so a site that never varies costs nothing anywhere.
#pragma once

#include "alphabet.hpp"
#include "tree.hpp"

#include <functional>
#include <string>
#include <vector>

struct SetDelta {
    AaSet    set;
    uint16_t site;  // 1-based
};

// The upward pass's V1/V2, kept only where they differ from what the node's own
// allele implies (V1 = {that allele}, V2 = {}), where "that allele" is the node's
// observation if it has one and the root's residue otherwise. That default is
// exact at every monomorphic site, for a free node as much as a pinned one, which
// is what makes it the right thing to store against. Step 3 needs these to resume
// the upward pass at an insertion point; the final W sets alone are not enough.
struct V12Delta {
    AaSet    v1;
    AaSet    v2;
    uint16_t site;  // 1-based
};

struct AlleleSets {
    int                                nSites = 0;
    std::vector<AaSet>                 rootSet;     // one per site; the pinned root
    std::vector<std::vector<SetDelta>> delta;       // per node, ascending site; empty at the root
    std::vector<std::vector<V12Delta>> v12;         // per node, ascending site; see V12Delta
    std::vector<bool>                  siteActive;  // more than one allele observed at this site
    long long                          parsimonyScore = 0;
};

// The allele a node is taken to carry when nothing is stored against it: its own
// observation where it has one, the root's residue otherwise. This is the one
// definition of both the pinning state and the V1/V2 default; it is a lookup, not
// a reconstruction, and at a free node it names nothing but the storage baseline.
char alleleAt(const Tree& t, int node, int site1, const std::string& rootSeq);

AlleleSets computeAlleleSets(const Tree& t, const std::string& rootSeq, int nSites);

// DFS carrying the full set vector materialised in a dense scratch buffer.
// Per-node work is proportional to that node's deltas.
void walkMaterialised(const Tree& t, const AlleleSets& sets,
                      const std::function<void(int node, const AaSet* sets)>& visit);

void writeAlleleSets(const Tree& t, const AlleleSets& sets, const std::string& path);
