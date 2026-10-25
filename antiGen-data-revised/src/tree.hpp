// The tree: the Nextstrain phylogeny, exactly as given. Nothing is collapsed and
// no node is committed to a genotype -- the mutation tree exists only at step 4,
// derived from the resolved sequences. See NOTES, "the collapse moves to the end".
//
// Observations are the leaves of the input tree, plus the root, which is pinned to
// the root-sequence sidecar. Every other node is free. An observed node stores its
// sequence as a sparse diff against the root sequence, so the allele at a
// (node, site) is a binary search rather than a walk down from the root.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// An amino-acid substitution on the S protein, e.g. "D614G" -> {614, 'D', 'G'}.
// Sites are 1-based, matching Nextstrain's numbering. Only the parse uses these:
// they build the leaf observations and then stop being part of the tree.
struct Mutation {
    int  site;
    char from;
    char to;
};

std::string formatMutation(const Mutation& m);

// A date that is not known: a free node has one, and so does any node the input
// tree left without a num_date. It compares below every real date, which is what
// the step-4 rule needs -- see NOTES, "a node with no date of its own".
inline constexpr double kNoDate = -1e18;
inline bool dated(double d) { return d > kNoDate; }

// A node of the input Nextstrain tree, stripped down to what we use.
struct RawNode {
    std::string           name;
    std::vector<Mutation> sMut;      // S mutations on the branch above this node
    std::vector<int>      children;  // indices into the RawNode array
    double                numDate = kNoDate;  // node_attrs.num_date.value, decimal year
    std::string           clade;     // node_attrs.clade_membership.value
};

// One residue of an observed sequence that differs from the root's. Missing data
// is an observation too and is stored as 'X', 'the state is unknown here'.
struct ObsAllele {
    uint16_t site;  // 1-based
    char     aa;
};

// Nodes live in flat, parallel arrays; node 0 is always the root. Input node ids
// are tree node ids, one to one, because nothing is collapsed.
struct Tree {
    std::vector<int>                      parent;    // -1 at the root
    std::vector<std::vector<int>>         children;
    std::vector<std::vector<std::string>> labels;    // the samples observed here
    std::vector<std::vector<ObsAllele>>   obs;       // ascending by site; empty at a free node
    std::vector<std::string>              srcName;   // the Nextstrain node, for diagnostics
    // Decimal year. The input tree's own num_date at every node it came with, the
    // collection date at a placed leaf, kNoDate at a node placement created. Step
    // 4 reads it, nothing else does.
    std::vector<double>                   date;
    // How many placed sequences matched this node's allele sets exactly and were
    // skipped rather than hung off it as a leaf. Provenance only: it pins nothing,
    // it is not an observation, and nothing in the algorithm reads it.
    std::vector<uint32_t>                 matched;
    // The earliest date among those skipped sequences. A skipped leaf would have
    // dated this node no later than itself, so dropping it must not lose that --
    // this is where the date survives when the node does not.
    std::vector<double>                   matchedDate;

    int size() const { return static_cast<int>(parent.size()); }

    // The central predicate. The root is the sidecar sequence; every other
    // observation arrived as a sample label, on an input leaf or a placement.
    // A placed leaf that later acquires children stays observed, which is why
    // this is not "is a leaf".
    bool observed(int v) const { return v == 0 || !labels[v].empty(); }

    int addNode(int par, const std::string& name);
};

// Reads a Nextstrain v2 dataset and keeps only name / children / <gene> mutations.
// Mutations beyond maxSite are dropped (maxSite == 0 keeps everything).
std::vector<RawNode> parseNextstrainTree(const std::string& path, const std::string& gene,
                                         int maxSite = 0);

// Reads the root-sequence sidecar and returns its <gene> entry.
std::string parseRootSequenceS(const std::string& path, const std::string& gene);

// The input tree, one node per input node, with each leaf's sequence recorded as
// its diff against rootSeq. `mismatches` collects branches whose from-allele
// disagrees with the state inherited down the path, which is a check on the input.
Tree buildTree(const std::vector<RawNode>& raw, const std::string& rootSeq, int nSites,
               std::vector<std::string>* mismatches);

// Nextstrain's own genotype at every node, walked down from the root sequence.
// This is an inference, not data, and nothing in the algorithm reads it; `sets`
// reports how often it lands inside the allele sets as a sanity statistic.
std::vector<std::string> nextstrainGenotypes(const std::vector<RawNode>& raw,
                                             const std::string& rootSeq, int nSites);

// Where a named clade begins: the one node carrying the clade whose parent does
// not. `variant` matches a clade label exactly, or as "(variant)" inside it --
// the same rule antiGen's own loader uses, so "BA.2" finds "21L (BA.2)".
//
// Returns the node id. `label` takes the matched clade label, `nMatched` the
// number of nodes carrying it, `entries` every entry point found. Throws if
// there is no match or more than one entry point: a clade that begins twice is
// not a variant root, and is for the caller to look at rather than for this to
// pick between.
int findCladeRoot(const std::vector<RawNode>& raw, const std::string& variant,
                  std::string* label, int* nMatched, std::vector<int>* entries);

void writeTree(const Tree& t, const std::string& path);
