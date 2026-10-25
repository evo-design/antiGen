// Step 4: the mutation tree as training examples.
//
// Three transformations, in order:
//   a. resolve   -- one parsimonious allele per (node, site), drawn at random
//                   where more than one is parsimonious. The draw is top-down and
//                   conditional on the parent's own draw, which is what makes the
//                   result jointly parsimonious; an independent draw from W(v) is
//                   not, because W(v) answers the full set W(p), not one state.
//   a'. collapse -- an edge whose resolved diff is empty disappears, its labels
//                   merged into the surviving node. Siblings that resolved to the
//                   same sequence are NOT merged: that would cost one mutation per
//                   merge against the parsimony score.
//   b. split     -- every edge carries exactly one mutation, order shuffled,
//                   intermediate nodes of degree 2.
//   c/d. write   -- one example per node **that has children**: its sequence, the
//                   mutation on each child edge as a (position, new_AA) pair,
//                   the position 0-based -- 0..1272 -- and so a direct index
//                   into the sequence on the line above it, and
//                   two flags -- which of the cutoff dates the node is at or
//                   after, and whether it is the root of, or inside, a named
//                   variant.
//                   A childless node's record would carry an empty
//                   MUTATIONS_TO_CHILDREN -- nothing to predict -- so it is left
//                   out. Nothing is lost: its sequence is its parent's with the
//                   parent's own listed child mutation applied.
//
// This is the only place the tree is collapsed. Edges are the diff between
// resolved parent and child, not anything recorded during placement, and most of
// them come out empty: the input tree keeps every zero-mutation edge, and a
// placement that matched an existing genotype is a leaf whose resolved sequence
// is its parent's. Collapsing those is what turns the attachment tree into the
// mutation tree, where every edge carries at least one mutation.
//
// Resolution draws from W, which never holds a gap or a stop, so every sequence
// written is fully specified -- which is also why an edge whose mutations all
// touched missing data resolves to nothing.
#pragma once

#include "hartigan.hpp"
#include "tree.hpp"

#include <cstdint>
#include <string>
#include <vector>

// One date the examples are labelled against: the YYYY-MM-DD label the file
// carries, and the decimal year the comparison is made in.
struct Cutoff {
    std::string label;
    double      when = 0.0;
};

// What step 4 needs beyond the tree and its sets.
struct ExampleOptions {
    uint64_t    seed        = 0;
    // The cutoffs POST_CUTOFF is taken against, ascending. A record lists the
    // ones every observed date below it is at or after, as "<date>,True" lines,
    // exactly as the variant fields list their label -- so an empty list means
    // the examples carry no cutoff flags at all, which is what the train arms
    // want: a train tree stops before its own cutoff, so the flag is vacuous
    // there, and which later cutoff a record falls after is a question only the
    // test tree, which holds every sequence, can answer.
    //
    // Because the flag is `earliest observed date below the node >= D`, the
    // cutoffs a record is at or after are a *prefix* of this list: raising D
    // only ever turns True into False. The writer relies on that and stops at
    // the first cutoff the record misses.
    std::vector<Cutoff> cutoffs;
    // The variant root, as a tree node id, or -1 for no variant at all. Its
    // record carries IS_VARIANT_ROOT, and it and everything below it carry
    // IS_DESCENDED_FROM_VARIANT, both as "<label>,True".
    int         variantRoot = -1;
    std::string variantLabel;
};

struct ExampleStats {
    long long resolvedNodes  = 0;  // nodes left after collapsing and merging
    long long collapsedEdges = 0;  // edges that resolved to no mutation at all
    long long drawn          = 0;  // (node, site) pairs where the draw had a choice
    long long splitEdges     = 0;  // edges that carried more than one mutation
    long long intermediates  = 0;  // degree-2 nodes the split introduced
    long long examples       = 0;  // records written
    long long leafRecords    = 0;  // childless nodes, which are not written
    std::vector<long long> postCutoff;  // records at or after each cutoff, in
                                        // opt.cutoffs order
    long long descended      = 0;  // records inside the variant
    long long variantRoots   = 0;  // records carrying IS_VARIANT_ROOT; must be 0 or 1
    long long mutations      = 0;  // edges after the split
    int       maxEdgeMut     = 0;  // most mutations on one edge before the split
};

ExampleStats writeExamples(const Tree& t, const AlleleSets& sets,
                           const std::string& rootSeq, const ExampleOptions& opt,
                           const std::string& path);
