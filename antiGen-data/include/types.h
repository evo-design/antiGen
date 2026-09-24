#pragma once

#include <vector>
#include <string>
#include <set>

struct Mutation {
    char from;
    int pos; // 0-indexed
    char to;
};

struct DatasetConfig {
    std::string type;                      // "sc2" or "h3n2"
    std::vector<std::string> segment_names; // {"S"} or {"SigPep", "HA1", "HA2"}
    std::vector<int> segment_lengths;        // {1273} or {16, 329, 221}
    std::vector<int> segment_offsets;        // {0} or {0, 16, 345}
    int total_length;                        // 1273 or 566
    std::string alphabet;                    // e.g. "ACDEFGHIKLMNPQRSTVWY*-" (protein) or "ACGT-N" (nucleotide)
    char ambig_char;                         // ambiguity sentinel: 'X' for protein, 'N' for nucleotide
};

struct Node {
    Node* parent = nullptr;
    std::vector<Node*> children;
    std::vector<Mutation> spike_mutations;
    std::vector<double> leaf_times;
    long long subtree_samples = 0; // samples at or below this node; set by set_subtree_samples()
    // Scratch for --intervals, rewritten once per time interval (see mark_interval).
    bool in_window = false;        // subtree holds a sample collected in interval i
    bool own_in_window = false;    // ...at this node itself
    long long cur_count = 0;       // samples at or below, collected in interval i
    long long next_count = 0;      // samples at or below, collected in interval i+1
    std::string name;
    std::vector<char> seq;
    int idx = -1;
    double time = 0.0;
    std::vector<int> mutation_pos_order;
    int example_id = -1;
    std::set<std::string> possible_clades;
    std::vector<bool> is_descended_from_variant;
    std::vector<bool> is_variant_root;
};

// Forward declare coordinate translation function
int translate_mutation_position(const DatasetConfig& config,
                               const std::string& segment,
                               int segment_pos);
