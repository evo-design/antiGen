#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <random>
#include <set>
#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <climits>
#include "json.hpp"
#include "types.h"
#include "read_fasta.h"
#include "attach.h"
#include "cache.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using json = nlohmann::json;

// Protein alphabet used by default. For nucleotide datasets this global is
// overwritten from the DatasetConfig alphabet in main() before it is used by
// aa_to_index / the PSSM code.
std::string ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY*-";

int aa_to_index(char aa) {
    size_t pos = ALLOWED_AAS.find(aa);
    if(pos == std::string::npos) {
        throw std::runtime_error(std::string("Invalid amino acid: ") + aa);
    }
    return (int)pos;
}

Mutation parse_mutation(const std::string& s) {
    char from = s.front();
    char to = s.back();
    int pos = std::stoi(s.substr(1, s.size() - 2)) - 1;
    return {from, pos, to};
}

int translate_mutation_position(const DatasetConfig& config,
                               const std::string& segment,
                               int segment_pos) {
    if (config.segment_names.size() == 1) {
        return segment_pos;  // Single-segment dataset, no translation needed
    }

    // Multi-segment: convert segment coordinates to concatenated coordinates
    for (size_t i = 0; i < config.segment_names.size(); i++) {
        if (config.segment_names[i] == segment) {
            return config.segment_offsets[i] + segment_pos;
        }
    }
    throw std::runtime_error("Unknown segment: " + segment);
}

DatasetConfig create_dataset_config(const std::string& type) {
    const std::string AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-";
    // Nucleotide alphabet: A/C/G/T + gap + N (ambiguity sentinel).
    const std::string NUC_ALPHABET = "ACGT-N";
    if (type == "sc2") {
        return {
            .type = "sc2",
            .segment_names = {"S"},
            .segment_lengths = {1273},
            .segment_offsets = {0},
            .total_length = 1273,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "sc2_nuc") {
        // Whole-genome SARS-CoV-2 nucleotide (Wuhan-Hu-1 / MN908947, 29903 nt).
        // Mutations live under branch_attrs.mutations.nuc in Auspice JSON; the
        // root sequence is taken from ref["nuc"] (no stop-codon stripping).
        return {
            .type = "sc2_nuc",
            .segment_names = {"nuc"},
            .segment_lengths = {29903},
            .segment_offsets = {0},
            .total_length = 29903,
            .alphabet = NUC_ALPHABET,
            .ambig_char = 'N'
        };
    } else if (type == "sc2_spike_nuc") {
        // Spike-only SARS-CoV-2 nucleotide (NC_045512 21563-25384, rebased to
        // 1..3822). Same handling as sc2_nuc, shorter total_length.
        return {
            .type = "sc2_spike_nuc",
            .segment_names = {"nuc"},
            .segment_lengths = {3822},
            .segment_offsets = {0},
            .total_length = 3822,
            .alphabet = NUC_ALPHABET,
            .ambig_char = 'N'
        };
    } else if (type == "toy") {
        // Tiny self-contained amino-acid dataset for the example/ demo + smoke test
        // (see example/). Single "protein" segment, length 25; emits AA examples so
        // the toy trains the model on the same 22-token alphabet as the real datasets.
        return {
            .type = "toy",
            .segment_names = {"protein"},
            .segment_lengths = {25},
            .segment_offsets = {0},
            .total_length = 25,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "h3n2") {
        return {
            .type = "h3n2",
            .segment_names = {"SigPep", "HA1", "HA2"},
            .segment_lengths = {16, 329, 221},
            .segment_offsets = {0, 16, 345},
            .total_length = 566,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "rsv_b") {
        return {
            .type = "rsv_b",
            .segment_names = {"F"},
            .segment_lengths = {574},
            .segment_offsets = {0},
            .total_length = 574,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "rsv_a") {
        return {
            .type = "rsv_a",
            .segment_names = {"F"},
            .segment_lengths = {574},
            .segment_offsets = {0},
            .total_length = 574,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else if (type == "denv2") {
        return {
            .type = "denv2",
            .segment_names = {"E"},
            .segment_lengths = {495},
            .segment_offsets = {0},
            .total_length = 495,
            .alphabet = AA_ALPHABET,
            .ambig_char = 'X'
        };
    } else {
        throw std::runtime_error("Unknown dataset type: " + type + ". Supported: sc2, sc2_nuc, sc2_spike_nuc, toy, h3n2, rsv_a, rsv_b, denv2");
    }
}

Node* build_tree(const json& j, Node* parent, const DatasetConfig& config) {
    Node* node = new Node();
    node->parent = parent;
    node->name = j.at("name").get<std::string>();

    // Parse clade membership - use subclade for H3N2 and DENV2, clade_membership otherwise
    if (j.contains("node_attrs")) {
        std::string clade_field = (config.type == "h3n2" || config.type == "denv2") ? "subclade" : "clade_membership";
        if (j["node_attrs"].contains(clade_field) &&
            j["node_attrs"][clade_field].contains("value")) {
            node->possible_clades = {j["node_attrs"][clade_field]["value"].get<std::string>()};
        }
    }

    // Parse mutations for all segments in this dataset
    if (j.contains("branch_attrs") && j["branch_attrs"].contains("mutations")) {
        for (const std::string& segment : config.segment_names) {
            if (j["branch_attrs"]["mutations"].contains(segment)) {
                for (const auto& m : j["branch_attrs"]["mutations"][segment]) {
                    std::string mut_str = m.get<std::string>();
                    Mutation mut = parse_mutation(mut_str);
                    mut.pos = translate_mutation_position(config, segment, mut.pos);

                    // Skip mutations beyond the sequence length (e.g. stop-codon
                    // position after stripping the trailing '*').
                    if (mut.pos < 0 || mut.pos >= config.total_length) continue;

                    // Skip mutations involving the ambiguity sentinel ('X'/'N').
                    if (mut.from == config.ambig_char || mut.to == config.ambig_char) continue;

                    node->spike_mutations.push_back(mut);  // Keep field name for compatibility
                }
            }
        }
    }

    // Recurse on children
    if (j.contains("children")) {
        for (const auto& child : j["children"]) {
            Node* c = build_tree(child, node, config);
            node->children.push_back(c);
        }
    }

    // Set estimated node time
    node->time = j.at("node_attrs").at("num_date").at("value").get<double>();

    // If this is too early compared to parent, update
    if(node->parent != nullptr) {
        if(node->time < node->parent->time) {
            std::cout << "Warning: child time before parent time; setting equal to parent time" << std::endl;
            node->time = node->parent->time;
        }
    }

    // If leaf node, also set leaf times
    if (node->children.empty()) {
        node->leaf_times.push_back(node->time);
    }

    return node;
}

void set_times_rel_root(Node* node, double root_time) {

    // Update node's time
    node->time -= root_time;
    if(node->time < 0.0) throw std::runtime_error("Negative node time");

    // Update leaf times
    for(int j = 0; j < node->leaf_times.size(); j++) {
        node->leaf_times[j] -= root_time;
        if(node->leaf_times[j] < 0.0) {
            throw std::runtime_error("Negative leaf time");
        }
    }
    for(Node* n : node->children) {
        set_times_rel_root(n, root_time);
    }
}

void set_node_times(Node* node) {

    std::vector<double> child_times;
    for(Node* n : node->children) {
        set_node_times(n);
        child_times.push_back(n->time);
    }

    if(node->children.size() == 0) {

        if(node->leaf_times.size() == 0) {
            throw std::runtime_error("Node with no children must have at least one leaf time");
        }

        node->time = *std::min_element(node->leaf_times.begin(), node->leaf_times.end());

    }else{
        node->time = *std::min_element(child_times.begin(), child_times.end());
    }
}

void count_nodes(const Node* node, int& total, int& leaves) {
    total++;
    if (node->children.empty()) leaves++;
    for (const Node* c : node->children) {
        count_nodes(c, total, leaves);
    }
}

void index_nodes(Node* node, int& idx, std::map<int, Node*>& idx_to_node) {
    node->idx = idx;
    idx_to_node[idx] = node;
    idx++;
    for(Node* n : node->children) {
        index_nodes(n, idx, idx_to_node);
    }
}

void spike_seqs_to_host(Node* node, char* h_sequences, std::vector<int>& h_parent, long long L) {
    std::memcpy(h_sequences + ((long long)node->idx * L), node->seq.data(), L * sizeof(char));
    if(node->parent != nullptr) {
        h_parent[node->idx] = node->parent->idx;
    }

    for(Node* n : node->children) {
        spike_seqs_to_host(n, h_sequences, h_parent, L);
    }

}

void delete_tree(Node* node) {
    for (Node* c : node->children) {
        delete_tree(c);
    }
    delete node;
}

// Via post-order traversal, prune branches with zero mutations
// Note this function doesn't affect node times, just may update leaves assigned to each node
void prune_branches_without_mutations(Node* node) {

    // post-order traversal (copy vector since children may be modified)
    std::vector<Node*> kids = node->children;
    for(Node* n : kids) {
        prune_branches_without_mutations(n);
    }

    // If node has no mutations and has a parent, prune
    if(node->spike_mutations.size() == 0 && node->parent != nullptr) {
        Node* parent = node->parent;
        parent->children.erase(std::remove(parent->children.begin(), parent->children.end(), node), parent->children.end());
        for(Node* n : node->children) {
            n->parent = parent;
            parent->children.push_back(n);
        }
        for(double t : node->leaf_times) {
            parent->leaf_times.push_back(t);
        }
        // Clade assignment: prioritize child
        for(std::string c : node->possible_clades) {
            parent->possible_clades.insert(c);
        }

        delete node;
    }
}

void check_prune(Node* node, int& total, int& leaf_count) {

    for(Node* n : node->children) {
        check_prune(n, total, leaf_count);
    }

    // If node has no mutations and has a parent, prune
    if(node->parent != nullptr) {
        if(node->spike_mutations.size() == 0) {
            throw std::runtime_error("Branch with 0 mutations");
        }
    }
    total++;
    leaf_count += node->leaf_times.size();
}

void assign_spike_sequences(Node* node) {

    if(node->parent != nullptr) {

        if(node->spike_mutations.size() == 0) {
            throw std::runtime_error("Branch with no mutations on original tree");
        }


        std::vector<char> new_seq = node->parent->seq;
        for(Mutation mut : node->spike_mutations) {
            if(new_seq[mut.pos] != mut.from) {
                throw std::runtime_error("Mutation mismatch!");
            }
            new_seq[mut.pos] = mut.to;
        }
        node->seq = new_seq;

        if(node->seq == node->parent->seq) {
            throw std::runtime_error("Sequences identical parent to child");
        }
    }

    for(Node* n : node->children) {
        assign_spike_sequences(n);
    }
}

void update_pssm(
    Node* node,
    char* h_sequences,
    double cutoff_time,
    double lower_cutoff_time,
    std::vector<int>& pssm,
    int L
) {

    if(node->leaf_times.size() == 0) return;

    int multiplicity = 0;
    for(double t : node->leaf_times) {
        if(t <= cutoff_time && t > lower_cutoff_time) {
            multiplicity++;
        }
    }

    if(multiplicity == 0) return;

    long long idx_offset = (long long)node->idx * L;
    for(int pos = 0; pos < L; pos++) {
        char aa = h_sequences[idx_offset + pos];
        pssm[pos * ALLOWED_AAS.size() + aa_to_index(aa)] += multiplicity;
    }
}

// Compile PSSMs for root and variant
void compile_pssms(
    Node* node,
    char* h_sequences,
    std::vector<int>& pssm_root, // matrix (flat)
    std::vector<std::vector<std::vector<int>>>& pssm_variants, // variant, timestep, matrix (flat)
    double cutoff,
    int L,
    const std::vector<std::string>& variants,
    double timestep = 0.1,
    int n_steps = 60,
    double start_time = 0.0,
    bool variant_pssms_post_cutoff = false
) {


    // Root PSSM
    update_pssm(node, h_sequences, cutoff, -INFINITY, pssm_root, L);

    // PSSM for each variant
    double variant_lower_time = variant_pssms_post_cutoff ? cutoff : -INFINITY;
    for(int v = 0; v < (int)variants.size(); v++) {
        if(!node->is_descended_from_variant[v]) continue;
        for(int i = 0; i < n_steps; i++) {
            update_pssm(node, h_sequences, start_time + ((double)i) * timestep, variant_lower_time, pssm_variants[v][i], L);
        }
    }

    for(Node* n : node->children) {
        compile_pssms(n, h_sequences, pssm_root, pssm_variants, cutoff, L, variants, timestep, n_steps, start_time, variant_pssms_post_cutoff);
    }
}




void set_variant_descendants(Node* node, const std::vector<std::string>& variants) {
    for(int i = 0; i < (int)variants.size(); i++) {
        if(node->possible_clades.find(variants[i]) != node->possible_clades.end()) {
            node->is_descended_from_variant.push_back(true);
        }else if(node->parent != nullptr) {
            node->is_descended_from_variant.push_back(node->parent->is_descended_from_variant[i]);
        }else{
            node->is_descended_from_variant.push_back(false);
        }

        if(
            node->parent != nullptr &&
            node->is_descended_from_variant[i] && 
            !node->parent->is_descended_from_variant[i]
        ) {
            node->is_variant_root.push_back(true);
        }else{
            node->is_variant_root.push_back(false);
        }
    }


    for(Node* n : node->children) {
        set_variant_descendants(n, variants);
    }
}

Node* prune_node(Node* node) {
    Node* original_parent = node->parent;
    if(original_parent == nullptr) throw std::runtime_error("Cannot prune a node with no parent");

    Node* new_node = new Node;
    new_node->children = {node};
    new_node->parent = nullptr;
    new_node->idx = original_parent->idx; // Map to the same sequence idx
    new_node->time = node->time;
    new_node->is_descended_from_variant = original_parent->is_descended_from_variant;
    new_node->is_variant_root = original_parent->is_variant_root;

    // Re-wire parent of node
    node->parent = new_node;

    // Update children of original parent
    original_parent->children.erase(std::remove(original_parent->children.begin(), original_parent->children.end(), node), original_parent->children.end());

    return new_node;

}

void prune_tree(Node* node, std::vector<Node*>& pruned_roots, double cutoff) {
    if(node->parent != nullptr) {
        if(node->time > cutoff && node->parent->time <= cutoff) {
            pruned_roots.push_back(prune_node(node));
        }
    }
    std::vector<Node*> kids = node->children;
    for(Node* n : kids) {
        prune_tree(n, pruned_roots, cutoff);
    }
}

void cleanup_pruned_tree(Node* node) {

    std::vector<Node*> kids = node->children;
    for(Node* n : kids) {
        cleanup_pruned_tree(n);
    }

    if(
        node->leaf_times.size() == 0 && 
        node->parent != nullptr &&
        node->children.size() <= 1
    ) {
        
        Node* parent = node->parent;
        parent->children.erase(std::remove(parent->children.begin(), parent->children.end(), node), parent->children.end());

        if(node->children.size() == 1) {
            Node* child = node->children[0];
            parent->children.push_back(child);
            child->parent = parent;
        }

        // Either way, no longer need this node
        delete node;
        
    }
}

// Number of samples at or below a node, cached on the node itself.
//
// Run this after prune_tree/cleanup_pruned_tree, so that each tree is counted on its own
// side of the train/test split: the training tree keeps the samples that were there before
// the split, and every pruned subtree holds only samples from after it. A sample is one
// entry of leaf_times -- one observed sequence, either an original tree tip or one attached
// from the alignment -- so a node carrying k identical sequences contributes k.
long long set_subtree_samples(Node* node) {
    long long total = (long long)node->leaf_times.size();
    for(Node* n : node->children) {
        total += set_subtree_samples(n);
    }
    node->subtree_samples = total;
    return total;
}

// Samples sitting in a tree whose own date is past the cutoff. The split is made on node
// times, and a node's time is the earliest date in its subtree, so a node kept on the
// training side may still carry a late duplicate of an early genotype. Reported as a
// diagnostic: it is the gap between "samples on the training tree" and "samples dated
// before the cutoff".
void count_samples_after(const Node* node, double cutoff, long long& late) {
    for(double t : node->leaf_times) {
        if(t > cutoff) late++;
    }
    for(const Node* n : node->children) {
        count_samples_after(n, cutoff, late);
    }
}

void set_mutation_pos_order(Node* node, char* h_sequences, long long L, std::mt19937& rng) {

    if(node->parent != nullptr) {
        std::vector<int> mutation_pos_order;
        long long idx = node->idx;
        long long parent_idx = node->parent->idx;
        for(int pos = 0; pos < L; pos++) {
            if(h_sequences[idx*L + pos] != h_sequences[parent_idx*L + pos]) {
                mutation_pos_order.push_back(pos);
            }
        }

        if(mutation_pos_order.size() == 0) {
            throw std::runtime_error("Should have at least one mutation from parent to child");
        }

        std::shuffle(mutation_pos_order.begin(), mutation_pos_order.end(), rng);

        node->mutation_pos_order = mutation_pos_order;

    }

    for(Node* n : node->children) {
        set_mutation_pos_order(n, h_sequences, L, rng);
    }
}




void write_training_examples(
    Node* node,
    char* h_sequences,
    std::ofstream& txt_out,
    long long L,
    bool is_testing,
    int& example_id,
    const std::vector<std::string>& variants
) {

    std::string seq;
    long long idx = node->idx; // Index of this node

    // Intermediate training examples along the branch
    if(node->parent != nullptr) {
        long long parent_idx = node->parent->idx;
        for(long long pos = 0; pos < L; pos++) {
            seq += h_sequences[parent_idx*L + pos];
        }

        // Apply first mutation
        seq[node->mutation_pos_order[0]] = h_sequences[idx*L + node->mutation_pos_order[0]];

        // If there are more mutations, generate a training example for each
        for(int j = 1; j < (int)node->mutation_pos_order.size(); j++) {
            // Write the sequence and its (one) mutation
            txt_out << "EXAMPLE_ID: " << example_id << std::endl;
            if(j == 1) {
                txt_out << "PARENT_EXAMPLE_ID: " << node->parent->example_id << std::endl;
            }else{
                txt_out << "PARENT_EXAMPLE_ID: " << example_id - 1 << std::endl;
            }
            example_id++;
            txt_out << "MUTATION_FROM_PARENT:" << std::endl;
            txt_out << h_sequences[parent_idx*L + node->mutation_pos_order[j-1]] << "," << node->mutation_pos_order[j-1] << "," << h_sequences[idx*L + node->mutation_pos_order[j-1]] << std::endl;

            if(is_testing) {
                txt_out << "TESTING" << std::endl;
            }else{
                txt_out << "TRAINING" << std::endl;
            }

            // Inherited from parent
            txt_out << "IS_DESCENDED_FROM_VARIANT:" << std::endl;
            for(int v = 0; v < variants.size(); v++) {
                txt_out << variants[v] << ",";
                if(node->parent->is_descended_from_variant[v] && is_testing) {
                    txt_out << "True" << std::endl;
                }else{
                    txt_out << "False" << std::endl;
                }
            }

            // Only an example at an actual node can ever be the variant root
            txt_out << "IS_VARIANT_ROOT:" << std::endl;
            for(int v = 0; v < variants.size(); v++) {
                txt_out << variants[v] << ",False" << std::endl;
            }
            
            txt_out << seq << std::endl;
            txt_out << "MUTATIONS_TO_CHILDREN:" << std::endl;
            txt_out << node->mutation_pos_order[j] << "," << h_sequences[idx*L + node->mutation_pos_order[j]] << "," << node->subtree_samples << std::endl;

            txt_out << std::endl;

            // Mutate the sequence
            seq[node->mutation_pos_order[j]] = h_sequences[idx*L + node->mutation_pos_order[j]];
        }


    }else{
        // If we're at the root, just get the sequence directly from the node
        for(long long pos = 0; pos < L; pos++) {
            seq += h_sequences[idx*L + pos];
        }
    }

    // Final example: the node itself
    if(node->children.size() > 0) {

        // Write the sequence and its (one) mutation
        txt_out << "EXAMPLE_ID: " << example_id << std::endl;

        
        /*
        txt_out << "NAME: " << node->name << std::endl;
        txt_out << "POSSIBLE_CLADES: " << std::endl;
        for(std::string c : node->possible_clades) {
            txt_out << c << std::endl;
        }
        txt_out << "LEAF_TIMES: " << std::endl;
        for(double t : node->leaf_times) {
            txt_out << t << " ";
        }
        txt_out << std::endl;
        */

        
        if(node->parent == nullptr) {
            txt_out << "PARENT_EXAMPLE_ID: None" << std::endl;
            txt_out << "MUTATION_FROM_PARENT:" << std::endl;
            txt_out << "None" << std::endl;

        }else{
            long long parent_idx = node->parent->idx;
            if(node->mutation_pos_order.size() == 1) {
                txt_out << "PARENT_EXAMPLE_ID: " << node->parent->example_id << std::endl;
            }else{
                txt_out << "PARENT_EXAMPLE_ID: " << example_id - 1 << std::endl;
            }

            txt_out << "MUTATION_FROM_PARENT:" << std::endl;
            txt_out << h_sequences[parent_idx*L + node->mutation_pos_order[node->mutation_pos_order.size()-1]] << "," << node->mutation_pos_order[node->mutation_pos_order.size()-1] << "," << h_sequences[idx*L + node->mutation_pos_order[node->mutation_pos_order.size()-1]] << std::endl;

        }
        node->example_id = example_id;
        example_id++;

        // Write the sequence and its mutations
        if(is_testing) {
            txt_out << "TESTING" << std::endl;
        }else{
            txt_out << "TRAINING" << std::endl;
        }

        txt_out << "IS_DESCENDED_FROM_VARIANT:" << std::endl;
        for(int v = 0; v < variants.size(); v++) {
            txt_out << variants[v] << ",";
            if(node->is_descended_from_variant[v] && is_testing) { // ATTEMPTED FIX: enforce these are in testing
                txt_out << "True" << std::endl;
            }else{
                txt_out << "False" << std::endl;
            }
        }

        txt_out << "IS_VARIANT_ROOT:" << std::endl;
        for(int v = 0; v < variants.size(); v++) {
            txt_out << variants[v] << ",";
            if(node->is_variant_root[v] && is_testing) { // ATTEMPTED FIX: enforce these are in testing
                txt_out << "True" << std::endl;
            }else{
                txt_out << "False" << std::endl;
            }
        }


        txt_out << seq << std::endl;

        txt_out << "MUTATIONS_TO_CHILDREN:" << std::endl;
        for(Node* n : node->children) {
            long long child_idx = n->idx;
            txt_out << n->mutation_pos_order[0] << "," << h_sequences[child_idx*L + n->mutation_pos_order[0]] << "," << n->subtree_samples << std::endl;
        }

        txt_out << std::endl;
    }

    for(Node* n : node->children) {
        write_training_examples(
            n,
            h_sequences,
            txt_out,
            L,
            is_testing,
            example_id,
            variants
        );
    }
}


struct CliOptions {
    std::string output_dir;
    double cutoff = 0.0;
    std::string dataset_type;
    std::string ref_json_path;
    std::string tree_json_path;
    std::string metadata_path;
    std::vector<std::string> alignment_files;
    int max_seqs = -1;
    std::string cache_path;
    bool no_cache = false;
    bool force_rebuild_cache = false;
    int grid_count = 0;
    bool grid_interior = false;
    std::string grid_output_dir;
    bool force_cpu = false;  // --cpu: force the CPU attachment path even if a GPU is present
    bool intervals = false;  // --intervals: one-year windows instead of a train/test cutoff
    bool seasons = false;    // --seasons: flu-season windows [Y-0.25, Y+0.25) for --intervals
    bool clades = false;     // --clades: write the Luksza-Lassig clade x season table
    double clade_cutoff = 0.15;  // --clade-cutoff: the paper's X_n(t) > 0.15 selection
    double clade_lca_floor = 0.02; // --clade-lca-floor: noise floor for a(t)
    double clade_clock_z = 0.0;   // --clade-clock-filter: outlier exclusion, 0 disables
    bool clade_merge = true;      // --clade-no-merge: keep one clade per genotype
    bool clade_haplotype = false; // --clade-haplotype: clade = carriers of a_n's alleles
    bool no_augment = false;      // --no-augment: guide tree only, skip attachment
    bool clade_refit_ancestors = false; // --clade-refit-ancestors: with the above, also
                                        // re-infer every ancestral genotype from the
                                        // surviving tips alone
    bool clade_prune_future = false; // --clade-prune-future: season t's clades from
                                     // a tree holding only samples dated before the
                                     // end of season t
};

bool is_integer_arg(const std::string& arg) {
    if (arg.empty()) return false;
    size_t start = arg[0] == '-' ? 1 : 0;
    if (start == arg.size()) return false;
    for (size_t i = start; i < arg.size(); i++) {
        if (!std::isdigit((unsigned char)arg[i])) return false;
    }
    return true;
}

bool is_dataset_type(const std::string& arg) {
    return arg == "sc2" || arg == "sc2_nuc" || arg == "sc2_spike_nuc" || arg == "toy" || arg == "h3n2" || arg == "rsv_a" || arg == "rsv_b" || arg == "denv2";
}

std::string infer_legacy_sc2_json_path(const std::filesystem::path& fasta_dir, const std::string& filename) {
    std::vector<std::filesystem::path> candidates = {
        fasta_dir / filename,
        std::filesystem::current_path() / filename,
        std::filesystem::path("/home/ivan.specht/antiGen/antiGen-data") / filename,
    };
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) return path.string();
    }
    return (fasta_dir / filename).string();
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " <output_dir> <cutoff> <dataset_type> <ref_json> <tree_json> <metadata_tsv> <alignment_files...> [max_seqs]"
              << " [--cache path] [--force-rebuild-cache] [--no-cache] [--grid N] [--grid-interior] [--grid-output-dir dir]"
              << " [--intervals] [--seasons]"
              << " [--clade-prune-future] [--no-augment]"
              << " [--clades] [--clade-cutoff X] [--clade-lca-floor f]"
              << " [--clade-clock-filter Z] [--clade-no-merge] [--clade-haplotype]"
              << std::endl;
    std::cerr << "Legacy SC2 usage: " << program
              << " <output_dir> <cutoff> <spike_fasta> <metadata_tsv> [max_seqs]"
              << " [--cache path] [--force-rebuild-cache] [--no-cache] [--grid N] [--grid-interior]"
              << std::endl;
    std::cerr << "  dataset_type: 'sc2', 'h3n2', 'rsv_a', 'rsv_b', or 'denv2'" << std::endl;
    std::cerr << "  --intervals: ignore <cutoff> and write examples_intervals.txt instead. The span"
              << std::endl
              << "      of sample dates is cut into one-year windows on the half-year; for each window"
              << std::endl
              << "      i, every genotype in the minimal subtree covering that window's samples becomes"
              << std::endl
              << "      an example whose child substitutions are counted by the window-(i+1) samples"
              << std::endl
              << "      below them. No train/test split -- the windows share one tree, so this leaks"
              << std::endl
              << "      by design; it is a proof of concept, not a held-out evaluation."
              << std::endl;
    std::cerr << "  --seasons: with --intervals, use influenza-season windows instead of"
              << std::endl
              << "      one-year ones: season Y covers [Y-0.25, Y+0.25), i.e. roughly Oct 1 of"
              << std::endl
              << "      year Y-1 to Apr 1 of year Y, and samples from Apr-Sep fall in no window."
              << std::endl
              << "      Output goes to examples_seasons.txt."
              << std::endl;
    std::cerr << "  --clade-prune-future: with --clades, read each season's clades off a"
              << std::endl
              << "      tree holding only the samples dated before that season ended, with"
              << std::endl
              << "      the internal nodes that only the later samples evidenced suppressed."
              << std::endl
              << "      Next-season frequencies -- the prediction target -- still come from"
              << std::endl
              << "      the full tree. This is not the tree anyone actually held at the time,"
              << std::endl
              << "      since the topology is still inferred once from everything; it removes"
              << std::endl
              << "      the later *samples* from the clade definitions, which is the part"
              << std::endl
              << "      that can be removed without re-running the phylogenetics."
              << std::endl;
    std::cerr << "  --no-augment: do not attach the alignment; use the guide tree as it"
              << std::endl
              << "      stands. The guide tree's tips are already dated samples, and at"
              << std::endl
              << "      ~150 a year it sits at the sampling depth the Luksza-Lassig clade"
              << std::endl
              << "      construction was designed for, where a season's clades are not a"
              << std::endl
              << "      long nested chain of near-duplicates. The alignment arguments are"
              << std::endl
              << "      still required but go unread."
              << std::endl;
    std::cerr << "  --clades: write the Luksza-Lassig clade x season table instead of (or,"
              << std::endl
              << "      together with --intervals, alongside) the examples file. Uses the same"
              << std::endl
              << "      windowing as --intervals, so --seasons gives the paper's own half-year"
              << std::endl
              << "      seasons. Output goes to clades_<seasons|intervals>.csv plus a"
              << std::endl
              << "      per-season summary."
              << std::endl;
    std::cerr << "  --clade-cutoff X: the frequency above which a clade is flagged for"
              << std::endl
              << "      prediction, X_n(t) > X. Default 0.15, the value used in the paper."
              << std::endl;
    std::cerr << "  --clade-lca-floor f: treat a branch carrying less than a fraction f of"
              << std::endl
              << "      a season as sampling noise when locating that season's last common"
              << std::endl
              << "      ancestor a(t), so that a handful of deep-branching sequences cannot"
              << std::endl
              << "      pin a(t) at the root of the tree. Default 0.02; 0 gives the paper's"
              << std::endl
              << "      exact last common ancestor."
              << std::endl;
    std::cerr << "  --clade-clock-filter Z: before extracting anything, drop the samples"
              << std::endl
              << "      whose divergence from the root is out of line with their collection"
              << std::endl
              << "      date by more than Z interquartile distances -- the outliers the paper"
              << std::endl
              << "      excludes by hand, and what otherwise pins a(t) at the root of the"
              << std::endl
              << "      tree. Applied once to the whole tree, so every season sees the same"
              << std::endl
              << "      data set. Defaults to 0, i.e. off: on this tree it costs 1.6% of"
              << std::endl
              << "      the samples and does not move a(t), because the sequences that pin"
              << std::endl
              << "      a(t) are not clock outliers. Use --clade-lca-floor instead."
              << std::endl;
    std::cerr << "  --clade-no-merge: index clades by genotype, as the paper does, instead"
              << std::endl
              << "      of merging the genotypes that hold the same strains in season t and"
              << std::endl
              << "      season t+1 into one unit. Without the merge a resolved trunk emits"
              << std::endl
              << "      the same population unit dozens of times over."
              << std::endl;
    std::cerr << "  --clade-haplotype: define a clade as the strains carrying a_n's mutant"
              << std::endl
              << "      alleles -- the paper's probabilistic version, which exists to average"
              << std::endl
              << "      clade membership over an ensemble of trees. The default is its"
              << std::endl
              << "      deterministic version, clade = the strains descending from a_n, which"
              << std::endl
              << "      is the one that applies to a single given tree. The two coincide on a"
              << std::endl
              << "      clean tree; they do not here."
              << std::endl;
}

std::vector<char> sequence_from_root_or_ref(
    const DatasetConfig& config,
    const json& ref,
    const json& tree_data
) {
    std::vector<char> root_protein_seq;
    if (tree_data.contains("root_sequence")) {
        std::cout << "Using root sequence from tree file" << std::endl;
        auto root_seq = tree_data["root_sequence"];
        if (config.type == "sc2") {
            std::string spike_str = root_seq["S"].get<std::string>();
            root_protein_seq = std::vector<char>(spike_str.begin(), spike_str.end() - 1);
        } else if (config.type == "sc2_nuc" || config.type == "sc2_spike_nuc") {
            std::string nuc_str = root_seq["nuc"].get<std::string>();
            root_protein_seq = std::vector<char>(nuc_str.begin(), nuc_str.end()); // nucleotide: no stop codon
        } else if (config.type == "toy") {
            std::string aa_str = root_seq["protein"].get<std::string>();
            root_protein_seq = std::vector<char>(aa_str.begin(), aa_str.end()); // toy protein: full sequence
        } else if (config.type == "h3n2") {
            std::string concat = root_seq["SigPep"].get<std::string>() +
                                 root_seq["HA1"].get<std::string>() +
                                 root_seq["HA2"].get<std::string>();
            root_protein_seq = std::vector<char>(concat.begin(), concat.end());
        } else if (config.type == "denv2") {
            std::string e_str = root_seq["E"].get<std::string>();
            root_protein_seq = std::vector<char>(e_str.begin(), e_str.end());
        } else {
            std::string f_str = root_seq["F"].get<std::string>();
            root_protein_seq = std::vector<char>(f_str.begin(), f_str.end() - 1);
        }
    } else {
        std::cout << "Using reference sequence from ref.json" << std::endl;
        if (config.type == "sc2") {
            std::string spike_str = ref["S"].get<std::string>();
            root_protein_seq = std::vector<char>(spike_str.begin(), spike_str.end() - 1);
        } else if (config.type == "sc2_nuc" || config.type == "sc2_spike_nuc") {
            std::string nuc_str = ref["nuc"].get<std::string>();
            root_protein_seq = std::vector<char>(nuc_str.begin(), nuc_str.end()); // nucleotide: no stop codon
        } else if (config.type == "toy") {
            std::string aa_str = ref["protein"].get<std::string>();
            root_protein_seq = std::vector<char>(aa_str.begin(), aa_str.end()); // toy protein: full sequence
        } else if (config.type == "h3n2") {
            std::string concat = ref["SigPep"].get<std::string>() +
                                 ref["HA1"].get<std::string>() +
                                 ref["HA2"].get<std::string>();
            root_protein_seq = std::vector<char>(concat.begin(), concat.end());
        } else if (config.type == "denv2") {
            std::string e_str = ref["E"].get<std::string>();
            root_protein_seq = std::vector<char>(e_str.begin(), e_str.end());
        } else {
            std::string f_str = ref["F"].get<std::string>();
            root_protein_seq = std::vector<char>(f_str.begin(), f_str.end() - 1);
        }
    }

    auto root_node = tree_data["tree"];
    if (root_node.contains("branch_attrs") && root_node["branch_attrs"].contains("mutations")) {
        std::cout << "Applying root mutations to get true phylogenetic root" << std::endl;
        auto root_mutations = root_node["branch_attrs"]["mutations"];
        for (const std::string& segment : config.segment_names) {
            if (!root_mutations.contains(segment) || root_mutations[segment].empty()) continue;
            for (const auto& m : root_mutations[segment]) {
                std::string mut_str = m.get<std::string>();
                Mutation mut = parse_mutation(mut_str);
                int translated_pos = translate_mutation_position(config, segment, mut.pos);
                if (mut.from == config.ambig_char || mut.to == config.ambig_char) continue;
                if (translated_pos >= 0 && translated_pos < (int)root_protein_seq.size()) {
                    if (root_protein_seq[translated_pos] != mut.from) {
                        std::cerr << "Warning: Root mutation " << segment << ":" << mut_str
                                  << " expects " << mut.from << " at position " << translated_pos
                                  << " but found " << root_protein_seq[translated_pos] << std::endl;
                    } else {
                        root_protein_seq[translated_pos] = mut.to;
                    }
                }
            }
        }
    }
    return root_protein_seq;
}

std::vector<std::string> dataset_variants(const DatasetConfig& config) {
    if (config.type == "sc2") return {"21L (BA.2)", "25C (XFG)"};
    if (config.type == "sc2_nuc" || config.type == "sc2_spike_nuc" || config.type == "toy") return {}; // nucleotide benchmark: no variant PSSMs
    if (config.type == "h3n2") return {"J", "K"};
    if (config.type == "rsv_b") return {"B.D.E.1"};
    if (config.type == "rsv_a") return {"A.D.1.6"};
    return {"2II_F.1.1.2_annotated"};
}

std::string variant_dir_name(std::string vname) {
    size_t paren = vname.find('(');
    if (paren != std::string::npos) {
        vname = vname.substr(paren + 1);
        if (!vname.empty() && vname.back() == ')') vname.pop_back();
    }
    return vname;
}

// Defined with the interval-mode helpers further down; needed here by --no-augment.
void sample_date_span(const Node* node, double root_time, double& lo, double& hi);

AttachedTree build_attached_tree(
    const DatasetConfig& config,
    const CliOptions& opts
) {
    std::ifstream rf(opts.ref_json_path);
    if (!rf.is_open()) throw std::runtime_error("Could not open " + opts.ref_json_path);
    json ref = json::parse(rf);

    std::ifstream tree_file(opts.tree_json_path);
    if (!tree_file.is_open()) throw std::runtime_error("Could not open " + opts.tree_json_path);
    json tree_data = json::parse(tree_file);

    std::vector<char> root_protein_seq = sequence_from_root_or_ref(config, ref, tree_data);
    std::cout << "Protein sequence length: " << root_protein_seq.size() << std::endl;
    std::cout << "Alignment files: ";
    for (const auto& file : opts.alignment_files) std::cout << file << " ";
    std::cout << std::endl;
    std::cout << "Metadata: " << opts.metadata_path << std::endl;

    Node* root = build_tree(tree_data["tree"], nullptr, config);
    double root_time = tree_data["tree"]["node_attrs"]["num_date"]["value"].get<double>();
    std::cout << "Root time: " << root_time << std::endl;

    set_times_rel_root(root, root_time);

    int total = 0, leaves = 0;
    count_nodes(root, total, leaves);
    std::cout << "Total nodes: " << total << std::endl;
    std::cout << "Leaf nodes: " << leaves << std::endl;

    prune_branches_without_mutations(root);
    int new_node_count = 0;
    int new_leaf_count = 0;
    check_prune(root, new_node_count, new_leaf_count);
    std::cout << "Pruning succeeded; tree now has " << new_node_count
              << " nodes and " << new_leaf_count << " leaves" << std::endl;

    std::cout << "Generating sequence at each node..." << std::endl;
    root->seq = root_protein_seq;
    assign_spike_sequences(root);
    std::cout << "Succeeded" << std::endl;

    // --no-augment: stop here and use the guide tree as it stands. Its tips already carry
    // leaf_times from node_attrs.num_date, so it is a complete, usable tree on its own.
    //
    // Why this is worth having: augmentation attaches every sequence in the alignment,
    // 203,729 of them for H3N2, and most are identical to something already on the tree.
    // That resolves each trunk branch into dozens of nodes and is what turns a season's
    // clades into a long nested chain of near-duplicates. The guide tree is 1,683 tips at
    // roughly 150 a year -- the same sampling depth as the 3,944 unique strains over ~20
    // years that Luksza & Lassig work with, which is the regime their clade construction
    // was designed for.
    if (opts.no_augment) {
        std::cout << "--no-augment: using the guide tree alone, no sequences attached"
                  << std::endl;
        int idx0 = 0;
        std::map<int, Node*> idx_to_node0;
        index_nodes(root, idx0, idx_to_node0);
        long long L0 = (long long)root_protein_seq.size();
        long long N0 = new_node_count;
        char* h_seq0 = (char*)malloc(L0 * N0);
        if (h_seq0 == nullptr) throw std::runtime_error("malloc h_sequences failed");
        std::vector<int> h_parent0(N0, -1);
        spike_seqs_to_host(root, h_seq0, h_parent0, L0);

        // Times were already made relative to the root above, so ask for the span with
        // the root offset added back and do not shift them again.
        double lo0 = std::numeric_limits<double>::infinity();
        double hi0 = -std::numeric_limits<double>::infinity();
        sample_date_span(root, root_time, lo0, hi0);
        set_node_times(root);

        AttachedTree plain;
        plain.root = root;
        plain.h_sequences = h_seq0;
        plain.L = L0;
        plain.node_count = N0;
        plain.root_time = root_time;
        plain.min_date = lo0;
        plain.max_date = hi0;
        compact_tree_indices(plain);
        check_branches_have_mutations(plain.root, plain.h_sequences, plain.L);
        std::cout << "Guide tree ready: " << plain.node_count << " nodes, sample dates "
                  << std::fixed << std::setprecision(3) << lo0 << " to " << hi0
                  << std::endl;
        return plain;
    }

    std::cout << "Loading date map from metadata TSV..." << std::endl;
    auto date_map = load_date_map(opts.metadata_path);

    std::cout << "Reading protein alignment(s)..." << std::endl;
    ProcessedFasta fasta;
    if (config.segment_names.size() == 1) {
        fasta = read_and_process_fasta(opts.alignment_files[0], date_map, config.total_length, opts.max_seqs);
    } else {
        fasta = read_multiple_alignments(opts.alignment_files, config.segment_lengths, date_map, opts.max_seqs);
    }

    double min_date = INFINITY, max_date = -INFINITY;
    int valid_dates = 0;
    for (int i = 0; i < fasta.n_seqs; i++) {
        double d = fasta.decimal_dates[i];
        if (std::isfinite(d)) {
            if (d < min_date) min_date = d;
            if (d > max_date) max_date = d;
            valid_dates++;
        }
    }
    if (valid_dates == 0) throw std::runtime_error("No finite sequence dates found");
    std::cout << "Dates: " << valid_dates << "/" << fasta.n_seqs << " valid, range: "
              << min_date << " - " << max_date << std::endl;

    std::cout << "Example dates:" << std::endl;
    for (int i = 0; i < std::min(10, fasta.n_seqs); i++) {
        std::cout << "  seq " << i << ": \"" << fasta.dates[i] << "\" -> " << fasta.decimal_dates[i] << std::endl;
    }

    std::cout << "Sorting sequences by date..." << std::endl;
    sort_by_date(fasta);
    std::cout << "Sorted. First 5 dates:";
    for (int i = 0; i < std::min(5, fasta.n_seqs); i++) std::cout << " " << fasta.dates[i];
    std::cout << std::endl;

    std::vector<double> est_times(fasta.decimal_dates.size());
    for (int i = 0; i < (int)est_times.size(); i++) {
        est_times[i] = fasta.decimal_dates[i] - root_time;
    }

    std::vector<char> protein_aa = flatten_protein_aa(fasta);
    long long L = root_protein_seq.size();
    long long N = new_node_count;
    long long M = N + (2 * (long long)fasta.n_seqs);

    int idx = 0;
    std::map<int, Node*> idx_to_node;
    index_nodes(root, idx, idx_to_node);

    // Decide GPU vs CPU: honor --cpu, else use a GPU only if one is actually present.
    // A build without CUDA (no nvcc) has no GPU path at all and always uses the CPU.
    bool use_gpu = !opts.force_cpu;
#ifdef USE_CUDA
    if (use_gpu) {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
            std::cout << "No CUDA device detected — falling back to the CPU attachment path." << std::endl;
            use_gpu = false;
        }
        cudaGetLastError();  // clear the no-device sticky error so nothing downstream trips on it
    } else {
        std::cout << "--cpu requested — using the CPU attachment path." << std::endl;
    }
#else
    use_gpu = false;  // built without CUDA (nvcc not available at build time)
    std::cout << "Built without CUDA — using the CPU attachment path." << std::endl;
#endif

    // Build the host sequence buffer (M*L): guide-tree sequences at their node indices,
    // then the N sequences to attach starting at row N. attach_sequences mirrors this to
    // the device internally when use_gpu is true.
    long long seq_bytes = L * M;
    char* h_sequences = (char*)malloc(seq_bytes);
    if (h_sequences == nullptr) throw std::runtime_error("malloc h_sequences failed");

    std::vector<int> h_parent(M, -1);
    spike_seqs_to_host(root, h_sequences, h_parent, L);
    std::memcpy(h_sequences + (L * N), protein_aa.data(), protein_aa.size() * sizeof(char));

    std::cout << "Root spike: ";
    for (long long j = 0; j < std::min(80LL, L); j++) std::cout << root_protein_seq[j];
    std::cout << "..." << std::endl;

    for (int i = 0; i < std::min(10, fasta.n_seqs); i++) {
        std::cout << "Seq " << i << " (date=" << fasta.dates[i] << "): ";
        for (long long j = 0; j < std::min(80LL, L); j++) {
            std::cout << protein_aa[i * L + j];
        }
        std::cout << "..." << std::endl;
    }

    auto attach_t0 = std::chrono::high_resolution_clock::now();
    h_sequences = attach_sequences(h_sequences, h_parent, L, N, M, est_times, idx_to_node, config.ambig_char, false, use_gpu);
    if (h_sequences == nullptr) throw std::runtime_error("attach_sequences failed");
    auto attach_t1 = std::chrono::high_resolution_clock::now();
    double attach_ms = std::chrono::duration_cast<std::chrono::milliseconds>(attach_t1 - attach_t0).count();
    std::cout << "AUGMENTATION_WALL_CLOCK_MS: " << (long long)attach_ms
              << "  (n_seqs=" << fasta.n_seqs << ", guide_nodes=" << N << ", L=" << L << ")" << std::endl;

    check_branches_have_mutations(root, h_sequences, L);
    std::cout << "Attachment wasn't the problem" << std::endl;

    std::cout << "Resetting node times to min of downstream leaf times..." << std::endl;
    set_node_times(root);
    std::cout << "Done resetting node times" << std::endl;

    AttachedTree attached;
    attached.root = root;
    attached.h_sequences = h_sequences;
    attached.L = L;
    attached.node_count = M;
    attached.root_time = root_time;
    attached.min_date = min_date;
    attached.max_date = max_date;
    compact_tree_indices(attached);
    std::cout << "Compacted attached tree to " << attached.node_count << " cached nodes" << std::endl;
    check_branches_have_mutations(attached.root, attached.h_sequences, attached.L);
    return attached;
}

void write_pssms(
    const std::string& output_dir,
    long long L,
    double timestep,
    double pssm_start_time,
    const std::vector<std::string>& variants,
    const std::vector<int>& pssm_root,
    const std::vector<std::vector<std::vector<int>>>& pssm_variants
) {
    int n_aas = ALLOWED_AAS.size();
    std::string pssm_dir = output_dir + "/pssms";
    std::filesystem::create_directories(pssm_dir);

    std::string root_path = pssm_dir + "/root.csv";
    std::ofstream root_out(root_path);
    root_out << "pos";
    for (int a = 0; a < n_aas; a++) root_out << "," << ALLOWED_AAS[a];
    root_out << std::endl;
    for (int pos = 0; pos < L; pos++) {
        root_out << pos;
        for (int a = 0; a < n_aas; a++) root_out << "," << pssm_root[pos * n_aas + a];
        root_out << std::endl;
    }
    std::cout << "Wrote " << root_path << std::endl;

    for (int v = 0; v < (int)variants.size(); v++) {
        std::string var_dir = pssm_dir + "/" + variant_dir_name(variants[v]);
        std::filesystem::create_directories(var_dir);
        int n_steps = pssm_variants[v].size();
        for (int i = 0; i < n_steps; i++) {
            std::ostringstream fname;
            fname << var_dir << "/time_" << std::fixed << std::setprecision(1)
                  << pssm_start_time + i * timestep << ".csv";
            std::ofstream vout(fname.str());
            vout << "pos";
            for (int a = 0; a < n_aas; a++) vout << "," << ALLOWED_AAS[a];
            vout << std::endl;
            for (int pos = 0; pos < L; pos++) {
                vout << pos;
                for (int a = 0; a < n_aas; a++) vout << "," << pssm_variants[v][i][pos * n_aas + a];
                vout << std::endl;
            }
        }
        std::cout << "Wrote " << n_steps << " PSSM files to " << var_dir << std::endl;
    }
}

// ── Time-interval mode (--intervals) ─────────────────────────────────────────
//
// An alternative to the single train/test cutoff. The whole span of sample dates is cut
// into one-year windows with boundaries on the half-year, so window Y covers
// [Y + 0.5, Y + 1.5) in absolute decimal years. For each window i:
//
//   T_i  = the minimal subtree containing every sample collected in window i, i.e. the
//          MRCA of those samples together with all nodes on the paths down to them;
//   each genotype in T_i becomes an example whose child substitutions are counted by how
//   many samples from window i+1 descend from them.
//
// The same node appears in as many windows as it is relevant to, each time with different
// counts, which is intended. There is no pruning: examples are drawn from the whole tree,
// so a model trained on window i and tested on window j shares the tree between them.

// Absolute decimal year of a time stored relative to the root.
static inline double absolute_time(double t_rel, double root_time) {
    return t_rel + root_time;
}

// Two windowing schemes.
//
//   year   (default)  window Y spans [Y + 0.5, Y + 1.5): contiguous one-year windows.
//   season (--seasons) window Y spans [Y - 0.25, Y + 0.25): roughly Oct 1 of year Y-1 to
//                      Apr 1 of year Y, i.e. one Northern Hemisphere influenza season.
//
// The season scheme is deliberately NOT a partition: consecutive windows are separated by
// the Apr-Oct off-season, and samples collected then belong to no window at all. That gap
// is the point. Under contiguous year windows a lineage merely has to still exist on the
// far side of a mid-year boundary to look "persistent", so the persistence baseline is
// rewarded for detecting continuity rather than growth. Putting the summer bottleneck
// between the window a genotype is observed in and the window its descendants are counted
// in asks a harder and more meaningful question.
static const long long WINDOW_NONE = LLONG_MIN;   // date falls outside every window

static inline long long window_index(double abs_t, bool seasons) {
    if(!seasons) return (long long)std::floor(abs_t - 0.5);
    double year = std::floor(abs_t);
    double frac = abs_t - year;
    if(frac >= 0.75) return (long long)year + 1;   // Oct-Dec belongs to the NEXT season
    if(frac < 0.25)  return (long long)year;       // Jan-Mar closes the season named for it
    return WINDOW_NONE;                            // Apr-Sep: the off-season
}

static inline double window_lo(long long idx, bool seasons) {
    return seasons ? (double)idx - 0.25 : (double)idx + 0.5;
}

static inline double window_hi(long long idx, bool seasons) {
    return seasons ? (double)idx + 0.25 : (double)idx + 1.5;
}

// Every sample dated at or after `horizon` is forgotten. The tree keeps its shape here;
// `cleanup_pruned_tree` is what then removes the nodes those samples were the only
// evidence for. Returns how many samples were dropped.
long long drop_samples_from(Node* node, double root_time, double horizon) {
    long long dropped = 0;
    std::vector<double> keep;
    keep.reserve(node->leaf_times.size());
    for(double t : node->leaf_times) {
        if(absolute_time(t, root_time) >= horizon) dropped++;
        else keep.push_back(t);
    }
    node->leaf_times.swap(keep);
    for(Node* n : node->children) dropped += drop_samples_from(n, root_time, horizon);
    return dropped;
}

long long count_nodes(const Node* node) {
    long long n = 1;
    for(const Node* c : node->children) n += count_nodes(c);
    return n;
}

// How much of the surviving tree could still be speaking for the samples that were
// pruned out of it. Two channels survive "drop the future-only branches, then suppress
// the degree-2 nodes and merge their mutations":
//
//   unsupported  an internal node joining two or more surviving lineages whose own
//                incoming branch carries no substitution. Nothing the survivors carry
//                distinguishes it from its parent, so the grouping is not evidenced by
//                them -- a future sequence sharing a mutation with two past ones can
//                create exactly this.
//   unparsimonious  a purely inferred ancestor (no observed samples of its own) with two
//                or more surviving children that agree on a residue it does not share.
//                Assigning it their residue would save a substitution on every child
//                branch and cost at most one on its own, so the surviving subtree alone
//                would never have reconstructed the state it has; something outside it
//                did, and the pruned descendants are the candidate.
//
// Neither is repaired by the pruning, so these counts are the honest size of what is
// left, not a diagnostic of a bug.
struct PruneAudit {
    long long internal = 0, joins = 0, unsupported = 0;
    long long inferred = 0, unparsimonious = 0, unparsimonious_nodes = 0;
    // The rigorous version of the second channel. The unanimity test above only fires
    // when the surviving children agree; it says nothing about the sites where they
    // disagree, and those are exactly the ties a pruned descendant could have broken.
    // Fitch's bottom-up pass answers it properly: the state of an inferred ancestor is
    // determined by the tree below it precisely when its Fitch set is a singleton.
    long long det_sites = 0, amb_sites = 0, mismatch_sites = 0, amb_nodes = 0;
    // ...and whether any of that uncertainty reaches a clade the analysis actually uses.
    long long touch_roots = 0, touch_hap = 0, touched_clades = 0;
};

// Fitch bottom-up over the *surviving* subtree, one pass, all positions at once.
// Returns the per-position state set of `v`, as a bitmask over `code`. A node carrying
// observed samples is pinned to what was observed; everything else is inferred from
// below. Sets are computed from the children alone, which is the whole point: they say
// what the surviving data can justify, independently of what the full-tree inference
// actually wrote down.
std::vector<uint32_t> fitch_up(const Node* v, const char* h_seq, long long L,
                               const int* code, PruneAudit& a,
                               std::map<int, std::set<int>>& uncertain,
                               std::vector<std::vector<uint32_t>>* keep = nullptr) {
    std::vector<std::vector<uint32_t>> kids;
    kids.reserve(v->children.size());
    for(const Node* c : v->children)
        kids.push_back(fitch_up(c, h_seq, L, code, a, uncertain, keep));

    const char* sv = h_seq + (long long)v->idx * L;
    std::vector<uint32_t> out(L, 0);
    bool pinned = !v->leaf_times.empty() || v->children.empty();
    bool judge = !pinned && v->children.size() >= 2;
    long long amb_here = 0;

    for(long long p = 0; p < L; p++) {
        uint32_t own = 1u << code[(unsigned char)sv[p]];
        uint32_t from_kids = 0;
        if(!kids.empty()) {
            uint32_t inter = ~0u, uni = 0;
            for(const auto& k : kids) { inter &= k[p]; uni |= k[p]; }
            from_kids = inter ? inter : uni;
        }
        if(judge) {
            if(from_kids & (from_kids - 1)) {        // more than one state possible
                a.amb_sites++;
                amb_here++;
                uncertain[v->idx].insert((int)p);
            } else if(from_kids == own) {
                a.det_sites++;
            } else {
                a.mismatch_sites++;
                uncertain[v->idx].insert((int)p);
            }
        }
        out[p] = pinned ? own : (from_kids ? from_kids : own);
    }
    if(amb_here) a.amb_nodes++;
    if(keep != nullptr) (*keep)[v->idx] = out;
    return out;
}

// Fitch down-pass: commit a state to every node from the sets the up-pass left, using
// nothing but the surviving tips and the surviving topology.
//
// Ties are broken toward the parent first -- the standard rule, and the one that keeps
// mutations off branches that do not need them -- and otherwise by the lowest residue
// code, which is arbitrary but, crucially, is *not* the state the full-tree inference
// chose. Preferring the incumbent there would let the pruned sequences back in through
// the tie-break and quietly guarantee the answer this is meant to test.
void fitch_down(const Node* v, char* h_out, long long L, const int* code,
                const std::vector<std::vector<uint32_t>>& up,
                const std::string& residues, const char* parent_seq, long long& changed) {
    char* sv = h_out + (long long)v->idx * L;
    bool pinned = !v->leaf_times.empty() || v->children.empty();
    if(!pinned) {
        const std::vector<uint32_t>& S = up[v->idx];
        for(long long p = 0; p < L; p++) {
            uint32_t set = S[p];
            if(!set) continue;
            char pick;
            if(parent_seq != nullptr && (set & (1u << code[(unsigned char)parent_seq[p]]))) {
                pick = parent_seq[p];
            } else {
                int bit = 0;
                while(!(set & (1u << bit))) bit++;
                pick = residues[bit];
            }
            if(pick != sv[p]) { sv[p] = pick; changed++; }
        }
    }
    for(const Node* c : v->children) fitch_down(c, h_out, L, code, up, residues, sv, changed);
}

void audit_pruned_tree(const Node* v, const char* h_seq, long long L, PruneAudit& a) {
    if(!v->children.empty()) {
        a.internal++;
        const char* sv = h_seq + (long long)v->idx * L;
        if(v->children.size() >= 2) {
            a.joins++;
            if(v->parent != nullptr) {
                const char* sp = h_seq + (long long)v->parent->idx * L;
                bool same = true;
                for(long long p = 0; p < L; p++) if(sv[p] != sp[p]) { same = false; break; }
                if(same) a.unsupported++;
            }
            if(v->leaf_times.empty()) {
                a.inferred++;
                long long bad = 0;
                for(long long p = 0; p < L; p++) {
                    char y = h_seq[(long long)v->children[0]->idx * L + p];
                    bool unanimous = true;
                    for(const Node* c : v->children) {
                        if(h_seq[(long long)c->idx * L + p] != y) { unanimous = false; break; }
                    }
                    if(unanimous && sv[p] != y) bad++;
                }
                a.unparsimonious += bad;
                if(bad) a.unparsimonious_nodes++;
            }
        }
    }
    for(const Node* c : v->children) audit_pruned_tree(c, h_seq, L, a);
}

// One post-order pass marking, for every node, whether its subtree holds a sample from
// window `cur`, and how many samples from window `cur` and from window `nxt` sit at or
// below it. The window-`cur` count is what the persistence baseline reads: this year's
// split between a genotype's child substitutions, used to predict next year's.
void mark_interval(Node* node, double root_time, long long cur, long long nxt,
                   bool seasons) {
    bool own = false;
    long long count = 0;
    long long now = 0;
    for(double t : node->leaf_times) {
        long long w = window_index(absolute_time(t, root_time), seasons);
        if(w == cur) { own = true; now++; }
        if(w == nxt) count++;
    }
    bool any = own;
    for(Node* n : node->children) {
        mark_interval(n, root_time, cur, nxt, seasons);
        any = any || n->in_window;
        count += n->next_count;
        now += n->cur_count;
    }
    node->own_in_window = own;
    node->in_window = any;
    node->cur_count = now;
    node->next_count = count;
}

// Deepest node whose subtree still contains every window-i sample: the root of T_i. Walk
// down while exactly one child carries the window and this node holds no sample itself.
Node* window_mrca(Node* root) {
    Node* node = root;
    while(true) {
        if(node->own_in_window) return node;
        Node* only = nullptr;
        int carrying = 0;
        for(Node* n : node->children) {
            if(n->in_window) { only = n; carrying++; }
        }
        if(carrying != 1) return node;
        node = only;
    }
}

// Window-w sequences sitting at this node itself, not below it.
static long long own_window_seqs(const Node* node, double root_time, long long w, bool seasons) {
    long long n = 0;
    for(double t : node->leaf_times) {
        if(window_index(absolute_time(t, root_time), seasons) == w) n++;
    }
    return n;
}

// ── Molecular-clock outlier filter ──────────────────────────────────────────
//
// Luksza & Lassig exclude by hand "17 sequences that are clearly recognizable as outliers
// in the reconstructed strain trees". The recognizable signature is a sample whose
// divergence from the root sequence is out of line with its collection date: a 2026
// sequence sitting a handful of substitutions from the 2008 root where its contemporaries
// sit at forty. Those are precisely the samples that pin a(t) to the root of the tree, so
// removing them once, globally, is the fix the paper itself applies.
//
// This is the standard clock filter (cf. Nextstrain's --clock-filter-iqd): regress
// divergence on date, then drop the samples whose residual falls more than z interquartile
// distances outside the quartiles. Both tails are cut -- too little divergence for the
// date is an ancient sequence wearing a recent label, too much is usually a bad read --
// and the line is refit on the survivors so that a cluster of bad sequences cannot tilt
// the fit that is meant to expose it. z = 0 disables the filter entirely.
struct ClockFilterReport {
    long long samples = 0;
    long long dropped_low = 0;
    long long dropped_high = 0;
    double intercept = 0.0, rate = 0.0, lo = 0.0, hi = 0.0;
};

void node_divergence(const Node* node, const char* h_sequences, const char* root_seq,
                     long long L, std::vector<int>& div) {
    const char* s = h_sequences + (long long)node->idx * L;
    int d = 0;
    for(long long p = 0; p < L; p++) if(s[p] != root_seq[p]) d++;
    div[node->idx] = d;
    for(const Node* n : node->children) node_divergence(n, h_sequences, root_seq, L, div);
}

void collect_clock_points(const Node* node, const std::vector<int>& div,
                          std::vector<std::pair<double, double>>& pts) {
    for(double t : node->leaf_times) pts.emplace_back(t, (double)div[node->idx]);
    for(const Node* n : node->children) collect_clock_points(n, div, pts);
}

// Least squares through the (date, divergence) cloud, using only the points whose residual
// against `a + b*t` is within [lo, hi]. Pass an infinite window for an unrestricted fit.
static void clock_fit(const std::vector<std::pair<double, double>>& pts,
                      double a_in, double b_in, double lo, double hi,
                      double& a_out, double& b_out) {
    double n = 0, st = 0, sd = 0, stt = 0, std_ = 0;
    for(const auto& p : pts) {
        double r = p.second - (a_in + b_in * p.first);
        if(r < lo || r > hi) continue;
        n++; st += p.first; sd += p.second; stt += p.first * p.first; std_ += p.first * p.second;
    }
    double den = n * stt - st * st;
    if(n < 3 || std::fabs(den) < 1e-12) { a_out = a_in; b_out = b_in; return; }
    b_out = (n * std_ - st * sd) / den;
    a_out = (sd - b_out * st) / n;
}

static void clock_bounds(const std::vector<std::pair<double, double>>& pts,
                         double a, double b, double z, double& lo, double& hi) {
    std::vector<double> r;
    r.reserve(pts.size());
    for(const auto& p : pts) r.push_back(p.second - (a + b * p.first));
    size_t q1 = r.size() / 4, q3 = (3 * r.size()) / 4;
    std::nth_element(r.begin(), r.begin() + q1, r.end());
    double lo_q = r[q1];
    std::nth_element(r.begin(), r.begin() + q3, r.end());
    double hi_q = r[q3];
    double iqr = hi_q - lo_q;
    lo = lo_q - z * iqr;
    hi = hi_q + z * iqr;
}

// Drop the flagged samples from the nodes that hold them. Reports up to `n_show` of them
// so the exclusions can be eyeballed rather than taken on trust.
void apply_clock_filter(Node* node, const std::vector<int>& div, double root_time,
                        double a, double b, double lo, double hi,
                        ClockFilterReport& rep, int& shown, int n_show) {
    std::vector<double> keep;
    keep.reserve(node->leaf_times.size());
    for(double t : node->leaf_times) {
        double d = (double)div[node->idx];
        double r = d - (a + b * t);
        if(r < lo || r > hi) {
            if(r < lo) rep.dropped_low++; else rep.dropped_high++;
            if(shown < n_show) {
                std::cout << "    dropped sample: date " << std::fixed << std::setprecision(2)
                          << absolute_time(t, root_time) << ", divergence " << (int)d
                          << " (clock expects " << std::setprecision(1) << (a + b * t)
                          << ")" << std::endl;
                shown++;
            }
            continue;
        }
        keep.push_back(t);
    }
    node->leaf_times = keep;
    for(Node* n : node->children) {
        apply_clock_filter(n, div, root_time, a, b, lo, hi, rep, shown, n_show);
    }
}

ClockFilterReport clock_filter_tree(Node* root, const char* h_sequences, long long L,
                                    long long node_count, double root_time, double z) {
    ClockFilterReport rep;
    std::vector<int> div(node_count, 0);
    const char* root_seq = h_sequences + (long long)root->idx * L;
    node_divergence(root, h_sequences, root_seq, L, div);

    std::vector<std::pair<double, double>> pts;
    collect_clock_points(root, div, pts);
    rep.samples = (long long)pts.size();
    if(pts.size() < 10) return rep;

    double inf = std::numeric_limits<double>::infinity();
    double a = 0, b = 0;
    clock_fit(pts, 0.0, 0.0, -inf, inf, a, b);
    clock_bounds(pts, a, b, z, rep.lo, rep.hi);
    clock_fit(pts, a, b, rep.lo, rep.hi, a, b);       // refit on the survivors
    clock_bounds(pts, a, b, z, rep.lo, rep.hi);
    rep.intercept = a;
    rep.rate = b;

    int shown = 0;
    apply_clock_filter(root, div, root_time, a, b, rep.lo, rep.hi, rep, shown, 10);
    return rep;
}

// a(t): the last common ancestor of the season's strains, up to a sampling-noise floor.
//
// window_mrca returns the exact last common ancestor, which is what Luksza & Lassig define
// a(t) to be -- but on a tree of this sampling depth a single deep-branching sequence
// drags it to the root, and every trunk genotype between the root and the lineage that
// actually circulates then becomes its own near-100% clade. Season 2026 is pinned at the
// 2008 root by 8 sequences out of 22,466.
//
// The fix is the paper's own argument for its 0.15 cutoff, applied one level up: a lineage
// carrying too little of the season is sampling noise, not a population unit. So descend
// while exactly one branch clears a floor of `floor_frac` of the season's sequences, and
// stop where two of them do -- that is where the season's real standing diversity begins.
// The node's own samples count as a branch, since a genotype observed at frequency is a
// unit in its own right. floor_frac = 0 reproduces window_mrca exactly.
//
// The frequency denominator stays the whole season, so lineages left above a(t) are not
// renormalized away -- they simply get no clades of their own.
Node* window_core_lca(Node* root, double root_time, long long window, bool seasons,
                      double floor_frac) {
    long long need = (long long)std::ceil(floor_frac * (double)root->cur_count);
    if(need < 1) need = 1;
    Node* node = root;
    while(true) {
        if(own_window_seqs(node, root_time, window, seasons) >= need) return node;
        Node* only = nullptr;
        int above = 0;
        for(Node* n : node->children) {
            if(n->in_window && n->cur_count >= need) { only = n; above++; }
        }
        if(above != 1) return node;
        node = only;
    }
}

// Span of sample dates over the whole tree. attached.min_date/max_date cover only the
// sequences read from the alignment, but the guide tree's own tips are samples too and can
// sit outside that range, so take the span from the tree itself.
void sample_date_span(const Node* node, double root_time, double& lo, double& hi) {
    for(double t : node->leaf_times) {
        double abs_t = absolute_time(t, root_time);
        if(abs_t < lo) lo = abs_t;
        if(abs_t > hi) hi = abs_t;
    }
    for(const Node* n : node->children) sample_date_span(n, root_time, lo, hi);
}

void count_window_nodes(const Node* node, long long& nodes) {
    nodes++;
    for(const Node* n : node->children) {
        if(n->in_window) count_window_nodes(n, nodes);
    }
}

// Writer for one window. Mirrors write_training_examples -- same fields, same synthesized
// intermediate genotypes along multi-mutation branches -- with three differences:
//   * it descends only into children that are in T_i, since a branch leaving T_i is not
//     part of the minimal subtree;
//   * MUTATIONS_TO_CHILDREN still lists EVERY child, in T_i or not. Those are the
//     substitutions available from this genotype, and a substitution whose descendants
//     all fall outside window i is exactly the kind the model should learn to score;
//   * each row carries three counts: samples from window i+1 below that child (the
//     training target), the all-time descendant total kept for reference, and samples
//     from window i below that child -- this year's split, which the persistence
//     baseline uses to predict next year's.
void write_interval_examples(
    Node* node,
    char* h_sequences,
    std::ofstream& txt_out,
    long long L,
    int& example_id,
    const std::vector<std::string>& variants,
    long long window,
    bool is_window_root,
    bool seasons
) {
    std::string seq;
    long long idx = node->idx;

    auto write_header = [&](int parent_field, bool have_parent, int mut_pos) {
        txt_out << "EXAMPLE_ID: " << example_id << std::endl;
        txt_out << "TIME_INTERVAL: " << window << std::endl;
        txt_out << "TIME_RANGE: " << std::fixed << std::setprecision(2)
                << window_lo(window, seasons) << "," << window_hi(window, seasons)
                << std::endl;
        if(!have_parent) {
            txt_out << "PARENT_EXAMPLE_ID: None" << std::endl;
            txt_out << "MUTATION_FROM_PARENT:" << std::endl << "None" << std::endl;
        } else {
            txt_out << "PARENT_EXAMPLE_ID: " << parent_field << std::endl;
            txt_out << "MUTATION_FROM_PARENT:" << std::endl;
            long long parent_idx = node->parent->idx;
            txt_out << h_sequences[parent_idx*L + mut_pos] << "," << mut_pos << ","
                    << h_sequences[idx*L + mut_pos] << std::endl;
        }
        txt_out << "TRAINING" << std::endl;
        txt_out << "IS_DESCENDED_FROM_VARIANT:" << std::endl;
        for(size_t v = 0; v < variants.size(); v++) {
            txt_out << variants[v] << "," << (node->is_descended_from_variant[v] ? "True" : "False")
                    << std::endl;
        }
        txt_out << "IS_VARIANT_ROOT:" << std::endl;
        for(size_t v = 0; v < variants.size(); v++) {
            txt_out << variants[v] << "," << (node->is_variant_root[v] ? "True" : "False")
                    << std::endl;
        }
    };

    // Intermediate genotypes along the branch into this node. Skipped at the root of T_i,
    // whose incoming branch lies outside the minimal subtree.
    if(node->parent != nullptr && !is_window_root) {
        long long parent_idx = node->parent->idx;
        for(long long pos = 0; pos < L; pos++) seq += h_sequences[parent_idx*L + pos];
        seq[node->mutation_pos_order[0]] = h_sequences[idx*L + node->mutation_pos_order[0]];

        for(int j = 1; j < (int)node->mutation_pos_order.size(); j++) {
            write_header(j == 1 ? node->parent->example_id : example_id - 1, true,
                         node->mutation_pos_order[j-1]);
            example_id++;
            txt_out << seq << std::endl;
            txt_out << "MUTATIONS_TO_CHILDREN:" << std::endl;
            txt_out << node->mutation_pos_order[j] << ","
                    << h_sequences[idx*L + node->mutation_pos_order[j]] << ","
                    << node->next_count << "," << node->subtree_samples << ","
                    << node->cur_count << std::endl;
            txt_out << std::endl;
            seq[node->mutation_pos_order[j]] = h_sequences[idx*L + node->mutation_pos_order[j]];
        }
    } else {
        for(long long pos = 0; pos < L; pos++) seq += h_sequences[idx*L + pos];
    }

    if(node->children.size() > 0) {
        bool have_parent = (node->parent != nullptr && !is_window_root);
        int mut_pos = have_parent
            ? node->mutation_pos_order[node->mutation_pos_order.size()-1] : 0;
        int parent_field = have_parent
            ? (node->mutation_pos_order.size() == 1 ? node->parent->example_id : example_id - 1)
            : 0;
        write_header(parent_field, have_parent, mut_pos);
        node->example_id = example_id;
        example_id++;
        txt_out << seq << std::endl;
        txt_out << "MUTATIONS_TO_CHILDREN:" << std::endl;
        for(Node* n : node->children) {
            long long child_idx = n->idx;
            txt_out << n->mutation_pos_order[0] << ","
                    << h_sequences[child_idx*L + n->mutation_pos_order[0]] << ","
                    << n->next_count << "," << n->subtree_samples << ","
                    << n->cur_count << std::endl;
        }
        txt_out << std::endl;
    }

    for(Node* n : node->children) {
        if(n->in_window) {
            write_interval_examples(n, h_sequences, txt_out, L, example_id, variants,
                                    window, false, seasons);
        }
    }
}

void generate_interval_outputs(
    const AttachedTree& attached,
    const DatasetConfig& config,
    const std::string& output_dir,
    bool seasons
) {
    std::filesystem::create_directories(output_dir);
    Node* root = clone_tree(attached.root);
    char* h_sequences = attached.h_sequences;
    long long L = attached.L;

    std::vector<std::string> variants = dataset_variants(config);
    set_variant_descendants(root, variants);

    // No pruning, so no train/test split -- but the same tidy-up the cutoff path applies
    // to its training tree, so the genotypes match what that path would have produced.
    std::cout << "Cleaning up tree..." << std::endl;
    cleanup_pruned_tree(root);
    check_branches_have_mutations(root, h_sequences, L);

    // One ordering for the whole run, so a node synthesizes the same intermediate
    // genotypes in every window it appears in.
    std::mt19937 rng(123);
    set_mutation_pos_order(root, h_sequences, L, rng);
    set_subtree_samples(root);

    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    sample_date_span(root, attached.root_time, lo, hi);
    if(!(lo <= hi)) throw std::runtime_error("No samples on the tree; nothing to window");
    // Candidate range. In season mode the span endpoints may themselves fall in the
    // off-season, so the first/last window cannot be read off them directly: take every
    // window that overlaps the span and let the empty-window skip below discard the rest.
    long long first, last;
    if(seasons) {
        first = (long long)std::floor(lo - 0.25) + 1;
        last  = (long long)std::ceil(hi + 0.25) - 1;
    } else {
        first = window_index(lo, false);
        last  = window_index(hi, false);
    }
    std::cout << std::fixed << std::setprecision(3)
              << "Sample dates span " << lo << " to " << hi << std::setprecision(2)
              << ", i.e. " << (seasons ? "seasons " : "windows ")
              << first << " [" << window_lo(first, seasons) << ","
              << window_hi(first, seasons) << ") through " << last << " ["
              << window_lo(last, seasons) << "," << window_hi(last, seasons) << ")"
              << std::endl;
    if(seasons) {
        std::cout << "Season Y runs Oct 1 of year Y-1 to Apr 1 of year Y; samples collected "
                  << "Apr-Sep\nfall in no window and are counted by nobody." << std::endl;
    }
    std::cout << "The final window has no successor to count, so it produces no examples."
              << std::endl;

    std::string txt_path = output_dir +
        (seasons ? "/examples_seasons.txt" : "/examples_intervals.txt");
    std::ofstream txt_out(txt_path);
    if(!txt_out.is_open()) throw std::runtime_error("Could not open " + txt_path + " for writing");

    int example_id = 0;
    for(long long w = first; w < last; w++) {
        mark_interval(root, attached.root_time, w, w + 1, seasons);
        if(!root->in_window) {
            std::cout << "  window " << w << ": no samples, skipped" << std::endl;
            continue;
        }
        Node* mrca = window_mrca(root);
        long long nodes = 0;
        count_window_nodes(mrca, nodes);
        int before = example_id;
        write_interval_examples(mrca, h_sequences, txt_out, L, example_id, variants, w,
                                true, seasons);
        std::cout << "  window " << w << " [" << window_lo(w, seasons) << ","
                  << window_hi(w, seasons)
                  << "): T_i has " << nodes << " nodes, " << root->cur_count
                  << " samples in it, wrote " << (example_id - before)
                  << " examples, " << root->next_count << " samples in the next window"
                  << std::endl;
    }
    txt_out.close();
    std::cout << "Wrote " << txt_path << " (" << example_id << " examples)" << std::endl;

    delete_tree(root);
}

// ── Łuksza–Lässig clade extraction (--clades) ────────────────────────────────
//
// Clades exactly as defined in Łuksza & Lässig, Nature 507:57-61 (2014), Methods §1
// ("Definition of clades", "Estimation of clade frequencies"), applied to the trees this
// tool builds. Clades are that paper's unit of prediction: its fitness model forecasts
// the frequency of clade n in season t+1 from the fitnesses and frequencies of its
// members in season t, so this is the table that framework needs.
//
// Per season t -- the same windows --intervals cuts, and with --seasons the same
// half-year convention the paper uses ("the Northern winter from October of year y-1 to
// April of year y is season t = y"):
//
//   a(t)  the last common ancestor of the strains observed in season t: the root of T_t,
//         the minimal spanning subtree the interval writer already builds;
//   K(t)  every HA genotype occurring in T_t. Genotypes, not nodes -- two nodes carrying
//         the same sequence are one element of K(t);
//   each a_n in K(t) is referred to a(t) and identified with its mutant haplotype, the
//         minimal set of (position, mutant allele) pairs from a(t) to a_n;
//   clade n  the strains descending from a_n. The paper gives this deterministic
//         definition first -- "this notion is intuitive and well-defined for a given
//         tree" -- and then replaces it with a probabilistic one, clade = the strains
//         carrying a_n's mutant alleles, delimited by T_n, the minimal set of disjoint
//         subtrees containing every carrier. That replacement exists to average membership
//         over an ensemble of trees, which we do not have, and on this tree it pools
//         convergent lineages into one "clade" 17% of the time. So the deterministic
//         reading is the default here and --clade-haplotype selects the other one; the
//         n_subtrees column reports the paper's "in most cases, this set consists of a
//         single subtree" either way;
//   X_n(t)  (season-t sequences in T_n) / (season-t sequences overall), which is eq. 6
//         fed by eq. 3 in the rho in {0,1} limit;
//   Kbar(t)  the clades with X_n(t) > --clade-cutoff (0.15 in the paper), flagged by the
//         above_cutoff column. Nesting is kept: the paper uses nested clades as units of
//         prediction on purpose, "because they can map evolutionary histories from
//         origination of new strains to fixation or loss".
//
// Three departures from the letter of the paper, each one measured in NOTES.md before it
// was made, because the definition as written does not survive this sampling depth:
//
//   * a(t) is located up to a sampling-noise floor (--clade-lca-floor, default 0.02)
//     rather than exactly, since a single deep-branching sequence otherwise drags it to
//     the root of the tree and turns the whole trunk into near-100% clades.
//   * K(t) is deduplicated by strain content, not by genotype (--clade-no-merge restores
//     the paper's indexing), since a trunk resolved by 20,000 sequences a season emits
//     the same population unit dozens of times over.
//   * Clades are defined by descent rather than by haplotype carriage, as above.
//
// Passing --clade-lca-floor 0 --clade-no-merge --clade-haplotype gives the definition
// exactly as published.
//
// Four further deviations, each forced by what this pipeline produces:
//
//   * One tree, not an ensemble. rho_{i,n} (eq. 5) is an average over 10-30
//     FastTree/PAUP trees; we have a single mutation-annotated guide tree with
//     parsimonious attachment, so rho collapses to 0/1 and every frequency is a plain
//     count ratio. The clade-membership uncertainty the ensemble exists to measure is
//     simply not available here.
//   * The tree is built from all strains, not only those up to season t. T_t itself is
//     unaffected -- it is spanned by the season-t samples, and every season-t sample sits
//     inside it -- but the internal genotypes along it were inferred with the help of
//     later sequences.
//   * Strains are not deduplicated by region: the paper counts sequences sharing a
//     genotype, season and location once, and our metadata carries no location, so m_i
//     here is the raw sequence count.
//   * Haplotypes are over amino acids (the gap '-' included, i.e. point indels count as
//     alleles); the paper's are over nucleotides.

// A mutant haplotype: (position, mutant allele) pairs relative to a(t), sorted by
// position, at most one entry per position.
using Haplotype = std::vector<std::pair<int, char>>;

// Is `sub` a subset of `sup`? Both sorted by position with unique positions, so one
// merge pass decides it. Carriage is exactly this test: a node v carries the haplotype
// H iff H is a subset of H(v), because every allele in H differs from a(t) by
// construction.
static bool hap_contains(const Haplotype& sup, const Haplotype& sub) {
    size_t i = 0;
    for(const auto& e : sub) {
        while(i < sup.size() && sup[i].first < e.first) i++;
        if(i >= sup.size() || sup[i].first != e.first || sup[i].second != e.second) return false;
        i++;
    }
    return true;
}

static std::string hap_key(const Haplotype& h) {
    std::string k;
    k.reserve(h.size() * 8);
    for(const auto& e : h) {
        k += std::to_string(e.first);
        k += e.second;
        k += ';';
    }
    return k;
}

// Human-readable site label, in the coordinates of the input tree's mutation lists:
// segment-local and 1-indexed, e.g. "HA1:145K". Single-segment datasets drop the prefix.
static std::string format_site(const DatasetConfig& config, int pos, char alt) {
    if(config.segment_names.size() == 1) {
        return std::to_string(pos + 1) + alt;
    }
    for(int s = (int)config.segment_names.size() - 1; s >= 0; s--) {
        if(pos >= config.segment_offsets[s]) {
            return config.segment_names[s] + ":" +
                   std::to_string(pos - config.segment_offsets[s] + 1) + alt;
        }
    }
    return std::to_string(pos + 1) + alt;
}

static std::string format_haplotype(const DatasetConfig& config, const Haplotype& h) {
    if(h.empty()) return "root";
    std::string out;
    for(size_t i = 0; i < h.size(); i++) {
        if(i) out += ';';
        out += format_site(config, h[i].first, h[i].second);
    }
    return out;
}

// One clade: a genotype of K(t), the disjoint subtrees that delimit it, and its counts.
struct Clade {
    Haplotype hap;
    std::vector<Node*> roots;        // T_n: maximal carriers, in pre-order
    long long seqs_t = 0;            // season-t sequences in T_n  (numerator of X_n(t))
    long long seqs_next = 0;         // season-(t+1) sequences in T_n
    long long samples_all = 0;       // all-time samples in T_n
    long long nodes_t = 0;           // nodes of T_t inside T_n
    long long noncarrier_seqs = 0;   // season-t sequences in T_n whose genotype no longer
                                     // holds every allele of hap; -1 when not evaluated
    std::vector<Node*> geno_nodes;   // nodes of T_t realizing this genotype exactly
    long long genotype_nodes = 0;    // ...how many
    int id = -1;
    int parent_id = -1;              // smallest clade strictly containing this one
    unsigned long long sig = 0;      // hash of the strain content in seasons t and t+1
    int rep_id = -1;                 // the clade this one is merged into (itself if kept)
    long long merged = 1;            // genotypes collapsed into this unit
};

// 64-bit mix, used to fingerprint a clade's strain content so that clades holding the
// same strains can be recognized as one unit.
static inline unsigned long long mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Per-node fingerprint of "which strains sit here, in season t and in season t+1", summed
// over the subtree. Nodes holding no strain in either season contribute nothing, so the
// fingerprint depends on strain content alone and not on the shape of the tree above it.
// Summing is safe across a clade's subtrees because they are disjoint.
unsigned long long strain_content_sig(const Node* node, double root_time, long long window,
                                      bool seasons, std::vector<unsigned long long>& sig) {
    long long own_t = 0, own_next = 0;
    for(double t : node->leaf_times) {
        long long w = window_index(absolute_time(t, root_time), seasons);
        if(w == window) own_t++;
        else if(w == window + 1) own_next++;
    }
    unsigned long long total = 0;
    if(own_t > 0 || own_next > 0) {
        total = mix64(((unsigned long long)node->idx + 1) * 0x100000001b3ULL ^
                      mix64((unsigned long long)own_t * 2654435761ULL +
                            (unsigned long long)own_next * 40503ULL + 1));
    }
    for(const Node* n : node->children) {
        total += strain_content_sig(n, root_time, window, seasons, sig);
    }
    sig[node->idx] = total;
    return total;
}

// Pre-order walk of T_t: the window MRCA plus every in-window node below it. Mirrors the
// descent write_interval_examples makes, so the genotype set is the same one that path
// turns into examples.
static void collect_window_subtree(Node* node, std::vector<Node*>& out) {
    out.push_back(node);
    for(Node* n : node->children) {
        if(n->in_window) collect_window_subtree(n, out);
    }
}

// Build K(t) and its clades for one window. Requires mark_interval(root, ., w, w+1, .) to
// have run, so in_window / cur_count / next_count are current, and `mrca` to be
// window_mrca(root).
static void extract_window_clades(
    Node* mrca,
    const char* h_sequences,
    long long L,
    long long node_count,
    double root_time,
    long long window,
    bool seasons,
    double cutoff,
    long long seqs_t_total,
    bool node_defined,
    bool merge_by_content,
    std::vector<Clade>& clades
) {
    std::vector<Node*> nodes;
    collect_window_subtree(mrca, nodes);

    std::vector<Haplotype> hap(node_count);
    std::vector<long long> order(node_count, -1);      // position in pre-order
    std::vector<long long> sub_nodes(node_count, 0);   // T_t nodes at or below
    const char* a_t = h_sequences + (long long)mrca->idx * L;

    // Haplotype of every genotype in T_t, and an index from each mutant allele to the
    // nodes whose incoming branch installs it. A node can only start carrying a
    // haplotype its parent did not carry by installing one of its alleles, so every
    // maximal carrier is in this index (the MRCA, which carries the empty haplotype, is
    // the one exception and is handled directly).
    std::map<std::pair<int, char>, std::vector<Node*>> installs;
    for(size_t i = 0; i < nodes.size(); i++) {
        Node* v = nodes[i];
        order[v->idx] = (long long)i;
        const char* sv = h_sequences + (long long)v->idx * L;
        Haplotype h;
        for(long long p = 0; p < L; p++) {
            if(sv[p] != a_t[p]) h.emplace_back((int)p, sv[p]);
        }
        hap[v->idx] = std::move(h);
        if(v != mrca) {
            const char* sp = h_sequences + (long long)v->parent->idx * L;
            for(const auto& e : hap[v->idx]) {
                if(sp[e.first] != sv[e.first]) installs[e].push_back(v);
            }
        }
    }
    for(size_t i = nodes.size(); i-- > 0;) {
        Node* v = nodes[i];
        long long n = 1;
        for(Node* c : v->children) {
            if(c->in_window) n += sub_nodes[c->idx];
        }
        sub_nodes[v->idx] = n;
    }

    // Collapse nodes to genotypes: K(t) is indexed by genotype, so identical sequences
    // at different nodes are one clade.
    std::unordered_map<std::string, int> by_key;
    for(Node* v : nodes) {
        std::string key = hap_key(hap[v->idx]);
        auto it = by_key.find(key);
        if(it != by_key.end()) {
            clades[it->second].genotype_nodes++;
            clades[it->second].geno_nodes.push_back(v);
            continue;
        }
        by_key[key] = (int)clades.size();
        Clade c;
        c.hap = hap[v->idx];
        c.id = (int)clades.size();
        c.genotype_nodes = 1;
        c.geno_nodes.push_back(v);
        clades.push_back(std::move(c));
    }

    // T_n for each clade: the maximal nodes of T_t carrying the haplotype. Maximality
    // makes the subtrees disjoint, which is what lets the counts below be summed.
    for(Clade& c : clades) {
        if(node_defined) {
            // The paper's deterministic definition: the clade is the set of strains
            // descending from a_n, so T_n is rooted at the nodes carrying that genotype
            // and nothing else. A lineage that converged on the same alleles from a
            // different background is not in it. Still filtered for disjointness, since
            // a reversion round-trip can put one occurrence of a genotype below another.
            for(Node* v : c.geno_nodes) {
                bool covered = false;
                for(Node* p = v->parent; p != nullptr && order[p->idx] >= 0; p = p->parent) {
                    if(hap_key(hap[p->idx]) == hap_key(c.hap)) { covered = true; break; }
                    if(p == mrca) break;
                }
                if(!covered) c.roots.push_back(v);
            }
        } else if(c.hap.empty()) {
            c.roots.push_back(mrca);
        } else {
            std::unordered_set<int> seen;
            std::vector<Node*> cand;
            for(const auto& e : c.hap) {
                auto it = installs.find(e);
                if(it == installs.end()) continue;
                for(Node* v : it->second) {
                    if(seen.insert(v->idx).second) cand.push_back(v);
                }
            }
            for(Node* v : cand) {
                if(!hap_contains(hap[v->idx], c.hap)) continue;
                bool ancestor_carries = false;
                for(Node* p = v->parent; p != nullptr && order[p->idx] >= 0; p = p->parent) {
                    if(hap_contains(hap[p->idx], c.hap)) { ancestor_carries = true; break; }
                    if(p == mrca) break;
                }
                if(!ancestor_carries) c.roots.push_back(v);
            }
            std::sort(c.roots.begin(), c.roots.end(),
                      [&](Node* a, Node* b) { return order[a->idx] < order[b->idx]; });
        }
        if(c.roots.empty()) {
            throw std::runtime_error("Clade with no delimiting subtree; haplotype index is incomplete");
        }
        for(Node* r : c.roots) {
            c.seqs_t += r->cur_count;
            c.seqs_next += r->next_count;
            c.samples_all += r->subtree_samples;
            c.nodes_t += sub_nodes[r->idx];
        }
    }

    // Independent recomputation of every T_n, gated on ANTIGEN_CLADE_BRUTE because it is
    // quadratic in |T_t|. The fast path above reaches the maximal carriers through an
    // allele index and a walk up to the first carrying ancestor; this one tests every
    // node of T_t against every genotype and every ancestor explicitly. They must agree.
    if(const char* brute = std::getenv("ANTIGEN_CLADE_BRUTE")) {
      if(brute[0] != '\0' && brute[0] != '0' && !node_defined) {
        for(const Clade& c : clades) {
            std::vector<Node*> want;
            for(Node* v : nodes) {
                if(!hap_contains(hap[v->idx], c.hap)) continue;
                bool covered = false;
                for(Node* p = v->parent; p != nullptr && order[p->idx] >= 0; p = p->parent) {
                    if(hap_contains(hap[p->idx], c.hap)) { covered = true; break; }
                    if(p == mrca) break;
                }
                if(!covered) want.push_back(v);
            }
            std::sort(want.begin(), want.end(),
                      [&](Node* a, Node* b) { return order[a->idx] < order[b->idx]; });
            if(want != c.roots) {
                throw std::runtime_error("Brute-force check failed for clade " +
                    std::to_string(c.id) + " in window " + std::to_string(window) +
                    ": fast path found " + std::to_string(c.roots.size()) +
                    " subtrees, brute force " + std::to_string(want.size()));
            }
        }
        std::cout << "  brute-force check passed for " << clades.size()
                  << " clades in window " << window << std::endl;
      }
    }

    // Nesting: the smallest clade strictly containing this one. Walking up from the
    // first of its subtrees, that is the nearest ancestor whose haplotype is a strict
    // subset -- reversions can put a non-subset genotype in between, so keep walking.
    for(Clade& c : clades) {
        if(c.hap.empty()) continue;
        for(Node* p = c.roots[0]->parent; p != nullptr && order[p->idx] >= 0; p = p->parent) {
            const Haplotype& ph = hap[p->idx];
            if(ph.size() < c.hap.size() && hap_contains(c.hap, ph)) {
                auto it = by_key.find(hap_key(ph));
                if(it != by_key.end()) { c.parent_id = it->second; break; }
            }
            if(p == mrca) break;
        }
    }

    // Deduplicate K(t) by strain content rather than by genotype.
    //
    // The paper indexes clades by genotype, which is the right thing to do when a season
    // holds 200 strains: each genotype in T_t is a distinct set of strains. At 20,000
    // sequences a season the trunk is resolved into dozens of successive genotypes that
    // all carry exactly the same observed strains, so the same population unit is emitted
    // 40 times over with a longer label each time. Any per-clade statistic then counts one
    // lineage repeatedly.
    //
    // Two clades are the same unit when they hold the same strains in season t and the
    // same strains in season t+1 -- the two sides of the prediction. Requiring both is
    // what makes the merge sound: a trunk node whose sister lineage is unsampled this
    // season but sampled next season is genuinely a different unit from its child, and
    // the season-(t+1) half of the fingerprint keeps them apart.
    //
    // The representative kept is the most derived genotype of the class, since that is the
    // genotype every observed member of the unit actually descends from; `merged` records
    // how many genotypes collapsed into it.
    if(merge_by_content) {
        std::vector<unsigned long long> nsig(node_count, 0);
        strain_content_sig(mrca, root_time, window, seasons, nsig);
        std::unordered_map<unsigned long long, int> rep_of;
        for(Clade& c : clades) {
            c.sig = 0;
            for(Node* r : c.roots) c.sig += nsig[r->idx];
            auto it = rep_of.find(c.sig);
            if(it == rep_of.end()) {
                rep_of[c.sig] = c.id;
                c.rep_id = c.id;
            } else {
                Clade& cur = clades[it->second];
                if(c.hap.size() > cur.hap.size()) {   // keep the most derived genotype
                    cur.rep_id = c.id;
                    c.rep_id = c.id;
                    rep_of[c.sig] = c.id;
                } else {
                    c.rep_id = cur.id;
                }
            }
        }
        // Collapse rep_id to a fixed point (a class can hand over its representative more
        // than once as deeper genotypes turn up) and total up the class sizes.
        for(Clade& c : clades) {
            int r = c.rep_id;
            while(clades[r].rep_id != r) r = clades[r].rep_id;
            c.rep_id = r;
        }
        for(Clade& c : clades) c.merged = 0;
        for(Clade& c : clades) clades[c.rep_id].merged++;
        // Re-aim the nesting at surviving units: walk up until the class changes.
        for(Clade& c : clades) {
            if(c.rep_id != c.id) continue;
            int p = c.parent_id;
            while(p >= 0 && clades[p].rep_id == c.id) p = clades[p].parent_id;
            c.parent_id = (p >= 0) ? clades[p].rep_id : -1;
        }
    } else {
        for(Clade& c : clades) c.rep_id = c.id;
    }

    // Reversion diagnostic, for the clades the paper would actually predict on: how many
    // of the season-t sequences inside T_n sit on a genotype that has since lost one of
    // the haplotype's alleles. T_n contains all carriers but need not consist only of
    // them, so this is the gap between "descends from a_n" and "carries a_n".
    std::vector<long long> own(node_count, -1);
    for(Clade& c : clades) {
        if(c.rep_id != c.id) continue;
        if(seqs_t_total <= 0 || (double)c.seqs_t / (double)seqs_t_total <= cutoff) continue;
        c.noncarrier_seqs = 0;
        std::vector<Node*> stack = c.roots;
        while(!stack.empty()) {
            Node* v = stack.back();
            stack.pop_back();
            if(!hap_contains(hap[v->idx], c.hap)) {
                if(own[v->idx] < 0) own[v->idx] = own_window_seqs(v, root_time, window, seasons);
                c.noncarrier_seqs += own[v->idx];
            }
            for(Node* n : v->children) {
                if(n->in_window) stack.push_back(n);
            }
        }
    }
}

// Why the season's last common ancestor sits where it does. window_mrca stops at the
// first node that either holds a season-t sample itself or has two children carrying the
// season, so these three numbers are exactly the evidence pinning a(t): the season-t
// sequences at a(t) itself, how many of its children carry the season, and how many
// season-t sequences fall outside its largest carrying child. When the last two are
// small, a(t) is being held in place by a handful of sequences rather than by real
// standing diversity, and the haplotypes are referred to a needlessly distant ancestor.
static void lca_evidence(
    Node* mrca, double root_time, long long window, bool seasons,
    long long& own_seqs, long long& branches, long long& minor_seqs
) {
    own_seqs = own_window_seqs(mrca, root_time, window, seasons);
    branches = 0;
    long long biggest = 0, total = 0;
    for(Node* n : mrca->children) {
        if(!n->in_window) continue;
        branches++;
        total += n->cur_count;
        if(n->cur_count > biggest) biggest = n->cur_count;
    }
    minor_seqs = own_seqs + total - biggest;
}

void generate_clade_outputs(
    const AttachedTree& attached,
    const DatasetConfig& config,
    const std::string& output_dir,
    bool seasons,
    double cutoff,
    double lca_floor,
    bool node_defined,
    double clock_z,
    bool merge_by_content,
    bool prune_future,
    bool refit_ancestors
) {
    if(!(lca_floor >= 0.0 && lca_floor < 0.5)) {
        throw std::runtime_error("--clade-lca-floor must be in [0, 0.5)");
    }
    std::filesystem::create_directories(output_dir);
    Node* root = clone_tree(attached.root);
    char* h_sequences = attached.h_sequences;
    long long L = attached.L;
    long long node_count = attached.node_count;

    // Same tidy-up the interval path applies, so the genotypes here are the genotypes
    // that path turns into examples.
    std::cout << "Cleaning up tree..." << std::endl;
    cleanup_pruned_tree(root);
    check_branches_have_mutations(root, h_sequences, L);

    // Outlier exclusion first and once, so that every season sees the same data set.
    if(clock_z > 0.0) {
        std::cout << "Clock filter at " << clock_z
                  << " interquartile distances:" << std::endl;
        ClockFilterReport rep = clock_filter_tree(root, h_sequences, L, node_count,
                                                  attached.root_time, clock_z);
        std::cout << "  divergence ~ " << std::fixed << std::setprecision(2) << rep.intercept
                  << " + " << rep.rate << " per year, keeping residuals in ["
                  << rep.lo << ", " << rep.hi << "]" << std::endl;
        std::cout << "  dropped " << (rep.dropped_low + rep.dropped_high) << " of "
                  << rep.samples << " samples ("
                  << std::setprecision(3)
                  << (100.0 * (double)(rep.dropped_low + rep.dropped_high) / (double)rep.samples)
                  << "%): " << rep.dropped_low << " too diverged for their date's "
                  << "expectation, " << rep.dropped_high << " too little" << std::endl;
        cleanup_pruned_tree(root);
        check_branches_have_mutations(root, h_sequences, L);
    }
    set_subtree_samples(root);

    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    sample_date_span(root, attached.root_time, lo, hi);
    if(!(lo <= hi)) throw std::runtime_error("No samples on the tree; nothing to window");
    long long first, last;
    if(seasons) {
        first = (long long)std::floor(lo - 0.25) + 1;
        last  = (long long)std::ceil(hi + 0.25) - 1;
    } else {
        first = window_index(lo, false);
        last  = window_index(hi, false);
    }
    std::cout << std::fixed << std::setprecision(3)
              << "Sample dates span " << lo << " to " << hi << std::setprecision(2)
              << ", i.e. " << (seasons ? "seasons " : "windows ")
              << first << " through " << last << std::endl;
    std::cout << "Clade cutoff: X_n(t) > " << cutoff << std::endl;
    std::cout << (node_defined
        ? "Clades defined by descent from a_n"
        : "Clades defined by carriage of a_n's mutant haplotype") << std::endl;
    if(!merge_by_content) {
        std::cout << "One clade per genotype; no merging by strain content" << std::endl;
    }
    if(lca_floor > 0.0) {
        std::cout << "a(t) noise floor: branches below " << lca_floor
                  << " of a season are not treated as standing diversity" << std::endl;
    }
    if(prune_future) {
        std::cout << "--clade-prune-future: each season's clades are read off a tree "
                  << "holding only samples dated before that season ends" << std::endl;
    }
    if(refit_ancestors) {
        if(!prune_future) throw std::runtime_error(
            "--clade-refit-ancestors needs --clade-prune-future");
        std::cout << "--clade-refit-ancestors: ancestral genotypes re-inferred by Fitch "
                  << "from the surviving tips alone, so only the topology is inherited"
                  << std::endl;
    }

    std::string tag = seasons ? "seasons" : "intervals";
    std::string csv_path = output_dir + "/clades_" + tag + ".csv";
    std::string sum_path = output_dir + "/clades_" + tag + "_summary.csv";
    std::ofstream csv(csv_path);
    if(!csv.is_open()) throw std::runtime_error("Could not open " + csv_path + " for writing");
    std::ofstream sum(sum_path);
    if(!sum.is_open()) throw std::runtime_error("Could not open " + sum_path + " for writing");

    csv << "season,lo,hi,clade_id,parent_clade_id,n_mut,n_subtrees,genotype_nodes,merged,"
        << "nodes,seqs_t,X_t,seqs_next,X_next,samples_all,above_cutoff,noncarrier_seqs_t,"
        << "root_nodes,haplotype,haplotype_idx" << std::endl;
    sum << "season,lo,hi,seqs_t,seqs_t_in_Tt,seqs_next,Tt_nodes,n_genotypes,n_clades,"
        << "n_above_cutoff,n_multi_subtree,lca_node,lca_n_mut,lca_own_seqs,lca_branches,"
        << "lca_minor_seqs" << std::endl;

    // Two companion tables, because the clade rows above carry frequencies but not the
    // sequences a fitness model needs. `clade_strains_*` lists the strains of each season
    // -- one row per genotype observed in it, with its multiplicity m_i -- and
    // `clade_nodes_*` gives every node those strains hang from, with its tree parent and
    // its sequence, so clade membership (is this strain below that clade's root?) and a
    // per-strain sequence feature are both recoverable. Nodes are keyed by the tree index
    // that `root_nodes` and `lca_node` refer to.
    std::string strains_path = output_dir + "/clade_strains_" + tag + ".csv";
    std::string nodes_path = output_dir + "/clade_nodes_" + tag + ".csv";
    std::ofstream strains_out(strains_path);
    if(!strains_out.is_open()) throw std::runtime_error("Could not open " + strains_path);
    strains_out << "season,node_id,m_i" << std::endl;
    std::set<int> seen_nodes;

    // Parents are taken from the unpruned tree, once, and not from whatever each season's
    // pruning leaves behind. Suppressing a node splices its child onto its parent, so the
    // *set* of surviving ancestors of any node is the same either way and membership
    // ("is this strain below that clade's root?") reads the same -- but a pruned parent
    // pointer can name a node no season ever emits, which would break the walk. This map
    // is the one the table is written from.
    std::map<int, int> true_parent;
    {
        std::function<void(Node*)> walk = [&](Node* v) {
            true_parent[v->idx] = v->parent ? v->parent->idx : -1;
            for(Node* c : v->children) walk(c);
        };
        walk(root);
    }

    const char* global_root_seq = h_sequences + (long long)root->idx * L;
    long long total_clades = 0, total_above = 0, total_multi = 0;
    PruneAudit audit;
    std::vector<char> h_local;
    long long refit_changed = 0;
    // One bit per residue for the Fitch masks; anything off the alphabet shares the last
    // slot, which is fine because those sites are ambiguity and count as ambiguity.
    std::vector<int> residue_code(256, 0);
    std::string residue_list;
    {
        int next = 0;
        for(char c : config.alphabet) { residue_code[(unsigned char)c] = next++; residue_list += c; }
        residue_code[(unsigned char)config.ambig_char] = next++;
        residue_list += config.ambig_char;
        residue_list += config.ambig_char;   // slot for anything off the alphabet
        if(next > 31) throw std::runtime_error("alphabet too large for the Fitch audit");
        for(int i = 0; i < 256; i++) {
            bool known = (config.alphabet.find((char)i) != std::string::npos)
                         || ((char)i == config.ambig_char);
            if(!known) residue_code[i] = next;
        }
    }

    for(long long w = first; w <= last; w++) {
        // The tree this season's clades are read off. Without --clade-prune-future that is
        // the whole tree, as before; with it, a copy holding only the samples that existed
        // when the season ended.
        //
        // The order matters. mark_interval runs *first*, on the unpruned copy, so that
        // cur_count and next_count are the real counts: next_count is the target and is
        // allowed to be the future, and every count sits on a node that survives, because
        // cleanup_pruned_tree only ever removes nodes carrying no samples at all. Dropping
        // the future samples and then cleaning up is exactly the two-step the question
        // asks for -- forget the sequences, then forget the internal nodes those sequences
        // were the only evidence for. A node whose own samples were all in the future but
        // which still joins two surviving lineages is kept, because it is still the MRCA
        // of two things we can see.
        Node* tree = root;
        const char* h_seq_season = h_sequences;
        long long dropped = 0, nodes_before = 0, nodes_after = 0;
        std::map<int, std::set<int>> uncertain;   // node idx -> positions not pinned by
                                                  // the surviving subtree
        if(prune_future) {
            tree = clone_tree(root);
            mark_interval(tree, attached.root_time, w, w + 1, seasons);
            nodes_before = count_nodes(tree);
            dropped = drop_samples_from(tree, attached.root_time,
                                        window_hi(w, seasons));
            cleanup_pruned_tree(tree);
            nodes_after = count_nodes(tree);
            audit_pruned_tree(tree, h_sequences, L, audit);
            if(refit_ancestors) {
                // A private copy of every genotype, because an ancestor's re-inferred
                // state is specific to this season's pruned tree. Observed nodes are
                // pinned to what was observed, so the strains and their features -- the
                // only sequences the model itself reads -- are untouched; what moves is
                // the ancestral states the clade definitions are drawn from.
                h_local.assign(h_sequences, h_sequences + L * node_count);
                std::vector<std::vector<uint32_t>> up(node_count);
                fitch_up(tree, h_sequences, L, residue_code.data(), audit, uncertain, &up);
                long long changed = 0;
                fitch_down(tree, h_local.data(), L, residue_code.data(), up,
                           residue_list, nullptr, changed);
                refit_changed += changed;
                h_seq_season = h_local.data();
            } else {
                fitch_up(tree, h_sequences, L, residue_code.data(), audit, uncertain);
            }
        } else {
            mark_interval(tree, attached.root_time, w, w + 1, seasons);
        }
        if(!tree->in_window) {
            std::cout << "  " << (seasons ? "season " : "window ") << w
                      << ": no samples, skipped" << std::endl;
            if(prune_future) delete_tree(tree);
            continue;
        }
        Node* mrca = window_core_lca(tree, attached.root_time, w, seasons, lca_floor);
        long long seqs_t = tree->cur_count;
        long long seqs_next = tree->next_count;

        std::vector<Clade> clades;
        extract_window_clades(mrca, h_seq_season, L, node_count, attached.root_time, w,
                              seasons, cutoff, seqs_t, node_defined,
                              merge_by_content, clades);

        const char* a_t = h_seq_season + (long long)mrca->idx * L;
        long long lca_n_mut = 0;
        for(long long p = 0; p < L; p++) if(a_t[p] != global_root_seq[p]) lca_n_mut++;

        // Does any of that residual uncertainty reach a clade the analysis uses? A
        // selected clade is a root node plus the haplotype separating it from a(t), so
        // the ways it could move are (i) an uncertain residue at its own root or at a(t),
        // which would relabel it, and (ii) an uncertain residue anywhere in T_t at one of
        // the positions its haplotype is built from, which could move a carrier.
        if(prune_future && !uncertain.empty()) {
            std::set<int> hap_pos, sel_roots;
            sel_roots.insert(mrca->idx);
            for(const Clade& c : clades) {
                if(c.rep_id != c.id) continue;
                if(!(seqs_t > 0 && (double)c.seqs_t / (double)seqs_t > cutoff)) continue;
                for(const auto& e : c.hap) hap_pos.insert(e.first);
                for(Node* r : c.roots) sel_roots.insert(r->idx);
            }
            for(int idx : sel_roots) {
                auto it = uncertain.find(idx);
                if(it != uncertain.end()) audit.touch_roots += (long long)it->second.size();
            }
            for(const auto& kv : uncertain) {
                for(int pos : kv.second) if(hap_pos.count(pos)) audit.touch_hap++;
            }
            for(const Clade& c : clades) {
                if(c.rep_id != c.id) continue;
                if(!(seqs_t > 0 && (double)c.seqs_t / (double)seqs_t > cutoff)) continue;
                bool hit = false;
                for(Node* r : c.roots) {
                    auto it = uncertain.find(r->idx);
                    if(it != uncertain.end() && !it->second.empty()) hit = true;
                }
                if(hit) audit.touched_clades++;
            }
        }

        // The empty haplotype is a(t) itself, so its clade is the whole of T_t.
        long long n_above = 0, n_multi = 0, tt_nodes = 0, n_units = 0, seqs_in_tt = 0;
        for(const Clade& c : clades) {
            if(c.hap.empty()) { tt_nodes = c.nodes_t; seqs_in_tt = c.seqs_t; }
        }
        for(const Clade& c : clades) {
            if(c.rep_id != c.id) continue;
            n_units++;
            double x_t = seqs_t > 0 ? (double)c.seqs_t / (double)seqs_t : 0.0;
            bool above = x_t > cutoff;
            if(above) n_above++;
            if(c.roots.size() > 1) n_multi++;
            csv << w << "," << std::fixed << std::setprecision(2)
                << window_lo(w, seasons) << "," << window_hi(w, seasons) << ","
                << c.id << ",";
            if(c.parent_id >= 0) csv << c.parent_id; else csv << "NA";
            csv << "," << c.hap.size() << "," << c.roots.size() << ","
                << c.genotype_nodes << "," << c.merged << "," << c.nodes_t << ","
                << c.seqs_t << "," << std::setprecision(6) << x_t << "," << c.seqs_next << ",";
            if(seqs_next > 0) csv << (double)c.seqs_next / (double)seqs_next; else csv << "NA";
            csv << "," << c.samples_all << "," << (above ? "True" : "False") << ",";
            if(c.noncarrier_seqs >= 0) csv << c.noncarrier_seqs; else csv << "NA";
            csv << ",";
            for(size_t i = 0; i < c.roots.size(); i++) {
                if(i) csv << ';';
                csv << c.roots[i]->idx;
            }
            csv << "," << format_haplotype(config, c.hap) << ",";
            for(size_t i = 0; i < c.hap.size(); i++) {
                if(i) csv << ';';
                csv << c.hap[i].first << ':' << c.hap[i].second;
            }
            if(c.hap.empty()) csv << "root";
            csv << std::endl;
        }

        // Strains of this season: every T_t node holding season-t sequences of its own.
        {
            std::vector<Node*> tt;
            collect_window_subtree(mrca, tt);
            for(Node* v : tt) {
                long long m = own_window_seqs(v, attached.root_time, w, seasons);
                if(m > 0) {
                    seen_nodes.insert(v->idx);
                    strains_out << w << "," << v->idx << "," << m << std::endl;
                }
            }
        }

        long long lca_own = 0, lca_branches = 0, lca_minor = 0;
        lca_evidence(mrca, attached.root_time, w, seasons, lca_own, lca_branches, lca_minor);
        sum << w << "," << std::fixed << std::setprecision(2)
            << window_lo(w, seasons) << "," << window_hi(w, seasons) << ","
            << seqs_t << "," << seqs_in_tt << "," << seqs_next << "," << tt_nodes << ","
            << clades.size() << "," << n_units << "," << n_above << "," << n_multi << ","
            << mrca->idx << "," << lca_n_mut << "," << lca_own << "," << lca_branches << ","
            << lca_minor << std::endl;

        total_clades += n_units;
        total_above += n_above;
        total_multi += n_multi;
        std::cout << "  " << (seasons ? "season " : "window ") << w
                  << " [" << window_lo(w, seasons) << "," << window_hi(w, seasons)
                  << "): " << seqs_t << " sequences (" << seqs_in_tt << " under a(t)), T_t has "
                  << tt_nodes << " nodes, " << clades.size() << " genotypes -> " << n_units
                  << " clades, " << n_above
                  << " above the cutoff, " << n_multi << " on more than one subtree";
        if(prune_future) {
            std::cout << "; pruned " << dropped << " later samples and "
                      << (nodes_before - nodes_after) << " of " << nodes_before
                      << " nodes";
        }
        std::cout << std::endl;
        if(prune_future) delete_tree(tree);
    }

    if(prune_future) {
        std::cout << "What the pruning cannot remove, summed over seasons:" << std::endl;
        std::cout << "  " << audit.unsupported << " of " << audit.joins
                  << " surviving nodes that join two or more lineages carry no "
                  << "substitution of their own" << std::endl;
        std::cout << "  " << audit.unparsimonious << " residues over "
                  << audit.unparsimonious_nodes << " of " << audit.inferred
                  << " purely inferred ancestors disagree with a unanimous set of their "
                  << "surviving children" << std::endl;
        long long judged = audit.det_sites + audit.amb_sites + audit.mismatch_sites;
        std::cout << "  Fitch over the surviving subtree alone, "
                  << judged << " (ancestor, position) pairs: "
                  << audit.det_sites << " determined and matching, "
                  << audit.mismatch_sites << " determined and NOT matching, "
                  << audit.amb_sites << " ambiguous (over " << audit.amb_nodes
                  << " ancestors) -- the ambiguous ones are the ties the pruned "
                  << "descendants could have broken" << std::endl;
        if(refit_ancestors) {
            std::cout << "  ancestral states re-inferred from the survivors: "
                      << refit_changed << " residues over all seasons differ from what "
                      << "the full-tree inference had written there" << std::endl;
        }
        std::cout << "  of that uncertainty, " << audit.touch_roots
                  << " residues sit on a selected clade's own root or on a(t) ("
                  << audit.touched_clades << " selected clades affected), and "
                  << audit.touch_hap << " sit anywhere in T_t at a position some "
                  << "selected clade's haplotype is built from" << std::endl;
    }

    strains_out.close();
    std::ofstream nodes_out(nodes_path);
    if(!nodes_out.is_open()) throw std::runtime_error("Could not open " + nodes_path);
    nodes_out << "node_id,parent_node_id,seq" << std::endl;
    // Every ancestor of an emitted node is emitted too, so the walk from a strain up to
    // a clade root never runs off the end of the table.
    std::set<int> emit = seen_nodes;
    for(int idx : seen_nodes) {
        for(int v = idx; v >= 0; ) {
            auto it = true_parent.find(v);
            if(it == true_parent.end()) break;
            v = it->second;
            if(v >= 0) emit.insert(v);
        }
    }
    for(int idx : emit) {
        nodes_out << idx << ",";
        int par = true_parent.count(idx) ? true_parent[idx] : -1;
        if(par >= 0) nodes_out << par; else nodes_out << "NA";
        nodes_out << ",";
        const char* sv = h_sequences + (long long)idx * L;
        for(long long p2 = 0; p2 < L; p2++) nodes_out << sv[p2];
        nodes_out << std::endl;
    }
    nodes_out.close();

    csv.close();
    sum.close();
    std::cout << "Wrote " << csv_path << " (" << total_clades << " clades, "
              << total_above << " above the cutoff, " << total_multi
              << " spanning more than one subtree)" << std::endl;
    std::cout << "Wrote " << sum_path << std::endl;
    std::cout << "Wrote " << strains_path << " and " << nodes_path << " ("
              << seen_nodes.size() << " nodes)" << std::endl;

    delete_tree(root);
}

void generate_cutoff_outputs(
    const AttachedTree& attached,
    const DatasetConfig& config,
    const std::string& output_dir,
    double cutoff
) {
    std::filesystem::create_directories(output_dir);
    Node* root = clone_tree(attached.root);
    char* h_sequences = attached.h_sequences;
    long long L = attached.L;

    std::vector<std::string> variants = dataset_variants(config);
    set_variant_descendants(root, variants);

    double timestep = 0.1;
    double pssm_start_time = 0.0;
    // SC2/H3N2 variant PSSMs are cumulative from root; newer datasets window after the cutoff.
    bool variant_pssms_post_cutoff = false;
    int num_timesteps;
    double max_time_rel = attached.max_date - attached.root_time;
    if (config.type == "h3n2") {
        num_timesteps = 190;
    } else if (config.type == "rsv_b" || config.type == "rsv_a" || config.type == "denv2") {
        pssm_start_time = cutoff;
        variant_pssms_post_cutoff = true;
        num_timesteps = (int)std::ceil((max_time_rel - cutoff) / timestep) + 1;
        if (num_timesteps < 1) num_timesteps = 1;
    } else {
        num_timesteps = (int)std::ceil(max_time_rel / timestep) + 1;
        if (num_timesteps < 60) num_timesteps = 60;
    }
    std::cout << "PSSM timesteps: " << num_timesteps
              << " starting at " << pssm_start_time
              << " (step " << timestep << ")" << std::endl;

    std::vector<int> blank_pssm(L * ALLOWED_AAS.size(), 0);
    std::vector<int> pssm_root = blank_pssm;
    std::vector<std::vector<int>> blank_pssms(num_timesteps, blank_pssm);
    std::vector<std::vector<std::vector<int>>> pssm_variants(variants.size(), blank_pssms);

    compile_pssms(root, h_sequences, pssm_root, pssm_variants, cutoff, L, variants, timestep, num_timesteps, pssm_start_time, variant_pssms_post_cutoff);
    write_pssms(output_dir, L, timestep, pssm_start_time, variants, pssm_root, pssm_variants);

    std::vector<Node*> pruned_roots;
    std::cout << "Pruning tree at cutoff " << cutoff << "..." << std::endl;
    prune_tree(root, pruned_roots, cutoff);
    std::cout << "Done pruning tree, " << pruned_roots.size() << " pruned roots" << std::endl;

    check_branches_have_mutations(root, h_sequences, L);
    for (Node* n : pruned_roots) check_branches_have_mutations(n, h_sequences, L);

    std::cout << "Cleaning up pruned tree..." << std::endl;
    cleanup_pruned_tree(root);
    std::cout << "Done cleaning up" << std::endl;

    check_branches_have_mutations(root, h_sequences, L);
    for (Node* n : pruned_roots) check_branches_have_mutations(n, h_sequences, L);

    long long train_samples = set_subtree_samples(root);
    long long test_samples = 0;
    for (Node* n : pruned_roots) test_samples += set_subtree_samples(n);
    long long train_late = 0;
    count_samples_after(root, cutoff, train_late);
    std::cout << "Samples by side of the split: " << train_samples << " training, "
              << test_samples << " testing"
              << " (" << root->leaf_times.size() << " training samples match the root genotype; "
              << train_late << " training samples are dated after the cutoff)" << std::endl;

    std::mt19937 rng(123);
    std::cout << "Setting mutation pos order..." << std::endl;
    set_mutation_pos_order(root, h_sequences, L, rng);
    for (Node* n : pruned_roots) set_mutation_pos_order(n, h_sequences, L, rng);
    std::cout << "Done setting mutation pos order" << std::endl;

    std::ostringstream cutoff_ss;
    cutoff_ss << cutoff;
    std::string txt_path = output_dir + "/examples_cutoff_" + cutoff_ss.str() + ".txt";

    std::ofstream txt_out(txt_path);
    if (!txt_out.is_open()) throw std::runtime_error("Could not open " + txt_path + " for writing");

    int example_id = 0;
    write_training_examples(root, h_sequences, txt_out, L, false, example_id, variants);
    for (Node* n : pruned_roots) {
        write_training_examples(n, h_sequences, txt_out, L, true, example_id, variants);
    }
    txt_out.close();
    std::cout << "Wrote " << txt_path << std::endl;

    delete_tree(root);
    for (Node* n : pruned_roots) delete_tree(n);
}

CliOptions parse_cli(int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        throw std::runtime_error("not enough arguments");
    }

    CliOptions opts;
    opts.output_dir = argv[1];
    opts.cutoff = std::atof(argv[2]);

    if (!is_dataset_type(argv[3])) {
        opts.dataset_type = "sc2";
        std::filesystem::path fasta_path(argv[3]);
        std::filesystem::path data_dir = fasta_path.parent_path();
        if (data_dir.empty()) data_dir = ".";
        opts.ref_json_path = infer_legacy_sc2_json_path(data_dir, "ref.json");
        opts.tree_json_path = infer_legacy_sc2_json_path(data_dir, "tree.json");
        opts.metadata_path = argv[4];
        opts.alignment_files.push_back(argv[3]);

        for (int i = 5; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--cache") {
                if (++i >= argc) throw std::runtime_error("--cache requires a path");
                opts.cache_path = argv[i];
            } else if (arg == "--force-rebuild-cache") {
                opts.force_rebuild_cache = true;
            } else if (arg == "--intervals") {
                opts.intervals = true;
            } else if (arg == "--seasons") {
                opts.seasons = true;
            } else if (arg == "--clades") {
                opts.clades = true;
            } else if (arg == "--clade-cutoff") {
                if (++i >= argc) throw std::runtime_error("--clade-cutoff requires a frequency");
                opts.clade_cutoff = std::atof(argv[i]);
            } else if (arg == "--clade-lca-floor") {
                if (++i >= argc) throw std::runtime_error("--clade-lca-floor requires a fraction");
                opts.clade_lca_floor = std::atof(argv[i]);
            } else if (arg == "--clade-haplotype") {
                opts.clade_haplotype = true;
            } else if (arg == "--clade-prune-future") {
                opts.clade_prune_future = true;
            } else if (arg == "--clade-refit-ancestors") {
                opts.clade_refit_ancestors = true;
            } else if (arg == "--no-augment") {
                opts.no_augment = true;
            } else if (arg == "--clade-no-merge") {
                opts.clade_merge = false;
            } else if (arg == "--clade-clock-filter") {
                if (++i >= argc) throw std::runtime_error("--clade-clock-filter requires a number of interquartile distances");
                opts.clade_clock_z = std::atof(argv[i]);
            } else if (arg == "--no-cache") {
                opts.no_cache = true;
            } else if (arg == "--cpu") {
                opts.force_cpu = true;
            } else if (arg == "--grid") {
                if (++i >= argc) throw std::runtime_error("--grid requires a count");
                opts.grid_count = std::atoi(argv[i]);
            } else if (arg == "--grid-interior") {
                opts.grid_interior = true;
            } else if (arg == "--grid-output-dir") {
                if (++i >= argc) throw std::runtime_error("--grid-output-dir requires a directory");
                opts.grid_output_dir = argv[i];
            } else if (is_integer_arg(arg)) {
                opts.max_seqs = std::atoi(arg.c_str());
            } else {
                throw std::runtime_error("Unexpected legacy argument: " + arg);
            }
        }

        if (opts.grid_count < 0) throw std::runtime_error("--grid must be non-negative");
        if (opts.grid_count > 0 && opts.grid_output_dir.empty()) opts.grid_output_dir = opts.output_dir;
        std::cout << "Using legacy SC2 CLI; inferred ref_json=" << opts.ref_json_path
                  << " tree_json=" << opts.tree_json_path << std::endl;
        return opts;
    }

    if (argc < 8) {
        print_usage(argv[0]);
        throw std::runtime_error("not enough arguments for dataset-aware CLI");
    }

    opts.dataset_type = argv[3];
    opts.ref_json_path = argv[4];
    opts.tree_json_path = argv[5];
    opts.metadata_path = argv[6];

    DatasetConfig config = create_dataset_config(opts.dataset_type);
    int expected_files = (int)config.segment_names.size();

    for (int i = 7; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--cache") {
            if (++i >= argc) throw std::runtime_error("--cache requires a path");
            opts.cache_path = argv[i];
        } else if (arg == "--force-rebuild-cache") {
            opts.force_rebuild_cache = true;
        } else if (arg == "--intervals") {
            opts.intervals = true;
        } else if (arg == "--seasons") {
            opts.seasons = true;
        } else if (arg == "--clades") {
            opts.clades = true;
        } else if (arg == "--clade-cutoff") {
            if (++i >= argc) throw std::runtime_error("--clade-cutoff requires a frequency");
            opts.clade_cutoff = std::atof(argv[i]);
        } else if (arg == "--clade-lca-floor") {
            if (++i >= argc) throw std::runtime_error("--clade-lca-floor requires a fraction");
            opts.clade_lca_floor = std::atof(argv[i]);
        } else if (arg == "--clade-haplotype") {
            opts.clade_haplotype = true;
        } else if (arg == "--clade-prune-future") {
            opts.clade_prune_future = true;
        } else if (arg == "--clade-refit-ancestors") {
            opts.clade_refit_ancestors = true;
        } else if (arg == "--no-augment") {
            opts.no_augment = true;
        } else if (arg == "--clade-no-merge") {
            opts.clade_merge = false;
        } else if (arg == "--clade-clock-filter") {
            if (++i >= argc) throw std::runtime_error("--clade-clock-filter requires a number of interquartile distances");
            opts.clade_clock_z = std::atof(argv[i]);
        } else if (arg == "--no-cache") {
            opts.no_cache = true;
        } else if (arg == "--cpu") {
            opts.force_cpu = true;
        } else if (arg == "--grid") {
            if (++i >= argc) throw std::runtime_error("--grid requires a count");
            opts.grid_count = std::atoi(argv[i]);
        } else if (arg == "--grid-interior") {
            opts.grid_interior = true;
        } else if (arg == "--grid-output-dir") {
            if (++i >= argc) throw std::runtime_error("--grid-output-dir requires a directory");
            opts.grid_output_dir = argv[i];
        } else if (is_integer_arg(arg) && (int)opts.alignment_files.size() >= expected_files) {
            opts.max_seqs = std::atoi(arg.c_str());
        } else {
            opts.alignment_files.push_back(arg);
        }
    }

    if ((int)opts.alignment_files.size() != expected_files) {
        throw std::runtime_error(config.type + " requires " + std::to_string(expected_files) +
                                 " alignment file(s), got " + std::to_string(opts.alignment_files.size()));
    }
    if (opts.grid_count < 0) throw std::runtime_error("--grid must be non-negative");
    if (opts.grid_count > 0 && opts.grid_output_dir.empty()) opts.grid_output_dir = opts.output_dir;
    return opts;
}

int main(int argc, char* argv[]) {
    try {
        CliOptions opts = parse_cli(argc, argv);
        DatasetConfig config = create_dataset_config(opts.dataset_type);

        // Point the alphabet used by aa_to_index / PSSM code at this dataset's
        // alphabet (protein by default, nucleotide for sc2_nuc).
        ALLOWED_AAS = config.alphabet;

        std::cout << "Cutoff: " << opts.cutoff << std::endl;
        std::cout << "Dataset: " << config.type << std::endl;
        bool use_cache = !opts.no_cache && !opts.cache_path.empty();
        if (use_cache) std::cout << "Attached-tree cache: " << opts.cache_path << std::endl;

        AttachedTree attached;
        CacheManifest requested_manifest = make_cache_manifest(
            config.type,
            config.total_length,
            opts.max_seqs,
            opts.ref_json_path,
            opts.tree_json_path,
            opts.metadata_path,
            opts.alignment_files
        );

        bool loaded_cache = false;
        if (use_cache && !opts.force_rebuild_cache) {
            CacheManifest saved_manifest;
            std::string reason;
            if (load_cache_manifest(opts.cache_path, saved_manifest, reason) &&
                cache_manifest_matches(saved_manifest, requested_manifest, reason)) {
                if (load_attached_tree_cache(opts.cache_path, attached, saved_manifest, reason)) {
                    loaded_cache = true;
                    std::cout << "Loaded attached-tree cache with " << attached.node_count << " nodes" << std::endl;
                    check_branches_have_mutations(attached.root, attached.h_sequences, attached.L);
                } else {
                    std::cout << "Cache payload unusable, rebuilding: " << reason << std::endl;
                }
            } else {
                std::cout << "Cache miss, rebuilding: " << reason << std::endl;
            }
        }

        if (!loaded_cache) {
            attached = build_attached_tree(config, opts);
            if (use_cache) {
                populate_cache_runtime_metadata(requested_manifest, attached);
                std::string reason;
                if (!save_attached_tree_cache(opts.cache_path, attached, requested_manifest, reason)) {
                    throw std::runtime_error("Could not save attached-tree cache: " + reason);
                }
                std::cout << "Saved attached-tree cache to " << opts.cache_path << std::endl;
            }
        }

        // Benchmark mode: time only the augmentation phase (measured inside
        // build_attached_tree); skip downstream PSSM / training-example output.
        if (const char* bench = std::getenv("ANTIGEN_BENCH_ATTACH_ONLY")) {
            if (bench[0] != '\0' && bench[0] != '0') {
                std::cout << "ANTIGEN_BENCH_ATTACH_ONLY set; skipping downstream output." << std::endl;
                free(attached.h_sequences);
                delete_tree(attached.root);
                return 0;
            }
        }

        if (opts.clades) {
            generate_clade_outputs(attached, config, opts.output_dir, opts.seasons,
                                   opts.clade_cutoff, opts.clade_lca_floor,
                                   !opts.clade_haplotype, opts.clade_clock_z,
                                   opts.clade_merge, opts.clade_prune_future,
                                   opts.clade_refit_ancestors);
        }
        if (opts.intervals) {
            generate_interval_outputs(attached, config, opts.output_dir, opts.seasons);
        } else if (opts.clades) {
            // --clades on its own: the clade table is the whole output.
        } else if (opts.grid_count > 0) {
            double start = attached.min_date - attached.root_time;
            double end = attached.max_date - attached.root_time;
            if (opts.grid_count == 1) {
                double cutoff = opts.grid_interior ? start + (end - start) / 2.0 : start;
                generate_cutoff_outputs(attached, config, opts.grid_output_dir + "/cutoff_" + std::to_string(cutoff), cutoff);
            } else {
                for (int i = 0; i < opts.grid_count; i++) {
                    double fraction = opts.grid_interior
                        ? ((double)(i + 1) / (double)(opts.grid_count + 1))
                        : ((double)i / (double)(opts.grid_count - 1));
                    double cutoff = start + (end - start) * fraction;
                    std::ostringstream cutoff_ss;
                    cutoff_ss << std::fixed << std::setprecision(3) << cutoff;
                    std::string output_dir = opts.grid_output_dir + "/cutoff_" + cutoff_ss.str();
                    generate_cutoff_outputs(attached, config, output_dir, cutoff);
                }
            }
        } else {
            generate_cutoff_outputs(attached, config, opts.output_dir, opts.cutoff);
        }

        free(attached.h_sequences);
        delete_tree(attached.root);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
