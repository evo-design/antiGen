#include "tree.hpp"

#include "json.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

std::string formatMutation(const Mutation& m) {
    return std::string(1, m.from) + std::to_string(m.site) + std::string(1, m.to);
}

static Mutation parseMutation(const std::string& s) {
    if (s.size() < 3) throw std::runtime_error("malformed mutation '" + s + "'");
    Mutation m;
    m.from = s.front();
    m.to   = s.back();
    m.site = 0;
    for (size_t i = 1; i + 1 < s.size(); ++i) {
        char c = s[i];
        if (c < '0' || c > '9') throw std::runtime_error("malformed mutation '" + s + "'");
        m.site = m.site * 10 + (c - '0');
    }
    if (m.site <= 0) throw std::runtime_error("malformed mutation '" + s + "'");
    return m;
}

// Parses one tree node. The node is appended before its children are read,
// because Nextstrain writes "children" ahead of "name".
static int parseNode(JsonReader& r, std::vector<RawNode>& out, const std::string& gene,
                     int maxSite) {
    int self = static_cast<int>(out.size());
    out.emplace_back();

    forEachMember(r, [&](const std::string& key) {
        if (key == "name") {
            out[self].name = r.readString();
        } else if (key == "children") {
            forEachElement(r, [&]() {
                int child = parseNode(r, out, gene, maxSite);
                out[self].children.push_back(child);
            });
        } else if (key == "node_attrs") {
            forEachMember(r, [&](const std::string& ak) {
                if (ak == "num_date") {
                    forEachMember(r, [&](const std::string& nk) {
                        if (nk == "value") {
                            const double v = r.readNumber();
                            if (v == v) out[self].numDate = v;  // NaN means absent
                        } else {
                            r.skipValue();
                        }
                    });
                } else if (ak == "clade_membership") {
                    forEachMember(r, [&](const std::string& ck) {
                        if (ck == "value") out[self].clade = r.readString();
                        else               r.skipValue();
                    });
                } else {
                    r.skipValue();
                }
            });
        } else if (key == "branch_attrs") {
            forEachMember(r, [&](const std::string& bk) {
                if (bk != "mutations") { r.skipValue(); return; }
                forEachMember(r, [&](const std::string& g) {
                    if (g != gene) { r.skipValue(); return; }
                    forEachElement(r, [&]() {
                        Mutation m = parseMutation(r.readString());
                        if (maxSite == 0 || m.site <= maxSite) out[self].sMut.push_back(m);
                    });
                });
            });
        } else {
            r.skipValue();
        }
    });
    return self;
}

std::vector<RawNode> parseNextstrainTree(const std::string& path, const std::string& gene,
                                         int maxSite) {
    std::string text = readWholeFile(path);
    JsonReader r(text);
    std::vector<RawNode> nodes;
    bool found = false;

    forEachMember(r, [&](const std::string& key) {
        if (key != "tree") { r.skipValue(); return; }
        parseNode(r, nodes, gene, maxSite);
        found = true;
    });
    if (!found) throw std::runtime_error("no \"tree\" member in " + path);
    return nodes;
}

std::string parseRootSequenceS(const std::string& path, const std::string& gene) {
    std::string text = readWholeFile(path);
    JsonReader r(text);
    std::string s;
    forEachMember(r, [&](const std::string& key) {
        if (key == gene) s = r.readString();
        else r.skipValue();
    });
    if (s.empty()) throw std::runtime_error("no \"" + gene + "\" sequence in " + path);
    return s;
}

int Tree::addNode(int par, const std::string& name) {
    int id = size();
    parent.push_back(par);
    children.emplace_back();
    labels.emplace_back();
    obs.emplace_back();
    srcName.push_back(name);
    date.push_back(kNoDate);
    matched.push_back(0);
    matchedDate.push_back(kNoDate);
    if (par >= 0) children[par].push_back(id);
    return id;
}

Tree buildTree(const std::vector<RawNode>& raw, const std::string& rootSeq, int nSites,
               std::vector<std::string>* mismatches) {
    Tree t;
    if (raw.empty()) return t;
    if (static_cast<int>(rootSeq.size()) < nSites) {
        throw std::runtime_error("root sequence is shorter than the site count");
    }

    // Input node ids are tree node ids, so the arrays are filled in input order
    // and the edges wired up afterwards. The root's own branch mutations are
    // relative to the reference, which the root sequence already accounts for, so
    // the root edge contributes nothing.
    t.parent.assign(raw.size(), -1);
    t.children.assign(raw.size(), {});
    t.labels.assign(raw.size(), {});
    t.obs.assign(raw.size(), {});
    t.matched.assign(raw.size(), 0);
    t.matchedDate.assign(raw.size(), kNoDate);
    t.date.assign(raw.size(), kNoDate);
    t.srcName.reserve(raw.size());
    for (const RawNode& n : raw) t.srcName.push_back(n.name);
    for (size_t i = 0; i < raw.size(); ++i) t.date[i] = raw[i].numDate;
    for (size_t i = 0; i < raw.size(); ++i) {
        for (int c : raw[i].children) {
            t.parent[c] = static_cast<int>(i);
            t.children[i].push_back(c);
        }
    }

    // One DFS accumulating the mutation path, so each leaf's observation is the
    // set of residues where its sequence differs from the root's. Nothing is
    // recorded at an internal node: Nextstrain's genotype there is an inference.
    std::vector<char> cur(rootSeq.begin(), rootSeq.begin() + nSites);
    std::vector<std::pair<int, char>> undo;

    struct Frame {
        int    node;
        size_t nextChild;
        size_t mark;
    };
    std::vector<Frame> stack;

    auto record = [&](int node) {
        if (!raw[node].children.empty()) return;
        t.labels[node].push_back(raw[node].name);
        for (int s = 0; s < nSites; ++s) {
            if (cur[s] != rootSeq[s]) {
                t.obs[node].push_back({static_cast<uint16_t>(s + 1), cur[s]});
            }
        }
    };

    record(0);
    stack.push_back({0, 0, 0});
    while (!stack.empty()) {
        Frame& f = stack.back();
        if (f.nextChild < raw[f.node].children.size()) {
            const int    c    = raw[f.node].children[f.nextChild++];
            const size_t mark = undo.size();
            for (const Mutation& m : raw[c].sMut) {
                if (m.site < 1 || m.site > nSites) continue;
                if (cur[m.site - 1] != m.from && mismatches) {
                    mismatches->push_back("node " + std::to_string(c) + " (" + raw[c].name +
                                          "): " + formatMutation(m) + " but the path has " +
                                          std::string(1, cur[m.site - 1]));
                }
                undo.emplace_back(m.site - 1, cur[m.site - 1]);
                cur[m.site - 1] = m.to;
            }
            record(c);
            stack.push_back({c, 0, mark});
        } else {
            const size_t mark = f.mark;
            stack.pop_back();
            while (undo.size() > mark) {
                cur[undo.back().first] = undo.back().second;
                undo.pop_back();
            }
        }
    }
    return t;
}

std::vector<std::string> nextstrainGenotypes(const std::vector<RawNode>& raw,
                                             const std::string& rootSeq, int nSites) {
    std::vector<std::string> geno(raw.size());
    if (raw.empty()) return geno;
    geno[0].assign(rootSeq.begin(), rootSeq.begin() + nSites);

    std::vector<int> stack{0};
    while (!stack.empty()) {
        const int node = stack.back();
        stack.pop_back();
        for (int c : raw[node].children) {
            std::string seq = geno[node];
            for (const Mutation& m : raw[c].sMut) {
                if (m.site >= 1 && m.site <= nSites) seq[m.site - 1] = m.to;
            }
            geno[c] = std::move(seq);
            stack.push_back(c);
        }
    }
    return geno;
}

// Exactly antiGen's own matcher: the full clade label, or the variant in
// parentheses inside it.
static bool matchVariant(const std::string& label, const std::string& variant) {
    if (label == variant) return true;
    return label.find("(" + variant + ")") != std::string::npos;
}

int findCladeRoot(const std::vector<RawNode>& raw, const std::string& variant,
                  std::string* label, int* nMatched, std::vector<int>* entries) {
    std::vector<int> parent(raw.size(), -1);
    for (size_t i = 0; i < raw.size(); ++i) {
        for (int c : raw[i].children) parent[c] = static_cast<int>(i);
    }

    std::vector<int> hits;
    std::string      seen;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (!raw[i].clade.empty() && matchVariant(raw[i].clade, variant)) {
            hits.push_back(static_cast<int>(i));
            if (seen.empty()) seen = raw[i].clade;
            else if (seen != raw[i].clade) {
                throw std::runtime_error("'" + variant + "' matches two different clade labels, '" +
                                         seen + "' and '" + raw[i].clade + "'");
            }
        }
    }
    if (hits.empty()) throw std::runtime_error("no node carries a clade matching '" + variant + "'");

    std::vector<int> ins;
    for (int v : hits) {
        const int p = parent[v];
        if (p < 0 || !matchVariant(raw[p].clade, variant)) ins.push_back(v);
    }
    if (label)    *label = seen;
    if (nMatched) *nMatched = static_cast<int>(hits.size());
    if (entries)  *entries = ins;
    if (ins.size() != 1) {
        std::string where;
        for (size_t k = 0; k < ins.size() && k < 8; ++k) {
            where += (k ? ", " : "") + std::to_string(ins[k]) + " (" + raw[ins[k]].name + ")";
        }
        throw std::runtime_error("clade '" + seen + "' begins at " + std::to_string(ins.size()) +
                                 " nodes, not one: " + where +
                                 " -- it is not a single variant root, so nothing is written");
    }
    return ins[0];
}

void writeTree(const Tree& t, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);

    std::fprintf(f, "#node\tparent\tobserved\tn_obs\tobs\tn_labels\tlabels\tn_matched\tdate\n");
    for (int i = 0; i < t.size(); ++i) {
        std::string o;
        for (size_t k = 0; k < t.obs[i].size(); ++k) {
            if (k) o += ',';
            o += std::to_string(t.obs[i][k].site);
            o += ':';
            o += t.obs[i][k].aa;
        }
        std::string labels;
        for (size_t k = 0; k < t.labels[i].size(); ++k) {
            if (k) labels += ',';
            labels += t.labels[i][k];
        }
        char when[32];
        if (dated(t.date[i])) std::snprintf(when, sizeof when, "%.4f", t.date[i]);
        else                  std::snprintf(when, sizeof when, ".");
        std::fprintf(f, "%d\t%d\t%d\t%zu\t%s\t%zu\t%s\t%u\t%s\n", i, t.parent[i],
                     t.observed(i) ? 1 : 0, t.obs[i].size(), o.empty() ? "." : o.c_str(),
                     t.labels[i].size(), labels.empty() ? "." : labels.c_str(), t.matched[i], when);
    }
    std::fclose(f);
}
