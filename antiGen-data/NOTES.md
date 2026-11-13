# NOTES

Working log for the `antigen` tool in this repo. Data provenance is in `README.md`.

## The task

A C++ command line tool over the Nextstrain tree in `data/`, in four steps:

1. **Allele sets.** Take the Nextstrain tree exactly as given — **no collapsing** — and
   record, at every (node, site), the set of alleles consistent with parsimony —
   Hartigan (1973). Gaps and stop codons are missing data. No node is committed to a
   genotype.
2. **Placement.** Append new sequences by parsimony placement, one at a time, updating
   the allele sets after each.
3. **Resolution.** Draw one allele per (node, site), top-down, so that the result is
   jointly parsimonious.
4. **The mutation tree, and the examples.** Only now: derive each edge as the diff
   between resolved parent and child, collapse the edges that carry nothing, explode the
   edges that carry more than one mutation, and write one example per node.

Everything is **S protein, amino acids only**. Nucleotides and the other genes are
ignored.

## Directive, 2026-10-21: the collapse moves to the end

> In step 1, don't bother with any collapsing of the tree at all — just record the allele
> sets at each (node, site) pair — no need to commit to any particular genotype at a node.
> Then run attachment as before, noting that some nodes may have identical allele sets and
> that's fine. Then resolve the ambiguities randomly to get a parsimonious configuration
> (as we do now). Finally, and only then, explode edges with multiple mutations (as we
> already do), collapse to a mutation tree, and write the examples.

The old pipeline collapsed twice: once up front, on Nextstrain's own genotypes, and again
at the end, on the resolved sequences. The first collapse is what forced every node to
carry a committed genotype, and the committed genotype is what three of the five
corrections below are about. Removing it removes them.

### What this changes

**1. Step 1 is a parse, not a collapse.** The tree is the Nextstrain tree: 7,123 nodes,
3,848 tips, 3,275 internal, 7,122 edges, largest multifurcation **19**. Tree node ids are
raw node ids, one to one. 6,145 of the edges carry no S mutation; they stay.

**2. A node has no genotype.** Observations are the **3,848 leaves**, plus the root, which
stays pinned to the sidecar. Every internal node is free. That includes the 35 degree-1
internal nodes named `<strain>_travel_history`: they are Nextstrain travel annotations
wrapping the real sample, not sequences, and they were never labels under the old code
either. So `observed(v) == isLeaf(v) || v == 0`, with no label bookkeeping needed to
decide it.

This is the substantive change, not just a bookkeeping one. In the collapsed tree a node
*was* a genotype and 969 of 978 carried samples, so the interior was almost entirely
pinned. Uncollapsed, the sample sits on its own leaf and the interior is free. The
observations are the same; the constraints on the interior are gone.

**3. `edgeMut` goes, and `obs` replaces it.** An observed node stores the sparse diff of
its sequence **against the root sequence** — `(site, allele)` pairs, missing residues
included as `X`. Not against the parent: the parent may be free and have nothing to diff
against. Root-relative also makes it random-access, so the allele at (node, site) needs no
path walk, which `genoAt` / `genoOn` / `genoAbove` all were. Size: a base-tree leaf is a
handful of substitutions plus its missing runs; a placed leaf is mean 4.5 substitutions
plus mean 11 missing sites, so ~16 entries, ~22 MB over 351k placements.

Nextstrain's own `branch_attrs` mutations are still what builds `obs` at the leaves (one
DFS accumulating down the tree) and still what `siteActive` is derived from. They are
input provenance after that, and the tree structure no longer holds them.

**4. The V1/V2 storage default becomes root-relative.** It was `V1 = {the node's
genotype}, V2 = {}`. It is now `V1 = {obs allele if observed, else rootSeq[s]}, V2 = {}`
— the same rule with "genotype" read as "observation, defaulting to the root". Exact at
every monomorphic site for free and observed nodes alike, which is what makes it the right
default; a free node inside a clade fixed for a derived allele now pays an entry, so the
`v12` store grows. Measure it.

**5. Placement always builds a pendant leaf.** The scan is unchanged — it reads `W` only,
and `cost(child of v) = #{s : x_s ∉ W(v)}`, `cost(new node on p→v) = #{s : x_s ∉ W(p)|W(v)}`
never depended on genotypes. The surgery loses almost everything:

- no merge path — a cost-0 placement is a new leaf under `v`, not a label on `v`, so
  **correction 3 is gone**;
- no genotype to commit at a new internal node, so `chooseState` and **correction 4** are
  gone;
- no redundant-node collapse at placement time, so `splitBranch`'s collapse pricing and
  **correction 5** are gone, along with `collapseSlips` / `slipExtra`.

A split inserts free `u` between `p` and `v` and hangs leaf `L` off `u`; `u` has exactly
two children and no edge mutations anywhere.

**6. Step 4 does the collapsing, once.** Resolve → derive every edge as the diff between
resolved parent and child → collapse the edges that come out empty, merging labels into
the surviving node → merge siblings that resolved to the same sequence → explode
multi-mutation edges into shuffled chains → write. That is what `src/examples.cpp` already
does; only the V1/V2 default inside it changes. The directive says explode then collapse;
the two commute (exploding an empty edge is a no-op) and the code collapses first, because
it is cheaper to not explode what is about to disappear.

The check that keeps step 4 honest survives untouched and is still the main validator: a
jointly parsimonious assignment changes state exactly `parsimonyScore` times, so the
resolved tree's mutation count must equal it on the nose, before the merge.

### Consequences, measured

Of the five things this section predicted before implementing, two were right about the
direction but badly wrong about the magnitude, one was right, and the two named risks did
not materialise at all. The predictions are kept alongside what happened, because the
arguments behind them still look right -- they were just arguments about sign, not size.

- **The base tree's parsimony score falls by 2, 1,153 → 1,151.** Predicted "visibly
  lower, not marginally"; wrong. Un-pinning the interior can only shorten the tree, and
  it does, but Nextstrain's own reconstruction is already parsimonious almost
  everywhere: exactly **2 (node, site) pairs** have a Nextstrain allele that no MPR
  admits, and each is worth 1. `sets` reports that count, which is the direct measure of
  how much the old pinning was costing.
- **Ambiguity rises 7×, and is still nothing: 4 → 29 pairs over 28 nodes**, out of 9.07M.
  Predicted "a lot"; true as a ratio, irrelevant as a quantity. One set reaches size 3,
  which the collapsed tree never did. New ambiguity is at sites 28 `{H,Y}`, 54 `{F,L}`,
  70 `{I,V}` and 704 `{L,S}`, all on interior nodes that used to be pinned.
- **The tree roughly doubles, as predicted.** 14,679 nodes after 5,000 placements against
  7,123 + 2 × 5,000 = 17,123 worst case, so a few placements land where no split is
  needed. Now measured outright rather than extrapolated: the full run is **351,344
  placements in 14,133 s = 3.93 h**, tree 535,892 nodes, 9.54 × 10^10 node visits at
  271,465 per placement. Both earlier projections were low — 80 min off the first 5,000,
  then 2.8 h off the first 100,000 — because the quadratic only shows itself late (827 s
  to 100k, 3,263 s to 200k, 9,165 s to 300k). The 12 h SLURM limit is the right size.
- **Upward propagation stays local without any pinned ancestor.** This was the main risk.
  Measured over 5,000 real placements: 35,669 ancestors moved, **7.1 per placement**, and
  the **longest single chain the upward pass ever walked was 5**. Over the full run the
  *mean* is flat — 2,446,195 ancestors moved over 351,344 placements, **7.0 per
  placement**, the same 7.1 the first 5,000 gave — but the **tail grows**: longest chain 5
  at 20k, 10 at 100k, **53** at 351k. A mid-run guess that it grew "like a logarithm" was
  not supported; what is supported is that the mean does not grow at all, which is what
  the cost depends on. The argument holds — a sibling's vote breaks the tie a new child
  creates, usually one hop up — and `upHops`/`upHopsMax` are reported by `place` so the
  tail stays watched rather than assumed.
- **Degrees do not grow.** `recomputeV12` is O(degree) at a free node and a node placement
  adds a child, so a popular attachment point could in principle run away. It does not:
  the largest multifurcation is **19 after all 351,344 placements, the same 19 the input
  tree started with**. The 173,919 node-placements spread over distinct parents. The
  mitigation on standby — break a branch/node tie in favour of the branch, which leaves
  every degree untouched — is not needed and has not been applied.
- `--verify` works unchanged: `computeAlleleSets` is still the independent from-scratch
  reference, and it now has less to reproduce.

### The old pipeline failed at full scale, and this is why

Worth recording as the directive's strongest motivation, found after it was written: the
old pipeline's one full 351,344-sequence run (job 3397843, `work/place_full.3397843.log`)
placed every sequence and then **died in step 4**:

```
error: resolved tree carries 14104 mutations against a parsimony score of 14103
```

That is step 4's `mutations == parsimonyScore` check — the one invariant that cannot be
argued around — failing by exactly 1 after 351k placements, with nothing written. It is
the same off-by-one signature as correction 5, in the same place: a committed genotype and
the sets disagreeing about whether a node is redundant. Correction 5's fix priced the
collapse it could see; this is one it could not. None of the 20-seed sweeps at 20,968
placements caught it, because it needs a tree the prefix never builds.

The directive deletes the mechanism rather than the symptom. There are no committed
genotypes to disagree with the sets, no collapse during placement to mis-price, and the
only collapse left runs after resolution, where an edge *is* the diff between two resolved
sequences. The check is unchanged and still the thing to watch, but what it was catching
cannot arise.

**Settled.** Job **3399137** ran the whole 351,344-sequence queue and finished: step 4's
`mutations == parsimonyScore` check passed, all three outputs were written, and
`check_examples.py` is clean over its 13,895 records. Where the old pipeline died, this
one does not, and it is the mechanism that is gone rather than the symptom that is
patched. (That run predates removing the sibling merge; job **3402849** is the re-run. It
finished on 2026-10-22 and wrote exactly what was projected for it: **14,108 mutations in
14,109 records**, on the same 535,892-node tree and the same sets, in 10,656 s.)

### What this obsoletes

Kept below for the record, but no longer describes the code:

- **A node carrying sample labels is pinned** — labels no longer pin anything; leaves do.
- **Dropping site 1274 does not change the topology** — there is no topology to change.
- **Corrections 3, 4 and 5** — all three are about the committed genotype, which is gone.
- **Why the update is local: pinned nodes absorb** — the mechanism is gone; see above.
- The `collapse` command, which collapses nothing. It becomes `tree`.
- `MutationTree` as a type name: the tree is now just a tree, and the mutation tree exists
  only inside step 4. Renamed to `Tree` in `src/tree.{hpp,cpp}`.
- Every **Results** figure, all of them re-measured below.

Corrections 1 and 2 survive intact: both are about Hartigan itself, not about genotypes.

Status: **implemented and validated.** See **Results** and **Validation** below.

## Directive, 2026-10-22: attachment is a match first, a branch second

> First, check if the new sequence being attached has allele sets that match exactly to the
> allele sets at an existing node/leaf. If so, simply attach at that node/leaf (if multiple
> options, choose randomly). This should be really cheap, essentially we just notate that
> an existing node picks up an additional leaf with identical allele sets. If such a node
> doesn't exist, only consider the "split a branch" option for attachment, as this will
> never be worse than attaching by hanging off an existing node. If multiple "split branch"
> options are equally parsimonious, choose randomly.

The scan is unchanged. What changed is what it draws from: it used to pool every candidate
at the minimum cost -- branches and nodes together -- into one argmin and draw uniformly
over all of it. It now decides in two stages, and the first one wins outright.

### The match condition, and what "exactly" means where the sequence is missing

Alongside the cost, the scan carries a second counter, incremental in the same deltas:

```
e(v) = #{ s : x_s observed, W(v) != {x_s} }
```

`e(v) == 0` is the match. It is exactly the condition under which `writeLeafSets` emits
nothing: a leaf hung under `v` would carry an **empty set-delta**, so `W(leaf) == W(v)` at
every one of the 1,273 sites. That is the directive's "identical allele sets", read as a
statement about the leaf rather than about the sequence -- which is what makes "an existing
node picks up an additional leaf with identical allele sets" literally the thing that
happens.

**A site where the sequence is missing is not required to match**, and that is the one
judgement call in here. The leaf inherits `v`'s set wherever it is missing -- `V1(L)` is all
20 states there, so the downward rule gives `W(L) = W(v)` -- so excluding those sites is
precisely what makes the leaf's sets identical to `v`'s. The literal alternative, requiring
`W(v)` to be all 20 states wherever the sequence is missing, would essentially never fire:
32% of the queue carries missing data and `W` is a singleton at 99.999% of (node, site)
pairs.

`e >= cost`, because a non-empty `{x_s}` outside `W(v)` also differs from it, so **a match
is always a zero-cost placement** and preferring one costs no parsimony. Once the DFS sees
a match the branch bookkeeping is dropped for the rest of the scan: the answer is already
known, and the argmin set it would otherwise go on building runs to thousands of entries.

### Why dropping the node candidates is free

`cost(new node on p->v) = #{s : x_s not in W(p) | W(v)}` is at most `#{s : x_s not in W(v)}`
and at most `#{s : x_s not in W(p)}`, so a branch is never worse than **either** of its
endpoints -- a strengthening of what the scoring section already said. Every node but the
root has a branch above it, and a child of the root is dominated by the branch down to any
child of the root, so no node candidate can beat every branch. Restricting the second stage
to branches cannot raise the cost, and measured it does not.

### Consequences, measured

Scores and record counts did not move at all; the tree got substantially smaller.

| placed | score | records | nodes, before | nodes, after | splits, after | matched exactly |
|---|---|---|---|---|---|---|
| 1,000 | 1,267 | 1,268 | 8,638 | **8,219** | 96 | 904 (90.4%) |
| 5,000 | 1,388 | 1,389 | 14,679 | **12,331** | 208 | 4,792 (95.8%) |
| 20,000 | 1,807 | 1,808 | 37,339 | **27,749** | 626 | 19,374 (96.9%) |

- **The parsimony score is identical** -- 1,267 / 1,388 / 1,807, the same three numbers the
  old draw gave, to the unit. So is the record count, which is `score + 1` by construction
  either way. The two rules pick different *positions* at the same cost, which is what a
  tie is.
- **The tree is 26% smaller at 20,000** and grows more slowly: a match adds one node, a
  split adds two, and 96.9% of placements now match. Where the old draw split 51% of the
  time (2,556 of 5,000), this one splits 4% (208 of 5,000) and 3% at 20,000. A smaller
  tree is a cheaper scan, so the run gets faster twice over -- 9,754 node visits per
  placement at 5,000 against 10,907 before, 1.5 s against 2.1 s.
- **Ambiguity moves a little, upward**: 36 / 38 / 40 pairs against 33 / 33 / 37. A match
  attaches a leaf without splitting the branch above it, so the interior keeps freedom a
  split would have taken from it. Still no set above size 3.
- **Degrees still do not grow.** Largest multifurcation is 19 at every prefix -- the input
  tree's own -- through 20,000 placements of which 19,374 added a child to an existing
  node. This was the standing worry about preferring node attachments over splits: a split
  grows no degree at all, a match grows one. The mitigation NOTES kept on standby was
  exactly "break the tie in favour of the branch", which this directive half adopts and
  half reverses -- and the 19 holds either way.
- **The ties that remain are between indistinguishable nodes.** The mean draw set rises to
  4,479 at 20,000, which looks worse than it is, because for a matching placement every
  candidate has *the same allele sets as the sequence*: the draw is between nodes the
  output cannot tell apart. A matched leaf's edge resolves to nothing, collapses at step 4,
  and -- the example format carrying no sample names -- contributes nothing to the written
  file at all. That is the honest reading of the old 88,173-position argmin too, but there
  the tie was between genuinely different topologies at equal cost; here most of it is
  between copies.

Status: **superseded the same day** by the directive below, which keeps the match
condition exactly as it is here and stops attaching the leaf at all. Everything above
about `e(v)`, about missing sites, and about why node candidates can be dropped still
describes the code; what changed is the surgery that follows a match. The table above is
the measurement of the intermediate rule, kept because the directive below compares
against it.

## Directive, 2026-10-22 (second): a match is not attached at all

> In the case where a new sequence being attached matches the sequence at an existing
> node, can we simply skip attaching that sequence without affecting the parsimony score?
> If so, this could be a drastic speedup to the algorithm and lead to much smaller trees
> without majorly affecting the resulting examples. Bear in mind we ultimately need this
> to work for nearly 10 million sequences being attached.

Yes, with one number attached. The directive above had already established that a match is
a zero-cost placement; what it did not say is the stronger fact, and it is the stronger
fact that makes the leaf discardable rather than merely cheap: **the leaf changes no
allele set anywhere**. So skipping it is exactly free *for that placement*. It is not
quite free for the run -- the leaf would have been in the tree for every later placement --
and the measured price is **+0.24% on the tree's length at 100,000 placements**, 10
mutations in 4,205, for a run that is **90x faster on a tree 8.5x smaller** at that same
prefix. Both halves of that are below.

### Why the leaf is a no-op, site by site

Hanging an observed leaf `L` under `v` adds, to the cost of every assignment `A` of the
tree, the term

```
[ A(v) != x_s ]   at each site x is observed at, and 0 where x is missing
```

-- `L` is pinned to `x_s` where `x` is observed, and free to copy `A(v)` where it is not.
The match condition is `W(v) == {x_s}` at exactly the observed sites, and `W(v)` is by
definition the set of states `v` takes in *some* most-parsimonious reconstruction. So
**every** MPR of the tree without `L` already carries `A(v) = x_s`, the added term is 0 on
all of them, and therefore

- the minimum does not move -- the score is unchanged, which is what "cost 0" already
  said; and
- the *set* of minimising assignments does not move either: `A` is optimal with `L` if and
  only if it was optimal without it. So `W` is identical at every pre-existing
  (node, site) pair, and so is the set of jointly parsimonious resolutions step 4 draws
  from.

The leaf carries nothing the tree does not already hold. It is counted against `v` and
dropped.

### What does move, and why it does not matter

`V1`/`V2` at `v`, and at a few ancestors, can move: the new child votes for `x_s`, which
can break a tie in the argmax that `V1` is. That is Hartigan's bookkeeping, not the
answer. The tree without the leaf has its own `V1`/`V2`, they are the correct ones for it,
and the `W` they produce is the same `W` -- which is the paragraph above.

`--verify` does not test that claim, and it is worth being clear about what it does test:
it recomputes every stored value from scratch on the tree **as it stands** and compares,
so it says the maintained state is the right state for the tree that was built, not that
two differently-built trees agree. It is clean over 300 real placements and all 8
synthetic sets, which is what rules out the skip corrupting the incremental bookkeeping.
The claim that the two trees agree is the argument above, and what can be measured of it
is in the tables below.

The lemma is also checked directly, against the binary rather than on paper. Run
`--attach-matches --out sets.tsv` at `--limit k` and at `--limit k+1`: where the extra
placement is a match it appends exactly one node, and then

- every pre-existing node's set-delta line is **byte-identical** between the two files,
  which is `W` unchanged at every (node, site) pair the two trees share; and
- the appended leaf's own delta count is **0**, which is `W(L) == W(v)`.

Clean over placements 2..30 (28 matches, 1 split) and 5,001..5,040 (40 matches), the
second window chosen because by then the tree carries ambiguity for the leaf to disturb.

### What it does *not* settle: the rest of the run

The leaf is a no-op on the tree at the moment it would be attached. It is **not** a no-op
on the run. Under the old rule that leaf stayed in the tree for every later placement, as
a candidate position of its own and -- if some later placement moved `W(v)` off `{x_s}` --
as an observation pulling `v` back towards `x_s`. Skipping removes both. Neither run is
"the" answer: both are greedy parsimony runs over the same queue, and they can end on
different totals. How different is a measurement, not an argument, and it is the one thing
here worth measuring carefully, because the per-placement argument above cannot reach it.

### Consequences, measured

Seed 1, tie-seed 1 throughout. "attach" is the rule this directive replaces, run from the
same binary with `--attach-matches`.

| placed | score, attach | score, skip | records, attach | records, skip | nodes, attach | nodes, skip | splits, skip | s, attach | s, skip |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 1,267 | **1,267** | 1,268 | **1,268** | 8,219 | **7,315** | 96 | 0.2 | **0.0** |
| 5,000 | 1,388 | **1,388** | 1,389 | **1,389** | 12,331 | **7,541** | 209 | 1.5 | **0.1** |
| 20,000 | 1,807 | **1,808** | 1,808 | **1,809** | 27,749 | **8,381** | 629 | 14.6 | **0.3** |
| 50,000 | 2,617 | **2,619** | - | - | 58,519 | **9,927** | 1,402 | 109.4 | **1.3** |
| 100,000 | 4,205 | **4,215** | - | - | 110,008 | **12,921** | 2,899 | 538.8 | **6.0** |
| 351,344 | 14,108 * | **14,112** | 14,109 * | **14,113** | 535,892 * | **31,613** | 12,245 | 10,656 * | **73.5** |

\* the full-queue row compares against job **3402849** (`work/*_2021.*`), which finished on
2026-10-22: the sibling merge is already gone there, so its 14,108 mutations and 14,109
records *are* like-for-like. What is not like-for-like is the placement rule -- 3402849
still draws uniformly over nodes and branches together, predating **both** 2026-10-22
directives -- and the wall clock, which is a SLURM node's rather than this one's. The
hardware-independent comparison is the scan: **271,465 node visits per placement there,
4,644 here**, 9.54 x 10^10 against 1.63 x 10^9. The control still missing is the
*middle* rule -- attach-on-match -- over the whole queue, which is what the 50,000 and
100,000 rows measure and this row does not.

- **The score is not quite unchanged over a whole run, and the gap grows with the queue.**
  Per placement it is exactly unchanged -- that is the lemma, and it is checked. Over a run
  it is not, for the reason the section above gives: a skipped leaf is not there to be
  matched later, once some placement has moved `W(v)` off `{x_s}`. The tie seed is swept so
  that the gap can be told apart from the draw:

  | placed | attach, added (tie-seed 1..n) | skip, added (tie-seed 1..n) | gap | of the whole tree |
  |---|---|---|---|---|
  | 1,000 | 116 | 116 | 0 | 0 |
  | 5,000 | 237 | 237 | 0 | 0 |
  | 20,000 | 656 x7, **657** x1 (1..8) | **657** x1, 656 x7 (1..8) | 0 | 0 |
  | 50,000 | 1,466, 1,466, 1,466 | 1,468, 1,468, 1,468 | **+2** | +0.08% of 2,617 |
  | 100,000 | 3,054, 3,052 | 3,064, 3,064, 3,065 | **+10 to +13** | ~+0.25% of 4,205 |
  | 351,344 | 12,957 * | 12,961, 12,964 | +4 * | +0.03% * |


  At 20,000 the two rules are indistinguishable: one tie seed in eight lands a unit high
  under *either* of them, which is what a tie looks like. From 50,000 the gap is
  systematic -- no tie seed closes it -- and it is also tiny: **a quarter of a percent of
  the tree's length at 100,000**, a dozen mutations in 4,205 and the same dozen extra
  records. The full-queue row
  is not like-for-like (see the footnote), so **the one measurement missing from this
  section is an attach-rule run of the whole queue under the current tie-break**, which is
  ~4 h of one core where the skip rule is 73 s. Until it exists, read the 100,000 row as
  the honest estimate of the price and expect it to be a little worse at 351,344.
- **The skip rule is also far less seed-dependent**, which is the same fact seen from the
  other side: 96.5% of placements are no longer drawn for at all, so three tie seeds give
  1,468 / 1,468 / 1,468 at 50,000 where the old rule's draw was over thousands of
  candidates.
- **On the synthetic stress sets the skip rule is never worse**, and is better on half of
  them: added score 2,151/2,114/2,244/2,189/1,952/2,172/2,111/2,052 against the attach
  rule's 2,153/2,115/2,246/2,189/1,952/2,173/2,111/2,052. These are the runs where
  matching is rare (4.5-10%), so they say the rest of the engine did not move -- and where
  matching is rare there is nothing for the gap above to arise from.
- **The tree stops growing with the queue.** It grows only on splits -- 7,123 + 2 x 12,245
  = 31,613 after 351,344 placements, against 535,892. The full queue adds 24,490 nodes
  where it used to add 528,769, and 96.51% of the queue adds nothing at all.
- **The examples file is not further from the old one than the old one is from itself.**
  At 5,000, the skip rule's 1,389 records share 1,273 of their sequences (as a multiset)
  with the attach rule's file at the same seed; the attach rule at `--tie-seed 2` shares
  1,260 with the attach rule at `--tie-seed 1`. The difference between the two rules is
  smaller than the difference a tie seed makes.
- **Ambiguity falls by an order of magnitude**: 378 of 40,243,349 (node, site) pairs over
  367 nodes at the full queue, against 7,357 over 7,343. Most of the old count was
  duplicate: a matched leaf inherited its parent's ambiguous sets and was counted again.
  Still no set above size 3 (`1:40242971  2:375  3:3`).
- **The upward pass all but disappears**: 104,446 ancestors moved over 351,344 placements,
  0.30 per placement against 7.0, because 96.51% of placements touch nothing. The longest
  chain is 27, against 53.

### Two things the scan can now do that it could not before

Both follow from the same observation: when the sequence is not being attached, *which*
matching node is found no longer matters, so the uniform draw over the matching nodes --
which is what forced the scan to visit every node -- is gone.

- **The scan returns at the first match**, rather than finishing the DFS to count the tie
  set it would have drawn from.
- **A 4-entry cache of the nodes the last few matches landed on is tested first**, each
  test being `materialisePath` plus one pass over the sites -- two passes over 1,273 sites
  against one over the whole tree. A chronological queue is full of consecutive duplicates,
  so it hits often: **180,383 of the 339,099 matches, 53%, were settled without a scan at
  all**, over 1,026,210 probes (17.6% of probes hit). It is a heuristic for *finding* a
  match, never for deciding one: a probe recomputes `e(v) == 0` against the current `W`,
  so a stale entry simply fails and costs one pass.

Together they take the scan from 271,465 node visits per placement to 4,644.

### What this means for 10 million

The cost per placement is now set by the number of *distinct genotypes* the tree holds,
not by the number of sequences placed into it, which is the scaling property the old rule
did not have. Three terms, measured over the full 2020 queue:

- a match found by the cache: two passes over 1,273 sites, no scan, no surgery.
  **51.3%** of placements (180,383);
- a match found by the scan: a partial DFS, ended at the first match. **45.2%** (158,716);
- no match: a full DFS over the tree, then the split. **3.5%** (12,245).

Extrapolating the 2020 numbers to the ~9.3M dated spike records: the split rate is roughly
flat at 2.9-3.5% over the whole queue, so ~300k splits and a tree of ~600k nodes; the scan
is then ~300k full DFSs over an average of perhaps 300k nodes, ~10^11 visits, which at the
measured 22M visits/s is **around an hour**. The attach rule at that scale would build a
tree of order 14M nodes -- it managed 1.5 nodes per placement -- and scan of order 10^13
visits, which is days on one core and no longer fits in memory either. This is the
difference between a run and no run, not a constant factor.

**The honest caveat is the same one the ambiguity section already learned**: this queue is
2020 only, and a prefix of it is not a small version of what follows. Delta and Omicron
are far more diverse than anything here, so the match rate will fall below 96.5% and the
split rate will rise, and the hour above is a floor rather than an estimate. What survives
the caveat is the shape of the cost, which is now the number of distinct genotypes.

### The duplicates are kept as counts, not as nodes

Dropping the sequence would otherwise drop its provenance, so two things record it, and
nothing in the algorithm reads either:

- `Tree::matched`, a per-node counter, written as `tree.tsv`'s `n_matched` column.
  6,921 of 31,613 nodes carry a count at the full queue, the largest being 63,613 on one
  node. It is deliberately **not** a label: a label makes a node observed, which would pin
  it and move its `V1`/`V2`, and the whole point is that nothing moves.
- `--out-matches <path>`, a `strain<TAB>node` row per skipped sequence, written as the run
  goes so nothing accumulates in memory. 339,099 rows, 9.4 MB, over the full queue.

The full-queue outputs under this rule are on disk, written in 73.5 s on one core:

```
work/examples_2021_skip.txt     4.2 MB   2,877 records, 14,112 mutations
work/tree_2021_skip.tsv         2.3 MB   31,613 nodes, with the n_matched column
work/sets_2021_skip.tsv         0.5 MB   the allele-set deltas
work/matches_2021_skip.tsv      9.4 MB   339,099 skipped sequences -> the node each matched
work/place_full_skip.log                 the run's own report
```

against `work/*_2021.*` (job 3402849, the attach-rule control: 20.2 MB / 48.7 MB / 9.5 MB
-- written before the childless records were dropped, so its 14,109 records are the old
count)
and `work/*_2021_merged.*` (the same run with the sibling merge still in). The sets file
is **17x smaller** and the tree file **21x smaller**, which is the node count showing up
on disk.

The node a sequence is counted against is *a* node whose allele sets are the sequence's,
not a draw over all of them -- the scan stops at the first, and the cache biases towards
the previous one. Where several nodes have identical sets the counts split between them
arbitrarily, so a per-genotype frequency wants aggregating by resolved sequence, not by
node id.

### What this obsoletes

- **"Pendant leaf: always created, even at cost 0"** -- it is created only where the
  sequence is not a duplicate of what the tree already holds.
- `placements - splits == exactHits`, the middle rule's invariant, becomes
  `placements == splits + skipped`.
- The directive above's measured table, and its "the tree is 26% smaller at 20,000": it is
  70% smaller now, and the `ties` block it reported is reported differently -- a match is
  no longer a draw, so the draw statistics cover only the placements that split.
- "**The tie sets are large and that is real**" in **Open for step 3**, insofar as it was
  about matches: the 4,479-entry mean draw at 20,000 was 96.9% matches, and those are not
  drawn for any more. What is left is the branch draw, mean 243 at 20,000 and 500 at the
  full queue, which is the real choice and is unchanged.

Status: **implemented and validated.** `place --verify` clean over the first 300 of the
prepared queue and over all 8 synthetic stress sets; `check_examples.py` clean at 1,000,
5,000, 20,000 and the full 351,344, the last against its own reported 14,112 mutations;
the W-unchanged lemma checked directly over two windows of the queue, above. And
`--attach-matches` reproduces the previous rule **byte for byte** -- its examples file at
5,000 placements is identical to the one the pre-directive binary writes -- so the rule
can be switched off without switching anything else off with it.

One measurement is open and is named above: the attach rule over the **whole** queue under
the current tie-break, which is the like-for-like control for the 14,112. Nothing else in
this section depends on it; the 50,000 and 100,000 rows are controlled, and they are what
the price should be read off.

## Directive, 2026-10-22 (third): a childless node is not an example

> Revision to the output examples: if a node on the ultimate mutation tree has no
> children, you don't need to write it as an example. Equivalently just delete examples
> where the mutations to children list is empty.

A record is a sequence together with the mutations that happened next, so a record whose
`MUTATIONS_TO_CHILDREN` is empty is an input with no label -- nothing to predict from, and
nothing to learn from. Those are not written any more.

### Nothing is lost, and the label is not the thing being deleted

The two halves of that are worth separating, because only one of them is obvious.

- **The record goes, the mutation stays.** A dropped node is somebody's child, and the
  parent's `MUTATIONS_TO_CHILDREN` still lists the mutation that leads to it. That listing
  *is* the training signal -- "from this sequence, this happened" -- and deleting the
  child's own record does not touch it. Every edge of the mutation tree is still in the
  file exactly once, which is why the mutation total is still the parsimony score.
- **The sequence stays recoverable.** A dropped node's sequence is its parent's with that
  listed mutation applied. The file loses no information at all, only records that carry
  none.

A dropped node is a leaf, so it is never anyone's parent, and the ids simply renumber:
they stay `0..n-1` in order with every parent resolving and preceding. The only degenerate
case is a tree with no mutations at all, where the root itself is childless and the file
comes out empty; that cannot happen on any real input here.

### This is the legacy convention, not a departure from it

Checked after implementing, and it settles the question: **all nine files under
`antiGen/experiments/multitree/data/forecast2021/` carry zero childless records** --
`examples_ncov@2021-01-01_train.txt` (347 records), `_test` (986), `_novel` (775), and the
h3n2, h1n1pdm, yam, evd68, ebola and zika files alike. The format this repo targets has
always been written this way; it is this repo that was adding records the legacy arm never
had.

### Consequences, measured

Seed 1. `mutations` is unchanged throughout -- it is the parsimony score, and the collapse
and this deletion both leave it alone.

| run | mutations | records before | records now | childless dropped |
|---|---|---|---|---|
| base tree | 1,151 | 1,152 | **320** | 832 |
| +1,000 | 1,267 | 1,268 | **357** | 911 |
| +5,000 | 1,388 | 1,389 | **375** | 1,014 |
| +20,000 | 1,808 | 1,809 | **424** | 1,385 |
| +351,344 | 14,112 | 14,113 | **2,877** | 11,236 |

**Four fifths of the file goes**: 20.2 MB to 4.2 MB over the full queue, 14,113 records to
2,877. The mutation tree is bushy -- 2,877 internal nodes hold 14,112 edges, so the
average written record lists 4.9 child mutations -- and under the old rule four out of
five records existed only to restate one of those mutations as a sequence.

### What this obsoletes

- `records == mutations + 1 == parsimonyScore + 1`, and with it the **seed-invariance of
  the record count**. The relation is now

  ```
  records + childless == mutations + 1 == parsimonyScore + 1
  ```

  and the record count moves with the seed through `childless`: 320 / 322 / 321 on the
  base tree for seeds 1 / 2 / 3, on an unchanged 1,151 mutations. What is seed-invariant
  is what always was -- the mutation total, which is the parsimony score. The invariant
  that matters, `mutations written == parsimonyScore`, is untouched: it is asserted inside
  step 4 before anything is written, and deleting childless records removes no edge.
- The `records` columns in the two tables above (**Directive, 2026-10-22 (second)**) are
  the counts under the old rule. The scores in them are unaffected.
- `check_examples.py`'s child-list test, which was an equality: `MUTATIONS_TO_CHILDREN` is
  now a **superset** of the mutations on the records beneath it. The checker counts the
  surplus, and cross-checks it against `edges + 1 - records`, which the tree shape forces
  -- two independent derivations of the number of childless nodes, which have to agree.
  That is a stronger test than the equality it replaces, not a weaker one: a record
  wrongly dropped (one that had children) would orphan its children and fail the
  parent-sequence test, and a record wrongly kept would break both counts at once.

Status: **implemented and validated.** `check_examples.py` clean on the base tree at seeds
1, 2 and 3 and at 1,000, 5,000, 20,000 and the full 351,344 placements, with the two
childless counts agreeing on every one. Outputs regenerated: `work/*_2021_skip.*`.

## Directive, 2026-10-22 (fourth): a 2026 tree, the whole queue, and two flags

> For this run: we want no date cutoff and no exclusion criteria based on the clock rate.
> However, we do want some extra features. First, identify the node in the starting tree
> that is the root of the BA.2 variant (should be unique -- stop and flag if not), and use
> this to determine the is_variant_root and is_descended_from_variant fields in the final
> output. In addition, we will want a flag in the output for whether an example is from
> before or after 2021-01-01. To determine this you can treat the node dates in the
> nextstrain tree as trustworthy, and propagate into newly attached regions whether they
> are before/after this date threshold.

Four changes, of which the last two are the substantive ones.

### The starting tree moves to 2026

`data/ncov_open_global_all-time@2026-03-15.json`, the open-data all-time build at the last
snapshot before the data's own last collection date. Provenance and guards are in the
README. What it costs the pipeline is nothing and what it changes is scale: 5,688 nodes
and 2,939 tips against 7,123 and 3,848, but **505 polymorphic sites against 386**, a
parsimony score of **2,133** against 1,151, ambiguity at **1,633 (node, site) pairs**
against 29 with sets reaching **size 4** where the old tree never passed 3, and a longest
resolved edge of **27** mutations against 5 -- that one is Omicron's. `sets` runs in
0.28 s and `place --verify` is clean over 200 real placements on it.

### No date cutoff, no clock filter, coverage kept

**Half superseded by Directive, 2026-10-22 (seventh)**: the clock is back on for the full
run, as a per-month envelope rather than the line this section rejects. The reasoning
below is why the *line* had to go, and it stands.

`--date-cutoff none` and `--no-clock`. The clock filter had to go rather than be
widened: it is `d_root <= 3.3417 (t - 2019.9849) + 8`, regressed on the 2020 tree's own
tips, and at 2022 that line allows about 7 S substitutions where Omicron carries 30-odd.
It would reject most of the queue, and the rejection would be the calibration's fault
rather than the data's.

The **coverage filter stays** -- fewer than 1,146 of 1,273 non-missing is still a skip. It
is not a clock criterion, and a mostly-`X` sequence matches almost anything while still
costing a placement.

### The variant root is found, and a second one is a stop

`--variant BA.2`. The matcher is antiGen's own, copied rather than approximated: a clade
label matches if it **is** the name or contains `(name)`, which is `_match_variant` in
`antiGen/antiGen/load.py`. So `BA.2` finds the clade labelled `21L (BA.2)`, and the label
is what gets written, so both spellings work downstream.

```
variant      : 'BA.2' matches clade '21L (BA.2)' on 337 nodes, which begin at
               exactly one of them: node 3430 (NODE_0001779)
```

The root is the one node carrying the clade whose parent does not. `findCladeRoot` throws
if there are none or more than one -- the directive's "stop and flag if not", and the right
behaviour either way, since a clade with two entry points is not a variant root and no
rule for picking between them would be anything but a guess. Here there is exactly one,
and the independent check agrees: node 3430 is also the MRCA of all 337 labelled nodes.
Its parent is clade `21M (Omicron)`, the B.1.1.529 stem, which is where BA.2 should hang.

**Descent is a tree relation, not a label.** Everything under node 3430 is descended,
which is 2,258 of 5,688 nodes and includes the clades labelled `22B (BA.5)`,
`23A (XBB.1.5)`, `24A (JN.1)` and `25C (XFG)` -- all of them BA.2 descendants that carry
their own names. Reading descent off the labels would have missed every one. The root
itself counts as descended as well as being the root.

A node whose own edge resolves to nothing is collapsed at step 4, which would leave no
record to *be* the variant root. **This was wrong**: it cannot happen on the tree alone,
but a placement splicing into the edge into node 3430 divides those 10 mutations and the
collapse happens after all. See **Directive, 2026-10-22 (ninth)**, where the variant root
becomes the resolved node the tree's variant root landed in and descent moves to the
resolved tree.

### `POST_CUTOFF` replaces `TRAINING`/`TESTING`

The legacy format's fifth line is a bare `TRAINING` or `TESTING` token, which
`antiGen/load.py` reads as `is_training`, and the legacy files split on exactly this
directive's date -- `examples_ncov@2021-01-01_{train,test}.txt`. The token is **gone**,
replaced by a field of its own:

```
POST_CUTOFF: True
```

**This is a format change `antiGen/load.py` has to be told about**: it currently does
`is_training = lines[i].strip() == "TRAINING"` at that position, which would now read
`POST_CUTOFF: True` and call every record testing. The two variant fields are unchanged in
shape -- zero or more `<label>,True` lines under each header, which is what that parser
already accepts -- so the variant work needs no change there, only this one line.

`--post-cutoff` sets the date, default 2021-01-01, and the threshold is **midnight at the
start of that day**: a sample collected on 2021-01-01 is at or after it, and so is an
inferred node placed anywhere within that day. (Taking the date's own decimal year, which
is midday, put one base-tree node on the wrong side -- 860 records against 861.)

### The date of a node that has none of its own

**Superseded by Directive, 2026-10-22 (eighth)**: an interior `num_date` is no longer
read at all, and the flag is the earliest observed date below the node, full stop. The
part of this section that survives is the second bullet -- why a skipped match's date
still has to count -- which the new rule keeps for the same reason.

The directive says to trust the tree's dates and propagate into the new regions. Every
node of the input tree has a `num_date` and every placed sequence has a collection date;
what has no date is a node **placement created** -- a free node spliced into a branch, and
the intermediates step 4 makes when it explodes a multi-mutation edge. The rule, chosen
over inheriting the parent's flag:

```
eff(v) = v's own date, if it has one
       = the earliest date anywhere below it, if it does not
```

A node existed no later than its earliest dated descendant, so this says `True` only when
everything below it is after the cutoff -- it never calls a node late when it provably
existed early. It is also monotone, which inheriting from the parent is too but which the
alternatives are not: no record before the cutoff hangs under one at or after it, and
`check_examples.py` now asserts that.

Two consequences worth stating, because both are places the rule could have been lost:

- **A skipped sequence leaves its date behind.** A match is not attached (second
  2026-10-22 directive), so there is no leaf to carry its date -- and the date is evidence
  that *this genotype existed then*, which is exactly what the rule reads. `Tree::matchedDate`
  keeps the earliest such date per node and `minDated` folds it in. Without it, 96% of the
  queue would contribute no date at all, and a node whose only early evidence was a skipped
  match would be called late.
- **An intermediate on an exploded edge is dated by the child's subtree, not by the
  child.** It stands above everything the child stands above, so the rule applied literally
  gives it the subtree's earliest date. Its *variant* flag goes the other way -- it takes
  the parent's, because an inferred ancestor on the edge *into* the variant root is not yet
  inside the variant. Each is its own definition applied literally, which is why they point
  in opposite directions.

### What the checker now verifies

`check_examples.py` reads the new fields and adds four invariants to the ones it had:

- every record carries `POST_CUTOFF: True` or `False`, and none before the cutoff hangs
  under one at or after it;
- the variant fields carry only `<label>,True` lines, under a single label;
- at most one record is `IS_VARIANT_ROOT`, and it is itself descended;
- the descended records form **one subtree hanging off that root** -- no descended record
  without a descended parent except the root, and no plain record under a descended one.

### The strain table did not grow, and a no-cutoff run is what found it

`prepare`'s metadata pass holds strain -> date in an open-addressing table sized once by
`table.reserve(1 << 20)` -- 2.1M slots, chosen when the date cutoff kept 377k strains.
With no cutoff it takes **9,335,628**, and the table had no growth path at all: past 2.1M
entries every slot is full, and `while (slot_[i].len) i = (i + 1) & mask_` then **never
terminates**. The first full run sat at 100% CPU, 80 MB of 338 MB read after three
minutes, and would have spun forever.

The fix is a `grow()` that doubles and reinserts; a slot carries its own hash and an
offset into the names blob, so nothing is re-hashed and nothing but the slots is copied.
`reserve` is now a starting size rather than a bound.

Worth recording for what it says about the failure mode rather than the fix: this was not
a slowdown that a bigger constant would have papered over. It was a hang, it needed no
unusual input -- just more rows than the constant allowed -- and the regression check is
that the old configuration still prints the **same six numbers byte for byte**
(9,424,942 rows, 377,063 in window, 25,551 coverage, 267 clock, 99 both, 351,344 kept).

### The queue, as measured 2026-10-22

```
metadata     : 9424942 rows, 9335628 dated, no date cutoff
fasta        : 9335628 records, 9335628 of them in the date window
filters      : 832137 fail coverage (<1146 of 1273 sites), 0 fail the clock (off)
attached     : 8503491 (91.09%)
missing sites: mean 17.42, 465974 of the kept are fully covered (5.5%)
read         : 35.1 s
```

**8,503,491 sequences**, 24x the 2020 queue, written in 52 s to a 12 GB prepared file.
Coverage is the only filter left and it drops 832,137, 8.9%. The missing-data profile is
much worse than the 2020 window's -- mean 17.4 sites against 11.0, and **5.5% fully
covered against 68%** -- which is sequencing dropout in the later years, and which the
match rule will feel: a sequence missing at a site matches a node there for free.

### The full run, and a second death by nohup

The first attempt at the 8.5M run was started with `nohup ... &` from inside an
interactive allocation and **died with that allocation at 490,000 placements**, having
written nothing but a partial matches file. That is the same failure, for the same
reason, as the very first full run of the 2020 queue -- which is why **Running it: SLURM,
not nohup** is already a section of this file. The lesson did not need re-learning; it
needed following. `slurm/place_all.sbatch` now exists for this queue, alongside the 2021
one, and the run goes through it.

Which partition is not a detail either: `cpu_batch`'s QoS caps the **group** at 288 CPUs,
and it was at 288 of 288 with 43 jobs queued, so a one-core job sat behind all of them.
The `cpu` partition is the same hardware with ~600 cores idle, and the job started
instantly there; its limit is 5 days rather than 14, so `--time` comes down to 4 days,
which is far more than this needs. Job **3405038**, logging to
`work/place_all.3405038.log`.

### How redundant the queue is, and what that buys

Measured over the prepared file rather than guessed, because the cost of the run turns on
it: **1,362,680 distinct spike sequences among 8,503,491 rows, a 6.2x redundancy.**

| year | rows | new distinct | share of its rows |
|---|---|---|---|
| 2020 | 351,495 | 56,441 | 16.1% |
| 2021 | 3,980,127 | 638,873 | 16.1% |
| 2022 | 3,482,360 | 479,624 | 13.8% |
| 2023 | 463,926 | 121,761 | 26.2% |
| 2024 | 160,271 | 45,256 | 28.2% |
| 2025 | 61,050 | 19,126 | 31.3% |
| 2026 | 4,245 | 1,598 | 37.6% |

The new-distinct rate is flat at ~16% through the two big years and only rises in the
tail, where there is little data left. It is also **4.7x higher than the split rate the
run is actually paying**: 71,860 splits over 2.09M placements is 3.44%, against ~335,000
cumulative distinct sequences over the same prefix. That gap is missing data doing the
work the match rule was built for -- a sequence missing at a site matches a node there for
free, and the late queue is missing a mean of 17.4 sites with only 5.5% fully covered.

### The cost model, and where it lands

The marginal cost per placement tracks the tree size, so the useful quantity is

```
u = marginal seconds per placement, per node of the tree
```

which settles rather than running away: 1.4e-8 at 270k placements, 2.3e-8 at 530k,
3.2e-8 at 1.05M, and **2.6-2.7e-8** from 1.5M on. The early climb is the scan's per-node
work growing as the tree takes on deltas; it then flattens.

With `u` flat, the run is an integral of the tree size over the queue. Projecting the
split rate through the measured diversity gives **~292,000 splits and a final tree near
589,000 nodes** -- against 535,892 for the 2020 queue under the old attach rule, on 24x
the sequences. The remaining work from 2.09M placements is then ~2.4 x 10^12
placement-node units, which at that `u` is **about 18 hours**, call it **19-20 hours end
to end**, comfortably inside the 4-day limit. That is a projection off a flat constant and
a measured diversity curve, not an extrapolation of a prefix's wall clock, which is what
this file has been wrong about twice before.

### The queue could be deduplicated, and what that would and would not be

Since only 1,362,680 of the 8,503,491 sequences are distinct, the obvious lever is to
place one representative per distinct sequence -- a **6.2x** cut, hours instead of a day.
It is worth being precise about when that is free, because it is free in one case and not
in the other:

- **The representative was attached.** Its leaf `L` is pinned to its own alleles, so
  `W(L) = {x_s}` at every site the sequence is observed at, for ever, whatever later
  placements do elsewhere. So `e(L) = 0` for every later copy, every later copy matches,
  and every later copy is skipped already. Dropping them changes nothing at all.
- **The representative was itself skipped**, having matched some node `v`. Then no leaf
  exists, and a later placement can move `W(v)` off `{x_s}`. A later copy would then
  *not* match, and would split a branch -- which deduplication would never do. This is
  the same second-order effect the skip rule itself has, measured there at +0.24% of tree
  length at 100,000 placements, and it would need measuring here rather than assuming.

So deduplication is exact for the first case and an approximation of a known kind for the
second, which is the majority. It is **not built**: the run is already going, the saving
is wall clock rather than correctness, and the way to settle it is the same as before --
run both over a prefix and compare the score and the record count. Recorded as the next
lever, with its caveat attached.

Status: **implemented**, validated on the base tree (1,055 records, 2,133 mutations, 1
variant root, 491 descended) and over 200 real placements with `--verify` clean. The full
8.5M run is job 3405038, in flight.

## Directive, 2026-10-22 (fifth): a position count matrix over the queue

> Write a position count matrix for all of the sequences that got attached. That is,
> maintain a matrix of dimension (seq_length x 20) initialized to zero, and each time a
> sequence gets attached (i.e. is in the correct date range and isn't excluded due to
> apparently being mis-dated), for each position on that sequence that is one of the 20
> standard AAs, increment the corresponding (position, AA) pair in the matrix by +1.
> Write this as a companion output in addition to the ones we have now.

`--out-counts <path>`, a 1,273 x 20 TSV. `ResidueCounts` in `src/sequences.{hpp,cpp}`
holds it and writes it; the increment is one table lookup per residue, `aaTable()`, the
same one the alphabet uses everywhere else.

### "Attached" here means the queue, not the leaves

The word is overloaded, and the directive's parenthetical settles which one it means:
*in the date range and not excluded* is the **`prepare` sense** -- the line `prepare`
already prints as `attached : 351344 (93.18%)`, which is every sequence that got through
the date, coverage and clock filters. It is **not** the placement sense, where since the
second 2026-10-22 directive a sequence that matches a node's allele sets is counted there
and **not attached as a leaf** -- 96.5% of the 2020 queue. Counting leaves would have
thrown away 339,099 of 351,344 sequences and made the matrix a description of the
tree's shape rather than of the data.

So every record of the queue is counted exactly once, whatever the placement then did
with it. Both commands write it and they agree byte for byte:

- `prepare --out-counts` counts the kept set it is already holding, which is the
  definition read literally and needs no placement at all;
- `place --out-counts` counts each record as it is consumed, so `--limit n` cuts the
  matrix at the same place it cuts the run.

### Missing data counts for nothing

`-`, `*` and `X` name no state -- the same rule as everywhere else in this tool -- so
they increment nothing rather than getting a 21st column. A site's row therefore sums to
the number of sequences **covering** that site, not to the sequence count, and the gap
between the two is that site's dropout. On the 2020 queue a sequence contributes 1,262.0
of its 1,273 positions on average and **no site at all is covered by every sequence** --
443,407,553 residues counted over 351,344 x 1,273 = 447,260,912 positions. That is the
same 10.97 mean missing sites `prepare` reports, arrived at down a different path.

The file is `#sequences`, `#sites`, then a `#site` header naming the 20 columns in
`kAlphabet` order and one row per 1-based site:

```
#sequences	351344
#sites	1273
#site	A	C	D	E	F	G	H	I	K	L	M	N	P	Q	R	S	T	V	W	Y
1	0	0	0	0	0	0	0	0	0	1	345002	0	...
614	3	0	15241	0	0	335733	0	0	0	0	0	15	...
```

Site 614 is the check worth keeping: D 15,241 against G 335,733 over 2020 is D614G going
to near-fixation, which is what that year looks like.

### Validated against an independent recount

A numpy pass over `work/prepared_2021.tsv` -- read the sequence field, compare against
each of the 20 bytes, sum the columns -- reproduces the written matrix **exactly**, all
1,273 x 20 cells, and the `prepare` and `place` matrices are byte-identical to each
other. `--out-counts` on `tree` or `sets` is an error rather than an all-zero file:
neither reads a sequence.

### The alignment the matrix was built from

`--out-msa <path>`, same two commands, same record set: the counted sequences written
back out as a FASTA, `>strain` and one line of residues each, which is the ingested
file's own layout. It is a subset of the input alignment and nothing else -- **gaps,
stops and X are written as they arrived**, since the point of having it next to the
matrix is to hold the data the matrix summarised, missing data included. That is the one
place in this tool where missing data is not folded away.

Two details worth stating, because each could have gone the other way:

- **The width is 1,273, not 1,274.** The terminal stop is dropped at the parse and is
  not a site anywhere else in the pipeline, so a 1,274th column would line up with
  nothing in the matrix, the examples or the tree. The input FASTA's own records are
  1,274 wide; this is the same records minus that column.
- **The order follows the command.** `prepare` writes in the prepared file's
  deterministic date-then-strain order; `place` writes as it consumes the queue, which
  is that order with each collection day shuffled by `--seed`. Same records either way.

Not added to `slurm/place_all.sbatch`: over the 8.5M queue the MSA is ~11 GB, which is
the prepared file again in another encoding, and the prepared file is already on disk.
It is a flag, so a run that wants it can ask.

Status: **implemented and validated.** `output/sc2-2021/train/` holds the finalized
2021-cutoff run -- examples, tree, sets, matches, the count matrix and the MSA. The
matrix written beside the MSA is byte-identical to the one the placement run wrote, a
numpy recount **off the MSA itself** reproduces all 1,273 x 20 cells, and the MSA's
strain set and order are the prepared queue's exactly. Job 3405038 was submitted before
any of this existed, so it will write neither; both for that queue are a
`prepare --out-counts --out-msa` over the same inputs, which costs a FASTA pass and no
placement.

## Directive, 2026-10-22 (sixth): the examples' positions are 0-based

> The output examples should zero-index the mutations.

Both position fields in the examples file -- the `pos` of `MUTATION_FROM_PARENT`'s
`from,pos,to` and the `pos` of each `MUTATIONS_TO_CHILDREN` `pos,newAA` -- are now a
0-based index into the 1,273-aa sequence, in `0..1272`, so `seq[pos]` is the residue the
mutation is about with no arithmetic in between. They were 1-based.

### This is the legacy convention, and the old files had it wrong

Measured, not assumed. Reading `MUTATION_FROM_PARENT` off
`antiGen/experiments/multitree/data/forecast2021/examples_ncov@2021-01-01_train.txt` --
the file the 2021 arm actually consumed -- against its own sequences:

| reading | records where parent[pos] == from and child[pos] == to |
|---|---|
| 0-based | **346 of 346** |
| 1-based | 0 of 346 |

So the legacy file is 0-based with nothing ambiguous about it, and everything this repo
wrote before this change was off by one against the format it was targeting. The quoted
record under **The legacy format is the target** was always 0-based -- `K,309,E` is the
0-based spelling of what the 1-based pipeline was writing as `K,310,E` -- which is why
the two never looked like they disagreed.

### What changed, and what did not

`ResMut` is `(site, allele)` with the site **0-based** throughout the resolved tree, so
the `+ 1` at the diff and the `- 1` at the split both disappear and the two write points
emit the stored value unchanged. Nothing else in the repo uses 1-based sites through this
path; the allele sets, the deltas and the `V1`/`V2` store keep their own 1-based site
numbering, which only the resolution walk reads and which never reaches the file.

Count-neutral by construction -- the change is the integer written, not which mutations
exist. The base tree at seed 1 gives the same 320 records, 1,151 mutations, 832 childless
nodes, 945 resolved nodes and 14 drawn alleles as before, and `check_examples.py` -- whose
range test and three sequence lookups moved with it -- is clean on the result.

**The files under `output/sc2-2021/train/` and `work/` predate this** and carry 1-based
positions. They are regenerated by rerunning, not patched.

## Directive, 2026-10-22 (seventh): one clock filter, calibrated over the whole span

> Rather than just applying the filter in the 2021 cutoff case and filtering sequences
> that should be post-cutoff, I instead want to skip over all sequences that deviate too
> much from the clock rate for their dates to be reasonable. This applies to both the
> 2021 cutoff version and the full 9M records version.

### No line works, and that is a measurement rather than an opinion

The old filter was `d_root <= 3.3417 (t - 2019.9849) + 8` and it is off (`--no-clock`) on
the full run for the reason the fourth directive gives. Widening it does not help, and
neither does refitting it, because divergence against time is not a line. Median
divergence over the 8.5M queue by month:

| 2020-03 | 2020-12 | 2021-06 | 2021-11 | **2021-12** | **2022-01** | 2023-06 | 2025-06 | 2026-03 |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 8 | 9 | **11** | **30** | 39 | 59 | 62 |

Omicron adds about 20 S substitutions between one month and the next, after which the
curve flattens to ~3/year. A line fitted anywhere on that either deletes BA.1 or admits a
2022 sequence as a January 2020 one, and November 2021 shows why no constant slack fixes
it either: that month's divergences run 0-16 for 585,715 records, then **nothing at all
between 17 and 24**, then 619 records at 25-34 -- genuine BA.1, correctly dated, sitting
20 above everything else collected that month.

### The rule: the divergence has to be plausible for the date, to within a tolerance

Per calendar month, over every coverage-passing record, take a low and a high quantile of
divergence from the pinned root. Make the high curve **non-decreasing**. Then keep a
record when its divergence falls inside the band of *some* month within `tol` of the date
it carries -- which, because the high curve is non-decreasing, is just an interval:

```
lo(t) = min over u >= t - tol of loQuantile(u)
hi(t) = hiCurve(t + tol) + slack
```

`tol = 3` months, `slack = 3` substitutions on the upper edge, `hiQ = 0.9999`,
`loQ = 0.001`, and a month thinner than 2,000 records widens its window until it is not.

**Non-decreasing is the whole trick.** It is true of the real envelope -- a lineage cannot
circulate before it arises, so the most diverged thing around only grows -- and it is what
makes the calibration robust to the very records the filter is for. January 2020 has 262
records of which about 6% are mislabelled recent sequences, so its own upper quantile is
31; clipped by the months that follow it, which are clean, its curve value is **9** and
its band is `[0,12]`. The contaminated month is bounded by its own future.

**The two quantiles are deliberately asymmetric** (0.9999 against 0.001) because only one
of them has that protection. The upper edge can sit near the maximum, since the isotonic
clip removes the spikes. The lower edge has no equivalent -- a lineage that lingers
genuinely lowers it, so there is no monotonicity to impose -- and so it has to be a
quantile the contamination cannot reach.

### Calibrated on the whole dataset, in both runs

The envelope is built from **every dated record in the FASTA**, before the date cutoff is
applied, so the cutoff run and the full run apply the identical criterion. That is the
directive's "applies to both", and it is not optional: calibrated on its own window
instead, the 2021 run would have no months after 2020-12 to justify Alpha's 9-10 S
changes and would delete B.1.1.7, which is the trap the old slack sweep already found.

It costs the cutoff run a full FASTA pass -- `prepare` at the 2021 cutoff goes from 4.7 s
to **31.7 s**, the full run from 35.1 s to **45.2 s**. Worth noting and not hidden: the
2021 arm's filter now depends on post-2021 data. It only ever *removes* records, and what
it removes is mislabelled, but a hermetic 2021 arm would want the calibration windowed
and Alpha protected some other way.

### The sweeps

Ground truth, both sides, chosen so that no judgement is involved: **5** records dated
January-February 2020 with `d >= 6` (the tree's own 2020Q1 tips reach 3, and the open
data's March-June p99 is 2-3, so 6+ is not 2020 virus), and **588** records dated
2022-03 or later with `d <= 5` (Omicron had swept globally; nothing Wuhan-identical was
circulating). Against those, with what must survive:

| slack | high | low | early caught | late caught | BA.1 lost | Delta lost |
|---|---|---|---|---|---|---|
| 0 | 1,238 | 241 | 4/5 | 225/588 | 0 | 0 |
| 1 | 344 | 241 | 4/5 | 225/588 | 0 | 0 |
| 2 | 274 | 241 | 4/5 | 225/588 | 0 | 0 |
| **3** | **254** | **241** | **4/5** | **225/588** | **0** | **0** |
| 5 | 232 | 241 | 4/5 | 225/588 | 0 | 0 |
| 12 | 158 | 241 | 4/5 | 225/588 | 0 | 0 |

| tol | high | low | early caught | late caught | BA.1 lost | Delta lost |
|---|---|---|---|---|---|---|
| 0 | 332 | 672 | 4/5 | 225/588 | **1** | 0 |
| 1 | 297 | 378 | 4/5 | 225/588 | 0 | 0 |
| **3** | **254** | **241** | **4/5** | **225/588** | **0** | **0** |
| 6 | 221 | 146 | 4/5 | 142/588 | 0 | 0 |
| 12 | 39 | 52 | 4/5 | 52/588 | 0 | 0 |

Slack 0 is quantile-edge noise, not signal: 894 of those 1,238 are outside by exactly 1,
and 707 of them are January-February 2022 at `d = 35` against a band ending at 34. Slack
2-8 is a plateau, so the answer is insensitive there. Tolerance 1-3 is likewise a plateau;
0 starts costing real BA.1 and 6 upwards starts letting the late errors back in.

`BA.1 lost` and `Delta lost` are the two classes a filter like this destroys if it is
wrong -- the 619 November 2021 BA.1 above, and the 28,985 records dated 2022 at `d` 10-19,
which are Delta and chronic infections persisting at a smooth 0.3-0.9% of each month
rather than a spike. **Both stay at zero everywhere in both sweeps**, which is the thing
worth checking: at `tol = 3, slack = 3` the filter removes 0.006% of the queue and none of
it is a variant.

The 363 late errors it does not catch are the limit of a 0.1% quantile against months
whose contamination is itself near 0.1%; they are also the harmless direction. A
too-ancestral record is placed near the root with a late date, and `POST_CUTOFF` takes the
**earliest** date below a node, so a late leaf changes nothing. The costly direction is
the other one -- a 2022 sequence dated 2020 lands in the Omicron clade and drags that
whole ancestral chain to pre-cutoff -- and that is the side the envelope is tight on.

### Measured, both runs

```
2021 cutoff : 377,063 in window, 25,551 fail coverage,
              88 too diverged, 14 too ancestral -> 351,410 (93.20%)
full queue  : 9,335,628 dated, 832,137 fail coverage,
              254 too diverged, 241 too ancestral -> 8,502,996 (91.08%)
```

The worst offenders are unambiguous. `USA/MN-MDH-41076/2020` and
`USA/CA-SEARCH-139716/2020`, both dated January 2020, carry `d = 52` against a band of
`[0,12]`; `USA/VT-23-002353-WGS-02/2020` is dated 2020-04-03 and carries 37 -- its own
name says 23. On the other side `PV935594` and three siblings are dated 2025-02-01 with a
spike identical to Wuhan over 1,273 sites, and `JPN/N-S79A-RevGen1-TI/2022` is a reverse
genetics construct.

The two runs agree on 351,410 of the 351,424 pre-2021 records either keeps. The 14 that
differ are **not** a difference in the rule: the two trees' root sidecars disagree at site
943 (`F` in the 2020 build, `S` in the 2026 one), which shifts every divergence and the
whole envelope by one, and the 14 are `d = 0` records from late 2020 sitting exactly on
the boundary. Same rule, one different reference residue.

### Landed

`--no-clock` still turns the whole thing off. `--clock-tol <months>` and
`--clock-slack <n>` move the two knobs, and `--out-clock <path>` dumps the calibration
month by month -- worth writing beside any run, since a filter calibrated from the data
should be auditable from the output rather than only from the code.

### What this obsoletes

- **The clock threshold is calibrated, not assumed** -- the regression, the slack-8 sweep
  and the 267/99 counts under **The data, as measured 2026-10-21**. The rate and origin
  constants are gone from `PrepareConfig`.
- **No date cutoff, no clock filter, coverage kept** (fourth directive) -- half of it. The
  cutoff still goes, and the coverage filter still stays, but the clock is back on for the
  full run, as an envelope rather than a line.
- `PrepareStats::failClock`/`failBoth`, replaced by `failClockHigh`/`failClockLow`, since
  the two directions mean different things and coverage is now tested first.

## Directive, 2026-10-22 (eighth): POST_CUTOFF is a statement about the leaves

> For all examples, it should simply be true if all leaves descended from the example are
> after the cutoff date (i.e. 2021-01-01) and otherwise false. So we can ignore the times
> in the original nextstrain tree (which were being used to define pre/post cutoff for
> some examples).

The rule the fourth directive set was `eff(v) = v's own date if it has one, else the
earliest date below it`, and the first half of that is now gone. `POST_CUTOFF` is True
exactly when the earliest **observed** date below the node is at or after the cutoff.

What counts as an observation is the leaves of the tree as it stands -- the input tree's
tips and the placed sequences -- **plus the sequences that matched a node and were never
attached**. That last part is kept deliberately: a matched sequence is a real sequence
with a real collection date whose genotype is that node's, its date has nowhere else to
live, and 96% of the queue is matched rather than attached. Dropping it would call a node
late when the data says the genotype existed early, which is the one thing the rule is
supposed to prevent.

What is no longer read is `num_date` at an interior node. That is an inference about a
node, not an observation of one, which is what the directive is pointing at.

### It only moves the flag one way on the tree, and the tree says why

Over the 2026 build's 2,749 interior nodes:

| | |
|---|---|
| own `num_date` before the cutoff, every tip below at or after it | **137** |
| own `num_date` at or after the cutoff, a tip below before it | 0 |
| own `num_date` later than its earliest tip | 0 |

TreeTime's interior dates are never later than the tips below them, so on the input tree
alone the change can only move records from False to True, and it does: `sets` on the
2026 build goes from 861 True / 194 False to **908 / 147**, 47 records flipping, none the
other way. The 137 and the 47 differ because most of those nodes collapse at step 4 or
are childless and never written.

### The fold was walking the wrong order, and placement is what exposed it

`minDated` was folded by walking the node index range backwards, which assumes a child's
index is above its parent's. **It is not.** A placement splices a new node *above* an
existing one, so the spliced node gets a fresh high index and its child keeps a low one:
8,348 of 31,613 nodes on the full 2021 queue, 3,332 of 14,058 on the subsample below. For
every one of those the fold read the parent before the child had been computed and so
missed that whole subtree's dates.

It never showed up because the only run that ever had post-cutoff records to get wrong is
the full 2026 one, which has not completed. On a 21,257-record date-spanning subsample of
the queue, placed on the 2026 tree:

| | old | new |
|---|---|---|
| records at or after the cutoff | 2,325 | **2,294** |
| flipped False -> True | | 31 |
| flipped True -> False | | 62 |
| **records before the cutoff hanging under one at or after it** | **15** | **0** |

That last row is the giveaway, and it is an invariant `check_examples.py` already
asserts: the old output violates its own checker, the new one is clean on every
invariant. The fold now runs over a real post-order, collected by an explicit DFS from
the root.

With the flag defined by the leaves below a node, a node and an inferred ancestor of it
on its own incoming edge stand above the same leaves and get the same flag, so `postSelf`
and `postSub` have collapsed into one. The *variant* flag still distinguishes them --
see the fourth directive -- because descent is a different question.

**No delivered artifact is affected.** `output/sc2-2021/train/` is a 2021-cutoff run in
which every record is pre-cutoff, so both the old rule and the broken fold are invisible
there; the counts are 0 at or after the cutoff either way.

## Directive, 2026-10-22 (ninth): the variant root survives the collapse

The full run would have died at the last step. With `--variant BA.2` on the 2026 tree
plus placements, `writeExamples` threw:

```
error: the variant root resolved to 0 records, not one: its own edge carried no
mutation, so it collapsed into its parent
```

The fourth directive said this could not happen because node 3430's branch carries 10 S
mutations. That reasoning predates the splice. A placement spliced a node into the edge
*into* the variant root -- node 3430's parent is 11078, which is a node placement created
-- which divides those 10 mutations between two edges. When they all land above the
splice, the variant root's own edge carries nothing, the two nodes collapse, and no
resolved node has the variant root as its representative. It reproduces on the previous
binary too, so it is not new; it had simply never been reached, because the only run that
would reach it is the full one, which has not completed. Job 3405038 would most likely
have spent its hours of placement and then thrown this.

### The fix: the topmost resolved node inside the variant

The variant root is now **the resolved node the tree's variant root landed in**, which is
itself whenever its own edge survived. That is unique by construction -- every tree node
lands in exactly one resolved node -- where "the resolved node whose representative *is*
the variant root" was not, because a collapse leaves no such node at all.

The merged node is the right answer rather than a fallback. An edge collapses exactly
when its two ends resolve to the same sequence, so the merged node **holds the variant
root's genotype**; checked rather than asserted, its record carries all six of
`T19I A27S V213G T376A D405N R408S` in both the spliced and the unspliced run.

**Descent moves to the resolved tree with it.** It was a walk of the input tree from the
variant root; it is now a walk of the resolved tree from the resolved variant root. This
is forced, not incidental: the spliced node's other child -- the placed leaf -- is a
sibling of the variant root in the input tree and so was not descended, but after the
collapse it hangs off a node carrying the variant root's genotype, so it is. Leaving
descent on the input tree would have put a plain record under a descended one, which
`check_examples.py` rejects, and rightly.

Descent is still a tree relation rather than a label, which is the fourth directive's
point. It is now the relation in the tree the examples are actually written from.

### Measured

| | old | new |
|---|---|---|
| `sets` on the 2026 tree, nothing spliced | 491 descended, 1 root | **491 descended, 1 root** |
| records whose variant flags differ, that run | | **0** |
| `place`, 21,257-record subsample | **throws** | 1,105 descended, 1 root |

So the no-collapse case is untouched, and the case that used to abort now writes and is
clean on every invariant the checker has, including the two the change could have broken:
one variant root, and the descended records forming a single subtree hanging off it.

A childless resolved variant root would still leave no record, since childless nodes are
not written. That takes a variant whose root genotype has nothing below it, which is not
a variant -- but it is now asserted after the write rather than assumed.

## Directive, 2026-10-23: the scan runs on many cores

> Since we need to do tree traversals iteratively, I imagine CPU parallelization, not
> GPU, is the way to go. And of course the step to parallelize is the search for the
> parsimonious attachment point (I presume that's the main overhead). [...] I imagine it
> will be important to try to balance a workload over subtrees.

The presumption is right, and `perf` says by how much. Over a 21,257-placement run:

| | |
|---|---|
| `Placer::scan` | 67.7% |
| the stack push inside it | 16.0% |
| `computeAlleleSets` (setup, once) | 5.6% |
| `Placer::materialisePath` | 1.3% |
| everything else, the set update included | < 1% each |

**83.7% in the scan**, and the update that follows it does not register. Placements
cannot run beside each other -- each one changes the sets the next one reads -- so the
parallelism has to live inside one scan, which is a DFS over the whole tree.

### The cut: subtrees, sized by the thread count rather than as small as possible

A walk down from the root stops wherever a subtree is small enough to be one task, and
the nodes it passes on the way -- the spine -- are evaluated as it goes, since the walk
has `W` in hand. Everything below is independent: the scan only reads the sets.

The task size is `N / (4 x threads)`, not the smallest that balances well, and the reason
is the cost of a cut. A worker starting a task has to rebuild `W` down the root path, so
the overhead of the whole scheme is `depth x tasks / N`. Tasks scaled to the thread count
keep that at a few percent; tasks scaled to *balance* -- 256 of them on a 14,058-node tree
-- would have cost 80%, which is the trap in this design and the thing to re-check if the
trees ever get much deeper. Dynamic scheduling does the rest.

Workers are permanent. Each keeps its `W` buffer across placements and is handed only the
sites the root's sets moved at, so a task never pays for a dense copy, and the resync
never becomes serial work that grows with the thread count.

### The thread count cannot be allowed to change the answer

Every task and every spine node takes an **ordinal** in DFS order, and the results are
merged by ordinal rather than by whoever finished first:

- the candidate list is concatenated in ordinal order, which is exactly the order a
  serial DFS would have built it in;
- the exact match that settles a placement is the one with the **lowest ordinal**, which
  is the node the serial scan's first-match-wins rule returned;
- a worker whose ordinal is above the best match found so far stops, because it cannot
  change the answer -- so how far anyone got affects the speed and nothing else.

**One change of substance.** The reservoir sampler is gone, replaced by one uniform draw
over the merged candidate list. It is the same distribution -- the sampler's own argmin
set was already the final best-cost candidates -- and it is what the header always said
the rule was, but it consumes the RNG once per placement instead of once per tie, so a
given seed now draws differently. On the 21,257-placement subsample: parsimony added
4,931 before against **4,927** after, 17,072 exact matches against 17,069. Same
distribution, different draw, and no longer order-dependent.

The property this buys is checked rather than claimed: `slurm/scaling.sbatch` runs the
same queue at every thread count and compares the written tree, and it is **identical at
1, 2, 4, 8, 16, 32 and 64 threads**, byte for byte.

### Scaling, measured

340,119 placements on the 2026 tree, which grows from 5,688 to 75,348 nodes, on one node
of `cpu`:

| threads | wall | speedup |
|---|---|---|
| 1 | 395.5 s | 1.00x |
| 2 | 215.7 s | 1.83x |
| 4 | 118.8 s | 3.33x |
| 8 | 72.0 s | 5.49x |
| **16** | **53.4 s** | **7.41x** |
| 32 | 55.5 s | 7.13x |
| 64 | 70.0 s | 5.65x |

It peaks at 16 and then goes backwards. Two things are in the way, and the second is the
one that bites at 64: Amdahl -- 7.41x at 16 implies the run is about **92% parallel**, so
13x is the ceiling whatever happens -- and the cost of starting a parallel region, once
per scanning placement, which grows with the thread count while the work inside each task
shrinks. The serial 8% is not the scan: it is the per-placement passes over the 1,273
sites, the set update, and above all the **match cache, which settles 48% of placements
without scanning at all** and cannot be helped by cores.

### Correctness, four ways

- `--verify`, which recomputes the Hartigan sets from scratch after every placement:
  **every placement matches a full recompute**, at 1 thread and at 4.
- ASan + UBSan over a 12,000-placement parallel run, past the point where the parallel
  path switches on: **clean**.
- The examples checker on the output: clean, at every thread count.
- ThreadSanitizer reports 189 races and **all of them are noise**. Each is a worker's
  write inside the parallel region against the main thread's read after it, with
  `libgomp.so.1` in the frame -- the implicit barrier that separates them is something
  TSan cannot see, because libgomp is not instrumented. Demonstrated rather than asserted:
  a twenty-line program that writes disjoint slots in an `omp parallel for` and reads them
  after the barrier -- correct by the standard -- draws 14 reports of the same shape.

### Two bugs found on the way

- `subtreeSize_` was indexed for nodes added since its last rebuild, which is **out of
  bounds**. It never misbehaved, because a vector's spare capacity absorbed it and the
  value only steers the cut, but it was undefined behaviour on the hot path.
- The scan's DFS stack was a local `std::vector`, so every placement allocated and freed
  one. That is the 16% above; it is a member now.

### Running it

`--threads <n>` on `place`, default 1, and it is refused on the other commands rather
than ignored. `slurm/place_all.sbatch` takes `--cpus-per-task` as the count, now 32.
Below 8,192 nodes the parallel region is skipped -- a scan that small is shorter than the
cost of starting one -- and because the cut is the same either way, that changes the
speed and not the tree.

### Status, 2026-10-23, and what is not settled

**Landed and checked**: the parallel scan, the ordinal merge, the `--threads` flag, the
two bugs above, `slurm/scaling.sbatch`, and `slurm/place_all.sbatch` on 32 cores. The
examples checker is clean on the base tree at seeds 1/2/3 and on the 2026 tree, and the
tree is byte-identical at 1/2/4/8/16/32/64 threads.

**`--threads 16` is the recommendation on the evidence so far**, not 32, and
`place_all.sbatch` asking for 32 is deliberately above it: the real queue builds a tree
of 300,000+ nodes against the 75,348 the curve was measured on, and a bigger tree is more
scan per placement and so a higher parallel fraction. Where the peak actually sits for
that tree is the open question below.

**Three measurements were still running when this was written**, and each would change
something:

- *The same curve on 850,299 placements* (`work/sc_big.<jobid>.log`, threads 1/8/32/64),
  where the tree reaches roughly twice the size. If the peak moves right, 32 is right for
  the full run and the 16 above is an artefact of the benchmark being too small.
- *The same curve with `OMP_WAIT_POLICY=active`* (`work/sc_spin.<jobid>.log`, threads
  8/16/32/64). The 32- and 64-thread regression looks like the cost of starting a
  parallel region once per scanning placement, with threads asleep in between. If
  spinning recovers it, that is a one-line environment setting for the real run and the
  answer is not "16 cores is the limit" but "libgomp's idle policy is". If it does not,
  the suspect is the node's `ThreadsPerCore=2`: 64 allocated CPUs may be 32 physical
  cores, which would explain a plateau at 32 exactly.
- *`--verify` with the 8,192-node gate compiled out*, so the parallel path runs from the
  first placement. The gate means the longest verify run so far (4,000 placements, clean)
  **never engaged the parallel path** -- it stopped at 7,374 nodes. ASan and UBSan did
  cross the gate, at 12,000 placements, and the determinism check crosses it at every
  thread count, so what is missing is specifically the semantic check against a full
  Hartigan recompute *while threads are running*. Worth closing before the full run.

**Not attempted**: a persistent thread pool. If the region-start cost is what the spin
test says it is, replacing libgomp's fork/join with a pool that parks on an atomic would
be the next lever, and the one that would make 64 cores worth asking for.

## Decisions

| | |
|---|---|
| Alphabet | the 20 standard amino acids, or 22 with `--alphabet 22` -- see "Directive, 2026-10-30" |
| Missing data | `-`, `*` and `X` all fold to missing: they name no state and constrain nothing. At 22 tokens only `X` does |
| Site 1274 | dropped. It is the terminal stop in the 1,274-aa sidecar, so S is **1,273 sites** |
| Tree | the Nextstrain tree as given. **Nothing is collapsed** until step 4 |
| Root | **pinned** to the root-sequence sidecar, not left free to parsimony |
| Observed nodes | the input tree's leaves and the placed sequences; everything else is free |
| Genotypes | none. A node carries allele *sets*; one allele per node is chosen at step 4 |
| Observation storage | sparse, **against the root sequence**, missing residues included |
| Set storage | per-node deltas against the parent, never a dense nodes × sites matrix |

### The leaves are the data; the interior is not

A sample's sequence is an observation and pins the node it sits on. In the Nextstrain
tree those nodes are exactly the 3,848 leaves: an internal node's genotype is Nextstrain's
own ancestral reconstruction, which this tool does not take as data. 3,274 interior nodes
are free, and the root is pinned to the sidecar.

The 35 internal nodes whose names are not `NODE_*` are all `<strain>_travel_history`
wrappers with exactly one child — Nextstrain travel annotations, not sequences. They are
free, degree-1, and resolve to the same sequence as their child, so step 4 collapses them.
Nothing special-cases them.

What this costs is 2 units of parsimony, no more: see **Consequences, measured**.

### Site 1274 is dropped at the parse

40 edges carry a mutation at 1274. The parser filters at `maxSite`, so a leaf's
observation holds only the 1,273 real residues, and `--root-seq` is required by every
command because it is what determines the site count.

## Data structures

Flat parallel arrays indexed by node id, node 0 = root (`src/tree.hpp`). Input node ids
*are* tree node ids, because nothing is collapsed:

```cpp
struct ObsAllele { uint16_t site; char aa; };    // 1-based; 'X' means "unknown here"

struct Tree {
    std::vector<int>                      parent;    // -1 at root
    std::vector<std::vector<int>>         children;
    std::vector<std::vector<std::string>> labels;    // the samples observed here
    std::vector<std::vector<ObsAllele>>   obs;       // vs the root seq; empty if free
    std::vector<std::string>              srcName;

    bool observed(int v) const { return v == 0 || !labels[v].empty(); }
};
```

There are no edge mutations. `obs` is the only sequence information the tree carries, it
exists only at observed nodes, and it is stored **against the root sequence** rather than
against the parent — the parent may be free and have nothing to diff against, and
root-relative also makes the allele at a (node, site) a binary search instead of a walk
down from the root. That is `alleleAt()`, and it is the single definition of both what
pins a node and what the `V1`/`V2` store is kept relative to.

The root sequence has no missing data, so it is a legitimate baseline at every site, and
the clock filter bounds how far any observation strays from it: a base-tree leaf costs
7,126/3,848 ≈ 1.9 entries, a placed leaf about 16 (mean 4.5 substitutions plus mean 11
missing residues), so ~22 MB over the full queue.

Allele sets as a 20-bit mask, stored sparsely (`src/alphabet.hpp`, `src/hartigan.hpp`):

```cpp
using AaSet = uint32_t;                          // bit i = i-th of the 20 AAs
struct SetDelta { AaSet set; uint16_t site; };   // only where a node differs from its parent
```

A dense nodes × sites matrix would be 3.6 GB at 710k nodes and is almost entirely
redundant: 887 of 1,273 sites are a single fixed state across the whole tree, and even at
the 386 polymorphic ones nearly every node matches its parent. The sparse form costs 9.4
KB here against 36 MB dense.

`walkMaterialised()` is the matching traversal: a DFS carrying one dense `AaSet[1273]`
scratch buffer (5 KB, L1-resident) plus an undo stack, so per-node work is proportional
to that node's deltas rather than to the site count. The placement scan is built on it.

## Algorithm, step 2

Hartigan's two passes, per site, over the tree as given.

- **Upward** (reverse preorder): `V1` = the states of minimum subtree cost, `V2` = those
  one step worse, from counts over the children's `V1`. An observed node takes
  `V1 = {its allele}`, `V2 = {}` regardless of its children; a free node is confined by
  nothing and reads only its children.
- **Downward** (preorder): `W(root)` = the pinned root state; otherwise
  `W(v) = W(p)` if `W(p) ⊆ V1(v)`, else `V1(v) | (W(p) & V2(v))`.
- Every site is solved, all 1,273 of them, not just the 386 polymorphic ones. A
  monomorphic site generates no set-deltas, but missing data still puts `V1`/`V2` off
  their default there, and step 3 resumes the upward pass from those (correction 1).

A missing-data observation is unconstrained, so `W(p) ⊆ V1(v) = all`, so its set is
exactly its parent's and it costs zero deltas. Missingness is free in this representation.

### The V1/V2 store is kept against the root, and that is what it costs

`V1`/`V2` is recorded only where it departs from `V1 = {alleleAt(v, s)}, V2 = {}`. For an
observed node that is its own residue, exactly as before. For a **free** node there is no
genotype to use, and the baseline becomes the root's residue — which is still exact at
every monomorphic site, for a free node as much as a pinned one, and that is what makes it
the right thing to store against.

The price is a free node inside a clade fixed for a derived allele, which now pays an
entry where the old genotype-relative default paid none: **51,260 entries over 3,103
nodes, 615 KB**, against 49 entries before. That is the whole cost of the simplification
in storage terms, it is 0.6 MB, and the alternative — a parent-relative store — would need
a path walk on every lookup, which is what the random-access upward pass is built to
avoid.

## Results

Base tree, `sets`:

```
S protein    : 1274 aa in the sidecar, 1273 sites kept
tree         : 7123 nodes, 3848 tips, 7122 edges, largest multifurcation 19
               977 edges carry S mutations (1282 mutations total); the other
               6145 carry none and are kept -- nothing is collapsed here
observations : 3849 nodes (3848 leaves plus the pinned root); 3274 free
               7126 residues recorded against the root, 411 of them missing
               every branch's from-allele matches the path above it
allele sets  : 386 of 1273 sites carry more than one allele
               parsimony score 1151 changes, against 1282 Nextstrain
               mutations on the same tree (136 of them touch missing data,
               which is free here)
               1170 set-deltas over 961 of 7123 nodes, 9360 bytes of payload
               51260 V1/V2 entries over 3103 nodes, 615120 bytes of payload
ambiguity    : 29 of 9067579 (node, site) pairs admit more than one allele
               spread over 28 of 7123 nodes
               set sizes:  1:9067550  2:28  3:1
check        : 0 observations outside their own allele set
               2 (node, site) pairs where Nextstrain's own reconstruction
               is outside the set, over 2 nodes
```

0.42 s end to end, including the 51 MB JSON parse — the 7.3× node count costs 2.5× the
old 0.17 s. 1,170 deltas against 1,282 mutations: the shortfall is mutations into or out
of missing data, which change no set.

Those last two lines are the pair that matters. Every observation is inside its own set,
as it must be. Nextstrain's reconstruction at a *free* node is not binding here and lands
outside the set at exactly 2 (node, site) pairs — which is the whole difference between
1,151 and the 1,153 the collapsed tree scored, measured directly rather than inferred.

Examples, base tree, seed 1:

```
examples     : 320 records, 1151 mutations, one per edge
               832 childless nodes left out, carrying nothing to predict
               945 resolved nodes (6178 edges collapsed to nothing),
               207 split from 171 multi-mutation edges (longest 5),
               14 (node, site) alleles drawn from more than one option
```

6,178 of 7,122 edges resolve to nothing: the 6,145 the input tree carried no S mutation
on, plus 33 whose every mutation touched missing data. That is the collapse the directive
moved here, and **945 resolved nodes against the old pipeline's 945** is the point — the
answer did not move, the two collapses just became one. 1,151 mutations written is the
parsimony score on the nose, because the sibling merge is gone.

Placement, first 5,000 of the prepared queue, seed 1:

```
placed       : 5000 sequences in 1.5 s, tree now 12331 nodes
               208 split a branch, 4792 hung straight off a node whose
               allele sets were already the sequence's own;
               largest multifurcation now 19
               added 237 to the parsimony score (mean 0.047 per sequence)
scan         : 48769082 node visits (9754 per placement)
update       : 151524 upward and 103453 downward (node, site) steps, 46391 set writes
               41575 ancestors actually moved, longest chain 4
ties         : 4792 placements matched an existing node exactly (95.84%)
               67 had a single best candidate (1.34%)
examples     : 1389 records, 1388 mutations, one per edge
               1156 resolved nodes (11175 edges collapsed to nothing)
```

The same 237 and the same 1,388 the old one-draw tie-break gave, on a tree 2,348 nodes
smaller. See **Directive, 2026-10-22**.

### The first full run, job 3399137 (superseded)

Run **before** the sibling merge was removed, so its written count is 214 short of the
score, and **before the 2026-10-22 tie-break** — so its tree size, split count and
tie statistics are all the old draw's. The score and the ambiguity counts are what is most
likely to survive a re-run; the node count will not. Kept because the outputs are preserved
as `work/*_2021_merged.*` for comparison.

Job **3402849** was the merge re-run and has since finished: same seed, same rule, so the
same tree and the same allele sets byte for byte -- `cmp` says so -- and only the examples
file differs, at **14,109 records / 14,108 mutations**, which is the score on the nose.
Its outputs are the unsuffixed `work/*_2021.*`. The **tie-break** re-run was never
submitted and has been overtaken: the current rule's full-queue outputs are
`work/*_2021_skip.*`, written in 73.5 s. So the three full-queue runs on disk are
`_merged` (merge, old draw), unsuffixed (no merge, old draw) and `_skip` (no merge,
current rule).

```
allele sets  : parsimony score 1151 over 7123 nodes
prepared     : 351344 sequences, 20191223 .. 20201231
placed       : 351344 sequences in 14132.5 s, tree now 535892 nodes
               177425 split a branch, 173919 hung straight off a node;
               largest multifurcation now 19
               added 12957 to the parsimony score (mean 0.037 per sequence)
scan         : 95377759589 node visits (271465 per placement)
update       : 8976708 upward and 10539300 downward (node, site) steps, 5864585 set writes
               2446195 ancestors actually moved, longest chain 53
ties         : 240 placements had a single best candidate (0.07%)
               argmin size mean 88172.7 max 618333
ambiguity    : 7357 of 682190516 (node, site) pairs admit more than one allele (0.001078%)
               spread over 7343 of 535892 nodes (4845 of them observed)
               set sizes:  1:682183159  2:7355  3:2
examples     : 13895 records, 13894 mutations, one per edge
               12985 resolved nodes (522693 edges collapsed to nothing,
               214 merged into a sibling's genotype),
               910 split from 747 multi-mutation edges (longest 7),
               82 (node, site) alleles drawn from more than one option
```

3.93 h, seed 1. Final parsimony score 1,151 + 12,957 = **14,108**, and step 4's resolved
tree carried exactly that — the check the old pipeline failed by 1 at this very scale. The
13,894 written is 14,108 less the 214 sibling merges, which is what prompted removing the
merge; the re-run writes 14,108.

522,693 of 535,891 edges resolve to nothing: 97.5%. The remaining 12,985 resolved nodes
become 13,895 records after the multi-mutation edges are exploded. Outputs, preserved
under `_merged` when job 3402849 was submitted:

```
work/examples_2021_merged.txt   19.8 MB   13,895 records
work/tree_2021_merged.tsv       48.7 MB   535,892 nodes
work/sets_2021_merged.tsv        9.5 MB   the allele-set deltas
work/place_full.3399137.log
```

The tie statistics are the honest embarrassment here: **0.07%** of placements had a single
best candidate, and the mean argmin was 88,173 positions over 43,805 distinct parents. A
sequence that matches the consensus at every covered site genuinely costs nothing almost
anywhere, so the topology of those placements is the shuffle's doing, not the data's. It
does not affect the score or any invariant, and 97.5% of those nodes collapse at step 4
anyway. It does mean the attachment *positions* carry much less information than the count
of placements suggests.

## Validation

- **Step 2, brute force**: 389 random trees — multifurcating, with missing data, free
  internal nodes, and mutations into and out of missing — run end to end through the
  binary, with every assignment enumerated over a 5-state alphabet (one state never
  observed, to confirm nothing invents a state). Every MPR set and every parsimony score
  matches. Predates the directive and tests Hartigan itself, which did not change.
- **Step 2, full scale**: an independent Sankoff implementation using explicit cost
  vectors and an outside-cost downward pass — a different formulation, no `V1`/`V2`.
  Also predates the directive. It was run against 1,153 on the collapsed tree; re-running
  it against 1,151 on the uncollapsed one is the one validation gap left in step 2.
- Score 1,151 ≤ 1,282 as it must be, and every observation's allele is in its own set.
- **Step 3**: `place --verify` clean over 300 real placements and 8 synthetic stress
  sets — see **Validation, step 3** — re-run clean on all nine after the
  2026-10-22 tie-break, and the full 351,344-sequence run completed with every invariant
  intact under the rule that preceded it.
- **Step 4**: an independent Python checker over the written examples file, clean on the
  base tree under seeds 1, 2 and 3 (same 1,152 records each; see below) and on
  5,000 placements. It re-derives everything from the file alone: ids `0..n-1` in order
  with one root and every parent resolving and preceding, every sequence 1,273 aa over
  the 20 amino acids with no `-`, `*` or `X`, every edge exactly one real mutation,
  every record its parent's sequence with that one mutation applied and differing
  nowhere else, every `MUTATIONS_TO_CHILDREN` the multiset of the mutations below it,
  and the total equal to the reported count — which, since the merge was removed, is the
  parsimony score itself. Clean at 1,000, 5,000, 20,000, 100,000 and the full **351,344**
  placements, and on all 8 synthetic stress sets.

`check_examples.py` is in the repo, run as `python3 -I check_examples.py <file>
[expected_mutations]`. The step-2 checkers were scratch and not kept.

## Layout

```
Makefile                      g++ -std=c++17 -O2 -fopenmp, no dependencies beyond OpenMP
src/json.{hpp,cpp}            minimal pull-style JSON reader; skips what it is not asked for
src/tree.{hpp,cpp}            parse -> the tree as given -> leaf observations -> TSV
src/alphabet.hpp              20-state AaSet, missing-data folding
src/hartigan.{hpp,cpp}        alleleAt, computeAlleleSets, walkMaterialised, TSV writer
src/sequences.{hpp,cpp}       step 3 input: FASTA + metadata -> the prepared file,
                              the (site x 20) count matrix and the MSA of the queue
src/placement.{hpp,cpp}       step 3: parsimony placement, the parallel scan, and
                              the set update
src/examples.{hpp,cpp}        step 4: resolve -> collapse -> split -> write
src/main.cpp                  CLI
check_examples.py             independent checker for the written examples file
guard.py                      the data guards; see README
slurm/place_all.sbatch        the full queue on the 2026 tree, multi-core
slurm/scaling.sbatch          thread scaling, asserting the tree is identical at
                              every count
```

```
make
./antigen tree     --tree data/ncov_global@2020-12-29.json \
                   --root-seq data/ncov_global@2020-12-29_root-sequence.json --out tree.tsv
./antigen sets     --tree data/ncov_global@2020-12-29.json \
                   --root-seq data/ncov_global@2020-12-29_root-sequence.json --out sets.tsv
./antigen prepare  --root-seq data/ncov_global@2020-12-29_root-sequence.json \
                   --fasta ~/sc2-open-data/spike.fasta \
                   --metadata ~/sc2-open-data/metadata.tsv --out prepared.tsv
./antigen place    --tree data/ncov_global@2020-12-29.json \
                   --root-seq data/ncov_global@2020-12-29_root-sequence.json \
                   --prepared prepared.tsv --seed 1 --out-tree tree.tsv --out sets.tsv \
                   --out-examples examples.txt
```

The whole open-data queue on the 2026 tree, which is what the fourth 2026-10-22 directive
asks for, is the same four steps with the filters off and the two flags on:

```
./antigen prepare  --root-seq "data/ncov_open_global_all-time@2026-03-15_root-sequence.json" \
                   --fasta ~/sc2-open-data/spike.fasta \
                   --metadata ~/sc2-open-data/metadata.tsv \
                   --date-cutoff none --out work/prepared_all.tsv \
                   --out-clock work/clock_all.tsv

sbatch --partition=cpu --time=4-00:00:00 slurm/place_all.sbatch
```

`prepare` takes `--date-cutoff <YYYY-MM-DD|none>`, `--no-clock`, `--clock-tol`,
`--clock-slack` and `--out-clock`; `sets` and `place`
both take `--variant <name>` (the clade whose root the examples flag) and
`--post-cutoff <YYYY-MM-DD>` (default 2021-01-01). `prepare` and `place` both take
`--out-counts <path>`, the (site x 20) count matrix over the queue, and `--out-msa
<path>`, those same sequences as a FASTA (fifth 2026-10-22 directive). `place` also takes `--limit n`,
`--tie-seed n` (the placement tie-break, seeded apart
from the queue shuffle so the two sources of variation can be told apart),
`--out-matches <path>` (a TSV of the sequences that matched a node and were not
attached, one `strain<TAB>node` row each), `--attach-matches` (attach them as leaves
anyway -- the rule before the second 2026-10-22 directive, kept for comparison) and
`--verify`, which recomputes every stored value from scratch after each placement and
compares. Verification is how every correction below was found, so it is worth
re-running after any change to the engine.

## Algorithm, step 3

### Scoring is a lookup against the step-2 sets

For an unweighted alphabet the score increase from hanging an observation `x` off node
`v` is 0 at every site where `x_s ∈ W(v)` and 1 at every other site — never more, since
the alternative to forcing `v` off its optimum is one mutation on the pendant branch.
Inserting a new node on the branch `p → v` gives that node the free run of `W(p) | W(v)`
(⊇ because any MPR realising `b` at `p` or at `v` extends with the new node at `b` for
free; ⊆ by the triangle inequality — a zero-cost assignment forces the new node onto one
endpoint, else its two half-edges cost 2 where the original edge cost ≤ 1). So

```
cost(child of v)       = #{ s : x_s observed, x_s ∉ W(v) }
cost(new node on p→v)  = #{ s : x_s observed, x_s ∉ W(p) | W(v) }
```

The second is never worse than the first -- nor worse than a child of `p` -- so **the
minimum always sits on a branch**, and a node placement can only ever join the tie set.

Which tied position is taken is settled in two stages, not one draw (**Directive,
2026-10-22**). The scan carries `e(v) = #{s : x_s observed, W(v) != {x_s}}` alongside the
cost; `e(v) == 0` says a leaf under `v` would have `W(leaf) == W(v)` at every site, and
such a **match wins outright** over every branch -- `e >= cost`, so a match is always free.
Failing that, only branch candidates are considered. Either draw is a reservoir sample, so
the choice is uniform over its set in O(1) memory.

Both costs are maintained incrementally down a DFS: `W` changes only at the sites in
`delta[v]`, so per-node scan work is proportional to that node's deltas, and the union
correction `#{delta sites : x_s ∉ W(v), x_s ∈ W(p)}` involves those same sites.

### The surgery: a pendant leaf, always

The sequence becomes a new observed leaf, under the chosen node or under a new free node
spliced into the chosen branch. Nothing is committed to a genotype and nothing is
collapsed. A split gives `p` a new child `u` in place of `v`, and `u` exactly two children
— `v` and the leaf — so no node's degree ever grows on a split, and `u` is precisely the
node Hartigan says may hold any of `W(p) | W(v)`, left to hold all of it.

The leaf's own sets are closed form, not a recompute. `V1(L) = {x_s}`, `V2(L) = {}`, so
the downward rule gives `W(L) = {x_s}` wherever `x` is observed and `W(L) = W(parent)`
wherever it is missing. So `delta[L]` is one pass over the sites, built straight in site
order: an entry wherever `x` is observed and the parent's set is not already exactly that
one state.

### Which sites can move

The upward pass reads only its **children's `V1`** — `gmin` feeds the score, never the
sets. A new child whose `V1` is exactly the baseline the store is kept against cannot move
its parent's `V1` at all: the baseline state's count rises by one, every other count holds,
so the argmax holds and `V2` holds with it. That is the whole locality argument now, and it
is what bounds the site set the update has to consider:

```
attach leaf under w : { s : x_s != rootSeq[s] }  ∪  sites(obs[w])  ∪  sites(v12[w])
split p -> v        : the above for v, plus sites(delta[v])
```

`sites(delta[v])` is there because `v`'s deltas were stored against `p` and have to be
rewritten against `u`, even at a site where nothing about `v` moved. `obs[w]` is there
because the baseline at an observed node is its own residue, not the root's. Off that set,
a short argument closes the loop: `delta[v]` empty at `s` forces `W(p) = {a}`, hence
`W(u) = W(v) = W(L) = {a}`, so neither `u` nor the leaf needs an entry there either.

Measured, the set is ~15–30 sites and the upward pass moves 7.1 ancestors per placement
over them, never more than 5 hops in one chain.

Caveat, unchanged from before: **"observed" is not "pinned at every site"**. A labelled
node that is missing at a site is free there, exactly as in `computeAlleleSets`, and that
site does propagate — which is why the refresh runs even from an observed node rather than
being skipped. The upward pass stops on its first comparison wherever the node really is
pinned, so this is cheap rather than skipped.

### Termination rules

- **Upward**, per site: recompute `V1`/`V2`; stop at the first ancestor whose **`V1`** is
  unchanged — not whose `W` is unchanged, which is a downward quantity not yet available.
  If `V1` holds but `V2` moved, store it and stop: the parent cannot see `V2`.
- **Downward**, per site: start at the highest node the upward pass touched. **Off the
  path it walked**, `V1`/`V2` are unchanged, so there the test collapses to `W` unchanged
  ⇒ prune the whole subtree. **On that path they are not unchanged**, and `W` can move
  below a node whose own `W` held still — a node whose `V2` moved while its `V1` did not
  is exactly that case. So the descent is *forced* along the path and only starts pruning
  once it steps off it. Either way the node's own delta may still need rewriting under a
  prune, since deltas are relative to the parent and the parent's `W` may have moved.
- Each site has exactly one such path, so the two passes never interact across sites and
  `W(parent(h))` is always the unchanged value. The two passes are run together, one site
  at a time, because the downward pass needs the path the upward pass took.

Measured cost per placement: 27 upward and 32 downward (node, site) steps, 7.1 ancestors
actually moved. The scan, at ~10,900 node visits, is the expense by three orders of
magnitude.

## Decisions, step 3

| | |
|---|---|
| Sequences | `~/sc2-open-data/spike.fasta`, dates from `metadata.tsv` alongside it |
| Date cutoff | `--date-cutoff`, default collection date < 2021-01-01, `none` to keep every dated sequence |
| Order | chronological; within a day, a shuffle seeded from `--seed` |
| Clock filter | the divergence must be plausible for the date to within 3 months, against an envelope calibrated per month from the data; **superseded** the 2020 regression -- see **Directive, 2026-10-22 (seventh)** |
| Coverage filter | skip if fewer than 1146 of 1273 sites are non-missing (90%) |
| Tree overlap | the 482 strains already present as tips are attached anyway |
| Attachment | if an existing node's allele sets are exactly the sequence's, the sequence is **counted there and not attached**; otherwise the cheapest branch to split. Never a plain child of a node that is not a match |
| New node | free, committed to nothing: it may hold any of `W(p) \| W(v)` and is left holding all of it |
| Pendant leaf | created only where the sequence is attached, which is where it is not a duplicate of what the tree already holds |
| Duplicates | `tree.tsv`'s `n_matched` column counts them per node; `--out-matches` writes one row per skipped sequence. Neither is read by anything |
| Intermediate file | `prepare` selects and orders; `place` consumes. No seed in the file |
| Count matrix | `--out-counts`, a (1273 x 20) TSV over **the queue**, not the leaves: every sequence that passed the filters counts once, whether it was attached or matched-and-skipped. Missing data increments nothing |
| MSA | `--out-msa`, the same records as a FASTA, 1,273 wide, **gaps and stops kept**: the alignment the matrix counts |

`d_root` is the Hamming distance to the **pinned root** over non-missing sites.

### The clock threshold is calibrated, not assumed

**Superseded by Directive, 2026-10-22 (seventh)**, which replaced the line with a
per-month envelope. The calibration below is why the line was right for 2020 and is kept
for that; it is not what the tool does now.

Regressing mutations-from-root on `num_date` over the input tree's 3,848 tips gives
**3.3417 S aa substitutions/year**, crossing zero at 2020.065. The open data agrees
independently: mean divergence 1.76 in March 2020 rising to 4.48 in December.

The slack of 8 was chosen off a sweep, not picked:

| slack | rejected | Jan–Feb 2020 | Nov–Dec 2020 |
|---|---|---|---|
| 4 | 22,647 | 87 | 21,858 |
| 5 | 5,534 | 80 | 4,953 |
| 6 | 906 | 76 | 554 |
| **8** | **267** | **75** | **104** |
| 12 | 184 | 73 | 87 |
| 15 | 135 | 28 | 86 |

Slack 4–5 is deleting B.1.1.7 and B.1.351, which genuinely carry 9–10 S changes in late
2020 — not date errors. 8–12 is a plateau, so the result is insensitive there; the
Nov–Dec rejects at slack 8 sit at divergence 11–53, well clear of Alpha. Slack 15 starts
losing the Jan–Feb signal, which is the thing the filter is for: 75 of the 1,165
sequences dated Jan–Feb 2020 carry up to 53 S changes, which no clock allows.

### The data, as measured 2026-10-21

```
spike.fasta    : 9,335,628 records, fixed length 1274 aa, one line each
                 alphabet ACDEFGHIKLMNPQRSTVWY plus - * X, already aligned,
                 so site 1274 drops and alphabet.hpp folds missing as-is
metadata.tsv   : 9,424,942 rows: strain + date, all YYYY-MM-DD or empty, strain
                 names unique; 9,335,628 dated, 89,314 undated. The FASTA holds
                 exactly the dated strains, which is why the two counts agree
date < 2021    : 377,063 strains, all of which have a sequence
filters        : 25,551 fail coverage, 267 fail the clock, 99 fail both
ATTACHED       : 351,344 (93.18%)
missing sites  : mean 10.97, median 0, p99 112, max 127 among those attached
                 68.0% are fully covered
root sequence  : no missing data at any of the 1273 sites, so the root is
                 pinned everywhere and is never the highest changed node
```

Missing residues are recorded in `obs` as `'X'`, alongside the substitutions, which costs
~11 extra entries per placed sequence — 15 MB over the full queue. No separate
observed-mask field. That matters for correctness, not just storage: a node that is
missing at a site must come out *unpinned* there, and `aaBit('X') == kAllStates` already
does that. Leaving the site out of `obs` instead would make it fall back to the root's
residue, which would fabricate an observation.

## Open for step 3

- ~~**Scan cost is the whole budget.**~~ It was, at ~2 nodes added per placement and
  9.5 × 10^10 visits over the full queue. Since a match is not attached (second
  2026-10-22 directive) the tree grows only on splits and 96.5% of placements never
  finish a scan at all: 1.63 × 10^9 visits over the same queue, 72.7 s end to end. The
  budget now sits with whatever the *next* scale brings, not with this one.
- **The prepared file is read whole into memory, and that is the next wall.**
  `PreparedSet` holds every selected sequence in one blob: 459 MB over 351,344 records,
  so ~12 GB over 9.3M, plus a `Record` and a strain string each. The tree is no longer
  the thing that does not fit; this is. It wants streaming `place` straight off the
  prepared file, which the format already allows -- the records are fixed-width and in
  a deterministic order -- except that the within-day shuffle needs the day's block,
  not the whole file.
- **Pruning is deliberately not built.** The bound is
  `lb(v) = D(v) − subDelta[v] − K`, with `subDelta[v]` the delta count in `v`'s subtree
  (maintained up the root path on insert) and `K` the largest per-node delta count. It
  only bites on subtrees of a handful of nodes, so it is worth perhaps 2×. 72.7 s is
  tolerable, so this stays unbuilt -- and the match rule took far more out of the scan
  than any bound would have.
- **Escape hatch if that is not enough.** A site-major index keyed by **(site, allele)**
  → the nodes where that allele's membership in `W` flips. Indexing by site alone does
  not reduce the total delta count — it is a pure transposition — whereas the flip lists
  for one sequence's allele vector are a small subset, and give a good incumbent early,
  which is what makes a bound bite.
- ~~**Recomputing `V1` at a free node is O(degree).**~~ Closed by measurement: the
  largest multifurcation is 19 on the input tree and still 19 after 5,000 placements, and
  a split never grows a degree at all. Reported by `place` as `largest multifurcation now`,
  so it stays watched.
- ~~**Genotypes are one reconstruction, the sets are the authority.**~~ Closed by deleting
  the genotypes. There are none, so there is nothing to reconcile against the sets, and
  corrections 3, 4 and 5 go with them.
- ~~**`siteActive` has to grow.**~~ Resolved, but not the way this said: the default is
  *not* exactly right at a monomorphic site, so step 2 solves every site (correction 1).
  `siteActive` is still maintained — a new observation whose allele differs from the
  root's flips it — but only as a statistic.
- **The tie sets were large, and what is left of them is real.** Since the second
  2026-10-22 directive a match is not drawn for at all, which removes 96.5% of the draw;
  the mean branch draw is 243 at 20,000 placements and 500 over the full queue, over 122
  and 251 distinct anchors. That is a genuine choice between topologies at equal cost,
  and still not worth "fixing" without a reason to prefer one tied branch over another.
  What follows describes the draw as it stood before, and is kept for the argument about
  what a large tie set means.

  Most of the tie is between copies. A sequence that matches the consensus at every covered site
  genuinely costs 0 almost everywhere, so the argmin was 88,173 positions over 43,805
  distinct parents on the full run under the old one-draw rule. The new rule does not
  shrink the count — the mean draw set is 4,479 at 20,000 placements — but it
  changes what the count is *of*: 96.9% of placements now draw among nodes whose allele
  sets are exactly the sequence's, which the output cannot tell apart, rather than among
  different topologies at equal cost. The residual 3% draw among branches, which is a real
  choice. Still not worth "fixing" further without a reason to prefer one tied branch over
  another.
- **Per-node `std::vector` will want replacing** with an arena or small-buffer inline
  array if the node count grows much past this; the accessors are narrow enough that the
  backing store can change. At ~32k nodes over the full queue it is further from being
  worth it than it was.

## Corrections from validating step 3

All but the last were found by `place --verify`; the fifth surfaced in a sweep over
seeds, through step 4's `mutations == parsimonyScore` check, and `--verify` then
located it exactly. Every one of them had a plausible argument behind it in this
file. They are recorded in full because the arguments still look right until you see
the counterexample.

### 1. The monomorphic-site default is not exact

This file claimed `V1 = {genotype}`, `V2 = {}` is "exactly right at a monomorphic site".
It is right only where there is no missing data. At a site where every observed node
carries the same allele `a`, a **free node whose children are missing there** has
`V1 = all 20 states`, not `{a}`; and a free node with exactly one child that does
constrain the site has `V2 = all \ {a}`, not `{}`. Both are ordinary Hartigan, and both
are invisible while the site stays monomorphic — `W` is `{a}` throughout either way — but
the moment a placement brings a second allele to that site, the upward pass reads those
stored defaults and gets the wrong answer.

The fix is to solve all 1,273 sites in `computeAlleleSets` rather than the 386 polymorphic
ones. Measured: same 0.16 s, the same 1,157 set-deltas and the same score of 1153, and 39
→ 49 `V1`/`V2` entries. The skip was never buying anything.

### 2. "W unchanged ⇒ prune" is wrong on the upward pass's own path

The downward rule as written pruned a subtree whenever a node's `W` came out unchanged.
That is sound *off* the path the upward pass walked, where `V1`/`V2` are untouched. On
that path it is not: the upward pass stops at a node whose `V1` held but whose `V2` moved,
and `V2` does not affect that node's own `W` at all — it affects its **children's**. The
descent therefore has to be forced down the path the upward pass took, and may only start
pruning once it steps off it.

Caught on a node whose `V2` went from `{D}` to `{}` while its `W` stayed `{A}`, leaving a
stale ambiguous `{A,D}` one level below.

### 3. A merge is only free when the node already carried a label

*Deleted with the merge path, which no longer exists: a placement is always a new
leaf. Kept for the argument, which was correct.*

"Its genotype already exists, so it is just a label there" is true for the 969 nodes that
already carry samples. For an **unlabelled** node it is not: a label turns an inferred
genotype into an observation, which pins the node, which collapses its `V1`/`V2` to the
default and moves `W` below it. The merge path now recomputes from that node whenever it
was free before.

### 4. The genotype choice has to defend the collapse

*Deleted with `chooseState`. There is no committed genotype and no collapse at
placement time. Kept for the argument, which was correct.*

The scan's branch cost is below the cost of hanging the sequence off either endpoint
exactly when some site has `x_s ∈ W(p) | W(v)` but `x_s ∉ W(p)` (or `∉ W(v)`). The old
rule — the member of `W(u)` minimising the *three* new edges — could pass over `x_s` at
such a site in favour of a majority between `geno(p)` and `geno(v)`. The new node then
came out a byte-for-byte copy of `p`, was collapsed as redundant, and the sequence was
hung off `p` after all: a strictly worse topology than the scan found, recorded with the
scan's cost. The score drifted by exactly 1 per occurrence.

Taking `x_s` whenever `W(u)` admits it puts a mutation on the half-edge at precisely those
sites, so a new node can no longer turn out to be a copy of the endpoint it beat — and a
new node that *is* a copy is one whose branch cost was a tie to begin with, so collapsing
it keeps the score the scan promised. The price is that the committed reconstruction is
no longer edge-minimal at a handful of sites, which is cosmetic: the tree's recorded
mutations were never an MPR anyway (1,282 against a score of 1,153).

### 5. A collapse is only free when the endpoint's genotype is in its own set

*Deleted, same reason as 4 -- and this is the one the directive was really aimed at.
The code it describes is gone, `collapseSlips` with it. Kept in full because the
argument is the clearest statement of why committing a genotype at a free node was
the wrong idea.*

Correction 4 left a premise unstated. The scan prices a branch insertion below
hanging the sequence off either endpoint exactly when some site has
`x_s in W(p)|W(v)` but `x_s not in W(v)`; `chooseState` takes `x_s` there, so the
new node differs from `v` at that site, so `mutsUV` cannot be empty and the
collapse cannot happen. That last step needs **`geno(v) in W(v)`** -- only then
does `x_s not in W(v)` imply `x_s != geno(v)`.

That invariant holds for every pinned node (a label forces `W = {its allele}`),
for every leaf, and on the input tree, where `sets` checks it outright. It does
**not** hold for a *free* node: an unlabelled node created by an earlier split
commits its genotype once, out of `W(u)` as it stood at insertion, and a later
placement can narrow `W(u)` past it. `edgeMut` and the sets then disagree, the
scan prices off the sets and the surgery collapses off the genotypes, and the
attachment quietly costs more than the branch the scan found.

Caught at placement 221 of seed 1 / tie-seed 777, site 708:

```
p = 1018 (labelled)     geno = S   W = {S}
v = 1031 (free, split)  geno = S   W = {F}   <- was {F,N,S}, narrowed at placement 220
x                       = S
```

Branch cost 0 there (`S in {S}|{F}`), child-of-`v` cost 1 (`S not in {F}`), so the
scan takes the branch and promises 3. `chooseState` returns `S`, which is also
`geno(v)`, so `mutsUV` is empty, the node looks redundant, and the sequence is
hung off `v` after all -- which really costs 4. The tree comes out one mutation
longer than the scan found and `parsimonyScore` one too low. Step 4's
`mutations == parsimonyScore` check is what fails downstream; `place --verify`
reports it directly as `score 1195 incremental vs 1196 recomputed`.

**The fix** is to price the collapse instead of assuming it. Both endpoints are
always legal targets for a plain attachment, and the scan already minimised over
both, so neither can come out below the promised cost. `splitBranch` now computes
`#{s : x_s observed, x_s not in W(w)}` at both endpoints before giving in to one,
keeps the forced endpoint whenever it meets the promise -- every case but the
defect, so nothing else moves -- and otherwise takes the cheaper of the two and
**reports the cost it actually paid**, which `place` adds to the score.

The remaining price is that in the defective case the placement can be one worse
than the theoretical optimum, because the optimum needs a node the genotype
bookkeeping cannot express: it would mean rewriting `geno(v)` to its MPR state,
which cascades into every descendant that inherits at that site. That is counted
and reported (`collapses cost more than the branch the scan priced`) rather than
hidden. Measured at 20,968 placements it fires **once** per affected run.

Rates, over the first 20,968 of the prepared queue (a whole-day boundary, so the
attached set does not move with the seed):

| sweep | runs | runs with a slip |
|---|---|---|
| `--seed 1..20` | 20 | 0 |
| `--seed 1 --tie-seed 1..20` | 20 | 6 |

Every slip was worth exactly 1. Byte-for-byte identical output on all 20 plain
seeds before and after the fix, so it is inert everywhere it is not needed.

The deeper invariant -- `geno(n) in W(n)` at every node -- is still only checked
on the input tree. It can only break at a free node, and this is the one place
that depended on it, but it is worth stating that it is an assumption and not a
guarantee.

## Step 3 progress

Landed:

- `AlleleSets::v12` — `computeAlleleSets` records `V1`/`V2` wherever they differ from what
  `alleleAt(v, s)` implies (`V1 = {that allele}`, `V2 = {}`), at every site rather than
  only the polymorphic ones (correction 1). The upward pass cannot resume from `W` alone,
  which is why this is needed.
- `src/tree.{hpp,cpp}` — the parse, and one DFS turning the Nextstrain mutation paths into
  each leaf's observation. `obs` is kept sorted by site, so a node can be binary-searched
  for one site; so are `delta` and `v12`.
- `src/sequences.{hpp,cpp}` — the metadata and FASTA readers, the three filters, and the
  prepared-file format. One streaming pass over 12 GB. Unchanged by the directive.
- `src/placement.{hpp,cpp}` — the scan, the surgery and the set update.
- `src/examples.{hpp,cpp}` — step 4, which is now where the collapse happens.
- `tree`, `sets`, `prepare` and `place` in `src/main.cpp`, with `--verify`.

### `prepare` writes the file; `place` owns the seed

The 12 GB pass is slow enough to be worth doing once, so `prepare` writes the selected
sequences to an intermediate file and `place` reads that. The file is in a **deterministic**
order — date, then strain name — and carries no seed. The within-day shuffle belongs to
`place`, so trying another seed does not mean re-reading the FASTA. 351,344 records,
459 MB, written in 4.7 s.

The three filters and the counts they produce reproduce the numbers in **The data** above
exactly: 25,551 fail coverage, 267 fail the clock, 99 fail both, 351,344 attached (93.18%),
mean 10.97 missing sites, 68.0% fully covered.

Two record counts in that block were wrong and are corrected above: `spike.fasta` holds
**9,335,628** records, not 9,424,942 — that figure is the metadata row count — and all
377,063 in-window strains have a sequence, not 377,062.

## Validation, step 3

`place --verify` recomputes **every stored value** from scratch after each placement — the
root sets, the parsimony score, every set-delta and every `V1`/`V2` entry — and compares.
Not just `W`: `V1`/`V2` is what lets the next placement resume the upward pass, so a drift
there is a real defect even when `W` agrees. Correction 1 was only visible in `V1`/`V2`.

Clean over, after both directives:

- **Real data**, the first 300 of the prepared queue. Mean cost 0.193 there, 0.047 over
  the first 5,000 — lower than the 0.17 the collapsed tree gave, because a less
  constrained tree has more places that cost nothing. 259 of the 300 matched an existing
  node's allele sets exactly.
- **Synthetic**, 8 seeds × 400 sequences generated off the root with up to 12 random
  substitutions and up to 3 missing runs of up to 40 residues. This is the stress case —
  almost every placement splits a branch, and it is what exercises deep ambiguity and the
  free-node paths that real data barely touches. Every previous correction was found here
  first. It matters more than ever now that *every* interior node is a free node, and it
  is also the one place where the 2026-10-22 match rule mostly *does not* fire: only
  4.5–10% of these placements match, against 90–97% on real data, because a random
  12-substitution sequence off the root is not a genotype the tree already holds.

All three directives' runs check the same thing in the same way, and in all nine runs
`placements == splits + skipped` exactly, which is the current rule's own invariant: a
placement either splits a branch or is not attached at all. (Under the middle rule, which
hung a leaf off the matched node, the same invariant read
`placements - splits == exactHits`.)

A failure prints the offending (node, site) and dumps that neighbourhood — parent,
grandparent, siblings, children — with `W` and `V1`/`V2` side by side from both the
incremental state and the recompute, each node marked `obs` or `free`. That dump is what
made each correction diagnosable; it is worth keeping.

## Ambiguity after placement, measured

`place` ends with the same materialised ambiguity walk `sets` does, so the count is
reported on the tree as it stands after attachment rather than only on the input tree.
Over a prefix of the prepared queue, seed 1:

Measured under the old one-draw tie-break. The 2026-10-22 rule moves the small counts up a
little and the node counts down a lot — 36 over 8,219 nodes at 1,000, 38 over 12,331
at 5,000, 40 over 27,749 at 20,000 — without changing the shape of any of this. The
rows beyond 20,000 have not been re-measured.

| placed | nodes | (node, site) pairs | ambiguous | share |
|---|---|---|---|---|
| 0 | 7,123 | 9,067,579 | 29 | 0.00032% |
| 1,000 | 8,638 | 10,996,174 | 33 | 0.00030% |
| 5,000 | 14,679 | 18,686,367 | 33 | 0.00018% |
| 20,000 | 37,339 | 47,532,547 | 37 | 0.000078% |
| 100,000 | 157,551 | 200,562,423 | 66 | 0.000033% |
| **351,344** | **535,892** | **682,190,516** | **7,357** | **0.0011%** |

**The plateau does not survive the full queue.** It held to 100,000 — 80,000 further
leaves added 29 pairs — and then the count went 66 → **7,357** over the last 251,344,
spread over 7,343 nodes of which 4,845 are observed. The *share* still falls, as it must,
and no set exceeds size 3 (`1:682183159  2:7355  3:2`), so nothing has broken; but the
raw count rises by two orders of magnitude.

That is the caveat below doing exactly what it warned about. A queue prefix is
chronologically the earliest, so every number above 100,000 is the first to include the
Alpha/Beta era, where many more sites carry a second allele and a free interior node has
something to be ambiguous *about*. The lesson is that a prefix of this queue is not a
small version of it, and a measurement taken on one should not be reported as a property
of the whole. The mechanism is the same
as before and survives the directive: a placement is an observation, and an observation
pins the node it lands on, so attachment mostly *removes* freedom even though it no longer
has a pinned interior to lean on.

Caveat, now vindicated: a queue prefix is chronologically the earliest, so anything
measured on one is pre-Alpha/Beta. The share can only fall as the denominator grows, but
the raw count may rise -- and did, 200-fold.

## Step 4: the mutation tree as examples

Turning the placed tree into the example file the downstream model consumes, and the only
place anything is collapsed. Five parts:

a. **Resolve the ambiguity.** Pick one parsimonious allele per (node, site). The pick
   must be **jointly** parsimonious, which an independent per-node draw from `W(v)` is
   not: `W(v)` was computed against the full set `W(p)`, so it is the right conditional
   set only when `|W(p)| = 1`. The correct draw is top-down — root from `W(root)` (pinned
   here), then each node uniformly from `V1(v) | ({a} & V2(v))` where `a` is the parent's
   **already-chosen** state. That is the same formula as the downward pass with `W(p)`
   replaced by the singleton draw, and it needs `V1`/`V2`, which `sets.v12` carries.
   With 37 ambiguous pairs over 36 nodes the two agree almost always, but "almost" is not
   a reason to write the wrong one.
a'. **Collapse to the mutation tree.** Derive each edge as the diff between resolved
   parent and child, and collapse every edge that comes out empty, merging its labels into
   the surviving node. This is where the input tree's 6,145 zero-mutation edges go, and
   where a placement that matched what was already there stops being a separate node. Then
   merge siblings that resolved to the same sequence.
b. **One mutation per edge.** Split every multi-mutation edge into a chain of
   single-mutation edges, order shuffled, intermediate nodes degree 2. Longest resolved
   edge is 5 mutations.
c. **One example per node that has children**: the resolved AA sequence at that node,
   plus the mutation on each child edge as a (position, new_AA) pair, the position
   **0-based** -- see **Directive, 2026-10-22 (sixth)**. A childless node is
   an input with no label and is not written -- see **Directive, 2026-10-22 (third)**. Its
   sequence stays recoverable from the parent's list, and its edge is still counted there.
d. **Write them** to a `.txt`.

The directive asks for the explode before the collapse. The two commute — exploding an
empty edge is a no-op — and the code collapses first, because it is cheaper not to explode
what is about to disappear.

### Resolution erases missing data, and that is what makes the collapse definable

A resolved sequence is drawn from `W`, which is always a non-empty set of real amino
acids, so **every node comes out fully specified** — no `-`, `*` or `X` anywhere. There is
nothing else to compare: the tree carries no edge mutations, so an edge *is* the diff
between the two resolved sequences, by construction rather than by re-derivation.

Three kinds of edge come out empty, and all three collapse:

- the 6,145 input edges that carried no S mutation at all — the directive's whole point;
- a placement whose resolved sequence is its parent's, which is most of them;
- an edge whose every recorded mutation touched missing data, which resolution erases.

At 20,000 placements, 35,765 of 37,338 edges resolve to nothing and 37,339 nodes become
**1,363** — the same 1,363 the two-collapse pipeline produced. The example count tracks
distinct resolved genotypes, not nodes, and that is the honest count: a sequence that is a
known genotype with some residues unknown carries no new information.

### What the collapse does not do: it removes no mutations

Easy to state backwards, so stated plainly. A collapsed edge is one whose resolved diff is
**empty** — that is the definition, and why it collapses. All 522,693 of them on the first
full run contributed zero mutations before the collapse and zero after. The collapse
removes *nodes*, 535,892 → 13,199, and not one mutation.

What used to remove mutations was the sibling merge, which is why it is gone — see below.

### Duplicate siblings are kept, so the count is the score

The merge is **removed**. Two children of one node that resolve to the same sequence —
because one was missing where the other was not, or because both are placements of the
same genotype — are now left as two records.

Merging them was the one thing standing between the written mutation count and the
parsimony score, at exactly one per merge: a redundant twin's edge still carries a *real*
mutation, so dropping the twin drops that mutation too. The identity was exact at every
scale measured:

| run | score | written, merged | merges | written, unmerged |
|---|---|---|---|---|
| base tree | 1,151 | 942 | 209 | **1,151** |
| +1,000 | 1,267 | 1,058 | 209 | **1,267** |
| +5,000 | 1,388 | 1,179 | 209 | **1,388** |
| +20,000 | 1,807 | 1,596 | 211 | **1,807** |
| +351,344 | 14,108 | 13,894 | 214 | — |

So `mutations written == parsimonyScore` is now true **by construction**, not nearly true,
and the same number the resolution check asserts is the number in the file. Record counts
rise by the merge count: base tree 943 → 1,152, 20k 1,596 → 1,808.

A second consequence, which the merge had been hiding: every record but the root is
reached by exactly one mutation, so

```
records == mutations == parsimonyScore, plus one for the root
```

and the record count was therefore **seed-invariant**: three seeds on the base tree gave
three byte-different files and the same 1,152 records, where under the merge it moved with
the seed. **Both halves of that are now superseded** -- a childless node is no longer
written at all (see **Directive, 2026-10-22 (third)**), so the relation becomes

```
records + childless == mutations + 1 == parsimonyScore + 1
```

and the record count moves with the seed again, through `childless` rather than through
the merge: 320 / 322 / 321 on the base tree for seeds 1 / 2 / 3. What is seed-invariant is
what always was -- the mutation total, which is the parsimony score.

The price, accepted deliberately: a node may list the same `(pos, newAA)` twice under
`MUTATIONS_TO_CHILDREN`, with two identical records beneath it. At 20,000 placements, 7
nodes carry a repeat, 287 repeats in all (concentrated on the big ancestral
multifurcations), and 418 of 1,808 records share a sequence with another record.

This is the legacy convention rather than a departure from it: the file the 2021 arm
actually consumed carries **68 sibling duplicates in 347 records**. It also makes the two
phenomena consistent — a split intermediate landing on a sibling's genotype was never
merged either, as the next section explains, so previously one kind of duplicate was
removed and the other kept for no principled reason.

### The legacy format is the target

`antiGen/experiments/multitree/data/forecast2021/examples_ncov@2021-01-01_train.txt` is
what the 2021 arm actually consumed, and it is exactly this structure — one mutation per
edge already, `(pos, newAA)` children, 4,664 lines, and **no record with an empty child
list**, which all nine files in that directory agree on (see **Directive, 2026-10-22
(third)**). Per record:

```
EXAMPLE_ID: 1
PARENT_EXAMPLE_ID: 0
MUTATION_FROM_PARENT:
K,309,E
TRAINING
IS_DESCENDED_FROM_VARIANT:
IS_VARIANT_ROOT:
<1273-aa sequence on one line>
MUTATIONS_TO_CHILDREN:
7,V
<blank line>
```

`MUTATION_FROM_PARENT` is `from,pos,to`; the root has `None` under it and
`PARENT_EXAMPLE_ID: None`. Both `pos` fields are **0-based** in the legacy file -- 346 of
its 346 edges read consistently that way and none reads consistently as 1-based -- which
is what **Directive, 2026-10-22 (sixth)** brought this repo into line with. Every record is written as `TRAINING` with both variant
fields left empty, which is what a record with no variant annotation looks like; the
split and the variant labelling are a downstream pass this repo does not have the inputs
for. Note the old file's alphabet includes `-` (e.g. `74,-`), which
resolution here removes. The split/variant fields are a downstream concern and are not
something this repo can fill honestly without being told how.

### Landed

`src/examples.{hpp,cpp}`, reached by `--out-examples <path>` on both `sets` and `place`,
sharing `--seed`. One pass per tree: resolve, collapse, merge, split, write. The only
thing the directive changed inside it is the `V1`/`V2` default, which is now
`alleleAt(c, s)` rather than a genotype buffer threaded down the DFS — so the walk carries
`W` and the drawn sequence and nothing else.

The resolution walk examines, per node, only the sites in its own set-delta plus the
sites where its **parent** had a choice. That is complete: a child's draw can differ from
its parent's only where `a` is outside `V1(c)`, and off a delta site `W(c) = W(p)`, which
forces `a` into `V1(c)` whenever `|W(p)| = 1`. The remaining case is a parent that drew
from more than one option, which is the second half of the candidate list.

**The check that makes this trustworthy**: a jointly parsimonious assignment changes
state exactly `parsimonyScore` times, so the resolved tree's mutation count must equal it
on the nose. It does — 1,151 on the base tree, and at every placement count tried — and a
site the candidate list failed to examine would show up at once as a shortfall.

Since the sibling merge was removed, the count the check asserts is also the count in the
written file: the collapse is count-neutral, and nothing else shortens the tree.

### Nothing is merged, including split intermediates

Two kinds of duplicate sibling arise, and neither is removed.

**Whole genotypes collide.** Two children of one node resolve to the same sequence when one
was missing where the other was not, or when both are placements of the same genotype.
These used to be merged; see **Duplicate siblings are kept** above for why they are not.

**Split intermediates collide.** Child `{5F}` and child `{5F, 614G}` meet at the first step
whenever the shuffle puts `5F` first. These were never merged: doing so would be building
a trie, and the random mutation order would then decide the topology rather than just the
path through it.

The legacy file settles both — it carries **68 sibling duplicates in 347 records** — so
duplication is the convention here. Removing the first kind while keeping the second was
the inconsistency, and it cost one mutation against the parsimony score per merge.

### Validation, step 4

An independent Python checker over the written file — `check_examples.py`, which reads
nothing but the file and re-derives everything from it. Clean on the base tree (seeds 1, 2
and 3) and on 5,000 and 20,000 placements:

- ids are `0..n-1` in order, one root, every `PARENT_EXAMPLE_ID` resolves
- every sequence is 1,273 aa over the 20 amino acids — no `-`, `*` or `X` anywhere
- every edge carries exactly one mutation, and it is a real change
- every record's sequence is its parent's with that one mutation applied, differing at
  that site and nowhere else
- every `MUTATIONS_TO_CHILDREN` **contains** the multiset of mutations in the records
  beneath it, and what it lists over and above them is a childless node, which is not
  written
- the two counts of those childless nodes agree: the surplus just described, and
  `edges + 1 - records`, which is what a tree of `edges` edges forces
- total mutations — counted from the child lists, since the records no longer carry one
  each — equals the reported parsimony score

Seeds 1, 2 and 3 give different mutation orders and different draws at the ambiguous
sites, and three byte-different files, all passing. The count relation is now

```
records written + childless == mutations written + 1 == parsimonyScore + 1
```

exactly, whatever the seed does, and the record count itself moves with the seed —
320 / 322 / 321 on the base tree — because which nodes end up childless depends on the
draw. Confirmed at 320+832/1,151, 357+911/1,267, 375+1,014/1,388, 424+1,385/1,808 and
2,877+11,236/14,112.

Of the 29 ambiguous (node, site) pairs on the base tree only **14** are free draws: once a
parent is drawn, its children's conditional sets usually collapse to that state. The
marginal count overstates the freedom; the joint count is what the draw actually sees.

### Running it: SLURM, not nohup

The first full run died at 190,000 of 351,344 placements with nothing written. It was
started with `nohup ... &` from inside an interactive allocation and went with it when
that allocation ended; the outputs are written only after the loop, so there was nothing
to salvage. Re-run as a batch job instead:

```
./slurm/submit_place_full.sh          # snapshots the binary, then sbatch
```

`slurm/place_full.sbatch` is `cpu_batch`, 1 CPU, 32 GB, 12 h, logging to
`work/place_full.<jobid>.log`, and writes `work/tree_2021.tsv`, `work/sets_2021.tsv`,
`work/examples_2021.txt` and `work/matches_2021.tsv`. **Those first three names are
currently occupied by job 3402849's outputs**, which are the attach-rule control the
directive table compares against, so move them aside (`_attach`) before submitting
anything that would overwrite them. The 12 h is now four orders of
magnitude of headroom -- the run is 73 s on one core -- and is left alone because the
queue it is sized for is the one that has not been built yet. The submit script copies `antigen` to `work/antigen_full` first,
so rebuilding the working tree cannot swap the binary out from under a queued or running
job. `ANTIGEN_SEED` overrides the seed, `ANTIGEN_BIN` the binary.

---
Last updated 2026-10-22. Ten changes that day, in order: the collapse moved to the end;
the sibling merge removed, so the written mutation count is the parsimony score;
attachment takes an exact allele-set match outright and otherwise only splits a branch; a
matched sequence is **not attached at all**, for +0.24% of tree length at 100,000
placements and a 90x speedup; a node with no children is no longer written as an example,
which is four fifths of the records; and the starting tree moved to the **2026-03-15**
open-data all-time build, with the whole 8.5M queue, the BA.2 variant flags and
`POST_CUTOFF` in place of `TRAINING`/`TESTING`; and `--out-counts`, a (site x 20)
position count matrix over every sequence that got through the filters, with `--out-msa`
writing that same subset of the input alignment beside it; and the examples' two position
fields became **0-based**, which is what the legacy file they target always was; and the
clock filter became a per-month envelope calibrated over the whole span and applied to
**both** runs, in place of the 2020 regression that only one of them used; and
`POST_CUTOFF` became a statement about the leaves below a record rather than about the
tree's own interior dates, which also fixed a fold that had been walking the node index
range as if it were a post-order; and the variant root became the resolved node the
tree's variant root collapsed into, with descent walked on the resolved tree, which is
what stops a spliced BA.2 stem from aborting the run at the last step.

The outputs under `output/sc2-2021/train/` and `work/` predate the last two changes: they
carry 1-based positions and were built from queues the old clock rule selected.

Two things to read with their dates attached: the full-queue figures under **The first
full run** predate the second and third of those, and **`antiGen/load.py` has not been
changed** to read `POST_CUTOFF`, so it would call every record testing until it is.

2026-10-23: the placement scan runs on many cores -- see the directive of that date. The
single-core timings everywhere above still stand as single-core timings.

2026-10-23 (second): the same train run at all 24 quarterly cutoffs from 2020-04-01 to
2026-01-01 -- see the directive of that date. The trees come from Nextstrain's own S3
archive rather than Charon or Wayback, neither of which spans it, and the 2021-01-01 arm
reproduces off a byte-identical tree. `output/sc2-<cutoff>/train/` per cutoff.

## Directive, 2026-10-23 (second): the same run at every quarter

> I want to repeat the same training data generation runs as we did for the 2021-01-01
> cutoff with other cutoffs every 1/4 year, starting with 1/4 of the way through 2020,
> up through the beginning of 2026. As a first step, check what Nextstrain trees are
> available [...] If a tree isn't available from the requested date, get the latest
> available tree preceding that date.

24 cutoffs, **2020-04-01 to 2026-01-01**. The tree work is in the README; this section is
the runs.

### Every cutoff has a tree, and the 2021 one is the same file

Neither route the two existing trees came by reaches the whole span. Charon's `@date`
bottoms out at **2022-04-30** (2022-04-29 is a 404, bisected) and does not serve the
retired names at all -- `/ncov/global` is a 404 at every date. Wayback's captures stop at
**2021-09-04**, leaving a hole over three cutoffs. The public `nextstrain-data` **S3
bucket** has neither problem: every build is still there under its own dated key, through
the four names the global all-time build has had (`ncov_` -> `ncov_global_` ->
`ncov_gisaid_global_` -> `ncov_gisaid_global_all-time_`).

All 24 land **1-4 days** before their cutoff, except 2026-01-01 at 83 days, because the
GISAID all-time build stopped on 2025-10-10. `fetch_trees.py` rebuilds the set and
`data/trees/manifest.tsv` indexes it, with the guards on every row.

The check that matters: `ncov_global_2020-12-28.json` from the bucket is **byte-identical**
to the Wayback copy this repo already had, `md5 4edfa514a7b7df6bf3a519c77f670be2`, sidecar
included. The series' 2021-01-01 arm therefore runs on the same tree as the original, not
a near-copy.

### `--post-cutoff` had to become explicit

`run_2021.sbatch` never passes it, and that was only safe because its default is the
hardcoded `2021-01-01` -- the same date as its own `--date-cutoff`. At any other cutoff
the default labels the train examples against the wrong date, so `run_series.sbatch`
passes `--post-cutoff "$D"`. At D = 2021-01-01 this is the default, so the original arm is
reproduced exactly; everywhere else it is the analogous thing. In the train arm the flag
is vacuous either way, since both the tree and the queue stop before D.

### Two departures from `run_2021.sbatch`, both for disk

296 GB free, and the later cutoffs each carry a 12 GB prepared file and would each write
an 11 GB MSA. So: **no `--out-msa`**, and the **prepared file is deleted** once `place` is
done with it. Both are regenerated by the `prepare` call directly above them in the same
script. Nothing derived is lost -- `counts_<D>.tsv` is the position count matrix over the
same sequences.

### The cluster eats tasks silently, and that needed its own script

Tasks kept failing to launch with

```
error: namespace_p_join: open failed for /media/8TBNVME/scratch/<jobid>/.ns
error: namespace_g_join(JobId=...) failed
*** JOB ... CANCELLED ... DUE TO JOB REQUEUE ***
```

-- node-local scratch, nothing to do with this pipeline. The failure mode is what makes it
worth a section: SLURM requeues the task and parks it in **`JobHeldUser`, where it sits for
ever**. The array then drains to empty with those cutoffs never having run, and
`squeue` says finished. Nothing in the job's own output says anything, because the job
never started.

`slurm/unhold.sh <arrayjobid>` runs detached beside the array and releases anything held,
every minute, until the array is empty. A released task may land on a healthy node or be
held again; either way it is not lost. `summarize_series.py` is the second guard: a cutoff
with no log, or a log with no `done` line, is reported **MISSING** rather than skipped,
so a silently-dropped cutoff cannot read as a clean run.

### Running it

```
sbatch --array=0-23%8 slurm/run_series.sbatch      # job array 3419037
./slurm/unhold.sh <arrayjobid> &                   # beside it, not optional here
python3 -I summarize_series.py                     # output/series_summary.tsv
```

`%8` is set by disk rather than by cores: 8 concurrent prepared files at 12 GB is 96 GB of
the 296 GB free. 32 cores and 96 GB per task, on `cpu_batch_high_mem`.

### Progress, 2026-10-23 17:51

**Superseded.** This array was cancelled at 7 of 24 and the whole series rerun; the
final numbers are in "Both arms complete" at the end of this file.

7 of 24 complete, checker clean on every one. The queue grows 300-fold across the series
and the record count with it:

| cutoff | tree | attached | % | records | mutations |
|---|---|---|---|---|---|
| 2020-04-01 | 2020-03-31 | 28,622 | 94.46 | 190 | 1,129 |
| 2020-07-01 | 2020-06-30 | 93,535 | 93.15 | 633 | 3,312 |
| 2020-10-01 | 2020-09-30 | 161,786 | 93.69 | 1,200 | 6,125 |
| **2021-01-01** | **2020-12-28** | **351,410** | **93.20** | **2,934** | **14,233** |
| 2021-04-01 | 2021-03-31 | 853,371 | 91.57 | 7,467 | 34,477 |
| 2021-07-01 | 2021-06-29 | 1,391,729 | 91.30 | 11,539 | 53,055 |

**351,410 (93.20%) is the number this file already records for the 2021 cutoff**, off the
same binary (`md5 ba2b1b69fafda3c448e032bbeb858dd0`) and the same seed. The series
reproduces the arm it is generalising, which is the only evidence available that the other
23 are right.

In flight, by attached count: 2022-04-01 at 6,170,340 (90.39%), 2023-01-01 at 7,813,585
(91.24%), 2024-01-01 at 8,277,482 (91.21%), 2025-10-01 at 8,488,981 (91.09%) -- against
8,502,996 for the no-cutoff run, so the last few cutoffs are the full queue in all but
name. The attached share falls from 94.5% to 91.1% across the series, which is the
coverage filter meeting the later years' sequencing dropout, not the clock.

## Directive, 2026-10-23 (third): one test tree, labelled against every cutoff

> All training sets can share the same testing tree, as long as we can make a small
> modification to the one test tree (which attaches all of the sequences). All we need
> to do is notate onto the test tree which examples are post which cutoffs, for every
> cutoff. [...] You can also update the code so that there are no "post-cutoff" flags at
> all in the training tree side, as these are irrelevant.

24 train trees, one test tree. `POST_CUTOFF` stops being a boolean and becomes a set.

### The flag was already one number per node, and nobody had noticed

The eighth directive made `POST_CUTOFF` a statement about the leaves: True when the
earliest observed date anywhere below the node is at or after the cutoff. `minLeafDated`
is that date, folded once over the resolved tree. The cutoff was then compared against it
and thrown away.

So the tree never held one cutoff's answer -- it held `minLeafDated`, from which the
answer at **every** cutoff follows. Labelling one tree against 24 of them costs one
comparison per (record, cutoff) and the lines written. Nothing is recomputed, nothing is
re-placed, and the mutation tree is bit-for-bit what it was: the draw and the shuffles
consume the same RNG stream in the same order, because the only thing that changed is
what gets printed.

### The flags a record carries are a prefix, which is checkable

The predicate is `minLeafDated(v) >= D`. Raising `D` can only turn True into False, so
with the cutoffs sorted ascending the ones a record is at or after are a **prefix** of
the list. The writer stops at the first miss; `check_examples.py` re-derives the sorted
set of cutoffs from the file and asserts every record's list is a prefix of it, in order.
That is a real constraint on 24 flags, not a restatement of how they were written.

The old monotonicity check -- no record before the cutoff hanging under one at or after
it -- generalises to set containment: a parent stands above a superset of its child's
leaves, so its earliest date is no later, so **a child's cutoff set contains its
parent's**. Checked at every cutoff at once.

### The format is the variant fields', not a new one

`POST_CUTOFF: True` was a one-line value. It is now a header with zero or more
`<date>,True` lines under it -- which is exactly what `IS_DESCENDED_FROM_VARIANT` and
`IS_VARIANT_ROOT` already are, so `check_examples.py`'s `_flags()` reads it unchanged and
the record keeps its field order and its nine-line floor.

```
EXAMPLE_ID: 900
PARENT_EXAMPLE_ID: 889
MUTATION_FROM_PARENT:
T,28,I
POST_CUTOFF:
2020-04-01,True
2020-07-01,True
...
2023-04-01,True
IS_DESCENDED_FROM_VARIANT:
21L (BA.2),True
IS_VARIANT_ROOT:
MFVFLVLLPLVSSQC...
MUTATIONS_TO_CHILDREN:
1207,H
```

Only the True lines are written, as with the variants. The False ones are the rest of the
list, and the prefix property says where the boundary is.

### The train side carries no flags at all

`--post-cutoff` now takes a comma-separated list and is repeatable, and **its default is
empty** rather than the hardcoded 2021-01-01. An empty list writes the header with
nothing under it, which is what every train arm gets: `run_series.sbatch` no longer
passes the flag and `run_2021.sbatch` never did.

It was always vacuous there. A train arm's tree and its queue both stop before `D`, so
every record read False; the flag was a column of one value. The question it was there to
answer -- which cutoffs a record falls after -- is a question only a tree holding every
sequence can answer, and now only that tree is asked it.

`run_all.sbatch` and `place_all.sbatch` build the list with
`cut -f1 slurm/cutoffs.tsv | paste -sd,`, from the same file the train array is driven
by, so the two cannot drift. Its output directory moved from `output/sc2-2021/test` to
**`output/sc2-test`**: it is not any one cutoff's any more.

### Measured on the base 2026 tree, no placements

`sets` on `ncov_open_global_all-time@2026-03-15`, all 24 cutoffs, 1,055 records:

| cutoff | at or after | before | | cutoff | at or after | before |
|---|---|---|---|---|---|---|
| 2020-04-01 | 1,049 | 6 | | 2023-04-01 | 282 | 773 |
| 2020-10-01 | 997 | 58 | | 2024-01-01 | 121 | 934 |
| **2021-01-01** | **908** | **147** | | 2025-01-01 | 60 | 995 |
| 2021-07-01 | 705 | 350 | | 2025-10-01 | 8 | 1,047 |
| 2022-01-01 | 516 | 539 | | 2026-01-01 | 3 | 1,052 |

**908 / 147 at 2021-01-01 is the number the eighth directive records** for this tree under
the old single-cutoff code. The generalisation reproduces the case it generalises, and the
C++ counts and `check_examples.py`'s independent re-derivation agree on all 24 columns.

The series is monotone decreasing by construction and is so in fact -- it is the survival
curve of "every observation below this node is at least this late", and it falls off a
cliff after 2024 because the tree's own tips run out at 2026-03.

### Running it

```
scancel 3419037                                    # the old array, 7 of 24 done
cp antigen work/antigen_series work/antigen_all    # md5 420ddd1f9c85e75c02b9f5852cade7b8
sbatch --array=0-23%8 slurm/run_series.sbatch      # job array 3419194 -- train, no flags
./slurm/unhold.sh 3419194 &
sbatch slurm/run_all.sbatch                        # job 3419195 -- the one test tree
```

The seven completed cutoffs were rerun rather than patched: they are 20 minutes each at
the early cutoffs and the point of the array is that all 24 come off one binary. The old
logs are in `work/series/old-3419037/`.

### `unhold.sh` was releasing arrays and nothing else

The test tree is one job, not an array, and it hit the same node-local scratch failure the
series does. `slurm/unhold.sh` did not release it, and would not have, ever: it matched the
reason code against `*Held*`, and SLURM spells it two ways.

| job | `scontrol show job` Reason |
|---|---|
| array task | `JobHeldUser` |
| plain job | **`launch_failed_requeued_held`** |

Only the first has a capital H. The pattern is now `*[Hh]eld*`. The bug was invisible for
as long as the only thing being babysat was an array, which is the whole of its history
before today -- and the failure mode is the one the script exists to prevent: a job parked
at `Priority=0` for ever while `squeue` shows it pending, politely.

Both babysitters run under `setsid`, so they outlive the shell that started them:

```
setsid nohup ./slurm/unhold.sh 3419194 > work/series/unhold_3419194b.log 2>&1 &
setsid nohup ./slurm/unhold.sh 3419195 > work/unhold_3419195.log 2>&1 &
```

### It is the idle nodes, and that is schedulable around

`unhold.sh` releasing the job was not enough on its own: SLURM kept handing it straight
back to the node that had just eaten it, because that node was **idle** and so the
cheapest place to put it. Twice, on two different nodes, until the launch-failure limit
tripped.

The split is clean, and it is not random:

| node | state | `/media/8TBNVME/scratch/<jobid>/.ns` |
|---|---|---|
| GPU1256, CPUDFDC84, CPUDFDE34, CPUDFE024 | mixed / allocated | fine -- 12 series tasks ran here |
| GPU115A, GPU1034 | **idle** | fails every time |
| GPU0F98 | drained | - |

A node that is already running something has the scratch tree; an idle one does not. So
the scheduler's preference and the job's requirement point in opposite directions, and
the babysitter alone would have cycled for ever.

```
scontrol update JobId=<id> ExcNodeList=GPU0F98,GPU115A,GPU1034,GPU104C
```

on the pending job, rather than a cancel and resubmit, which keeps the id the babysitter
is already watching. `--exclude=...` on `sbatch` does the same at submission.

### Progress, 2026-10-23 18:30

**Partial.** Six of 24 at the time of writing; the finished series is in "Both arms
complete" at the end of this file.

The six cutoffs the old array had finished came back off the new binary **identical in
every number the summary records**, which is the check that the change is a change to
what is printed and nothing else:

| cutoff | attached | % | records | mutations |
|---|---|---|---|---|
| 2020-04-01 | 28,622 | 94.46 | 190 | 1,129 |
| 2020-07-01 | 93,535 | 93.15 | 633 | 3,312 |
| 2020-10-01 | 161,786 | 93.69 | 1,200 | 6,125 |
| 2021-01-01 | 351,410 | 93.20 | 2,934 | 14,233 |
| 2021-04-01 | 853,371 | 91.57 | 7,467 | 34,477 |
| 2021-07-01 | 1,391,729 | 91.30 | 11,539 | 53,055 |

Every row matches the table in the previous section, placement for placement. The only
difference in the files is the `POST_CUTOFF:` header, now empty on all of them:
`grep -c ,True` is 0 over every train arm.

### A launch-failure hold cannot be released, so `--exclude` is the fix and the babysitter is not

The three sections above are one bug chased through three disguises. The root cause is the
node table: four nodes in `cpu_batch_high_mem` have no `/media/8TBNVME/scratch`, they are
the **idle** ones, and so they are exactly where the scheduler puts a new job.

`unhold.sh` cannot help with that, and this run is what showed why. After five failed
launches SLURM stops requeueing and sets its own hold:

```
$ scontrol release 3419194_6
3419194_6: Access/permission denied
```

`JobHeldUser` is the user's hold and releases. `launch_failure_limit_exceeded_requeued_held`
is **SLURM's**, and nothing the user can run will lift it. The script discarded the exit
code, so its log said `released 3419194_6` once a minute for twenty minutes while the task
sat at `Priority=0` and `Restarts=5`, going nowhere. Twelve of the twenty-four did this.
They were cancelled and resubmitted as array 3419443; there is no other move.

So:

* **`#SBATCH --exclude=GPU0F98,GPU115A,GPU1034,GPU104C`** is in `run_series.sbatch` and
  `run_all.sbatch`. It prevents the failed launches, which prevents the hold, which is the
  only thing that works. Drop it when the cluster is fixed.
* `unhold.sh` now reports a denied release instead of claiming success, and sees
  `REQUEUE_HOLD` (`-t PD,RH`) and the lowercase spelling (`*[Hh]eld*`). All three were
  bugs. But with `--exclude` in place there is nothing left for it to do, and it was not
  run for the jobs below.
* `summarize_series.py` remains the real guard: a cutoff with no log, or a log with no
  `done` line, reads MISSING rather than being skipped.

## Validation, 2026-10-23: records that share the root's sequence are reversions, not a bug

> If we look at the unique example identified as the testing root, there are other
> examples whose starting sequences are exactly the same. This seems suspicious -- I would
> expect that the "collapse to mutation tree" step would make such examples very rare, as
> we'd require a mutation followed by a reversion.

The expectation is right, the mechanism is exactly the one named, and the frequency is
what "very rare" predicts. On `output/sc2-2021/test/examples_all_cur.txt`, 75,606 records:

| | |
|---|---|
| distinct resolved sequences | 75,182 |
| records sharing a sequence with any other | 691 (**0.9%**) |
| records sharing the **root's** sequence | **3** -- ids 0, 25143, 29375 |

### The two are grandchildren of D614G, and holding G is parsimonious by 600-fold

Both reach the root's sequence by `D,613,G` then `G,613,D` -- D614G in 1-based, and back.
The intermediate is node 518, which is the D614G node itself:

| | |
|---|---|
| child edges below 518 | **2,452** |
| of those, edges mutating site 613 | 3 (two to `D`, one to `A`) |
| cost of 518 holding `G` | 1 (edge in) + 3 (edges out) = **4** |
| cost of 518 holding `D` | 0 + **2,449** |

So `G` is not merely an allowed draw, it is the only sane one, and the two children the
data puts at `D` genuinely require a reversion. Their resolved sequence equals the root's
because site 614 is the only site that ever moved on that path.

The BA.2 variant root (record 32082) is the same story at a different site: L24S, 355
child edges, 353 keeping `S`, 2 reverting, cost 3 against 353.

They are two distinct clades, not one record written twice -- 25143 has 11 child edges
and 5 records below it, 29375 has 2 and 2.

### The parsimony score is the proof, and it is already asserted

A gratuitous mutation-and-reversion would be a mutation the score does not pay for, and
`writeExamples` throws if the resolved tree carries a count other than
`sets.parsimonyScore`. On this run:

```
allele sets  : parsimony score 2133 over 5688 nodes
               added 317765 to the parsimony score
examples     : 75606 records, 319898 mutations, one per edge
```

2,133 + 317,765 = **319,898**, exactly the edges written. The resolution spends the
parsimony score and not one mutation more, so no reversion in this file is free-floating.

### The shape of the duplicates is what the biology predicts

| duplicate groups (267 in all) | count |
|---|---|
| all siblings under one parent -- the documented "duplicate siblings are kept" case | 125 |
| unrelated records in different clades -- convergent evolution | 103 |
| containing an ancestor/descendant pair -- a true reversion | 39 |

98 ancestor/descendant identical pairs in total. The edge count between such a pair is
`{2: 71, 4: 15, 5: 1, 6: 8, 8: 3}` -- the 5 is not a counter-example: site 23 goes
`L->S->L` in two edges while site 69 runs `V->L->T->V` in three, so every site returns to
its start and only the *total* is odd. The sites involved are the known spike homoplasy
hotspots -- 95, 142/144/157/158 (the NTD loop), 24, 70, 614, 484, 332, 950 -- which is
where recurrent mutation and reversion are expected.

**Caveat.** Resolution draws uniformly among equally parsimonious alleles, so *which*
records come out identical is a function of the seed. The rate is a property of the data;
the identity of any individual pair is not, and nothing downstream should key on it.

## Both arms complete, 2026-10-24: 24 train cutoffs and the one test tree

Everything the two 2026-10-23 directives asked for has run. The two Progress sections
above are snapshots taken while it was in flight; this is the end state.

### The series: 24 of 24, checker clean on every one

`python3 -I summarize_series.py` reports **24/24 OK, none MISSING, none NO-DONE**, with
`check_examples.py` exiting 0 on every arm. `output/series_summary.tsv` is the full
table; the six columns that matter:

| cutoff | tree build | attached | % | records | mutations |
|---|---|---|---|---|---|
| 2020-04-01 | 2020-03-31 | 28,622 | 94.46 | 190 | 1,129 |
| 2020-07-01 | 2020-06-30 | 93,535 | 93.15 | 633 | 3,312 |
| 2020-10-01 | 2020-09-30 | 161,786 | 93.69 | 1,200 | 6,125 |
| **2021-01-01** | **2020-12-28** | **351,410** | **93.20** | **2,934** | **14,233** |
| 2021-04-01 | 2021-03-31 | 853,371 | 91.57 | 7,467 | 34,477 |
| 2021-07-01 | 2021-06-29 | 1,391,729 | 91.30 | 11,539 | 53,055 |
| 2021-10-01 | 2021-09-29 | 2,640,997 | 90.58 | 19,925 | 96,898 |
| 2022-01-01 | 2021-12-31 | 4,331,397 | 90.94 | 32,500 | 151,790 |
| 2022-04-01 | 2022-03-31 | 6,170,340 | 90.39 | 40,497 | 185,937 |
| 2022-07-01 | 2022-06-30 | 6,904,984 | 90.87 | 44,481 | 204,172 |
| 2022-10-01 | 2022-09-30 | 7,441,350 | 91.12 | 48,723 | 223,176 |
| 2023-01-01 | 2022-12-30 | 7,813,585 | 91.24 | 53,916 | 244,460 |
| 2023-04-01 | 2023-03-31 | 8,045,730 | 91.28 | 59,593 | 264,757 |
| 2023-07-01 | 2023-06-30 | 8,106,235 | 91.24 | 61,945 | 271,989 |
| 2023-10-01 | 2023-09-29 | 8,182,388 | 91.22 | 64,715 | 281,819 |
| 2024-01-01 | 2023-12-30 | 8,277,482 | 91.21 | 68,342 | 293,748 |
| 2024-04-01 | 2024-03-29 | 8,331,825 | 91.19 | 69,776 | 298,901 |
| 2024-07-01 | 2024-06-30 | 8,360,832 | 91.17 | 71,036 | 302,846 |
| 2024-10-01 | 2024-09-29 | 8,414,336 | 91.13 | 72,734 | 308,849 |
| 2025-01-01 | 2024-12-28 | 8,437,717 | 91.12 | 73,329 | 311,061 |
| 2025-04-01 | 2025-03-30 | 8,458,115 | 91.12 | 74,047 | 313,652 |
| 2025-07-01 | 2025-06-29 | 8,469,445 | 91.11 | 74,466 | 315,742 |
| 2025-10-01 | 2025-09-28 | 8,488,981 | 91.09 | 75,405 | 318,174 |
| 2026-01-01 | 2025-10-10 | 8,498,752 | 91.09 | 75,647 | 319,699 |

The 2021-01-01 row is still placement-for-placement the number this file records for the
original run. The attached share falls 94.46% -> 91.09% across the series, and the record
count rises 400-fold; both are the shapes the earlier sections predicted.

`grep -c ,True` is **0 on all 24** train arms, which is the directive's "no post-cutoff
flags at all in the training tree side", checked rather than assumed.

The arms came off two arrays, for the reason the `--exclude` section gives: **3419194**
finished tasks 0-5, 11, 12, 13, 16, 17, 23, and the twelve SLURM had parked in
`launch_failure_limit_exceeded_requeued_held` were resubmitted as **3419443**, which
finished 6-10, 14, 15, 18-22. Same binary (`420ddd1f9c85e75c02b9f5852cade7b8`) and same
seed on both, so the split is an accident of scheduling and not of the data. 38.9 core-hours
over 24 tasks, longest 3.28 h (2022-04-01), 18:02 Wed to 04:55 Thu at `%8`.

### The test tree: job 3419409, 8,502,996 attached, labelled against all 24 cutoffs

The run that matters for the test side is **3419409**, not the two job ids the previous
section names: 3419195 and 3419321 were both eaten by the idle-node scratch bug before
they started, and 3419409 is the submission carrying `--exclude`. It ran 18:23:21 to
21:33:55 on CPUDFDC84, 3 h 10 m, **checker exit 0**.

| | |
|---|---|
| dated FASTA records, no cutoff | 9,335,628 |
| fail coverage (<1146 of 1273 sites) | 832,137 |
| too diverged / too ancestral for their date | 254 / 241 |
| **attached** | **8,502,996 (91.08%)** |
| matched an existing node exactly | 8,212,132 (96.58%) |
| split a branch | 290,864 |
| **records / mutations** | **75,606 / 319,898**, one per edge |
| BA.2 | 1 variant root, 35,395 descended |

The POST_CUTOFF ladder runs 75,143 records at or after 2020-04-01 down to 170 at
2026-01-01, falling at every one of the 24. That is the aggregate and not the check; the
check is `check_examples.py`, which asserts per record that its flag list is a prefix of
the sorted cutoffs and that a child's set contains its parent's, and it passed.

### Reading a cutoff's test set

```
output/sc2-test/examples_all_cur.txt      116 MB   the one test tree, all 24 cutoffs flagged
output/sc2-test/tree_all_cur.tsv           92 MB
output/sc2-test/sets_all_cur.tsv           11 MB
output/sc2-test/counts_all_cur.tsv         70 KB
output/sc2-<D>/train/examples_<D>_cur.txt           the train arm at cutoff D, no flags
output/series_summary.tsv                           all 24 rows, 20 columns
work/matches_all.tsv                      246 MB   8.5M rows, left where it was written
work/run_all.3419409.log                            the test run's log
```

A train arm at cutoff D takes its test set by keeping the records of
`examples_all_cur.txt` whose `POST_CUTOFF` list carries D. Nothing else has to be
regenerated per cutoff, which was the point of the third directive.

## Directive, 2026-10-24: the BA.2 count matrices, one per cutoff

> For every 1/4 year cutoff, form the position count matrices of the sequences that are
> descended from the BA.2 root. That is, each time a sequence gets appended to the tree,
> if its placement location is such that the BA.2 root is an ancestor of it, then for
> every (site, AA) pair on the new sequence, increment the corresponding (site, AA) pair
> on every position count matrix from a time before the time of that sequence. [...]
> Write a folder of these position count matrices alongside the existing testing output.

24 matrices, 1,273 x 20 each, written beside the one test tree. `M_D` counts the placed
sequences that sit at or below the BA.2 root and are dated at or after `D`.

### Two readings settled before writing any of it

**Which sequences.** "Appended to the tree" is literally the 290,864 placements that
created a node -- the other 8,212,132 match an existing node's allele sets exactly and
are counted against it rather than attached, by the 2026-10-22 skip rule. Counting only
the appended ones would make the matrix a tally over distinct genotypes, not over
observations. Confirmed: **every placed sequence**, keyed by its placement location,
which a skipped sequence has as much as an attached one -- the node it matched.

**Which side of the boundary.** `M_D` takes the sequences dated **at or after** `D`, the
same `>=` POST_CUTOFF uses. That is what makes the matrix the complement of its own train
arm: `prepare --date-cutoff D` keeps the dates *strictly before* `D`, so train and matrix
partition the queue at `D` with nothing falling between them. Under a strict `>` a
sequence collected exactly on a cutoff date would be in neither.

### The prefix property again, and this time it is what makes it affordable

The cutoffs a sequence belongs to are `{D : D <= date}` -- a **prefix** of the ascending
list, for the reason POST_CUTOFF rests on: raising `D` can only turn "at or after" into
"before". So a sequence is not added to up to 24 matrices. It is added once, to the
bucket `B_m` where `m` is how many cutoffs it is at or after, and

```
M_k = sum over m > k of B_m
```

is a suffix sum formed once at the end. One pass over 1,273 sites per sequence instead of
twenty-four: over the ~4M BA.2-descended sequences that is the difference between the
existing `--out-counts` (free) and roughly 1e11 increments (hours).

### Descent is maintained, not re-walked

`inVariant[v]` is `v == variantRoot` or `inVariant[parent(v)]`. Placement only ever
appends nodes and an appended node's parent already has its answer, so the array is kept
current with one lookup per new node -- no walk to the root per placement.

Splicing a node into a branch cannot change any existing node's answer: it lengthens the
path to the root without altering what is on it. That includes the case where the branch
split is the one *above* the variant root -- the new node takes the old parent's answer
(False) and the variant root keeps its own (True). This is the one thing a checker reading
the finished tree has to be careful about in the other direction: `splitBranch` re-parents
`v` under a node with a **higher** id, so after placement a parent's id is not always below
its child's, and descent cannot be derived by one pass in id order. It cost the first
version of the verification script a wrong answer.

### Checked against an independent recomputation, cell for cell

A 34,012-sequence subset of the real queue (every 250th record of `work/prepared_all.tsv`,
which keeps the date order and spans 2019-12 to 2026-02) placed on the real 2026-03-15
tree, then recomputed in Python from nothing but the written `tree_all.tsv`, the matches
file and the prepared records: descendants of BA.2 by an explicit walk down the finished
tree, `m` by string comparison on the dates, matrices accumulated directly.

**All 24 matrices match, every cell, and the sequence totals with them** -- 11,976 inside
the variant, falling to 16 at 2026-01-01, with the plateau at 11,976 for every cutoff at
or before 2021-10-01 and the first drop (6 sequences) at 2022-01-01, which is BA.2's own
arrival.

`check_variant_counts.py` is the standing checker, and it reads the directory and nothing
else. Its one real constraint is **nestedness**: `M_D'` is a subset of `M_D` for `D < D'`,
so every cell of the later matrix is at most the matching cell of the earlier one. That is
a statement about the data over 24 x 1,273 x 20 cells, not a restatement of how they were
written. It also re-derives the sequence totals from the drops between consecutive
matrices. Perturbing one cell by hand makes it exit 1 and name the cell.

### The existing outputs do not move

`exampleOptions` is now resolved before the placement loop rather than at step 4, because
the variant root and the cutoff list are what the matrices are keyed by. It consumes no
randomness and the loop consumes none that it did not before, so the run is bit-identical
where it was: on the subset above, **`tree`, `sets`, `examples` and `counts` have the same
md5 under the old binary (`420ddd1f...`) and the new one (`5c9c85af...`)**. The rerun
reproduces the 2026-10-23 test tree and adds the folder.

### Running it

```
make
cp antigen work/antigen_all                        # md5 5c9c85af59a32059f78cd8b3d9e8b1b1
sbatch slurm/run_all.sbatch
```

`prepare` is rerun rather than skipped: it is 71 seconds of the 3 h 10 m, and
`$HOME/sc2-open-data` has not moved since April, so it rewrites the same queue.

### Reading them

```
output/sc2-test/counts_ba2/counts_<cutoff>.tsv     24 files, one per cutoff
```

Same layout as `counts_all_cur.tsv` -- a `#sequences` line, a `#sites` line, the 20
residues as columns, one row per 1-based site. `counts_<D>.tsv` is the residue
distribution of BA.2 and everything below it over exactly the window the train arm at `D`
does not see, which is the window its test set describes.

## Directive, 2026-10-28: three more pathogens, and their historical trees

> We are now going to run this pipeline on three other datasets: `~/h3n2`, `~/rsv-a`,
> `~/denv2`. First off I want to download historical snapshots of the trees in these
> folders dated to approximately RSV 2023-11-11, H3N2 2022-11-09, DENV2 2019-09-15,
> then rounded to the nearest quarter year.

Rounded to the nearest quarter: **2023-10-01**, **2022-10-01**, **2019-10-01**. Each
then resolved the way `fetch_trees.py` resolves an ncov cutoff — the latest build
**strictly before** it. `fetch_snapshots.py` does the fetching and the guards; the
provenance is a README section, as the other trees' is.

| arm | cutoff | build | lag | tips | headline gene | mutations |
|---|---|---|---|---|---|---|
| rsv-a | 2023-10-01 | 2023-09-26 | 5 d | 1,918 | F | 768 |
| h3n2 | 2022-10-01 | 2022-09-23 | 8 d | 2,344 | HA1 | 1,465 |
| denv2 | 2019-10-01 | 2019-08-16 | 46 d | 1,198 | E | 661 |

### What is different from ncov

**The archive is object versions, not dated keys.** ncov is the only build Nextstrain
ever wrote under dated S3 keys. These three overwrite one undated key per rebuild, so
"the build as of D" is an *S3 object version*, and the dated-key machinery in
`fetch_trees.py` does not transfer. Keys get renamed across the years exactly as
`/ncov/global` did, so each arm is still a chain.

**denv2 at 2019 predates auspice v2.** The dengue build was rebuilt fourteen times in
six years and the only version before the cutoff is 2019-08-16, in v1 format — a tree
file and a meta file, `attr`/`muts`/`aa_muts`, CDS starts from 0. `from_v1()` converts
it. The check is the 2019-11 build, which Nextstrain exported *both* ways a day apart:
of the 1,022 tips the two share, 1,021 reconstruct to an identical E protein.

**The sidecar is a re-inferred root here, not a fixed reference.** For ncov it is
Wuhan-Hu-1 and one sidecar serves years. For rsv and dengue it is this build's inferred
ancestor and it moves — F differs at 7 residues against today's, E at 11, `nuc` at 956
of 10,723 — and `nuc`'s *length* moves too (1,701 → 1,737 for h3n2). Each tree is
therefore paired with the sidecar version from its own build day, and reading a snapshot
against the folder's current `ref.json` would be wrong in the coordinate frame as well as
in the bases. The 2023 rsv-a build also has one `M2` CDS where today's has `M2-1` and
`M2-2`, so gene names have to come from each build's `genome_annotations`.

### What the guards say

All three PASS: no tips at or after the cutoff, mutations present on the headline gene,
sidecar carrying `nuc` and that gene. The strong one is ncov's "every branch's
from-allele matches the path above it", run here against the sidecar for every CDS:
**0 mismatches on h3n2 (8,066 mutations) and rsv-a (47,435)**, and on denv2 0 on every
one of 4,801 amino-acid mutations with **18 on `nuc`**.

Those 18 are the whole cost of the one fallback this needed. 2019-08-16 predates every
version of `dengue_denv2_root-sequence.json` (the first is 2019-11-10), so it gets that
first one — the same fallback `fetch_trees.py` applies to the three 2020 ncov builds.
The 18 land on **two sites**, 6732 (NS4A) and 8289 (NS5), both third codon positions and
both synonymous, which is what 0 mismatches across every amino-acid mutation says
independently. The amino-acid frame, which is all this pipeline reads, is exact.

### Still open

The pipeline itself is ncov-shaped in one place that matters: `src/` hard-codes S as the
gene and 1,273 as its length. Per arm that becomes HA1/329, F/575, E/495. Nothing here
has touched it yet — this directive was the trees.

## Directive, 2026-10-28 (second): train and test arms for the three new pathogens

> Now for each of the three datasets, we will do two runs: training using these
> historical trees, and testing using the trees that were pre-existing in the repos I
> pointed you to. So, similar train/test setup to sc2. No need to identify any
> variants/variant roots.

The sc2 shape, with one cutoff per pathogen instead of 24. Train: the historical tree
fetched above, queue filtered to collection dates strictly before `D`. Test: the
`tree.json` already in the folder, the **whole** queue, `--post-cutoff D` — the test set
is then the records whose POST_CUTOFF carries `D`, exactly as the sc2 test tree works.
No `--variant`, so no `--out-variant-counts` and no `counts_<cutoff>` directory.

### The four decisions, settled 2026-10-28

| question | answer |
|---|---|
| gene | **h3n2 HA1+HA2**, rsv-a **F**, denv2 **E** |
| cutoff `D` | the **rounded quarter**: 2022-10-01, 2023-10-01, 2019-10-01 |
| denv2 partial dates | drop year-only; **year-month → the 15th** |
| clock filter | **on**, as sc2, and report what each arm cuts |

h3n2 is the only one that needs more than one CDS. The seasonal-flu tree annotates
`SigPep`/`HA1`/`HA2` separately and has no `HA` key at all, so HA1+HA2 is a
**concatenation**: 329 + 221 = 550 sites, HA1 at 1..329 and HA2 at 330..550.

### The inputs, audited before any of that was decided

The queue is each folder's `output/nextclade.cds_translation.<GENE>.fasta` plus
`metadata.tsv` — the drop-in for sc2's `spike.fasta`. Nextclade translations are
reference-length with insertions stripped, which is the fixed-width alignment `prepare`
already assumes:

| arm | gene | aa | records | ≥90% covered | metadata rows |
|---|---|---|---|---|---|
| h3n2 | HA1 / HA2 | 329 / 221 | 202,049 / 202,048 | 99.9% | 202,493 |
| rsv-a | F | 575 | 18,780 | 96.4% | 18,780 |
| denv2 | E | 495 | 26,403 | 85.2% | 26,645 |

`HA.fasta` is 566 = SigPep 16 + HA1 329 + HA2 221, so concatenating the two halves and
slicing `HA` are the same thing; the two halves are what the tree is keyed by.

**The frame was checked, not assumed.** The queues are GISAID `EPI_ISL_*` while the
trees carry strain names (h3n2, rsv-a) or GenBank accessions (denv2), so **no strain
name is shared** and a tip-by-tip comparison is not available. The check is statistical
instead — mean identity of 3,000 sampled queue translations to the tree's own sidecar,
at offset 0 and at ±1:

| arm | −1 | **0** | +1 |
|---|---|---|---|
| h3n2 HA1 | 4.6% | **89.8%** | 5.4% |
| rsv-a F | 7.3% | **99.7%** | 7.3% |
| denv2 E | 5.3% | **96.3%** | 5.1% |

A sharp peak at 0 is the frame being right; HA1's 89.8% is H3N2 drift, not a shift.
The historical sidecars carry the same gene lengths as the current ones (HA1 329,
HA2 221, F 575, E 495), so train and test share one coordinate frame per arm.

No shared names also means the queue never matches a tip by identity, only by allele
set — unlike sc2, where the open-data queue and the open build overlapped by name.

### Three things the audit turned up that the code has to handle

**rsv-a F carries a terminal stop.** Both `ref.json` and the translations are 575
characters ending in `*`, so `sitesOf()` drops it and F is **574 sites**. h3n2 HA1/HA2
and denv2 E have no terminal stop and their lengths are the site counts. This is the S
situation exactly, and `src/sequences.cpp:368` already accepts a record of `nSites` or
`nSites + 1`. The 13 F mutations at site 575 in the 2023 tree are stop-codon mutations,
which are missing data anyway and fall outside `maxSite`.

**denv2's `tree.json` has two E branches whose from-state is `*`** — `*122L` and
`*122K`, both at depth 1 — against `ref.json`'s `K`. Stop codons are missing data here,
so this is not a sidecar mismatch; every other CDS and all 77,563 `nuc` mutations check
out. Worth knowing that `tree_annotated.json` carries 1,484 E mutations where
`tree.json` carries 1,490 and reports 0 mismatches, so the two are not interchangeable.
The arms want plain `tree.json`, since no variant root is needed.

**`~/h3n2/ref.json` is valid for both h3n2 trees.** It was downloaded with
`tree_2026-04-02.json` in April and `tree_2026-09-24.json` arrived in September, so this
needed checking rather than assuming — the seasonal-flu HA root-sequence did not move,
and both trees give 0 from-state mismatches against it on every CDS.

### The queues the decisions imply

| arm | `D` | train (dates < `D`) | test (every dated record) |
|---|---|---|---|
| h3n2 | 2022-10-01 | 96,757 | 202,493 |
| rsv-a | 2023-10-01 | 8,707 | 18,780 |
| denv2 | 2019-10-01 | 4,004 + year-month imputed | 11,877 + 2,823 imputed |

denv2 is the thin one by construction: 11,877 of 26,645 rows carry a full `YYYY-MM-DD`,
2,823 are year-month, 11,129 are year-only and 801 are `unknown`. The decision keeps the
year-month records at the 15th and drops the year-only ones. A month is the resolution
both `D` and the clock envelope work at anyway, so imputing the day cannot move a record
across either; imputing a *year* could, which is why year-only goes.

There are also **15 denv2 rows that are complete but unpadded** — `2023-5-6`,
`2024-05-2` and so on. `parseDateCode` (`src/sequences.cpp:172`) takes a length of
exactly 10 and rejects them, so they are currently lost for no good reason. Normalising
them is part of the same change.

### What the code needs, and what has not been done

**Nothing in `src/` has been touched yet.** The three places that are ncov-shaped:

| where | now | wants |
|---|---|---|
| `src/tree.cpp:67` | `if (gene != "S")` on branch mutations | a gene *list*, each with a site offset |
| `src/tree.cpp:101` | `parseRootSequenceS`, `key == "S"` | the same list, concatenated in order |
| `src/sequences.hpp:46` | `minCovered = 1146` | a fraction (0.9) of `nSites` |

The shape of the change: `--gene G1[,G2,...]` defaulting to `S`, with the sidecar and
the branch mutations both read as the concatenation in that order, and `--fasta` taking
a parallel comma-separated list so `prepare` can join `HA1` and `HA2` per strain. Plus
`--date-impute month` for denv2 and the unpadded-date fix. Then an arms table and an
sbatch in the shape of `slurm/cutoffs.tsv` + `slurm/run_series.sbatch`, with
`--post-cutoff D` on the test side and nothing on the train side, as
"Directive, 2026-10-23 (third)" settled for sc2.

No runs have been started.

## Both arms, three pathogens, 2026-10-28: done, and what the code needed

The directive above is run. Six arms -- a train and a test for each of h3n2,
rsv-a and denv2 -- all written, `check_examples.py` **exit 0 on every one**.

### The three code changes the previous section listed, plus one it did not

| where | now |
|---|---|
| `--gene G1[,G2,...]` | a `GeneLayout`: the named CDSs, each trimmed of its terminal stop, concatenated in the order given. The sidecar, the tree's branch mutations and `--fasta` all follow it. Defaults to `S` |
| `--fasta f1[,f2,...]` | one path per gene. Several are joined **by strain name**, not by position |
| coverage | `minCoveredFrac = 0.9` of `nSites`, not the constant 1,146. `ceil(0.9 x 1273) = 1146`, so sc2 is unmoved |
| `--date-impute month` | `YYYY-MM` -> the 15th. `parseDateCode` also stops rejecting unpadded complete dates |
| **`pinnedRoot`** | **the fourth, and it was not predicted** -- see below |

`check_examples.py`'s `NSITES = 1273` went too: it reads the length off the
first record and checks every other against it.

### The sidecar is a reference, and two of these trees say so out loud

`buildTree` carried this since the first commit:

> The root's own branch mutations are relative to the reference, which the root
> sequence already accounts for, so the root edge contributes nothing.

That is true of ncov and of nothing else here. The **current** h3n2 and rsv-a
builds put branch mutations on the root node itself:

| tree | root node | gene mutations on it |
|---|---|---|
| `~/h3n2/tree_2026-09-24.json` | NODE_0000001 | 13: HA1 10, HA2 3 |
| `~/rsv-a/tree.json` | NODE_0000452 | 7 on F |
| the four others | — | none |

Reading the sidecar as the root then makes every one of those a mismatch on the
first edge below it. `antigen tree` reported **105** on the h3n2 test tree and
**103** on `tree_2026-04-02.json`, and the shape gave it away: `N158K but the
path has K`, i.e. a child reverting a change the root had already made and the
reader had not. So the pin is now the root node's own sequence -- the sidecar
with `raw[0].mut` applied -- and `pinnedRoot` is where that happens. It is an
identity wherever the root carries nothing, which is ncov everywhere and four of
these six trees, and the sc2 outputs are bit-identical across it (below).

This also explains the earlier "0 from-state mismatches on both h3n2 trees":
that check walked from the sidecar **applying the root node's mutations**, which
is the same thing this now does. The two were never in disagreement about the
data, only about which sequence the root holds.

`prepare` keeps measuring divergence against the sidecar, because it reads no
tree. That costs nothing: the clock envelope is quantiles of the same divergence
the records are then tested against, so a shifted origin moves the band and the
record together. It is a calibration scale, not a coordinate.

### Reading the queue across two files

h3n2 is the only arm that needs it, and it cannot be done positionally:
`HA1.fasta` has **202,049** records, `HA2.fasta` has **202,048**, and the orders
first diverge at record 5,884. So `prepare` branches:

* **one gene** -- the path unchanged. A record is complete when the next header
  arrives, so it is judged and dropped on the spot. This is the one that meets
  a 12 GB FASTA.
* **several** -- one `nSites`-wide staging slot per dated strain, keyed off the
  same `StrainTable` the metadata built, filled gene by gene and judged once
  every file has been read. A gene a strain lacks stays `X`, which is missing
  data and which coverage already handles. 202,049 x 550 = **111 MB**.

The one strain HA2 does not hold is `EPI_ISL_238598`, and it is the arm's single
`1 of them missing a gene`.

**Checked against the files, not inferred.** All **95,788** h3n2 train records
equal `HA1[strain] + HA2[strain]` character for character, and all of them also
equal `HA.fasta[16:566]` -- so concatenating the halves and slicing `HA` past
its 16-residue SigPep really are the same thing, as the audit predicted. All
**8,453** rsv-a records are the first 574 of the 575-character F translation,
the terminal stop dropped. Of the **4,663** denv2 train records, **1,329** came
from a year-month date and every one of them landed on the 15th.

### sc2 is bit-identical, checked twice

`work/antigen_all` (`5c9c85af...`) against the new binary, on the 2026-03-15
open tree with `--variant BA.2` and all 24 `--post-cutoff` dates: the `sets`,
`tree` and `examples` files have the **same md5**, before and again after the
`pinnedRoot` change. `prepare --date-cutoff 2020-04-01` over the real 12 GB
queue likewise: same `prepared` and same `clock`, **28,622 attached (94.46%)**,
which is the 2020-04-01 row of the series table above, placement for placement.

The relaxed `parseDateCode` cannot move sc2 either: of 9,424,942 metadata rows,
9,335,628 carry a padded `YYYY-MM-DD` and **zero** carry an unpadded one.

### The six arms

Train: the historical snapshot, queue filtered to dates strictly before `D`, no
`--post-cutoff`. Test: the tree already in the folder, the whole queue,
`--post-cutoff D`. No `--variant` on either side.

| arm | side | tree nodes/tips | queue | cov fail | clock hi/lo | attached | % | final nodes | records | mutations |
|---|---|---|---|---|---|---|---|---|---|---|
| h3n2 | train | 4,034 / 2,344 | 96,757 | 97 | 489 / 0 | 95,788 | 99.00 | 40,326 | 11,934 | 27,744 |
| h3n2 | test | 3,250 / 1,664 | 202,493 | 187 | 588 / 8 | 201,266 | 99.39 | 79,558 | 19,953 | 51,909 |
| rsv-a | train | 3,504 / 1,918 | 8,707 | 211 | 42 / 1 | 8,453 | 97.08 | 5,326 | 549 | 1,736 |
| rsv-a | test | 5,672 / 2,864 | 18,780 | 671 | 86 / 0 | 18,023 | 95.97 | 9,042 | 913 | 3,012 |
| denv2 | train | 2,203 / 1,198 | 5,393 | 669 | 54 / 1 | 4,663 | 86.46 | 3,941 | 827 | 1,812 |
| denv2 | test | 7,165 / 3,938 | 14,715 | 2,444 | 50 / 2 | 12,196 | 82.88 | 10,177 | 1,305 | 3,382 |

**The parsimony identity holds on all six.** The resolution spends the score and
not one mutation more, so `score + added == mutations written`:

| arm | side | score | added | sum | written |
|---|---|---|---|---|---|
| h3n2 | train | 1,916 | 25,828 | 27,744 | 27,744 |
| h3n2 | test | 1,705 | 50,204 | 51,909 | 51,909 |
| rsv-a | train | 710 | 1,026 | 1,736 | 1,736 |
| rsv-a | test | 1,080 | 1,932 | 3,012 | 3,012 |
| denv2 | train | 661 | 1,151 | 1,812 | 1,812 |
| denv2 | test | 1,486 | 1,896 | 3,382 | 3,382 |

POST_CUTOFF behaves exactly as sc2's: `grep -c ,True` is **0 on all three train
arms**, and the test arms split at their own `D` -- h3n2 7,974 at or after
2022-10-01 against 11,979 before, rsv-a 316 / 597, denv2 419 / 886.

### Three things worth knowing about the numbers

**These are not sc2-shaped queues.** h3n2's test tree holds 1,045 distinct HA1+HA2
sequences and the queue holds 202k records, so unlike sc2 -- where 96.6% of
placements matched a node exactly -- here 38,154 of 201,266 **split a branch**
and the tree grows from 3,250 nodes to 79,558. Flu HA carries far more diversity
per sequence than the subsampled build does.

**denv2 attaches worst, and it is coverage.** 2,444 of 14,715 test records fail
the 90% bar, against 187 of 202,493 for h3n2. That is the dengue data, not the
filter: the input audit already put only **85.2%** of denv2's E translations at
90% coverage or better, against 99.9% for h3n2 and 96.4% for rsv-a.

**rsv-a's train parsimony score, 710, is below the tree's own 756 F mutations.**
Hartigan finds a cheaper reconstruction than the build states, which is the same
thing the sc2 arms do (2,133 against the build's larger count) and is what the
allele sets are for. denv2's train tree is the one where the two agree exactly,
at 661.

### Running them

```
make
cp antigen work/antigen_arms                       # md5 1e775d5fde681ded8a9752ee95bbc9fa
sbatch --array=0-2 slurm/run_arms.sbatch           # or: bash slurm/run_arms.sbatch <0|1|2>
```

`slurm/arms.tsv` is the table, one row per pathogen, ten columns: arm, cutoff,
gene, impute, train tree, train sidecar, test tree, test sidecar, FASTA list,
metadata. Paths in it are relative to `$HOME`, where the three folders live.
`slurm/run_arms.sbatch` runs **both** sides of one arm in one task, because they
are small: the whole set took **3 min 22 s wall on four cores** of a login node,
of which h3n2's test place is 174 s and everything else together is 28 s. That
is also how these six were produced -- run in place rather than submitted, with
`ANTIGEN_THREADS` left to `nproc`. The thread count does not change what is
written.

### The train MSAs, added after the fact

The first pass of these six arms did **not** write `--out-msa`, which was an
omission and not a decision: `slurm/run_2021.sbatch` writes one for sc2's train
arm, and `run_series.sbatch` and `run_all.sbatch` drop it only because 24 arms
and one 8.5M-sequence queue come to ~11 GB each. Nothing of that kind applies
here, so the train side now writes one and the test side still does not.

| arm | records | width | size |
|---|---|---|---|
| h3n2 | 95,788 | 550 | 54.4 MB |
| rsv-a | 8,453 | 574 | 5.0 MB |
| denv2 | 4,663 | 495 | 2.4 MB |

It hangs off `place`, as sc2's does, so it is written as the queue is consumed
and is the alignment `counts_<D>_cur.tsv` was built from, record for record --
gaps, stops and X kept, since what it records is the data that entered.

**Checked, not asserted.** Re-tallying each MSA by hand into a (site x 20)
matrix reproduces the `counts` file beside it **cell for cell** on all three,
the record counts equal the arms table's `attached` exactly, every strain name
is distinct, and every line is its gene's width. Re-running with the flag added
left all **30** previously written files bit-identical and all six checkers at
exit 0, so the MSA is an addition and not a different run.

### Reading an arm

```
output/<arm>-<D>/train/examples_<D>_cur.txt     the train arm, no POST_CUTOFF flags
output/<arm>-<D>/train/msa_<D>_cur.fasta        the training window's alignment
output/<arm>-<D>/train/{tree,sets,counts,clock}_<D>_cur.tsv
output/<arm>-test/examples_all_cur.txt          the test tree, flagged at D
output/<arm>-test/{tree,sets,counts,clock}_all_cur.tsv
output/<arm>-test-backbone/                      the same tree with nothing placed
                                                 on it -- see the next section
work/arms/<arm>/                                 the logs, and matches_{train,test}.tsv
```

with `<arm>-<D>` one of `h3n2-2022-10-01`, `rsv-a-2023-10-01`,
`denv2-2019-10-01`. The test set at `D` is the records of
`examples_all_cur.txt` whose POST_CUTOFF carries `D` -- the same rule the sc2
test tree is read by, with one cutoff in the list instead of twenty-four. There
is no `counts_<cutoff>/` directory here: that was the BA.2 matrices, and this
directive asks for no variant roots.

### Still open

Nothing from this directive. The one thing left unsaid is that the denv2 test
tree still carries the two `*122L` / `*122K` branches the audit found: `antigen
tree` reports them as the file's only two from-allele mismatches. Stop codons
are missing data, so they cost nothing and are reported rather than fixed.


## Directive, 2026-10-30: the test trees with no attachment step

> For the three new datasets, generate a new version of the testing trees in
> which we skip the attachment step entirely, i.e. just generate examples based
> on the underlying backbone tree. As before we notate which sequences are post
> cutoff.

Nothing in `src/` needed touching: this is the `sets` subcommand, which has
taken `--post-cutoff` and `--out-examples` since the sc2 work. `sets` is
`place` without a queue -- it reads the Nextstrain tree, pins the root, puts
Hartigan allele sets at every (node, site), and hands that straight to step 4,
which resolves, collapses and splits it exactly as it does after a placement.
So the new arm is one command per pathogen and `slurm/run_arms_backbone.sbatch`
reads the **same** `slurm/arms.tsv`: the tree, the sidecar, the gene and the
cutoff are by construction the ones the placed test arm used.

```
output/<arm>-test-backbone/examples_all_cur.txt
output/<arm>-test-backbone/{tree,sets}_all_cur.tsv
```

No `counts`, no `clock`, no MSA: all three are functions of the open-data
queue, and this arm never opens it. The file names keep `_all_cur` so the arm
reads like the other two.

| arm | tree nodes / tips | edges with mutations | parsimony | records | mutations | post-cutoff |
|---|---|---|---|---|---|---|
| h3n2 | 3,250 / 1,664 | 1,137 of 3,249 | 1,705 | 873 | 1,705 | 311 / 562 |
| rsv-a | 5,672 / 2,864 | 862 of 5,671 | 1,080 | 362 | 1,080 | 48 / 314 |
| denv2 | 7,165 / 3,938 | 1,100 of 7,164 | 1,486 | 609 | 1,486 | 119 / 490 |

Three seconds for all three, which is the point: the whole cost of the placed
arm was the placement. `check_examples.py` exits 0 on all three.

### What POST_CUTOFF means when nothing is placed

The same thing it always meant -- a record is at or after `D` when **every
observed date below it** is -- but the observed dates are now only the tree's
own tips' `num_date`, because those are the only leaves there are. Every tip of
all three trees is dated (1,664 / 2,864 / 3,938, none missing), so no node
falls back to `kNoDate` and the flag is defined everywhere.

That makes the two test sets **not nested**. The backbone tree's tips are a
subset of what the placed arm saw, so a node's earliest date below can only be
later here, and a node the placed arm called pre-cutoff can be post-cutoff on
the backbone. They are two different trees to be read the same way, not a
coarse and a fine version of one.

### Checked

- `parsimony score == mutations written` on all three, with no `added` term to
  carry: 1,705 / 1,080 / 1,486 each appear three times -- as the Hartigan score,
  as Nextstrain's own mutation count on the same tree net of the ones touching
  missing data, and as the edges step 4 wrote.
- POST_CUTOFF is downward-closed in every examples file (0 violations over
  873 + 362 + 609 records) and the root is unflagged in all three, which is what
  "every date below" forces and is worth confirming rather than assuming.
- Re-reading the auspice JSONs in Python: every tip dated, 624 / 517 / 1,170 of
  them at or after `D`.
- All **33** files the placed arms had already written are bit-identical. The
  new arm writes to new directories and to `work/arms/<arm>/*_backbone.*`, so it
  cannot collide with them.
- The denv2 run still reports the two `*122L` / `*122K` from-allele mismatches,
  as it should -- stop codons are missing data and cost nothing.

### How to run

```
sbatch --array=0-2 slurm/run_arms_backbone.sbatch     # or
bash slurm/run_arms_backbone.sbatch 0                 # one arm, no scheduler
```


## Directive, 2026-10-30: gaps and stops as states

> We are currently treating `-` and `*` the same as an unknown AA, i.e. no data.
> Keep the default behavior as is, but create a new option that allows for an
> alphabet of the 20 standard AAs plus gap and stop. The complication is that in
> the sequences being appended, a `-` could reflect an actual gap, or could
> reflect unknown AAs, especially when we see a long string of gaps together.

`--alphabet 22` on every subcommand. The default is `--alphabet 20` and is
**bit-identical** to everything written before this change; see "What was
checked" below.

### The backbones needed no rule at all

The directive guessed this and the guess was right. Replaying every branch of
all eight trees with `-` and `*` as real states -- pinned-root semantics, the
sidecar plus the root node's own mutations -- produces **zero** new from-allele
mismatches:

| tree | muts to `-` | muts to `*` | longest adjacent-deletion run on one branch | new mismatches |
|---|---|---|---|---|
| h3n2 `tree_2026-09-24` / `tree_2022-09-23` | 0 / 0 | 0 / 0 | -- | 0 / 0 |
| rsv-a `tree.json` / `tree_2023-09-26` | 0 / 19 | 0 / 1 | -- / 4 | 0 / 0 |
| denv2 `tree.json` / `tree_2019-08-16` | 0 / 0 | 2 / 0 | -- | 2 / 0 |
| sc2 `ncov_open@2026-03-15` | 262 | 0 | 10 | 0 |
| sc2 `ncov_global@2020-12-29` | 31 | 18 | 8 | 0 |

The longest run of adjacent deletions on any one branch of any of them is 10.
There is nothing in a Nextstrain tree that looks like a coverage gap, so
backbone `-` and `*` are taken at face value and no heuristic touches them.

The two denv2 branches `*122L` and `*122K` are the exception, and they were
already known (see above). At 20 tokens they are a benign report; at 22 they are
genuine from-allele mismatches, and the `tree`/`place` report says so. Nothing
else moves.

### What a `-` means in the queue, and how it is decided

Nextclade writes `-` for two different things, and this is documented rather
than inferred. Its alignment page defines an unsequenced region as *"consecutive
gap (`-`) character ranges on the 5' and 3' ends"*, and `translate.rs` emits `-`
only for a fully-gapped codon `---`, `*` for a stop (IUPAC-aware), and `X` for
everything else -- including any codon holding an `N`, and any codon with one or
two gaps (`protect_codon_in_place`). So a submission that simply stops early
arrives looking exactly like a 450-residue deletion.

Measured on `~/denv2/output/nextclade.tsv`, three records that make the point:

| record | gap run in the FASTA | `totalAminoacidDeletions` | `cdsCoverage E` |
|---|---|---|---|
| `EPI_ISL_17609500` | leading, 452 of 495 | **0** | 0.085 |
| `EPI_ISL_408937` | internal, 1 at 355 | 1 (`E:N355-`) | 0.321 |
| `EPI_ISL_17569112` | internal, 159 at 158 | **159** | 0.279 |

A terminal run falls outside `[alignmentStart, alignmentEnd]`, so Nextclade's
own `aaDeletions` -- which is gated on `is_sequenced` -- excludes it even though
the FASTA still writes `-`. And `aaDeletions` is no oracle either: it calls a
159-codon deletion across a sequencing gap in a 28%-covered record.

**The rule**, in `classifyGaps` (`src/sequences.cpp`), applied per gene before
the concatenation and only under `--alphabet 22`:

```
a run touching either end of the gene   -> missing coverage, rewritten to 'X'
an internal run of at most 10 residues  -> a deletion, left as '-'
a longer internal run                   -> missing coverage, rewritten to 'X'
'*'                                     -> always a state
'X' and anything else                   -> missing, as before
```

Per gene, because h3n2's HA1/HA2 seam would otherwise turn two terminal runs
into one internal one. **Rewritten to `X` rather than dropped**, and before
`consider()` weighs the record, because with `-` a state it would stop counting
against coverage and a half-truncated denv2 record would start passing the 90%
filter. The coverage test therefore means exactly what it meant.

`--max-gap-run` sets the 10.

### Why this rule, and what it can and cannot cite

- **Terminal to missing is the Nextstrain stack's own convention, and it is the
  rule that produced the `-` states on the trees we read.** TreeTime's
  `SequenceData(fill_overhangs=True)` -- *"treat gaps at either end of sequence
  as missing data"*, converting to `X` for amino acids -- is on by default, and
  augur exposes the off switch as `--keep-overhangs` in `ancestral.py`. The
  Nextstrain guide "Missing sequence data (gaps, indels, ambiguity)" states it
  outright, naming the cause: *"aligners introducing gaps at the start and end
  of sequences if they are not fully complete (which is very common)"*.
  Applying the same rule to the queue is what makes queue and backbone agree.
- **Coding internal gaps as a state rather than as missing** is what Simmons &
  Ochoterena 2000, *Syst Biol* 49(2):369-381 and Ogden & Rosenberg 2007, *Mol
  Phylogenet Evol* 42(3):817-826 argue for; the latter found gaps-as-state beat
  gaps-as-missing about 90% of the time in the cases where the two differ.
  Those are citations for the *alphabet*, not for the terminal rule -- Simmons &
  Ochoterena's `?` rule is about overlapping non-homologous gaps and is a weak
  source for terminal-versus-internal.
- **The length cap is an engineering heuristic with no literature behind it,
  and it should always be described as one.** There is no published maximum
  plausible in-frame deletion. There are published counterexamples, one of them
  on our own gene: viable H3N2 HA mutants carrying 12-aa and 16-aa deletions in
  the receptor-binding region (mBio 12(6), 2021), and SARS-CoV-2 ORF8 Δ382
  (Young et al. 2020, *Lancet* 396:603-611). 10 is the smallest value admitting
  every adjacent-deletion run the backbones carry. The better-supported prior
  would be positional -- the recurrent deletion regions RDR1-4 of the S NTD,
  McCarthy et al. 2021, *Science* 371:1139-1142 -- but that is per-pathogen
  curation we do not have for HA, F or E.

### The rule against Nextclade's own calls

Restricted to the records the pipeline keeps, comparing every call against
`aaDeletions` from each pathogen's `nextclade.tsv`:

| | h3n2 HA1+HA2 | rsv-a F | denv2 E |
|---|---|---|---|
| kept, Nextclade agrees it is a deletion | 526 | 62 | 67 |
| **kept, Nextclade does not** | **0** | **0** | **0** |
| terminal-masked, Nextclade agrees | 8 | 0 | 12,844 |
| terminal-masked, Nextclade calls it a deletion | 9 | 38 | 529 |
| over-length-masked, Nextclade calls it a deletion | 144 | 0 | 38 |

**Precision on the keep side is 100%** across all three: every `-` kept as a
state is one Nextclade independently called a deletion. The cost is on the mask
side, and it is real:

- Terminal masking loses a few genuine terminal deletions, some in records
  Nextclade reports at `cdsCoverage 1`. Broken down by run length on denv2, a
  terminal run of 1-3 is a coin flip (213 called against 296 not) while one of
  4 or more is almost never a deletion (8 against 596). The rule masks all of
  them anyway, and deliberately: a masked site becomes `X`, which is free and
  constrains nothing, whereas a truncation kept as a deletion is a *fabricated
  observation* that costs parsimony and can pull a placement. Missing data is
  free in this representation; a wrong state is not. `fill_overhangs` makes the
  same trade, unconditionally and with no length test.
- The cap at 10 masks 144 h3n2 sites and 38 denv2 sites that Nextclade calls
  deletions, several in records it reports at `cdsCoverage 1`. These are the
  11/14/17/18-residue HA1 runs and denv2's 14 and 24. **`--max-gap-run 20` would
  keep them**, and the mBio paper above says lengths in that range are not
  biologically impossible for HA. This is the one knob worth revisiting.

### What the arms look like at 22 tokens

`slurm/run_arms_aa22.sbatch`, the same `slurm/arms.tsv`, writing to
`output/<arm>-<D>-aa22/train` and `output/<arm>-test-aa22`. The 20-token
directories beside them are untouched.

Gap calls over each whole FASTA pass (`--max-gap-run 10`):

| arm | kept as deletions | masked at a gene's end | masked as too long |
|---|---|---|---|
| h3n2 | 422 runs / 644 sites | 11 runs / 1,115 sites | 30 runs / 1,944 sites |
| rsv-a | 79 runs / 99 sites | 13 runs / 177 sites | 2 runs / 160 sites |
| denv2 | 88 runs / 110 sites | 1,382 runs / 189,588 sites | 10 runs / 529 sites |

denv2 is the whole reason the terminal clause exists: 189,588 sites of
truncation that would otherwise have become deletion states.

| arm | side | parsimony, 20 -> 22 | added | records | mutations | attached |
|---|---|---|---|---|---|---|
| h3n2 | train | 1916 -> 1916 | 25,828 -> 25,900 | 11,934 -> 12,018 | 27,744 -> 27,816 | 95,788 -> 95,788 |
| h3n2 | test | 1705 -> 1705 | 50,204 -> 50,422 | 19,953 -> 20,113 | 51,909 -> 52,127 | 201,266 -> 201,257 |
| rsv-a | train | **710 -> 730** | 1,026 -> 1,026 | 549 -> 566 | 1,736 -> 1,756 | 8,453 -> 8,453 |
| rsv-a | test | 1080 -> 1080 | 1,932 -> 1,943 | 913 -> 920 | 3,012 -> 3,023 | 18,023 -> 18,025 |
| denv2 | train | 661 -> 661 | 1,151 -> 1,163 | 827 -> 836 | 1,812 -> 1,824 | 4,663 -> 4,663 |
| denv2 | test | **1486 -> 1488** | 1,896 -> 1,918 | 1,305 -> 1,315 | 3,382 -> 3,406 | 12,196 -> 12,196 |

The two scores that move are exactly the ones predicted: rsv-a's 2023 train tree
gains 20, its 19 `-` mutations plus its one `*`, which were free before; denv2's
test tree gains 2, the `*122` pair. h3n2's trees carry neither and do not move.

The gap state is **sparse on the queue side for these three pathogens**: the
22-column count matrices hold 122 / 0 / 15 gap cells on the train side and
268 / 28 / 19 on the test side, against 52.7M, 4.8M and 2.3M residues. Where the
alphabet earns its keep here is the backbone (rsv-a's 19 F deletions) -- and,
when it is ever run, sc2, where 88% of coverage-passing spike sequences carry a
deletion and the internal runs reproduce the published ones to four significant
figures (Δ156-157 166,209; Δ69-70 147,880; Δ24-26 133,771; Δ142-144 58,468;
Δ211 57,799; Δ144 44,390; Δ241-243 577; and exactly three runs longer than 15,
each seen once).

Exact-match rates barely move (h3n2 81.06% -> 81.04%, rsv-a 90.65% -> 90.63%,
denv2 87.65% -> 87.59%) and the trees grow by 0.1%. The cost NOTES:222-227
predicts for sc2 does not materialise here, because so few queue records carry a
kept gap.

Making the alphabet width a runtime value rather than a `constexpr` costs
**1-3% on the placement scan**, measured on the h3n2 train placement with the
same queue and seed, alternating binaries on an otherwise idle node: the old
`work/antigen_arms` at 29.8 s and 29.9 s against the new binary at 30.2 s and
30.8 s. That is the price of the whole option and it is paid whether or not it
is used.

### The clock moved by one substitution, in a few months

A deletion now counts toward `dRoot`, so the envelope is built from a slightly
larger quantity. The effect is one substitution of band width in the recent
months of two arms -- rsv-a 2023 `band_hi` 58 -> 59, denv2 2023 `band_hi`
18 -> 19 -- and the filters move by at most ten records (h3n2 test: 588 -> 598
too diverged, 8 -> 7 too ancestral; rsv-a test: 671 -> 670 fail coverage). The
envelope is quantiles of the same quantity the record is tested against, so band
and record move together; this is the same argument `pinnedRoot` rests on.

### What was checked

1. **The 20-token default is bit-identical.** `sets --variant BA.2` on the sc2
   2026-03-15 open tree with all 24 cutoffs gives the same md5 on `tree`, `sets`
   and `examples` as the reference binary `work/antigen_all`
   (`5c9c85af59a32059f78cd8b3d9e8b1b1`). `prepare --date-cutoff 2020-04-01` over
   the 12 GB queue reproduces 28,622 attached / 94.46%, a bit-identical clock
   file, and a prepared file identical apart from the one new `#alphabet` header
   line. Re-running all six arms and all three backbone arms at the default left
   **all 42 output files bit-identical** and all nine checkers at exit 0.
2. **The invariant holds on all six 22-token arms**: `parsimony score + added ==
   mutations written`, which `writeExamples` asserts, and `check_examples.py
   --alphabet 22` exits 0 on all six.
3. **The count matrices are 22 columns and correct**: re-tallying each train MSA
   by hand into a (site x 22) matrix reproduces the adjacent
   `counts_<D>_cur.tsv` cell for cell on all three, with record counts equal to
   `attached` and every line at the gene's width.
4. **The flags refuse what they should**: `--alphabet 21`, `--max-gap-run` on
   `sets`, `--max-gap-run` at 20 tokens, and -- the one that matters -- feeding a
   20-token `prepared` file to a 22-token `place`, which the `#alphabet` header
   catches.

### Still open

- `--max-gap-run 20` instead of 10, per the Nextclade comparison above. One flag,
  one re-run.
- Masking gap runs flanked by `X`. In the h3n2 / rsv-a / denv2 queues 92% / 100%
  / 81% of short runs sit within 3 positions of an `X`, against 0.3% in sc2 --
  consistent with Nextclade's `protect_codon_in_place` (a partly-gapped codon is
  forced to `X`) and `mask_peptide_frame_shifts_in_place` (a frameshift is
  written as an `X` run flanked by `-`). Left out as a third non-standard
  criterion for a few hundred runs; this is the thing to try if the gap state
  ever looks noisy.
- There is no 22-token backbone arm. `slurm/run_arms_backbone.sbatch` plus
  `--alphabet 22` is all it would take.
- sc2 has not been run at 22 tokens. It is where the change actually bites.


## Directive, 2026-10-30 (second): the same three datasets at a 2025-01-01 split

> Regenerate the same datasets (with attachment in the testing data), just like
> before, but the only difference being a 2025-01-01 split. You'll need to first
> source the correct Nextstrain builds from 2025-01-01, or as close to that date
> as possible without exceeding it.

20-token alphabet, `place` on both sides, `--post-cutoff 2025-01-01` on the test
side: the run_arms shape exactly, with a different table and a different cutoff.

```
slurm/arms_2025.tsv          the three rows
slurm/run_arms_2025.sbatch   sbatch --array=0-2, or `bash ... 0` by hand
output/<arm>-2025-01-01/train/{examples,msa,tree,sets,counts,clock}_2025-01-01_cur.*
output/<arm>-2025-01-01/test/{examples,tree,sets,counts,clock}_all_cur.*
```

The runner differs from `run_arms.sbatch` in exactly three lines -- the table it
reads and the two output directories. **The test side had to move.**
`output/<arm>-test` is not cutoff-scoped, and a test examples file carries the
flags for exactly one cutoff, so writing there would have overwritten the
2022/2023/2019 test sets with differently flagged files. Both sides now sit
under one cutoff-named directory, which is the `output/sc2-2021/{train,test}`
layout.

### Sourcing the builds: the sidecar key is dead, the root sequence is inline

`fetch_snapshots.py` gained `--cutoff D`, which overrides all three arms at
once, and its manifest now **upserts** on (arm, cutoff) instead of being
rewritten, so `data/snapshots_manifest.tsv` is the record of every snapshot ever
fetched. The three 2019/2022/2023 rows are byte-identical to what they were.

| arm | key | build | lag | tips | gene mutations | post-cutoff tips |
|---|---|---|---|---|---|---|
| h3n2 | `seasonal-flu_h3n2_ha_12y.json` | 2024-12-29 | 3 d | 1,576 | 1,212 | 0 |
| rsv-a | `rsv_a_genome_all-time.json` | 2024-12-24 | 8 d | 2,901 | 1,076 | 0 |
| denv2 | `dengue_denv2_genome.json` | 2024-12-31 | 1 d | 2,555 | 1,074 | 0 |

The first fetch produced **two broken sidecars**, and the reason is worth
recording. The pick is still "the latest build strictly before D", so the trees
were right; the sidecars were not. `antigen tree` on them reported 3 from-allele
mismatches for rsv-a and **66 for denv2, 28 of which were real conflicts** --
the path carrying a third allele, not merely the one the branch was moving to.

The cause: **Nextstrain stopped publishing `<key>_root-sequence.json` during
2024.** Listing the bucket, dengue's last is 2024-01-16 and RSV's 2024-05-21,
while the tree keys themselves run to today. The fetcher's "latest sidecar at or
before the build" rule therefore reached back seven and eleven months, and the
root-sequence sidecar is not a fixed reference -- it is *that build's inferred
root*, which moves every rebuild.

What replaced it is a top-level `"root_sequence"` **inside the tree JSON**. All
three 2024-12 builds carry one; none of the 2019/2022/2023 builds do. So
`fetch_snapshots.py` now prefers the inline one and falls back to the sidecar
key only when there is none -- which leaves the three older rows on exactly the
path they were fetched by. With the inline root sequence all three new trees
report **zero** from-allele mismatches.

This supersedes the README's "Every snapshot needs its own sidecar" for builds
from mid-2024 on: the snapshot still needs its own root sequence, but it now
arrives in the tree file rather than beside it.

### Results

These are the numbers **after** the test-backbone refresh below; the test side
was run twice and only the second run is on disk.

| arm | side | tree nodes/tips | queue | cov fail | clock hi/lo | attached | % | final nodes | records | mutations |
|---|---|---|---|---|---|---|---|---|---|---|
| h3n2 | train | 3,082 / 1,576 | 157,303 | 148 | 576/0 | 156,140 | 99.26 | 62,704 | 16,639 | 41,715 |
| h3n2 | test | 3,318 / 1,698 | 201,266 | 187 | 588/8 | 201,266 | 99.39 | 79,570 | 19,917 | 51,899 |
| rsv-a | train | 5,717 / 2,901 | 14,721 | 468 | 86/0 | 14,167 | 96.24 | 8,403 | 843 | 2,646 |
| rsv-a | test | 5,748 / 2,898 | 18,780 | 671 | 86/0 | 18,023 | 95.97 | 9,142 | 966 | 3,092 |
| denv2 | train | 4,690 / 2,555 | 12,877 | 1,780 | 61/2 | 11,011 | 85.51 | 7,746 | 1,165 | 3,005 |
| denv2 | test | 7,235 / 3,958 | 14,715 | 2,444 | 47/2 | 12,199 | 82.90 | 10,255 | 1,320 | 3,425 |

`score + added == mutations written` on all six (1501+40214, 1708+50191,
1063+1583, 1154+1938, 1073+1932, 1535+1890), all six checkers exit 0, and
POST_CUTOFF is empty on every train record.

### The 2025-01-01 test sets are small, and that is structural

| arm | queue dated at or after D | flagged example records | the same at the old split |
|---|---|---|---|
| h3n2 | 45,126 | 3,001 of 19,917 | 7,974 at 2022-10-01 |
| rsv-a | 3,856 | **71** of 966 | 316 at 2023-10-01 |
| denv2 | 1,174 | **27** of 1,320 | 419 at 2019-10-01 |

The sequences are there; the tree structure to hold them is not. POST_CUTOFF is
`every observed date below the node is at or after D`, so a post-2025 sequence
that matches a node already carrying pre-2025 leaves is counted there and flags
nothing. How much post-cutoff structure each backbone has decides the rest:

| test backbone | tips | dated 2025 or later | newest tip |
|---|---|---|---|
| h3n2 `tree_2026-10-29` | 1,698 | 261 (15.4%) | 2026.76 |
| rsv-a `tree_2026-10-27` | 2,898 | 232 (8.0%) | 2026.52 |
| denv2 `tree_2026-10-29` | 3,958 | **80 (2.0%)** | 2026.66 |

denv2's test backbone holds 80 tips from 2025 or later out of 3,958, so there is
very little 2025 clade for its 1,174 post-cutoff sequences to form, and 27
records is what falls out. **The denv2 arm at this cutoff is probably too small
to train or evaluate against**; rsv-a's 71 is marginal. h3n2 at 3,001 is
comfortable. Refreshing the backbone was tried and is not the fix -- see below.

### What was checked

- The three new train trees report "every branch's from-allele matches the path
  above it" -- after the inline-root-sequence fix, and not before it.
- Every output file outside `output/<arm>-2025-01-01/` is bit-identical; the
  only files this directive wrote are that directory's 17.
- Re-tallying each train MSA into a (site x 20) matrix reproduces the adjacent
  `counts_2025-01-01_cur.tsv` cell for cell on all three, with record counts
  equal to `attached` (156,140 / 14,167 / 11,011) and every line at the gene's
  width.
- The three older manifest rows are unchanged.

### Refreshing the test backbones, 2026-10-30

The test backbones in the folders were `~/h3n2/tree_2026-09-24.json` (five weeks
old), `~/rsv-a/tree.json` at meta.updated 2026-04-21 and `~/denv2/tree.json` at
2026-04-22 -- both six months old. `fetch_snapshots.py` gained `--latest`, which
drops the date bound and takes the newest version there is, writing its manifest
rows under the key `latest`. The three it pulled are the **same keys the train
snapshots came from**, so each arm is now one build family end to end:

| arm | key | build | tips | from-allele check |
|---|---|---|---|---|
| h3n2 | `seasonal-flu_h3n2_ha_12y.json` | 2026-10-29 | 1,698 | clean |
| rsv-a | `rsv_a_genome_all-time.json` | 2026-10-27 | 2,898 | clean |
| denv2 | `dengue_denv2_genome.json` | 2026-10-29 | 3,958 | clean |

All three carry their root sequence inline, so the sidecar question does not
arise. Over the headline gene the new h3n2 and rsv-a root sequences are
**identical** to the folders' `ref.json`, which is why their `counts` and
`clock` files did not move; denv2's differs at 10 of 495 residues, which shifts
its divergence a little and with it the clock (47 rather than 50 too diverged,
12,199 rather than 12,196 attached).

**It is not the fix for the small test sets.** Six months of new builds moved
the post-cutoff record counts from 2,992 / 58 / 19 to 3,001 / 71 / 27. The
reason is that these builds subsample across their whole time range rather than
weighting the present: denv2's tree grew by 20 tips in six months, of which the
2025-or-later share went from 22 to 80 out of 3,958. A bigger denv2 test set
needs a build that keeps more recent tips, or a cutoff further from the end of
the data -- not a fresher copy of the same build.

The 2022/2023/2019 arms in `slurm/arms.tsv`, the backbone-only arms and the
22-token arms still point at the older test trees and were **not** re-run.

## Directive, 2026-11-09: benchmark against UShER, runtime and parsimony

Two plots, both against the number of sequences attached: runtime, and parsimony
score, with our tool and UShER each a line. SARS-CoV-2 whole genomes, so the tool
first needed a nucleotide mode. UShER on its fastest settings, every expensive
step skipped on both sides.

### The nucleotide mode

- `--alphabet nuc`: A, C, G, T as four states (`src/alphabet.hpp`,
  `AlphabetKind::Nuc`). N, `-`, lowercase and the IUPAC partial codes are all
  missing. A partial code is read as no constraint rather than as a 2- or 3-state
  mask: sound, since it only widens what a node may hold, and it keeps every
  observation one state or none, which `hartigan.cpp`'s pinned internal node and
  `placement.cpp`'s `writeLeafSets` both assume. Masks would need those two and
  `recomputeV12` reworked. The benchmark samples have no partial codes, no N and
  no gaps at all (below), so nothing is lost here.
- `--gene nuc` needed no change: the tree reader and the sidecar reader already
  take any key, and Nextstrain trees carry `nuc` mutations and a 29,903-nt `nuc`
  root.
- `prepare --min-covered <f>` (the 0.9 coverage fraction was hard-coded);
  `prepare/place --keep-order` (no date sort, no within-day shuffle -- the FASTA's
  order is the placement order); `place --progress-every <n>` (the progress line
  now carries the running parsimony score); `place --verify-final` (one full
  recompute after the last placement, for any run size).

### Two hot spots, both fixed exactly

- `attachLeaf`/`splitBranch` handed `refresh()` every site where the new leaf
  differs from the root, missing sites included. A missing leaf has V1 = every
  state, which adds one to every count at its parent: best rises by one and the
  states at best and best - 1 are unchanged, so V1/V2 hold and the upward pass
  stops on its first comparison -- but that comparison loops over every child of
  the parent. `liveSites()` drops them before `refresh()`. In `splitBranch` the
  new node u is still written at every touched site (a missing L leaves V2(u) off
  its default); only the refresh of p skips them, since p reads V1(u) = V1(v).
- The end-of-`place` ambiguity summary walked nodes x sites. It now walks only the
  sites that are active or whose root set is not a single state; a site with one
  allele observed is that allele everywhere and adds exactly one per node to the
  size-1 count.

**Bit-identity, checked.** Old binary (`42daeeae`, the one every current output
was made with) against new, same commands and inputs: sc2 2021-01-01 train arm (20
tokens), h3n2 test side (HA1+HA2, post-cutoff flags), denv2 test side (22 tokens,
month imputation). All 18 output files byte-identical. Log differences are only
the progress line (now with the score), the scan's node-visit counter (+-0.05%, it
depends on when parallel workers see the first exact match), and fewer upward steps
(sc2: 362,409 -> 217,994) with downward steps and set writes unchanged -- which is
the skipped no-op comparisons and nothing else. The ambiguity block is identical.

### The benchmark (`bench/`, data in `/large_storage/hielab/ivan.specht/usher-bench/`)

Reuses the earlier comparison's data (`~/antiGen-data/bench/`): samples from UCSC
`public-latest.all.masked.pb` (2026-07-03), 1M dated samples in a fixed md5 order,
every N a prefix of that list; 3k/30k/300k added with the same `matUtils extract` +
`reconstruct_fasta.py` path (`bench/prep_inputs.sbatch`). Guide tree
`~/sc2-open-data/tree.json` (4,549 tips); UShER's `guide.pb` was built from it with
`-c`, so identical sequences are condensed.

- **The samples carry no missing data.** All 299,030,000 characters of the 10k
  FASTA are A/C/G/T: the genotypes come from the UCSC tree, where missing calls
  were imputed at placement. Only the guide tips have missing data (Nextstrain
  gaps, `.` in `guide.vcf`), read as missing by both tools.
- **Settings.** 32 threads for every run, fixed in the script (SLURM hands out more
  CPUs with more memory, and two early jobs ran at 50 and 75 threads before this
  was pinned). Each N runs all its tools back to back in one allocation, since the
  nodes are shared. usher-sampled with `--optimization_radius 0
  --optimization_minutes 0 --last_optimization_minutes 0`, MPI as a singleton
  (SLURM/PMI variables unset). Classic usher to 30k only.
- **One scorer for every tree.** Each final tree goes to Newick (ours: the
  `--out-tree` TSV plus every matched sequence put back as a zero-length child of
  its node, `bench/ours_to_newick.py`; UShER: `matUtils extract -t`, uncondensed),
  and UShER's own Fitch-Sankoff re-scores it over one VCF of every tip
  (`bench/merge_vcf.py`: guide tips + samples, a position absent from one file is
  REF there). Self-reported scores are kept alongside.
- **The scorer and ours agree.** On our tree without the matched sequences the
  common score equals our incremental score exactly (30k: 129,519 both), and
  `--verify-final` confirms the incremental score and every set against a full
  recompute. UShER scores the bare guide at 23,486 without `-c` -- our number --
  and 23,483 with it: the collapse gives UShER a 3-mutation head start.
- **The skip rule is exact when it fires, not afterwards.** With the matched
  sequences put back the 30k tree scores 129,520, one more: a sequence counted at
  node v because it equalled W(v) then, whose v's sets later placements changed.
  The cost of dropping it is that later placements never see it. 1 in 129,520.
- **matUtils and usher prefix their output directory to output paths**, so an
  absolute output path becomes `.//large_storage/...`, relative, and the write fails
  without an error. Every call runs from inside its directory with bare names.
- **usher-sampled is not deterministic**: it commits parallel searches in completion
  order. 1k at 16 threads scored 28,902, at 32 threads 29,073-29,107 over three
  runs. Three runs per N, plotted as mean and range.
- **usher-sampled cannot place 1M in one run** (v0.6.6): "node id Mismatch" then a
  segfault within a minute, before any placement, optimization off -- as in the
  earlier comparison with it on. At 1M it runs as 10 chained 100k batches, each
  onto the tree the last saved, times summed (`ushsb`).
- **The inputs were in UCSC tree order.** `matUtils extract` writes samples in the
  order of its tree, not of the sample list (rank correlation -0.006 with the md5
  list; the 10k file opens with March 2020 genomes of one clade). Fair within one
  N, but each sample lands next to the last, which flatters locality -- our match
  cache above all. Directive: place chronologically, with a random tie-break within
  a day. `bench/reorder_inputs.py` computes that order once (date, then
  md5(seed + name)) and writes it into the FASTA records and the VCF columns
  alike; our tool reads with `--keep-order`, UShER in VCF order, and the 1M/300k
  batches are consecutive 100k slices of the same order. The tree-order results
  (1k-100k) are kept in `bench/treeorder/` and `results_treeorder/`, not plotted.

## Directive, 2026-11-10: seed stability of the sc2 mutation trees (dropped 2026-11-11)

Dropped in favour of measuring the seed's effect on model performance directly,
which is the question that matters. Kept here because the runs are done and the
findings are not obvious.

Five seeds per arm, amino-acid sc2 train arms, the series' exact commands
(`slurm/seed_study.sbatch`, analysis `seed_study.py`, outputs in
`output/seed_study/`). Seed 1 reproduces the shipped `output/sc2-<D>/train`
examples and tree byte for byte at both arms; every run passes the checker.
`--examples-seed <n[,n...]>` was added to `place` so the resolution (allele draw,
mutation order) can be seeded apart from the placement; without it nothing
changes (2021-01-01 reproduced exactly with no flag, with `--examples-seed 1`,
and with a list).

"Recurs" = another seed's tree has a branching node (2+ children) with the
identical genotype; weighted = by mutations below the node.

| arm | what varies | weighted recurrence | parsimony range |
|---|---|---|---|
| 2021-01-01 | everything | 94.5-96.4% | 14,214-14,236 (0.15%) |
| 2021-01-01 | placement only | 94.3-96.5% | same |
| 2021-01-01 | resolution only | 99.1-99.7% | 14,233 all |
| 2023-01-01 | everything | 40.2-52.1% | 244,186-244,515 (0.13%) |
| 2023-01-01 | placement only | 39.9-52.0% | same |
| 2023-01-01 | resolution only | 91.9-95.2% | 244,460 all |

- The parsimony score barely moves with the seed (sd 11 at 2021, 139 at 2023).
- The variation is placement: ~91% of non-matching placements are ties between
  different places (hundreds of options), and different tie choices give
  equally parsimonious but different trees. At 2021 that is confined to the
  tips (small clades recur 44%, clades with >20 mutations below 86%); at 2023 it
  reaches deep nodes (44% even above 100 mutations below).
- An earlier reading of mine -- that the 2023 instability came from the 3,717
  random allele draws at resolution (vs 95 at 2021) -- was wrong: with the tree
  held fixed, resolution alone leaves 92-95% recurring.
- Masking ambiguous sites is not a fix: at 2023 it merges 17,824 distinct
  branching genotypes into 1,226. A "within one amino acid" match is not one
  either: nearly every genotype is one mutation from its own parent.
- `--tie-seed` alone and "everything but ties" give the same instability: a
  changed queue order makes the same tie stream fall on different placements.
