#!/usr/bin/env python
"""
Simple fitness training: predict which mutations were observed from each ancestral sequence.
"""

import os
import gc
import random
import math

import argparse
import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from antiGen.config import ModelConfig, ModelType, TransformerConfig, AlphabetConfig, ESM2Config, EVcouplingsConfig, EVEConfig, TranceptionConfig, PSSMConfig, GatedFitnessConfig, ResidualConfig
from antiGen.models.factory import create_model
from antiGen.load import load_examples, split_by_window
from antiGen.eval_utils import (compute_probs, compute_loss_batch, compute_conditional_loss_batch,
                                compute_unconditional_loss_batch, compute_kl_loss_batch)
from antiGen import datasets


# ── Dataset ───────────────────────────────────────────────────────────────────

# How the descendant counts are weighted before they reach a count-based loss. The raw
# counts span five orders of magnitude, so under the plain conditional loss the ~100
# largest examples carry almost all the gradient (Kish effective sample size ~22 of 3,092
# test examples with two or more substitutions). A sublinear transform sits between that
# and weighting every mutation equally: log1p takes the effective sample size to ~123,
# sqrt to ~116. Note it flattens the target split within an example as well as the weight
# across examples, so the model is fitting a compressed version of the observed split --
# evaluation stays on raw counts.
COUNT_TRANSFORMS = {
    'none': lambda n: n,
    'log1p': torch.log1p,
    'sqrt': torch.sqrt,
}


class MutationDataset(Dataset):
    """Yields ``(aa_idx, mat, n_mat, cur_mat)``.

    ``n_mat`` holds the descendants behind each child substitution -- the training target
    -- put through ``count_transform``. It is all-zero for examples files written before
    that field existed; only the count-based losses read it.

    ``cur_mat`` holds the same counts one time window earlier, i.e. descendants collected
    in the example's *own* window rather than the next one. Only interval files carry it
    and only the residual model reads it, so it is all-zero otherwise. It is deliberately
    NOT put through ``count_transform``: that transform reweights the training target, and
    this is a model input.
    """

    def __init__(self, training_examples, count_transform='none'):
        fn = COUNT_TRANSFORMS[count_transform]
        self.aa_indices_list = [e.aa_idx for e in training_examples]
        self.mutation_matrices = [e.mat for e in training_examples]
        self.count_matrices = [
            fn(e.n_mat) if e.n_mat is not None else torch.zeros_like(e.mat)
            for e in training_examples
        ]
        self.own_window_matrices = [
            e.cur_mat if e.cur_mat is not None else torch.zeros_like(e.mat)
            for e in training_examples
        ]

    def __len__(self):
        return len(self.aa_indices_list)

    def __getitem__(self, idx):
        return (self.aa_indices_list[idx], self.mutation_matrices[idx],
                self.count_matrices[idx], self.own_window_matrices[idx])


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train fitness model on observed mutations')
    parser.add_argument('--cutoff', type=float, default=1.0, help='Train/test cutoff (years since MRCA)')
    parser.add_argument('--ignore-dead-ends', action='store_true', default=False, help='Do not make examples for sequences with no downstream mutants')
    parser.add_argument('--loss', type=str, default='multinomial',
                        choices=['multinomial', 'conditional', 'unconditional', 'kl'],
                        help="Training objective; the architecture and forward pass are the same "
                             "for all three. 'multinomial' (default, unchanged): each observed "
                             "child substitution is a draw over the whole position x AA grid with "
                             "wild-type masked out, so the model learns which substitutions occur. "
                             "'conditional': condition on the substitutions that did occur and learn "
                             "only how their descendants divide between them -- each of the n_i "
                             "sequences below mutation m_i is an i.i.d. categorical draw over "
                             "m_1..m_k, renormalized over those k cells. 'unconditional': the same "
                             "descendant counts but drawn over the whole grid, i.e. the multinomial "
                             "loss reweighted from child counts to descendant counts. 'kl': the "
                             "conditional split again, but with the counts normalized per example "
                             "first -- KL(n_i/sum(n) || predicted), averaged over the examples with "
                             "at least two substitutions. Same gradients as 'conditional' except "
                             "every example counts once instead of once per descendant. All three "
                             "count-based losses need an examples file carrying descendant counts, "
                             "and under 'conditional'/'kl' an example with one child substitution "
                             "gives no gradient at all.")

    parser.add_argument('--model-type', type=str, default='transformer', choices=['transformer', 'esm2', 'blosum62', 'pssm', 'evcouplings', 'eve', 'tranception', 'gated', 'residual'],
                        help='Model architecture: from-scratch transformer, ESM-2 finetune, BLOSUM62 (eval only), PSSM (eval only), EVcouplings (eval only), EVE VAE, Tranception LM, gated blend of two predictors, or residual (persistence counts + learned sequence residual; needs an --intervals examples file)')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension (transformer only)')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers (transformer only)')
    parser.add_argument('--esm2-model-name', type=str, default='facebook/esm2_t12_35M_UR50D',
                        help='HuggingFace ESM-2 model name (esm2 only)')
    parser.add_argument('--freeze-backbone', action='store_true', default=False,
                        help='Freeze pretrained backbone, only train output head (esm2 only)')
    parser.add_argument('--finetune-layers', type=int, default=0,
                        help='Number of last encoder layers to unfreeze (requires --freeze-backbone, esm2 only)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-batches', type=int, default=None, help='Stop after N batches total')
    parser.add_argument('--warmup-steps', type=int, default=0, help='LR warmup steps')
    parser.add_argument('--count-transform', type=str, default='none',
                        choices=['none', 'log1p', 'sqrt'],
                        help="Transform applied to the descendant counts before they enter a "
                             "count-based loss (no effect on --loss multinomial). 'none' keeps "
                             "the raw counts, which means the handful of examples with the most "
                             "descendants carry nearly all the gradient. 'log1p' and 'sqrt' "
                             "compress that: they sit between raw counts and weighting every "
                             "mutation equally. The transform also flattens the target split "
                             "inside an example, so the model fits a compressed version of the "
                             "observed proportions -- evaluation is still on raw counts.")
    parser.add_argument('--residual-alpha', type=float, default=1.0,
                        help="Pseudocount in the persistence term of --model-type residual: "
                             "s = beta*log(c + alpha) + lam*f(seq). Keeps log(c + alpha) "
                             "finite for the c = 0 cells, which are common. Small alpha makes "
                             "the prevalence term near-deterministic on this window's winner, "
                             "large alpha collapses it toward uniform. Swept, not learned.")
    parser.add_argument('--residual-beta-init', type=float, default=1.0,
                        help="Initial prevalence exponent. 1.0 is pure replicator dynamics; a "
                             "trained value below 1 means large lineages saturate.")
    parser.add_argument('--residual-lambda-init', type=float, default=0.0,
                        help="Initial sequence-residual gate. 0.0 starts the model exactly at "
                             "the persistence baseline, so any improvement is attributable to "
                             "sequence -- at the cost of the inner network seeing no gradient "
                             "until lam moves off zero. Set small-but-nonzero to skip that.")
    parser.add_argument('--freeze-beta', action='store_true', default=False,
                        help='Hold the prevalence exponent at its initial value (residual only).')
    parser.add_argument('--freeze-lambda', action='store_true', default=False,
                        help='Hold the sequence-residual gate at its initial value (residual '
                             'only). With --residual-lambda-init 0 this is the pure persistence '
                             'baseline with a learned beta and nothing else -- the ablation that '
                             'isolates saturation from sequence.')
    parser.add_argument('--checkpoint-every-epoch', action='store_true', default=False,
                        help='Also write <output-dir>/epochs/epoch<NNN>.pt after every epoch. '
                             'Lets an evaluator walk the whole training curve (e.g. to find '
                             'where a held-out metric bottoms out) instead of only seeing the '
                             'final weights. Costs one file per epoch.')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--examples', type=str, default=None,
                        help='Path to an examples_cutoff_*.txt file. Overrides dataset routing '
                             '(use this to train on any protein/pathogen without code edits).')
    parser.add_argument('--test-window', type=int, default=None, metavar='YEAR',
                        help="Split an --intervals examples file on its time windows instead "
                             "of on the TRAINING/TESTING label (which is TRAINING for every "
                             "example in such a file). Window YEAR is held out for testing and "
                             "every earlier window trains; later windows are dropped, since the "
                             "generator emits the most recent window against a partially "
                             "observed successor year and its counts are undercounts.")
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing examples_cutoff_{cutoff}.txt for a named '
                             'dataset (else resolved from $FITNESS_DATA_ROOT).')
    parser.add_argument('--date-cutoff', type=str, default=None,
                        help='YYYY-MM-DD MSA date cutoff for evcouplings/eve/tranception '
                             '(else derived from the dataset MRCA + --cutoff years).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hit-rate-buckets', type=int, default=10,
                        help='Number of quantile buckets for hit-rate evaluation')
    parser.add_argument('--variant', type=str, default=None,
                        help='Variant name (default: BA.2 for SARS-CoV-2, J for flu)')
    parser.add_argument('--flu', action='store_true', default=False,
                        help='Flu mode: uses flu data, full-sequence eval, variant J or K')
    parser.add_argument('--rsv-a', action='store_true', default=False,
                        help='RSV-A mode: uses RSV-A data, full-sequence eval, variant A.D.1.6')
    parser.add_argument('--denv2', action='store_true', default=False,
                        help='DENV2 mode: uses DENV2 data, full-sequence eval, variant 2II_F.1.1.2_annotated')
    parser.add_argument('--msa', type=str, default=None,
                        help='Path to pandemic MSA FASTA (required for evcouplings/eve)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata TSV with strain/date columns (required for evcouplings/eve)')
    parser.add_argument('--max-seqs', type=int, default=0,
                        help='Max MSA sequences for evcouplings/eve/tranception (0=unlimited)')
    parser.add_argument('--evc-pseudo-count', type=float, default=0.5,
                        help='Pseudo-count for EVcouplings mean-field DCA regularization (default: 0.5)')
    parser.add_argument('--scoring-direction', type=str, default='both',
                        choices=['forward', 'backward', 'both'],
                        help='Autoregressive scoring direction for Tranception (default: both)')
    _partner_choices = ['fitness_predictor', 'esm2_fitness_predictor',
                        'blosum62', 'pssm', 'evcouplings', 'eve', 'tranception']
    parser.add_argument('--gate-partner-a', type=str, default='fitness_predictor',
                        choices=_partner_choices,
                        help='ModelType.value of the first partner for --model-type gated '
                             '(reuses that model type\'s CLI flags).')
    parser.add_argument('--gate-partner-b', type=str, default='blosum62',
                        choices=_partner_choices,
                        help='ModelType.value of the second partner for --model-type gated.')
    parser.add_argument('--gate-variant', type=str, default='static',
                        choices=['static', 'uncertainty'],
                        help='Gate module: "static" (per-position learnable, data-blind) or '
                             '"uncertainty" (MLP on partner softmax entropies).')
    return parser.parse_args()


# ── Run-identity helpers ──────────────────────────────────────────────────────

_PARTNER_SHORT = {
    'esm2_fitness_predictor': 'esm2',
    'fitness_predictor': 'xfmr',
    'blosum62': 'blosum62',
    'pssm': 'pssm',
    'evcouplings': 'evc',
    'eve': 'eve',
    'tranception': 'tran',
}


def compute_run_family(args) -> str:
    """Stable run-family identifier (no timestamps) for grouping outputs.

    Non-gated: the model_type. Gated: gated_<gate_variant>_<partner_a>_<partner_b>.
    """
    if args.model_type == 'gated':
        a = _PARTNER_SHORT.get(args.gate_partner_a, args.gate_partner_a)
        b = _PARTNER_SHORT.get(args.gate_partner_b, args.gate_partner_b)
        return f"gated_{args.gate_variant}_{a}_{b}"
    return args.model_type


def write_manifest(args, model_config, output_dir, run_prefix) -> None:
    """Write manifest.json next to the run's outputs capturing the full recipe.

    Aggregators consume this rather than guessing from filenames, so new gate
    variants / model combos don't break discovery.
    """
    import json
    import subprocess
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    manifest = {
        "schema_version": 1,
        "run_family": compute_run_family(args),
        "run_prefix": run_prefix,
        "model_type": args.model_type,
        "model_config": model_config.to_dict(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_batches": args.max_batches,
            "warmup_steps": args.warmup_steps,
            "loss": args.loss,
            "count_transform": args.count_transform,
            "residual_alpha": args.residual_alpha,
            "residual_beta_init": args.residual_beta_init,
            "residual_lambda_init": args.residual_lambda_init,
            "freeze_beta": args.freeze_beta,
            "freeze_lambda": args.freeze_lambda,
            "freeze_backbone": args.freeze_backbone,
            "finetune_layers": args.finetune_layers,
        },
        "data": {
            "cutoff_years": args.cutoff,
            "data_dir": args.data_dir,
            "examples": args.examples,
            "test_window": args.test_window,
            "ignore_dead_ends": args.ignore_dead_ends,
            "variant": args.variant,
            "dataset_mode": (
                "flu" if args.flu else
                "rsv_a" if args.rsv_a else
                "denv2" if args.denv2 else
                "sc2"
            ),
        },
        "git_commit": git_commit,
    }
    if args.model_type == 'gated':
        manifest["gating"] = {
            "gate_variant": args.gate_variant,
            "partner_a": args.gate_partner_a,
            "partner_b": args.gate_partner_b,
        }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    allowed_aas = "ACDEFGHIKLMNPQRSTVWY*-"
    E = len(allowed_aas)
    print(f"Alphabet size: {E}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if sum([args.flu, args.rsv_a, args.denv2]) > 1:
        raise ValueError("Cannot specify more than one of --flu, --rsv-a, --denv2")

    # Dataset name (for defaults + MSA date-cutoff). Custom data via --examples still uses
    # 'sc2' defaults for the (train-irrelevant) variant unless overridden.
    dataset_name = 'flu' if args.flu else 'rsv_a' if args.rsv_a else 'denv2' if args.denv2 else 'sc2'
    if args.variant is None:
        args.variant = datasets.default_variant(dataset_name)
    cutoff_int = int(args.cutoff)

    if args.examples:
        # Fully custom: train on any examples file (new protein/pathogen, zero code edits).
        examples_path = args.examples
        data_dir = args.data_dir  # may be None; pssm/gated partners need it (pssms/root.csv)
    else:
        datasets.validate(dataset_name, cutoff_int, args.variant)
        data_dir = args.data_dir or datasets.default_data_dir(dataset_name, cutoff_int)
        examples_path = datasets.examples_path(data_dir, cutoff_int)
    print(f"Loading examples from {examples_path}...")
    all_examples = load_examples(examples_path, allowed_aas, args.variant)


    seq_len_aa = len(all_examples[0].aa_idx)
    print(f"Sequence length: {seq_len_aa}")

    training_examples = []
    testing_examples = []
    total_mutations_train = 0.0

    if args.test_window is not None:
        train_pool, test_pool, _ = split_by_window(all_examples, args.test_window)
    else:
        train_pool = [e for e in all_examples if e.is_training]
        test_pool = [e for e in all_examples if not e.is_training]

    for example in train_pool:
        if args.ignore_dead_ends and example.mat.sum() < 0.5:
            continue
        training_examples.append(example)
        total_mutations_train += example.mat.sum()
    for example in test_pool:
        if args.ignore_dead_ends and example.mat.sum() < 0.5:
            continue
        testing_examples.append(example)

    print(f"Training examples: {len(training_examples)}, Testing examples: {len(testing_examples)}")
    print(f"Total mutations in training set: {total_mutations_train:.0f}")


    train_dataset = MutationDataset(training_examples, args.count_transform)
    test_dataset = MutationDataset(testing_examples, args.count_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    # Per-batch test-loss monitoring is optional: a small or high-cutoff training set
    # can have no TESTING examples, in which case we skip it (an empty DataLoader would
    # raise num_samples=0).
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=False,
        )
        test_iter = cycle(test_loader)
    else:
        test_iter = None
        print("No TESTING examples found — skipping per-batch test-loss monitoring.")

    print(f"Train batches per epoch: {math.ceil(len(train_dataset) / args.batch_size)}")

    allowed_aas_str = allowed_aas
    alphabet_config = AlphabetConfig(amino_acids=allowed_aas_str)

    if args.model_type == 'esm2':
        model_config = ModelConfig(
            model_type=ModelType.ESM2_FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            esm2=ESM2Config(
                model_name=args.esm2_model_name,
                freeze_backbone=args.freeze_backbone,
                finetune_layers=args.finetune_layers,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: ESM-2 ({args.esm2_model_name}), freeze_backbone={args.freeze_backbone}, finetune_layers={args.finetune_layers}")
    elif args.model_type == 'blosum62':
        model_config = ModelConfig(
            model_type=ModelType.BLOSUM62,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
        )
        model = create_model(model_config, device)
        print("Model: BLOSUM62 (no training, eval only)")
    elif args.model_type == 'pssm':
        model_config = ModelConfig(
            model_type=ModelType.PSSM,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            pssm=PSSMConfig(data_dir=data_dir),
        )
        model = create_model(model_config, device)
        print(f"Model: PSSM log(count+1) (no training, eval only) — source: {data_dir}/pssms/root.csv")
    elif args.model_type == 'evcouplings':
        if args.msa is None or args.metadata is None:
            raise ValueError("--msa and --metadata are required for evcouplings model type")
        date_cutoff = args.date_cutoff or datasets.date_cutoff_str(dataset_name, args.cutoff)
        print(f"EVcouplings: date cutoff = {date_cutoff} (MRCA + {args.cutoff} years)")
        model_config = ModelConfig(
            model_type=ModelType.EVCOUPLINGS,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            evcouplings=EVcouplingsConfig(
                msa_path=args.msa,
                metadata_path=args.metadata,
                date_cutoff=date_cutoff,
                max_seqs=args.max_seqs,
                pseudo_count=args.evc_pseudo_count,
            ),
        )
        model = create_model(model_config, device)
        print("Model: EVcouplings mean-field DCA (no training, eval only)")
    elif args.model_type == 'eve':
        if args.msa is None or args.metadata is None:
            raise ValueError("--msa and --metadata are required for eve model type")
        date_cutoff = args.date_cutoff or datasets.date_cutoff_str(dataset_name, args.cutoff)
        print(f"EVE: date cutoff = {date_cutoff} (MRCA + {args.cutoff} years)")
        model_config = ModelConfig(
            model_type=ModelType.EVE,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            eve=EVEConfig(
                msa_path=args.msa,
                metadata_path=args.metadata,
                date_cutoff=date_cutoff,
                freeze_backbone=args.freeze_backbone,
                max_seqs=args.max_seqs,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: EVE VAE, freeze_backbone={args.freeze_backbone}")
    elif args.model_type == 'tranception':
        if args.msa is None or args.metadata is None:
            raise ValueError("--msa and --metadata are required for tranception model type")
        date_cutoff = args.date_cutoff or datasets.date_cutoff_str(dataset_name, args.cutoff)
        print(f"Tranception: date cutoff = {date_cutoff} (MRCA + {args.cutoff} years)")
        model_config = ModelConfig(
            model_type=ModelType.TRANCEPTION,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            tranception=TranceptionConfig(
                msa_path=args.msa,
                metadata_path=args.metadata,
                date_cutoff=date_cutoff,
                freeze_backbone=args.freeze_backbone,
                max_seqs=args.max_seqs,
                scoring_direction=args.scoring_direction,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: Tranception LM, scoring={args.scoring_direction}, freeze_backbone={args.freeze_backbone}")
    elif args.model_type == 'gated':
        # Populate every per-partner sub-config so any partner_type works; the selected
        # partners pick up their CLI flags through their existing sub-config dataclass.
        model_config = ModelConfig(
            model_type=ModelType.GATED_FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            transformer=TransformerConfig(
                d_model=args.d_model, nhead=4,
                dim_feedforward=4 * args.d_model, num_layers=args.num_layers,
            ),
            esm2=ESM2Config(
                model_name=args.esm2_model_name,
                freeze_backbone=args.freeze_backbone, finetune_layers=args.finetune_layers,
            ),
            pssm=PSSMConfig(data_dir=data_dir),
            gated=GatedFitnessConfig(
                partner_a_type=args.gate_partner_a,
                partner_b_type=args.gate_partner_b,
                gate_variant=args.gate_variant,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: Gated[{args.gate_partner_a} + {args.gate_partner_b}, gate={args.gate_variant}], "
              f"freeze_backbone={args.freeze_backbone}, finetune_layers={args.finetune_layers}")
    elif args.model_type == 'residual':
        if all_examples[0].cur_mat is None:
            raise ValueError(
                f"--model-type residual needs the own-window counts, and {examples_path} has "
                f"none. Regenerate with a build of antiGen-data that writes the fifth "
                f"'<pos>,<alt>,<next>,<all-time>,<own-window>' field.")
        model_config = ModelConfig(
            model_type=ModelType.RESIDUAL_FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            transformer=TransformerConfig(
                d_model=args.d_model, nhead=4,
                dim_feedforward=4 * args.d_model, num_layers=args.num_layers,
            ),
            residual=ResidualConfig(
                alpha=args.residual_alpha,
                beta_init=args.residual_beta_init,
                lambda_init=args.residual_lambda_init,
                learn_beta=not args.freeze_beta,
                learn_lambda=not args.freeze_lambda,
            ),
        )
        model = create_model(model_config, device)
        print(f"Model: Residual[persistence(alpha={args.residual_alpha}) + sequence], "
              f"beta init {args.residual_beta_init}"
              f"{' (frozen)' if args.freeze_beta else ''}, "
              f"lambda init {args.residual_lambda_init}"
              f"{' (frozen)' if args.freeze_lambda else ''}")
    else:
        model_config = ModelConfig(
            model_type=ModelType.FITNESS_PREDICTOR,
            seq_len=seq_len_aa,
            alphabet=alphabet_config,
            transformer=TransformerConfig(
                d_model=args.d_model,
                nhead=4,
                dim_feedforward=4 * args.d_model,
                num_layers=args.num_layers,
            ),
        )
        model = create_model(model_config, device)
    print(f"Model: {model.get_num_parameters(trainable_only=True):,} trainable params")

    # The objective. Same architecture and same forward pass either way -- only how the
    # logits are turned into a number changes.
    if args.loss in ('conditional', 'unconditional', 'kl'):
        if all_examples[0].n_mat is None:
            raise ValueError(
                f"--loss {args.loss} needs descendant counts, and {examples_path} has none. "
                f"Regenerate the examples with a build of antiGen-data that writes "
                f"'<pos>,<alt>,<descendants>' rows.")

        if args.loss == 'conditional':
            def batch_loss(logits, aa_batch, target_batch, count_batch):
                return compute_conditional_loss_batch(logits, target_batch, count_batch)
        elif args.loss == 'kl':
            def batch_loss(logits, aa_batch, target_batch, count_batch):
                return compute_kl_loss_batch(logits, target_batch, count_batch)
        else:
            def batch_loss(logits, aa_batch, target_batch, count_batch):
                return compute_unconditional_loss_batch(logits, aa_batch, count_batch)
    else:
        def batch_loss(logits, aa_batch, target_batch, count_batch):
            return compute_loss_batch(compute_probs(logits, target_batch, aa_batch), target_batch)
    if args.count_transform != 'none' and args.loss == 'multinomial':
        raise ValueError("--count-transform applies to the descendant counts, which "
                         "--loss multinomial does not read. Pick a count-based loss.")
    print(f"Loss: {args.loss} (count transform: {args.count_transform})")

    # Only the residual model takes the own-window counts; every other architecture keeps
    # the one-argument forward contract.
    takes_counts = args.model_type == 'residual'

    def forward(aa_batch, cur_batch):
        return (model(aa_batch, cur_batch) if takes_counts else model(aa_batch)).float()

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    epoch = 0
    avg_train_loss = float('nan')
    optimizer_state = None

    skip_training = args.model_type in ('blosum62', 'pssm', 'evcouplings') or \
        (args.model_type == 'eve' and args.freeze_backbone) or \
        (args.model_type == 'tranception' and args.freeze_backbone)
    if skip_training:
        print(f"{args.model_type} mode: skipping training (no learnable weights); saving checkpoint only.")
        model.eval()
        del train_loader, train_dataset, training_examples
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.warmup_steps > 0:
            def warmup_fn(step):
                if step < args.warmup_steps:
                    return float(step) / float(max(1, args.warmup_steps))
                return 1.0
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
        else:
            scheduler = None

        print("Starting training...")

        for epoch in range(args.epochs):
            total_train_loss = 0.0
            n_batches = 0

            for batch_num, (aa_batch, target_batch, count_batch, cur_batch) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()

                aa_batch = aa_batch.to(device)
                target_batch = target_batch.to(device)
                count_batch = count_batch.to(device)
                cur_batch = cur_batch.to(device)

                logits = forward(aa_batch, cur_batch)
                loss = batch_loss(logits, aa_batch, target_batch, count_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_loss_val = loss.item()
                total_train_loss += train_loss_val
                n_batches += 1
                global_step += 1

                # Test minibatch loss (cheap, per-batch monitoring; skipped if no test set)
                test_batch_loss_val = None
                if test_iter is not None:
                    test_aa, test_target, test_count, test_cur = next(test_iter)
                    with torch.no_grad():
                        model.eval()
                        test_aa = test_aa.to(device)
                        test_target = test_target.to(device)
                        test_count = test_count.to(device)
                        test_cur = test_cur.to(device)
                        test_logits = forward(test_aa, test_cur)
                        test_batch_loss_val = batch_loss(
                            test_logits, test_aa, test_target, test_count).item()

                if batch_num % 10 == 0:
                    if test_batch_loss_val is not None:
                        print(f"  Epoch {epoch}, batch {batch_num}: train_loss={train_loss_val:.8f}, test_batch_loss={test_batch_loss_val:.8f}")
                    else:
                        print(f"  Epoch {epoch}, batch {batch_num}: train_loss={train_loss_val:.8f}")

                if args.max_batches is not None and global_step >= args.max_batches:
                    print(f"Reached max batches ({args.max_batches}), stopping.")
                    break

            avg_train_loss = total_train_loss / max(n_batches, 1)
            print(f"Epoch {epoch}: avg_train_loss={avg_train_loss:.8f}")

            if args.checkpoint_every_epoch:
                epoch_dir = os.path.join(args.output_dir, "epochs")
                os.makedirs(epoch_dir, exist_ok=True)
                epoch_path = os.path.join(epoch_dir, f"epoch{epoch:03d}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss': avg_train_loss,
                    'config': model_config.to_dict(),
                    'train_args': vars(args),
                }, epoch_path)
                print(f"  wrote {epoch_path}")

            if args.max_batches is not None and global_step >= args.max_batches:
                break

        # Free training-only objects to reclaim memory before saving
        optimizer_state = optimizer.state_dict()
        del optimizer, scheduler, train_loader, train_dataset, training_examples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Freed training objects (optimizer, scheduler, train_loader, etc.) before saving")

    # Build run-specific prefix for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.model_type == 'esm2':
        model_name_short = args.esm2_model_name.split('/')[-1]
        run_prefix = f"esm2_{model_name_short}_lr{args.lr}_{timestamp}"
    elif args.model_type == 'blosum62':
        run_prefix = f"blosum62_{timestamp}"
    elif args.model_type == 'evcouplings':
        run_prefix = f"evcouplings_{timestamp}"
    elif args.model_type == 'eve':
        if args.freeze_backbone:
            run_prefix = f"eve_frozen_{timestamp}"
        else:
            run_prefix = f"eve_ft_lr{args.lr}_{timestamp}"
    elif args.model_type == 'tranception':
        if args.freeze_backbone:
            run_prefix = f"tranception_frozen_{args.scoring_direction}_{timestamp}"
        else:
            run_prefix = f"tranception_ft_{args.scoring_direction}_lr{args.lr}_{timestamp}"
    elif args.model_type == 'gated':
        a = _PARTNER_SHORT.get(args.gate_partner_a, args.gate_partner_a)
        b = _PARTNER_SHORT.get(args.gate_partner_b, args.gate_partner_b)
        run_prefix = f"gated_{args.gate_variant}_{a}_{b}_lr{args.lr}_{timestamp}"
    elif args.model_type == 'residual':
        tag = f"a{args.residual_alpha:g}_l{args.residual_lambda_init:g}"
        if args.freeze_lambda:
            tag += "_fixlam"
        if args.freeze_beta:
            tag += "_fixbeta"
        run_prefix = (f"residual_d{args.d_model}_n{args.num_layers}_{tag}_{args.loss}"
                      f"_lr{args.lr}_{timestamp}")
    else:
        loss_tag = "" if args.loss == 'multinomial' else f"_{args.loss}"
        if args.count_transform != 'none':
            loss_tag += f"_{args.count_transform}"
        run_prefix = f"simple_d{args.d_model}_n{args.num_layers}_lr{args.lr}{loss_tag}_{timestamp}"

    write_manifest(args, model_config, args.output_dir, run_prefix)

    output_path = os.path.join(args.output_dir, f"fitness_{run_prefix}.pt")

    print(f"Saving model to {output_path}...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_state,
        'epoch': epoch,
        'global_step': global_step,
        'loss': avg_train_loss,
        'config': model_config.to_dict(),
        'train_args': vars(args),
        'run_prefix': run_prefix,
    }

    # EVE/Tranception: persist the MSA-derived dims so eval.py can rebuild the
    # architecture and load weights without re-running MSA pretraining. The
    # alphabet-mapping buffers live in state_dict already; only L_focus and the
    # focus-column indices are needed to reconstruct the module shapes.
    if args.model_type in ('eve', 'tranception'):
        checkpoint['eve_state'] = {
            'L_focus': model._L_focus,
            'focus_cols': model._focus_to_aligned.tolist(),
        }

    torch.save(checkpoint, output_path)

    print("Done.")


if __name__ == "__main__":
    main()