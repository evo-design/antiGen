"""Configuration dataclasses for fitness prediction models."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


def _default_plmc_binary() -> Optional[str]:
    """Locate the external plmc binary from $PLMC_BINARY or PATH (None if absent)."""
    return os.environ.get("PLMC_BINARY") or shutil.which("plmc")


class ModelType(str, Enum):
    """Supported model architectures."""
    FITNESS_PREDICTOR = "fitness_predictor"
    ESM2_FITNESS_PREDICTOR = "esm2_fitness_predictor"
    BLOSUM62 = "blosum62"  # BLOSUM62 substitution matrix (no training)
    PSSM = "pssm"  # Per-position log(count+1) from variant-root PSSM (no training)
    CODON = "codon"  # Genetic-code single-nucleotide accessibility (no training)
    SUBSTITUTION = "substitution"  # Learned codon x BLOSUM62 log-linear mixture (5 params)
    EVCOUPLINGS = "evcouplings"  # EVcouplings mean-field DCA (no training)
    EVE = "eve"  # EVE VAE (pre-train on MSA, optional finetune)
    TRANCEPTION = "tranception"  # Tranception autoregressive LM (pre-train on MSA, optional finetune)
    GATED_FITNESS_PREDICTOR = "gated_fitness_predictor"  # two predictors blended via per-position gate


class TrainingMode(str, Enum):
    """Training paradigms for fitness prediction."""
    PHYLODYNAMIC = "phylodynamic"      # Current approach: learn from birth-death dynamics
    PER_MUTATION = "per_mutation"      # New approach: treat each unique mutation as one example


@dataclass(frozen=True)
class AlphabetConfig:
    """Configuration for amino acid and nucleotide alphabets.

    Attributes:
        nucleotides: Allowed nucleotide characters.
        amino_acids: Allowed amino acid characters including special tokens.
    """
    nucleotides: str = "ACGT-"
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY*-"

    @property
    def vocab_size(self) -> int:
        """Number of amino acid tokens (E in the codebase)."""
        return len(self.amino_acids)

    @property
    def nuc_vocab_size(self) -> int:
        """Number of nucleotide tokens (D in the codebase)."""
        return len(self.nucleotides)


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the from-scratch transformer model.

    Attributes:
        d_model: Embedding dimension.
        nhead: Number of attention heads.
        dim_feedforward: Feedforward network dimension.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability.
    """
    d_model: int = 128
    nhead: int = 4
    dim_feedforward: int = 512
    num_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )


@dataclass(frozen=True)
class ESM2Config:
    """Configuration for ESM-2 based model.

    Attributes:
        model_name: HuggingFace model identifier.
        hidden_size: ESM-2 hidden dimension (determined by model).
        freeze_backbone: Whether to freeze ESM-2 weights during training.
        finetune_layers: Number of last encoder layers to unfreeze when
            freeze_backbone is True. 0 means freeze entire backbone.
            Ignored when freeze_backbone is False.
    """
    model_name: str = "facebook/esm2_t12_35M_UR50D"
    hidden_size: int = 480  # Fixed by the 35M model
    freeze_backbone: bool = True
    finetune_layers: int = 0


@dataclass(frozen=True)
class EVcouplingsConfig:
    """Configuration for EVcouplings (plmc-based) DCA model.

    Attributes:
        msa_path: Path to pandemic MSA in FASTA format.
        metadata_path: Path to metadata TSV with 'strain' and 'date' columns.
        date_cutoff: YYYY-MM-DD cutoff for filtering MSA to training window.
        theta: Sequence reweighting threshold (evcouplings convention; plmc uses 1-theta).
        pseudo_count: (Legacy/unused under plmc.) Retained for backwards compatibility.
        plmc_binary: Path to the plmc executable.
        lambda_h, lambda_J: L2 regularization on fields / couplings (None = plmc defaults).
        plmc_iterations: Max L-BFGS iterations (None = plmc default).
        plmc_cpu: OpenMP threads ("max" or int); None = single-threaded.
    """
    msa_path: str = ""
    metadata_path: str = ""
    date_cutoff: str = ""
    theta: float = 0.8
    pseudo_count: float = 0.5
    max_seqs: int = 10000
    plmc_binary: Optional[str] = field(default_factory=_default_plmc_binary)
    lambda_h: Optional[float] = None
    lambda_J: Optional[float] = None
    plmc_iterations: Optional[int] = None
    plmc_cpu: Optional[str] = "max"
    plmc_fast: bool = False


@dataclass(frozen=True)
class EVEConfig:
    """Configuration for EVE VAE model.

    Attributes:
        msa_path: Path to pandemic MSA in FASTA format.
        metadata_path: Path to metadata TSV with 'strain' and 'date' columns.
        date_cutoff: YYYY-MM-DD cutoff for filtering MSA to training window.
        theta: Sequence reweighting threshold (0.01 for viruses per EVE paper).
        z_dim: Latent space dimension.
        pretrain_steps: Number of MSA pre-training steps.
        pretrain_lr: Learning rate for MSA pre-training.
        pretrain_batch_size: Batch size for MSA pre-training.
        freeze_backbone: If True, freeze all parameters after MSA pre-training.
    """
    msa_path: str = ""
    metadata_path: str = ""
    date_cutoff: str = ""
    theta: float = 0.01
    z_dim: int = 50
    pretrain_steps: int = 400000
    pretrain_lr: float = 0.0001
    pretrain_batch_size: int = 256
    freeze_backbone: bool = False
    max_seqs: int = 0
    # Set at checkpoint-load time (from checkpoint 'eve_state') to skip MSA
    # pretraining and rebuild the architecture from stored dims:
    # {'L_focus': int, 'focus_cols': List[int]}. None => normal MSA-pretraining path.
    pretrained_state: Optional[dict] = None


@dataclass(frozen=True)
class TranceptionConfig:
    """Configuration for Tranception autoregressive transformer model.

    Attributes:
        msa_path: Path to pandemic MSA in FASTA format.
        metadata_path: Path to metadata TSV with 'strain' and 'date' columns.
        date_cutoff: YYYY-MM-DD cutoff for filtering MSA to training window.
        theta: Sequence reweighting threshold.
        n_embd: Embedding dimension.
        n_head: Number of attention heads (must be divisible by 4).
        n_layer: Number of transformer layers.
        dropout: Dropout probability.
        pretrain_steps: Number of MSA pre-training steps.
        pretrain_lr: Peak learning rate for MSA pre-training.
        pretrain_batch_size: Batch size for MSA pre-training.
        freeze_backbone: If True, freeze all parameters after MSA pre-training.
        max_seqs: Max MSA sequences (0=unlimited).
        scoring_direction: "forward", "backward", or "both".
    """
    msa_path: str = ""
    metadata_path: str = ""
    date_cutoff: str = ""
    theta: float = 0.01
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1
    pretrain_steps: int = 150000
    pretrain_lr: float = 3e-4
    pretrain_batch_size: int = 32
    freeze_backbone: bool = False
    max_seqs: int = 0
    scoring_direction: str = "both"
    # Set at checkpoint-load time (from checkpoint 'eve_state') to skip MSA
    # pretraining and rebuild the architecture from stored dims:
    # {'L_focus': int, 'focus_cols': List[int]}. None => normal MSA-pretraining path.
    pretrained_state: Optional[dict] = None


@dataclass(frozen=True)
class CodonConfig:
    """Configuration for the codon-accessibility predictor.

    Attributes:
        kappa: Transition/transversion ratio. 1.0 counts every nucleotide change
            equally; >1 up-weights transitions (A<->G, C<->T).
        pseudocount: Floor added to every entry so inaccessible substitutions rank
            last without becoming -inf.
    """
    kappa: float = 1.0
    pseudocount: float = 0.01


@dataclass(frozen=True)
class SubstitutionConfig:
    """Configuration for the learned codon x BLOSUM62 substitution baseline.

    Attributes:
        init_kappa: Starting transition/transversion ratio; it is then fitted.
        pseudocount: Floor inside the log of the codon accessibility counts.
        use_blosum: Include the BLOSUM62 term. False leaves a pure learned-kappa codon
            model, which is the useful ablation.
    """
    init_kappa: float = 1.0
    pseudocount: float = 0.01
    use_blosum: bool = True


@dataclass(frozen=True)
class PSSMConfig:
    """Configuration for the PSSM (per-position log-count) predictor.

    The predictor reads ``<data_dir>/pssms/root.csv`` at construction time (the same
    per-cutoff PSSM auto-loaded as the ``pssm_root`` baseline). No learnable parameters.

    Attributes:
        path: Explicit path to the PSSM CSV (used by tests; overrides data_dir).
        data_dir: Directory containing ``pssms/root.csv`` (auto-resolved in train.py).
    """
    path: Optional[str] = None
    data_dir: Optional[str] = None


@dataclass(frozen=True)
class GatedFitnessConfig:
    """Configuration for a gated mixture of two fitness predictors.

    The gate produces a per-(batch, position) sigmoid weight blending z-normalized
    partner_a logits with z-normalized partner_b logits. Each partner inherits the outer
    ModelConfig, so its sub-config (e.g. ESM2Config, TransformerConfig) routes through the
    existing CLI flags for that model type.

    Attributes:
        partner_a_type: ModelType.value for the first partner (any registered type except
            ``gated_fitness_predictor``).
        partner_b_type: ModelType.value for the second partner.
        gate_variant: "static" = per-position learnable logit (data-blind);
            "uncertainty" = per-position MLP on partner softmax entropies (data-conditioned).
    """
    partner_a_type: str = "fitness_predictor"
    partner_b_type: str = "blosum62"
    gate_variant: str = "static"


@dataclass(frozen=True)
class ModelConfig:
    """Unified model configuration.

    Attributes:
        model_type: Which model architecture to use.
        seq_len: Expected sequence length (in amino acids, N/3 in codebase).
        alphabet: Alphabet configuration.
        transformer: Config for from-scratch transformer (used if model_type is FITNESS_PREDICTOR).
        esm2: Config for ESM-2 model (used if model_type is ESM2_FITNESS_PREDICTOR).
    """
    model_type: ModelType = ModelType.FITNESS_PREDICTOR
    seq_len: int = 1273  # Default: (25381 - 21562) / 3 for spike protein
    alphabet: AlphabetConfig = field(default_factory=AlphabetConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    esm2: ESM2Config = field(default_factory=ESM2Config)
    evcouplings: EVcouplingsConfig = field(default_factory=EVcouplingsConfig)
    eve: EVEConfig = field(default_factory=EVEConfig)
    tranception: TranceptionConfig = field(default_factory=TranceptionConfig)
    pssm: PSSMConfig = field(default_factory=PSSMConfig)
    codon: CodonConfig = field(default_factory=CodonConfig)
    substitution: SubstitutionConfig = field(default_factory=SubstitutionConfig)
    gated: GatedFitnessConfig = field(default_factory=GatedFitnessConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary for checkpoint storage."""
        return {
            "model_type": self.model_type.value,
            "seq_len": self.seq_len,
            "alphabet": {
                "nucleotides": self.alphabet.nucleotides,
                "amino_acids": self.alphabet.amino_acids,
            },
            "transformer": {
                "d_model": self.transformer.d_model,
                "nhead": self.transformer.nhead,
                "dim_feedforward": self.transformer.dim_feedforward,
                "num_layers": self.transformer.num_layers,
                "dropout": self.transformer.dropout,
            },
            "esm2": {
                "model_name": self.esm2.model_name,
                "hidden_size": self.esm2.hidden_size,
                "freeze_backbone": self.esm2.freeze_backbone,
                "finetune_layers": self.esm2.finetune_layers,
            },
            "evcouplings": {
                "msa_path": self.evcouplings.msa_path,
                "metadata_path": self.evcouplings.metadata_path,
                "date_cutoff": self.evcouplings.date_cutoff,
                "theta": self.evcouplings.theta,
                "pseudo_count": self.evcouplings.pseudo_count,
                "max_seqs": self.evcouplings.max_seqs,
            },
            "eve": {
                "msa_path": self.eve.msa_path,
                "metadata_path": self.eve.metadata_path,
                "date_cutoff": self.eve.date_cutoff,
                "theta": self.eve.theta,
                "z_dim": self.eve.z_dim,
                "pretrain_steps": self.eve.pretrain_steps,
                "pretrain_lr": self.eve.pretrain_lr,
                "pretrain_batch_size": self.eve.pretrain_batch_size,
                "freeze_backbone": self.eve.freeze_backbone,
                "max_seqs": self.eve.max_seqs,
            },
            "tranception": {
                "msa_path": self.tranception.msa_path,
                "metadata_path": self.tranception.metadata_path,
                "date_cutoff": self.tranception.date_cutoff,
                "theta": self.tranception.theta,
                "n_embd": self.tranception.n_embd,
                "n_head": self.tranception.n_head,
                "n_layer": self.tranception.n_layer,
                "dropout": self.tranception.dropout,
                "pretrain_steps": self.tranception.pretrain_steps,
                "pretrain_lr": self.tranception.pretrain_lr,
                "pretrain_batch_size": self.tranception.pretrain_batch_size,
                "freeze_backbone": self.tranception.freeze_backbone,
                "max_seqs": self.tranception.max_seqs,
                "scoring_direction": self.tranception.scoring_direction,
            },
            "codon": {
                "kappa": self.codon.kappa,
                "pseudocount": self.codon.pseudocount,
            },
            "substitution": {
                "init_kappa": self.substitution.init_kappa,
                "pseudocount": self.substitution.pseudocount,
                "use_blosum": self.substitution.use_blosum,
            },
            "pssm": {
                "path": self.pssm.path,
                "data_dir": self.pssm.data_dir,
            },
            "gated": {
                "partner_a_type": self.gated.partner_a_type,
                "partner_b_type": self.gated.partner_b_type,
                "gate_variant": self.gated.gate_variant,
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Deserialize config from dictionary."""
        # Filter ESM2 config to only known fields (backward compat)
        esm2_fields = {'model_name', 'hidden_size', 'freeze_backbone', 'finetune_layers'}
        esm2_dict = {k: v for k, v in d["esm2"].items() if k in esm2_fields}

        # Parse EVcouplings config (with backward compat for old checkpoints)
        evc_dict = d.get("evcouplings", {})
        evcouplings_config = EVcouplingsConfig(
            msa_path=evc_dict.get("msa_path", ""),
            metadata_path=evc_dict.get("metadata_path", ""),
            date_cutoff=evc_dict.get("date_cutoff", ""),
            theta=evc_dict.get("theta", 1.0),
            pseudo_count=evc_dict.get("pseudo_count", 0.5),
            max_seqs=evc_dict.get("max_seqs", 10000),
        )

        # Parse EVE config (with backward compat for old checkpoints)
        eve_dict = d.get("eve", {})
        eve_config = EVEConfig(
            msa_path=eve_dict.get("msa_path", ""),
            metadata_path=eve_dict.get("metadata_path", ""),
            date_cutoff=eve_dict.get("date_cutoff", ""),
            theta=eve_dict.get("theta", 0.01),
            z_dim=eve_dict.get("z_dim", 50),
            pretrain_steps=eve_dict.get("pretrain_steps", 400000),
            pretrain_lr=eve_dict.get("pretrain_lr", 0.0001),
            pretrain_batch_size=eve_dict.get("pretrain_batch_size", 256),
            freeze_backbone=eve_dict.get("freeze_backbone", False),
            max_seqs=eve_dict.get("max_seqs", 0),
        )

        # Parse Tranception config (with backward compat for old checkpoints)
        tc_dict = d.get("tranception", {})
        tranception_config = TranceptionConfig(
            msa_path=tc_dict.get("msa_path", ""),
            metadata_path=tc_dict.get("metadata_path", ""),
            date_cutoff=tc_dict.get("date_cutoff", ""),
            theta=tc_dict.get("theta", 0.01),
            n_embd=tc_dict.get("n_embd", 256),
            n_head=tc_dict.get("n_head", 8),
            n_layer=tc_dict.get("n_layer", 6),
            dropout=tc_dict.get("dropout", 0.1),
            pretrain_steps=tc_dict.get("pretrain_steps", 150000),
            pretrain_lr=tc_dict.get("pretrain_lr", 3e-4),
            pretrain_batch_size=tc_dict.get("pretrain_batch_size", 32),
            freeze_backbone=tc_dict.get("freeze_backbone", False),
            max_seqs=tc_dict.get("max_seqs", 0),
            scoring_direction=tc_dict.get("scoring_direction", "both"),
        )

        codon_dict = d.get("codon", {})
        codon_config = CodonConfig(
            kappa=codon_dict.get("kappa", 1.0),
            pseudocount=codon_dict.get("pseudocount", 0.01),
        )

        sub_dict = d.get("substitution", {})
        substitution_config = SubstitutionConfig(
            init_kappa=sub_dict.get("init_kappa", 1.0),
            pseudocount=sub_dict.get("pseudocount", 0.01),
            use_blosum=sub_dict.get("use_blosum", True),
        )

        pssm_dict = d.get("pssm", {})
        pssm_config = PSSMConfig(
            path=pssm_dict.get("path"),
            data_dir=pssm_dict.get("data_dir"),
        )

        gated_dict = d.get("gated", {})
        gated_config = GatedFitnessConfig(
            partner_a_type=gated_dict.get("partner_a_type", "fitness_predictor"),
            partner_b_type=gated_dict.get("partner_b_type", "blosum62"),
            gate_variant=gated_dict.get("gate_variant", "static"),
        )

        return cls(
            model_type=ModelType(d["model_type"]),
            seq_len=d["seq_len"],
            alphabet=AlphabetConfig(**d["alphabet"]),
            transformer=TransformerConfig(**d["transformer"]),
            esm2=ESM2Config(**esm2_dict),
            evcouplings=evcouplings_config,
            eve=eve_config,
            tranception=tranception_config,
            pssm=pssm_config,
            codon=codon_config,
            substitution=substitution_config,
            gated=gated_config,
        )


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 0.005
    warmup_steps: int = 0
    save_every: int = 0  # Save checkpoint every N batches (0 = only at end)
    seed: int = 42

    # Training mode: phylodynamic (default) or per_mutation (new approach)
    training_mode: TrainingMode = TrainingMode.PHYLODYNAMIC

    # Fitness-weighted mutation loss (phylodynamic mode only):
    # treats observed mutations as "selected" from alternatives based on fitness
    mutation_fitness_loss: bool = False  # Enable fitness-dependent mutation likelihood
    mutation_fitness_weight: float = 1.0  # Weight of this loss term relative to main loss

    # Per-mutation mode specific settings
    mutation_success_metric: str = "descendants"  # "descendants" or "time_on_tree"


@dataclass
class DataConfig:
    """Configuration for data processing."""
    start_pos: int = 21562
    end_pos: int = 25381
    dt: float = 0.01


@dataclass
class ExperimentConfig:
    """Full experiment configuration combining model, training, and data."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "checkpoints"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "warmup_steps": self.training.warmup_steps,
                "save_every": self.training.save_every,
                "seed": self.training.seed,
                "training_mode": self.training.training_mode.value,
                "mutation_fitness_loss": self.training.mutation_fitness_loss,
                "mutation_fitness_weight": self.training.mutation_fitness_weight,
                "mutation_success_metric": self.training.mutation_success_metric,
            },
            "data": {
                "start_pos": self.data.start_pos,
                "end_pos": self.data.end_pos,
                "dt": self.data.dt,
            },
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Deserialize from dictionary."""
        # Handle training_mode enum conversion
        training_d = d["training"].copy()
        if "training_mode" in training_d:
            training_d["training_mode"] = TrainingMode(training_d["training_mode"])
        return cls(
            model=ModelConfig.from_dict(d["model"]),
            training=TrainingConfig(**training_d),
            data=DataConfig(**d["data"]),
            output_dir=d.get("output_dir", "checkpoints"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)

        # Parse model config from YAML structure
        model_d = d.get("model", {})
        model_type_str = model_d.get("type", "transformer")

        # Defaults for configs not specified
        transformer_config = TransformerConfig()
        esm2_config = ESM2Config()

        if model_type_str == "esm2":
            model_type = ModelType.ESM2_FITNESS_PREDICTOR
            esm2_d = model_d.get("esm2", {})
            esm2_config = ESM2Config(
                model_name=esm2_d.get("model_name", "facebook/esm2_t12_35M_UR50D"),
                freeze_backbone=esm2_d.get("freeze_backbone", True),
                finetune_layers=esm2_d.get("finetune_layers", 0),
            )
        else:
            model_type = ModelType.FITNESS_PREDICTOR
            transformer_d = model_d.get("transformer", {})
            transformer_config = TransformerConfig(
                d_model=transformer_d.get("d_model", 128),
                nhead=transformer_d.get("nhead", 4),
                dim_feedforward=transformer_d.get("dim_feedforward", 512),
                num_layers=transformer_d.get("num_layers", 1),
                dropout=transformer_d.get("dropout", 0.1),
            )

        # seq_len will be computed from data config
        data_d = d.get("data", {})
        start_pos = data_d.get("start_pos", 21562)
        end_pos = data_d.get("end_pos", 25381)
        seq_len = (end_pos - start_pos) // 3

        model_config = ModelConfig(
            model_type=model_type,
            seq_len=seq_len,
            transformer=transformer_config,
            esm2=esm2_config,
        )

        training_d = d.get("training", {})
        training_mode_str = training_d.get("training_mode", "phylodynamic")
        training_config = TrainingConfig(
            batch_size=training_d.get("batch_size", 64),
            epochs=training_d.get("epochs", 1),
            learning_rate=training_d.get("learning_rate", 0.005),
            warmup_steps=training_d.get("warmup_steps", 0),
            save_every=training_d.get("save_every", 0),
            seed=training_d.get("seed", 42),
            training_mode=TrainingMode(training_mode_str),
            mutation_fitness_loss=training_d.get("mutation_fitness_loss", False),
            mutation_fitness_weight=training_d.get("mutation_fitness_weight", 1.0),
            mutation_success_metric=training_d.get("mutation_success_metric", "descendants"),
        )

        data_config = DataConfig(
            start_pos=start_pos,
            end_pos=end_pos,
            dt=data_d.get("dt", 0.01),
        )

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            output_dir=d.get("output_dir", "checkpoints"),
        )

    def save_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml

        # Convert to YAML-friendly structure
        if self.model.model_type == ModelType.ESM2_FITNESS_PREDICTOR:
            model_type_str = "esm2"
        else:
            model_type_str = "transformer"

        d = {
            "model": {
                "type": model_type_str,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "warmup_steps": self.training.warmup_steps,
                "save_every": self.training.save_every,
                "seed": self.training.seed,
                "training_mode": self.training.training_mode.value,
                "mutation_fitness_loss": self.training.mutation_fitness_loss,
                "mutation_fitness_weight": self.training.mutation_fitness_weight,
                "mutation_success_metric": self.training.mutation_success_metric,
            },
            "data": {
                "start_pos": self.data.start_pos,
                "end_pos": self.data.end_pos,
                "dt": self.data.dt,
            },
            "output_dir": self.output_dir,
        }

        if model_type_str == "esm2":
            d["model"]["esm2"] = {
                "model_name": self.model.esm2.model_name,
                "freeze_backbone": self.model.esm2.freeze_backbone,
                "finetune_layers": self.model.esm2.finetune_layers,
            }
        else:
            d["model"]["transformer"] = {
                "d_model": self.model.transformer.d_model,
                "nhead": self.model.transformer.nhead,
                "dim_feedforward": self.model.transformer.dim_feedforward,
                "num_layers": self.model.transformer.num_layers,
                "dropout": self.model.transformer.dropout,
            }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
