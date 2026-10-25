"""Model factory functions for creating and loading fitness predictors."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from antiGen.config import ExperimentConfig, ModelConfig, ModelType, AlphabetConfig, TransformerConfig, LEGACY_AAS
from antiGen.models.base import BaseFitnessPredictor


def create_model(
    config: ModelConfig,
    device: Optional[Union[str, torch.device]] = None,
) -> BaseFitnessPredictor:
    """Factory function to create fitness prediction models.

    Args:
        config: Model configuration specifying architecture and hyperparameters.
        device: Target device for the model. If None, uses CUDA if available.

    Returns:
        Initialized model instance moved to the specified device.

    Raises:
        ValueError: If model_type is not recognized.

    Example:
        >>> from antiGen.config import ModelConfig, ModelType
        >>> config = ModelConfig(model_type=ModelType.ESM2_FITNESS_PREDICTOR)
        >>> model = create_model(config, device="cuda")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if config.model_type == ModelType.FITNESS_PREDICTOR:
        from antiGen.models.fitness_predictor import FitnessPredictor
        model = FitnessPredictor(config)
    elif config.model_type == ModelType.ESM2_FITNESS_PREDICTOR:
        from antiGen.models.esm2_fitness_predictor import ESM2FitnessPredictor
        model = ESM2FitnessPredictor(config)
    elif config.model_type == ModelType.BLOSUM62:
        from antiGen.models.blosum62_fitness_predictor import BLOSUM62FitnessPredictor
        model = BLOSUM62FitnessPredictor(config)
    elif config.model_type == ModelType.CODON:
        from antiGen.models.codon_fitness_predictor import CodonFitnessPredictor
        model = CodonFitnessPredictor(config)
    elif config.model_type == ModelType.SUBSTITUTION:
        from antiGen.models.substitution_fitness_predictor import SubstitutionFitnessPredictor
        model = SubstitutionFitnessPredictor(config)
    elif config.model_type == ModelType.PSSM:
        from antiGen.models.pssm_fitness_predictor import PSSMFitnessPredictor
        model = PSSMFitnessPredictor(config)
    elif config.model_type == ModelType.CODON_BLOSUM:
        from antiGen.models.codon_blosum_fitness_predictor import CodonBLOSUMFitnessPredictor
        model = CodonBLOSUMFitnessPredictor(config)
    elif config.model_type == ModelType.SITE_PROPENSITY:
        from antiGen.models.site_propensity_fitness_predictor import SitePropensityFitnessPredictor
        model = SitePropensityFitnessPredictor(config)
    elif config.model_type == ModelType.COUNTED_PSSM:
        from antiGen.models.counted_pssm_fitness_predictor import CountedPSSMFitnessPredictor
        model = CountedPSSMFitnessPredictor(config)
    elif config.model_type == ModelType.EVCOUPLINGS:
        from antiGen.models.evcouplings_fitness_predictor import EVcouplingsFitnessPredictor
        model = EVcouplingsFitnessPredictor(config)
    elif config.model_type == ModelType.EVE:
        from antiGen.models.eve_fitness_predictor import EVEFitnessPredictor
        model = EVEFitnessPredictor(config)
    elif config.model_type == ModelType.TRANCEPTION:
        from antiGen.models.tranception_fitness_predictor import TranceptionFitnessPredictor
        model = TranceptionFitnessPredictor(config)
    elif config.model_type == ModelType.GATED_FITNESS_PREDICTOR:
        from antiGen.models.gated_fitness_predictor import GatedFitnessPredictor
        model = GatedFitnessPredictor(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model.to(device)


def create_model_from_yaml(
    config_path: str,
    device: Optional[Union[str, torch.device]] = None,
) -> BaseFitnessPredictor:
    """Create a model from a YAML config file.

    Useful for constructing a model from a config without a training checkpoint.

    Args:
        config_path: Path to YAML config file.
        device: Target device for the model.

    Returns:
        Initialized model instance.

    Example:
        >>> model = create_model_from_yaml("configs/esm2_default.yaml")
    """
    exp_config = ExperimentConfig.from_yaml(config_path)
    return create_model(exp_config.model, device)


def load_model(
    checkpoint_path: str,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> tuple[BaseFitnessPredictor, dict[str, Any]]:
    """Load a model from checkpoint or YAML config.

    Handles:
    - .pt/.pth checkpoint files (loads trained weights)
    - .yaml/.yml config files (creates model with pretrained weights if available)
    - Legacy checkpoint formats for backward compatibility

    Args:
        checkpoint_path: Path to checkpoint (.pt) or config (.yaml) file.
        device: Target device for the model.
        strict: Whether to strictly enforce state_dict key matching.

    Returns:
        Tuple of (model, checkpoint_dict) where checkpoint_dict contains
        all saved state including population genetics parameters.

    Example:
        >>> model, ckpt = load_model("trained_model.pt")
        >>> mu = torch.exp(ckpt["log_mu"])

        >>> # Or load from a YAML config
        >>> model, ckpt = load_model("configs/esm2_default.yaml")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    path = Path(checkpoint_path)

    # Handle YAML config files (for pretrained models)
    if path.suffix in ('.yaml', '.yml'):
        model = create_model_from_yaml(checkpoint_path, device)
        model.eval()
        # Return empty checkpoint dict for compatibility
        return model, {"config": model._config.to_dict()}

    # Handle checkpoint files
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle legacy checkpoint format
    if "config" not in checkpoint:
        config = _convert_legacy_checkpoint(checkpoint)
    else:
        config = ModelConfig.from_dict(checkpoint["config"])

    model = create_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.eval()

    return model, checkpoint


def _convert_legacy_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig:
    """Convert legacy checkpoint format to new config format.

    Handles two legacy formats:
    - 'model_args' dict: from train.py's transformer path (vocab_size, seq_len, d_model, etc.)
    - Top-level keys (d_model, E, N): oldest format
    """
    if "model_args" in checkpoint:
        ma = checkpoint["model_args"]
        return ModelConfig(
            model_type=ModelType.FITNESS_PREDICTOR,
            seq_len=ma["seq_len"],
            alphabet=AlphabetConfig(
                amino_acids=LEGACY_AAS,
            ),
            transformer=TransformerConfig(
                d_model=ma["d_model"],
                nhead=ma.get("nhead", 4),
                dim_feedforward=ma.get("dim_feedforward", 4 * ma["d_model"]),
                num_layers=ma.get("num_layers", 1),
                dropout=0.1,
            ),
        )

    # Oldest legacy format with top-level keys
    d_model = checkpoint.get("d_model", 128)
    E = checkpoint.get("E", 22)
    N = checkpoint.get("N", 3819)  # nucleotide length

    return ModelConfig(
        model_type=ModelType.FITNESS_PREDICTOR,
        seq_len=N // 3,
        alphabet=AlphabetConfig(
            nucleotides="ACGT-",
            amino_acids=LEGACY_AAS,
        ),
        transformer=TransformerConfig(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            num_layers=1,
            dropout=0.1,
        ),
    )


def save_checkpoint(
    path: str,
    model: BaseFitnessPredictor,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    **extra,
) -> None:
    """Save a training checkpoint.

    Args:
        path: Output path for the checkpoint file.
        model: The model to save.
        model_config: Model configuration.
        optimizer: Optimizer state to save.
        **extra: Additional data to include (epoch, loss, population params, etc).
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config.to_dict(),
        # Legacy keys for backward compatibility
        'd_model': model_config.transformer.d_model,
    }

    # Add extra data
    checkpoint.update(extra)

    # Handle experiment config separately if provided
    if 'experiment_config' in checkpoint:
        exp_config = checkpoint.pop('experiment_config')
        checkpoint['experiment_config'] = exp_config.to_dict()

    torch.save(checkpoint, path)
