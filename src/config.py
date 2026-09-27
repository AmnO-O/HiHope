"""Typed configuration for Target-Aware Query Attention compositionality pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

Mode = Literal['train80']
ModelBackend = Literal['query_attention']

_MODES = ('train80',)
_MODEL_BACKENDS = ('query_attention',)


@dataclass
class Config:
    """Configuration for Target-Aware Query Attention training pipeline."""

    # === run ===
    mode: Mode = 'train80'
    seed: int = 42

    # === model ===
    backbone: str = 'jhu-clsp/mmBERT-base'
    hidden_size: int = 768
    model_backend: ModelBackend = 'query_attention'
    head_hidden: int = 128
    dropout: float = 0.1
    num_roles: int = 4            # 0: Context, 1: Mod, 2: Head, 3: Compound
    num_cross_heads: int = 8
    num_self_heads: int = 4
    shared_head: bool = True
    sigma_floor: float = 0.04
    use_pre_ln: bool = True       # Pre-LN architecture for stable gradient flow
    use_rms_norm: bool = True     # RMSNorm (Root Mean Square Layer Normalization)

    # === data / paths ===
    data_path: Optional[str] = None
    output_dir: Optional[str] = None
    max_context_length: int = 256

    # Train datasets (EN / DE, NN + PV)
    en_nn_train: str = 'en-nn-train.tsv'
    de_nn_train: str = 'de-nn-train.tsv'
    en_pv_train: str = 'en-pv-train.tsv'
    de_pv_train: str = 'de-pv-train.tsv'
    train_aux: Optional[str] = None
    use_aux: bool = False
    test_size: float = 0.2
    val_ratio: float = 0.2
    load_from: Optional[str] = None

    # Multi-task Trial Datasets (EN / DE)
    en_nn_trial: str = 'en-nn-trial.tsv'
    de_nn_trial: str = 'de-nn-trial.tsv'
    en_pv_trial: str = 'en-pv-trial.tsv'
    de_pv_trial: str = 'de-pv-trial.tsv'

    # === training phases ===
    freeze_epochs: int = 3
    unfreeze_epochs: int = 9
    unfreeze_layers: int = 4      # Number of top transformer layers to unfreeze (4 for top-4, -1 for all)
    targets: List[str] = field(default_factory=lambda: ['mod', 'head', 'pv'])

    # Optimization
    batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    lr_heads: float = 3e-4
    lr_backbone: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    fp16: bool = True

    # Loss terms
    ccc_weight: float = 0.7
    ccc_var_floor: float = 0.05
    bin_sigma: float = 0.5
    use_label_std: bool = True
    kl_weight: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any], strict: bool = False) -> Config:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> Config:
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        save_config(self, path)

    def validate(self) -> None:
        assert self.backbone, "Backbone must not be empty"
        assert self.batch_size > 0, "Batch size must be positive"

    def build_tokenizer(self):
        """Construct HuggingFace AutoTokenizer for configured backbone."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.backbone)


def coerce_value(key: str, val: str | Any) -> Any:
    """Coerce string value from CLI --set key=value into appropriate Python types."""
    if not isinstance(val, str):
        return val
    v_str = val.strip()
    if v_str.lower() in ('true', 'yes', 'on'):
        return True
    if v_str.lower() in ('false', 'no', 'off'):
        return False
    if v_str.lower() in ('none', 'null'):
        return None
    try:
        return int(v_str)
    except ValueError:
        pass
    try:
        return float(v_str)
    except ValueError:
        pass
    if (v_str.startswith('[') and v_str.endswith(']')) or (v_str.startswith('{') and v_str.endswith('}')):
        try:
            return json.loads(v_str)
        except Exception:
            pass
    if key in ('targets',) and ',' in v_str:
        return [item.strip() for item in v_str.split(',')]
    return v_str


def config_from_dict(d: Dict[str, Any]) -> Config:
    """Construct a Config dataclass from a raw dictionary."""
    return Config.from_dict(d)


def config_from_json(path: str | Path) -> Config:
    """Load Config from a JSON file."""
    return Config.from_json(path)


def save_config(cfg: Config, path: str | Path) -> None:
    """Save Config to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(cfg), f, indent=2)
