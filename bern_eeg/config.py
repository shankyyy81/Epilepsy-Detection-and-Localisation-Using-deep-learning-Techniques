"""
Central configuration for the Bern-Barcelona focal / non-focal EEG pipeline.

Configuration is resolved in this order (later entries override earlier ones):
    DEFAULT_CONFIG  ->  MODEL_PRESETS[model]  ->  YAML/JSON file  ->  CLI "--set key=value"

Everything is a plain nested dict wrapped in `Cfg` (attribute access).
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from typing import Any, Dict, Iterable, Optional

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# DATA_ROOT: edit this, or export BERN_DATA_ROOT=/path/to/BERNBarcelona
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "BERN_DATA_ROOT",
    os.path.join(os.path.dirname(PROJECT_DIR), "BERNBarcelona"),
)


class Cfg(dict):
    """dict with attribute access; nested dicts are converted recursively."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, v):
        if isinstance(v, dict) and not isinstance(v, Cfg):
            return Cfg(v)
        if isinstance(v, list):
            return [cls._wrap(x) for x in v]
        return v

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __deepcopy__(self, memo):
        return Cfg(copy.deepcopy(dict(self), memo))

    def to_dict(self) -> Dict[str, Any]:
        def unwrap(v):
            if isinstance(v, dict):
                return {k: unwrap(x) for k, x in v.items()}
            if isinstance(v, list):
                return [unwrap(x) for x in v]
            return v
        return unwrap(self)


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment_name": None,          # defaults to preset exp_name
    "model_preset": "full",
    "seed": 42,
    "seeds": [42],                    # multi-seed runs: [42, 123, 2024, 3407, 999]
    "device": "auto",                 # auto | cuda | mps | cpu
    "output_root": os.path.join(PROJECT_DIR, "experiments"),

    "data": {
        "root": DATA_ROOT,            # directory or single .npy/.npz/.mat file
        "labels_path": None,          # optional separate label file for bulk arrays
        "patient_map": None,          # optional CSV: file,patient_id  (enables LOPO)
        "fs": 512,                    # used only if the files do not carry a sampling rate
        "cache_dir": os.path.join(PROJECT_DIR, "outputs", "cache"),
        "use_cache": True,
        "channel_mode": "all",        # all | select | average
        "channels": None,             # list of channel indices when channel_mode == select
        "length_policy": "error",     # error | crop_min  (if segment lengths differ)
        "strict": False,              # raise on malformed files instead of excluding+reporting
        "max_samples": None,          # stratified subset for smoke tests
        "dup_check": "block",         # none | exact | block
        "dup_block": 64,
    },

    "preprocess": {
        "dc_remove": True,            # per-sample mean removal (no learned parameters)
        "detrend": False,
        "notch_hz": None,             # e.g. 50
        "notch_q": 30.0,
        "bandpass": None,             # e.g. [0.5, 128]
        "filter_order": 4,
        "normalization": "global",    # none | global | channel | robust | window
    },

    "stft": {
        "short": {"win": 256, "hop": 128},
        "long": {"win": 512, "hop": 256},
        "fmax": 128,                  # 80 | 100 | 128 | 160 | null (= full range)
        "crop": "fmax",               # fmax | half (baseline: len(f)//2 bins)
        "representation": "log_power",  # magnitude | power | log_power | log_magnitude
        "scaling": "scipy",           # scipy (1/sum(window), matches baseline) | none
        "input_bn": True,             # learned input BatchNorm (statistics from train only)
    },

    "grp": {
        "decimate": 8,
        "m": 3,                       # int or "auto" (false-nearest-neighbours on inner-train)
        "tau": 2,                     # int or "auto" (first minimum of AMI on inner-train)
        "threshold_mode": "percentile",  # percentile | fixed | adaptive | none
        "percentile": 10.0,
        "fixed_eps": None,            # fitted on inner-train when threshold_mode == fixed
        "adaptive_frac": 0.2,         # eps = frac * std(D) for adaptive mode
        "soft_kappa": 0.05,           # smooth Heaviside width (relative to eps)
        "image_size": 128,
        "chunk": 8,                   # signals per cdist chunk (memory control)
        "auto_subset": 300,           # segments used to estimate m / tau
    },

    "hilbert": {
        "n_bands": 16,
        "fmin": 0.5,
        "fmax": 128.0,
        "time_bins": 80,
    },

    "model": {
        "arch": "multimodal",         # multimodal | baseline
        "use_raw": True,
        "use_stft": True,
        "use_grp": True,
        "use_hilbert": False,
        "fusion": "cross_attention",  # gated | cross_attention | concat
        "use_transformer": True,
        "embed_dim": 256,
        "heads": 4,
        "ca_layers": 2,
        "tf_layers": 2,
        "ffn_dim": 512,
        "dropout": 0.25,
        "attn_dropout": 0.1,
        "norm": "batch",              # batch | group
        "stft_depth": 18,             # 18 | 34 | 50
        "stft_widths": [32, 64, 128, 256],
        "grp_widths": [32, 64, 128],
        "hilbert_widths": [32, 64, 128],
        "raw_widths": [32, 64, 128, 256, 256],
        "raw_strides": [2, 4, 4, 4, 2],
        "raw_kernels": [7, 15, 31],
        "raw_dilations": [1, 2, 4, 8],
        "token_freq_rows": 8,
        "encoder_dropout": 0.1,
        "channel_attention": False,
        "head_hidden": 128,
    },

    "loss": {
        "name": "bce",                # bce | focal | weighted_bce | smooth_bce
        "gamma": 2.0,
        "alpha": None,                # focal alpha (positive-class weight); None = no weighting
        "smoothing": 0.1,
        "pos_weight": None,           # weighted_bce; None -> n_neg/n_pos from TRAIN labels
    },

    "augment": {
        "enabled": True,
        "p_noise": 0.5, "noise_std": 0.05,          # relative to per-sample std
        "p_scale": 0.5, "scale_range": [0.8, 1.2],
        "p_time_mask": 0.3, "time_mask_frac": 0.05,
        "p_shift": 0.3, "shift_max_s": 0.25,
        "p_crop": 0.0, "crop_frac": 0.9,
        "p_warp": 0.0, "warp_max": 0.05,
        "p_bandstop": 0.0, "bandstop_width_hz": [2.0, 8.0], "bandstop_range_hz": [1.0, 100.0],
        "p_channel_dropout": 0.0,
        "p_polarity": 0.0,            # off by default: bipolar sign carries information
        "p_mixup": 0.0, "mixup_alpha": 0.2,
        "p_specaug": 0.5, "specaug_frac": 0.08,
    },

    "train": {
        "epochs": 100,
        "batch_size": 16,
        "auto_batch": False,
        "accum_steps": 1,
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine_warmup",  # cosine_warmup | cosine | onecycle | none
        "warmup_epochs": 5,
        "min_lr_ratio": 0.01,
        "patience": 12,
        "monitor": "val_auc",          # val_auc | val_loss | val_accuracy | val_f1 | val_balanced_accuracy
        "grad_clip": 1.0,
        "amp": True,
        "ema": False,
        "ema_decay": 0.999,
        "num_workers": "auto",
        "drop_last": True,
        "freeze_epochs": 0,            # Phase 3: freeze encoders for N epochs (used with SSL init)
        "finetune_lr_mult": 0.1,       # encoder LR multiplier when initialised from SSL
        "save_optimizer": True,
    },

    "cv": {
        "mode": "kfold",               # kfold | lopo | cluster
        "n_folds": 10,
        "inner_val_frac": 0.15,
        "folds": None,                 # subset of outer folds to run, e.g. [0, 1]
        "n_clusters": 50,              # cluster stress-test only
    },

    "threshold": {
        "start": 0.05, "stop": 0.95, "step": 0.05,
        "criterion": "youden",         # youden | accuracy | f1 | mcc | balanced_accuracy
    },

    "calibration": {"temperature": True, "n_bins": 15},

    "ssl": {
        "enabled": False,
        "mode": "strict",              # strict (per fold, inner-train only) | transductive
        "epochs": 50,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "temperature": 0.1,
        "proj_dim": 128,
        "aug": {
            "p_noise": 0.8, "noise_std": 0.1,
            "p_scale": 0.8, "scale_range": [0.7, 1.3],
            "p_time_mask": 0.5, "time_mask_frac": 0.1,
            "p_shift": 0.5, "shift_max_s": 0.5,
            "p_crop": 0.5, "crop_frac": 0.85,
            "p_warp": 0.3, "warp_max": 0.05,
            "p_bandstop": 0.3, "bandstop_width_hz": [2.0, 8.0], "bandstop_range_hz": [1.0, 100.0],
            "p_channel_dropout": 0.2,
            "p_polarity": 0.0,
            "p_specaug": 0.5, "specaug_frac": 0.1,
        },
    },

    "ddqn": {
        "enabled": False,
        "win_s": 4.0,
        "hop_s": 2.0,
        "history": 4,
        "encoder_epochs": 15,
        "encoder_lr": 5e-5,
        "encoder_patience": 4,
        "rewards": {
            "correct": 10.0,
            "false_focal": -10.0,
            "missed_focal": -12.0,
            "defer": -1.0,
            "early_bonus": 3.0,
            "early_wrong_penalty": 5.0,
        },
        "gamma": 0.95,
        "lr": 1e-3,
        "batch_size": 128,
        "buffer_size": 50000,
        "train_steps": 20000,
        "warmup_steps": 1000,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_steps": 10000,
        "target_tau": 0.005,
        "grad_clip": 1.0,
        "hidden": [256, 128],
        "dueling": True,
        "prioritized": False,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "eval_every": 1000,
        "select_frac": 0.2,            # part of inner-val used to select the DDQN policy
    },

    "hpo": {
        "n_trials": 20,
        "max_epochs": 30,
        "stage": "optim",
        "metric": "val_auc",
    },

    "eval": {
        "bootstrap": 2000,
        "explain": True,
        "explain_fold": 0,
        "n_error_examples": 5,
        "tsne": True,
        "ig_steps": 32,
    },

    "sanity": {
        "enabled": True,
        "epochs": 5,
        "max_train": 2000,
        "auc_limit": 0.62,
    },

    "logging": {"tensorboard": True},
}


# ---------------------------------------------------------------------------
# Model presets (ablation identities). Keys are CLI names for --model.
# ---------------------------------------------------------------------------
_BASELINE = {
    "exp_name": "baseline",
    "ablation_id": "A",
    "model": {"arch": "baseline", "use_raw": False, "use_grp": False, "use_transformer": False,
              "fusion": "gated"},
    "stft": {"representation": "log_magnitude", "crop": "half", "scaling": "scipy",
             "input_bn": False},
    "preprocess": {"normalization": "none", "dc_remove": False},
    "loss": {"name": "focal", "gamma": 2.0, "alpha": None},
    "augment": {"enabled": True, "p_noise": 0.0, "p_scale": 0.0, "p_time_mask": 0.0,
                "p_shift": 0.0, "p_specaug": 1.0, "specaug_frac": 0.08},
    "train": {"lr": 5e-4, "weight_decay": 1e-3, "batch_size": 32, "epochs": 40,
              "patience": 7, "scheduler": "cosine", "warmup_epochs": 0,
              "monitor": "val_accuracy"},
}

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": _BASELINE,
    "raw_stft": {
        "exp_name": "raw_stft", "ablation_id": "B",
        "model": {"use_raw": True, "use_stft": True, "use_grp": False,
                  "fusion": "gated", "use_transformer": False},
    },
    "stft_grp": {
        "exp_name": "stft_grp", "ablation_id": "C",
        "model": {"use_raw": False, "use_stft": True, "use_grp": True,
                  "fusion": "gated", "use_transformer": False},
    },
    "raw_stft_grp": {
        "exp_name": "raw_stft_grp", "ablation_id": "D",
        "model": {"use_raw": True, "use_stft": True, "use_grp": True,
                  "fusion": "gated", "use_transformer": False},
    },
    "cross_attention": {
        "exp_name": "cross_attention", "ablation_id": "E",
        "model": {"use_raw": True, "use_stft": True, "use_grp": True,
                  "fusion": "cross_attention", "use_transformer": False},
    },
    "full": {
        "exp_name": "transformer", "ablation_id": "F",
        "model": {"use_raw": True, "use_stft": True, "use_grp": True,
                  "fusion": "cross_attention", "use_transformer": True},
    },
    "full_ssl": {
        "exp_name": "ssl", "ablation_id": "G",
        "model": {"use_raw": True, "use_stft": True, "use_grp": True,
                  "fusion": "cross_attention", "use_transformer": True},
        "ssl": {"enabled": True},
        "train": {"freeze_epochs": 3},
    },
    "full_ddqn": {
        "exp_name": "ddqn", "ablation_id": "H",
        "model": {"use_raw": True, "use_stft": True, "use_grp": True,
                  "fusion": "cross_attention", "use_transformer": True},
        "ddqn": {"enabled": True},
    },
    "stft_hilbert_grp": {
        "exp_name": "stft_hilbert_grp", "ablation_id": "I",
        "model": {"use_raw": False, "use_stft": True, "use_grp": True, "use_hilbert": True,
                  "fusion": "cross_attention", "use_transformer": True},
    },
    "raw_stft_hilbert_grp": {
        "exp_name": "raw_stft_hilbert_grp", "ablation_id": "J",
        "model": {"use_raw": True, "use_stft": True, "use_grp": True, "use_hilbert": True,
                  "fusion": "cross_attention", "use_transformer": True},
    },
}

# letter aliases used by run_ablation.py
ABLATION_LETTERS = {v["ablation_id"]: k for k, v in MODEL_PRESETS.items()}


def deep_update(base: Dict[str, Any], upd: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not upd:
        return base
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _read_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as e:  # pragma: no cover
                raise ImportError("PyYAML is required for .yaml configs: pip install pyyaml") from e
            return yaml.safe_load(fh) or {}
        return json.load(fh)


def _parse_value(s: str) -> Any:
    low = s.strip().lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply CLI overrides of the form 'train.lr=3e-4'."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def load_config(path: Optional[str] = None, model: Optional[str] = None,
                overrides: Optional[Iterable[str]] = None) -> Cfg:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    file_cfg = _read_file(path) if path else {}
    preset_name = model or file_cfg.get("model_preset") or cfg["model_preset"]
    if preset_name in ABLATION_LETTERS:
        preset_name = ABLATION_LETTERS[preset_name]
    if preset_name not in MODEL_PRESETS:
        raise KeyError(f"Unknown model preset {preset_name!r}; choose from {list(MODEL_PRESETS)}")
    preset = copy.deepcopy(MODEL_PRESETS[preset_name])
    exp_name = preset.pop("exp_name")
    ablation_id = preset.pop("ablation_id")
    deep_update(cfg, preset)
    deep_update(cfg, file_cfg)
    apply_overrides(cfg, overrides)
    cfg["model_preset"] = preset_name
    cfg["ablation_id"] = ablation_id
    if not cfg.get("experiment_name"):
        cfg["experiment_name"] = exp_name
    if cfg["seed"] not in cfg["seeds"] and len(cfg["seeds"]) == 1:
        cfg["seeds"] = [cfg["seed"]]
    return Cfg(cfg)


def config_hash(cfg: Dict[str, Any]) -> str:
    d = cfg.to_dict() if isinstance(cfg, Cfg) else cfg
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def save_config(cfg: Cfg, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, default=str)
