import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
import copy

from bern_eeg.config import load_config
from bern_eeg.data.dataset import load_dataset
from bern_eeg.models.full_model import MultiModalEEGModel
from bern_eeg.train import train_fold
from bern_eeg.utils.metrics import compute_metrics
from bern_eeg.utils.seed import set_seed

class TorchEEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        # Return dict matching train.py expectations
        return {'signal': torch.tensor(self.X[idx]), 'label': torch.tensor(self.y[idx])}

def run_cross_validation(cfg, data_root=None):
    if data_root is None:
        data_root = cfg.data.root
        
    print(f"Starting Cross-Validation for model: {cfg.experiment_name} (Ablation {cfg.ablation_id})")
    
    # 1. Load full dataset
    data = load_dataset(cfg)
    
    # StratifiedGroupKFold (groups are duplicate clusters)
    cv = StratifiedGroupKFold(n_splits=cfg.cv.n_folds, shuffle=True, random_state=cfg.seed)
    
    X = data.X
    y = data.y
    groups = data.groups
    
    all_results = []
    oof_probs = np.zeros(len(y))
    oof_true = np.zeros(len(y))
    
    out_dir = os.path.join(cfg.output_root, cfg.experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
        
    for fold, (train_val_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        if cfg.cv.folds is not None and fold not in cfg.cv.folds:
            continue
            
        print(f"\n{'='*40}\nFold {fold+1}/{cfg.cv.n_folds}\n{'='*40}")
        
        # Inner split: 85% train, 15% val
        inner_cv = StratifiedGroupKFold(n_splits=int(1 / cfg.cv.inner_val_frac), shuffle=True, random_state=cfg.seed)
        inner_train_idx, inner_val_idx = next(inner_cv.split(
            X[train_val_idx], 
            y[train_val_idx], 
            groups[train_val_idx]
        ))
        
        # Map back to original indices
        train_idx = train_val_idx[inner_train_idx]
        val_idx = train_val_idx[inner_val_idx]
        
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Datasets
        train_ds = TorchEEGDataset(X[train_idx], y[train_idx])
        val_ds = TorchEEGDataset(X[val_idx], y[val_idx])
        test_ds = TorchEEGDataset(X[test_idx], y[test_idx])
        
        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4, drop_last=cfg.train.drop_last, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Create model
        set_seed(cfg.seed + fold)
        model = MultiModalEEGModel(cfg)
        
        # Train fold
        fold_res, test_probs, test_true = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            cfg=cfg,
            fold=fold,
            out_dir=out_dir
        )
        
        fold_res['fold'] = fold
        all_results.append(fold_res)
        
        oof_probs[test_idx] = test_probs
        oof_true[test_idx] = test_true
        
        # Save per-fold results
        pd.DataFrame(all_results).to_csv(os.path.join(out_dir, "fold_results.csv"), index=False)
        
    print("\nCross-Validation Complete.")
    
    # Compute overall OOF metrics
    # Only compute on indices that were actually tested (useful if cfg.cv.folds is used)
    tested_mask = (oof_probs != 0) | (oof_true != 0) # Simplistic mask
    if np.any(tested_mask):
        overall_metrics = compute_metrics(oof_true[tested_mask], oof_probs[tested_mask], threshold=0.5)
        print("Overall OOF Metrics:")
        for k, v in overall_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        pd.DataFrame([overall_metrics]).to_csv(os.path.join(out_dir, "overall_metrics.csv"), index=False)

if __name__ == "__main__":
    cfg = load_config(model="baseline")
    run_cross_validation(cfg)
