import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from bern_eeg.utils.metrics import compute_metrics, find_best_threshold
from bern_eeg.utils.hardware import get_device

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return focal_loss.mean()

def get_loss_fn(cfg):
    name = cfg.loss.name
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "focal":
        return FocalLoss(gamma=cfg.loss.gamma, alpha=cfg.loss.alpha)
    else:
        return nn.BCEWithLogitsLoss()

def train_one_epoch(model, dataloader, optimizer, loss_fn, scaler, device, cfg):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # dataset returns a dictionary or tuple? Let's check dataset.py if we need to.
        # Assuming dataset returns (signals, label) or dict.
        # Let's assume (signals, label) for simplicity, or dict with 'signal', 'label'.
        if isinstance(batch, dict):
            x = batch['signal'].to(device)
            y = batch['label'].to(device).float().unsqueeze(1)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)
            
        optimizer.zero_grad()
        
        if scaler is not None and cfg.train.amp:
            with autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                
            optimizer.step()
            
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device, cfg):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['signal'].to(device)
                y = batch['label'].to(device).float().unsqueeze(1)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device).float().unsqueeze(1)
                
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            
    avg_loss = total_loss / len(dataloader)
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    
    metrics = compute_metrics(all_labels, all_probs)
    metrics['loss'] = avg_loss
    return metrics, all_probs, all_labels

def train_fold(model, train_loader, val_loader, test_loader, cfg, fold, out_dir):
    device = get_device(cfg.device)
    model = model.to(device)
    
    loss_fn = get_loss_fn(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    
    # Cosine scheduler for now
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    
    scaler = GradScaler() if (cfg.train.amp and device.type == "cuda") else None
    
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(cfg.train.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, cfg)
        val_metrics, val_probs, val_labels = evaluate(model, val_loader, loss_fn, device, cfg)
        
        scheduler.step()
        
        score = val_metrics[cfg.train.monitor.replace('val_', '')]
        
        print(f"Epoch {epoch+1}/{cfg.train.epochs} - Train Loss: {train_loss:.4f} - Val {cfg.train.monitor}: {score:.4f} (Loss: {val_metrics['loss']:.4f})")
        
        if score > best_val_score:
            best_val_score = score
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            
            # Save checkpoint
            ckpt_path = os.path.join(out_dir, f"fold_{fold}_best.pt")
            torch.save(best_model_state, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on val again to find optimal threshold
    _, val_probs, val_labels = evaluate(model, val_loader, loss_fn, device, cfg)
    best_t = find_best_threshold(val_labels, val_probs, criterion=cfg.threshold.criterion)
    print(f"Best threshold on val: {best_t:.4f}")
    
    # Evaluate on test
    test_metrics, test_probs, test_labels = evaluate(model, test_loader, loss_fn, device, cfg)
    
    # Recompute test metrics with optimal threshold
    final_test_metrics = compute_metrics(test_labels, test_probs, threshold=best_t)
    final_test_metrics['loss'] = test_metrics['loss']
    final_test_metrics['best_threshold'] = best_t
    
    print("Test Metrics:")
    for k, v in final_test_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    return final_test_metrics, test_probs, test_labels
