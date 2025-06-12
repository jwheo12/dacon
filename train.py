
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
import timm
from dataload import get_loaders, CFG, seed_everything
from sklearn.metrics import log_loss
from tqdm import tqdm

# Initialize W&B


# Seed and device
seed_everything(CFG['SEED'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data loaders
train_loader, val_loader, _, class_names = get_loaders()

# Model definition
class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = timm.create_model(
            CFG['model'],
            pretrained=True,
            num_classes=num_classes
        )
    def forward(self, x):
        return self.backbone(x)

# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        at = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
        fl = at * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return fl.mean()
        elif self.reduction == 'sum':
            return fl.sum()
        else:
            return fl
def main():
    run = wandb.init(project="kon")
    wandb.config.update({
        "learning_rate": CFG['LEARNING_RATE'],
        "epochs": CFG['EPOCHS'],
        "batch_size": CFG['BATCH_SIZE'],
        "optimizer": "AdamW",
        "loss_function": "FocalLoss",
        "model": CFG['model'],
    })
    # Instantiate model and freeze backbone initial epochs
    model = BaseModel(num_classes=len(class_names)).to(device)
    freeze_epochs = 10
    for name, param in model.backbone.named_parameters():
        if 'head' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Optimizer, scheduler, scaler
    optimizer = optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=0.05)
    best_logloss = float('inf')
    total_steps = len(train_loader) * CFG['EPOCHS']
    warmup_steps = len(train_loader) * 3
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = GradScaler()
    patience = CFG['patience']
    trigger_times = 0
    best_val = float('inf')
    
    # Class weights for focal loss
    # Compute class counts from train_loader.dataset.dataset.samples if using Subset
    try:
        samples = train_loader.dataset.dataset.samples
    except:
        samples = train_loader.dataset.samples
    labels = [label for _, label in samples]
    cls_counts = np.bincount(labels)
    total_count = sum(cls_counts)
    criterion = FocalLoss(alpha=torch.tensor(total_count / (len(cls_counts) * cls_counts)).to(device), gamma=2.0)
    
    # Training loop
    for epoch in range(CFG['EPOCHS']):
        if epoch == freeze_epochs:
            print(f"▶ Epoch {epoch+1}: Feature Extractor unfreeze and full-model fine-tuning 시작")
            for name, param in model.backbone.named_parameters():
                param.requires_grad = True
    
        model.train()
        train_loss = 0.0
        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training", leave=False)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    
            train_loss += loss.item()
            if (step + 1) % 10 == 0:
                wandb.log({
                    "epoch": epoch + 1,
                    "step": epoch * len(train_loader) + step + 1,
                    "lr": scheduler.get_last_lr()[0]
                })
    
        avg_train_loss = train_loss / len(train_loader)
    
        model.eval()
        val_loss = 0.0
        correct = 0
        total_samples = 0
        all_probs, all_labels = [], []
    
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total_samples
        val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))
    
        print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")
        if val_logloss < best_logloss:
            best_logloss = val_logloss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"📦 Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")
    
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_logloss": val_logloss
        })
    
        if val_logloss < best_val:
            best_val = val_logloss
            #torch.save(model.state_dict(), 'best_swin.pth')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                wandb.save('best_model.pth')
                break
    
    wandb.finish()
if __name__ == "__main__":
    main()
