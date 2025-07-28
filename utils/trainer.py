#!/usr/bin/env python3
"""
Training and evaluation utilities for TAU synthetic data classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
import pandas as pd
import time
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class MetricsTracker:
    """Track and compute metrics during training"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
    def update(self, preds, targets, probs):
        self.predictions.extend(preds.detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())
        self.probabilities.extend(probs.detach().cpu().numpy())
    
    def compute_metrics(self):
        if len(self.predictions) == 0:
            return {}
            
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        probs = np.array(self.probabilities)[:, 1]  # Probability of positive class
        
        # Check for NaN values and replace them
        if np.any(np.isnan(probs)):
            print(f"Warning: Found {np.sum(np.isnan(probs))} NaN values in probabilities, replacing with 0.5")
            probs = np.nan_to_num(probs, nan=0.5)
            
        if np.any(np.isnan(preds)):
            print(f"Warning: Found {np.sum(np.isnan(preds))} NaN values in predictions, replacing with 0")
            preds = np.nan_to_num(preds, nan=0)
        
        # Ensure probabilities are in valid range [0, 1]
        probs = np.clip(probs, 0.0, 1.0)
        
        # Compute metrics
        auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.5
        f1 = f1_score(targets, preds, average='weighted')
        bacc = balanced_accuracy_score(targets, preds)
        
        # Confusion matrix
        cm = confusion_matrix(targets, preds)
        
        return {
            'auc': auc,
            'f1': f1,
            'bacc': bacc,
            'confusion_matrix': cm,
            'accuracy': (preds == targets).mean()
        }


class TAUTrainer:
    """
    Trainer class for TAU synthetic data classification
    """
    
    def __init__(
        self,
        model,
        device,
        criterion,
        optimizer,
        scheduler=None,
        early_stopping_patience=15,
        class_weights=None,
        results_dir='./results'
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.class_weights = class_weights
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Training history
        self.train_history = []
        self.val_history = []
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        train_metrics = MetricsTracker()
        
        total_loss = 0
        num_batches = len(train_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"Warning: NaN/Inf gradient in {name}, skipping update...")
                    has_nan_grad = True
                    break
                    
            if has_nan_grad:
                self.optimizer.zero_grad()
                continue
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            train_metrics.update(preds, labels, probs)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                current_metrics = train_metrics.compute_metrics()
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'AUC': f'{current_metrics.get("auc", 0):.3f}',
                    'F1': f'{current_metrics.get("f1", 0):.3f}',
                    'BACC': f'{current_metrics.get("bacc", 0):.3f}'
                })
        
        # Compute final metrics
        avg_loss = total_loss / num_batches
        final_metrics = train_metrics.compute_metrics()
        final_metrics['loss'] = avg_loss
        
        return final_metrics
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        val_metrics = MetricsTracker()
        
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            # Progress bar
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} - Validation')
            
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                val_metrics.update(preds, labels, probs)
        
        # Compute final metrics
        avg_loss = total_loss / num_batches
        final_metrics = val_metrics.compute_metrics()
        final_metrics['loss'] = avg_loss
        
        return final_metrics
    
    def test(self, test_loader):
        """Test the model"""
        self.model.eval()
        test_metrics = MetricsTracker()
        
        total_loss = 0
        num_batches = len(test_loader)
        
        all_subject_ids = []
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                test_metrics.update(preds, labels, probs)
                
                # Store detailed results
                all_subject_ids.extend(batch['subject_id'])
                all_probs.extend(probs.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
        
        # Compute final metrics
        avg_loss = total_loss / num_batches
        final_metrics = test_metrics.compute_metrics()
        final_metrics['loss'] = avg_loss
        
        # Create detailed results DataFrame
        detailed_results = pd.DataFrame({
            'subject_id': all_subject_ids,
            'true_label': all_labels,
            'predicted_label': all_preds,
            'prob_CN': [prob[0] for prob in all_probs],
            'prob_AD': [prob[1] for prob in all_probs]
        })
        
        return final_metrics, detailed_results
    
    def train(self, train_loader, val_loader, num_epochs, fold_idx, model_name):
        """Full training loop"""
        print(f"\nStarting training for {model_name} - Fold {fold_idx}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("-" * 50)
        
        best_val_auc = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Store history
            train_metrics['epoch'] = epoch + 1
            val_metrics['epoch'] = epoch + 1
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # Print epoch results
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{num_epochs} ({epoch_time:.1f}s) - "
                  f"Train: Loss={train_metrics['loss']:.4f}, AUC={train_metrics['auc']:.3f}, "
                  f"F1={train_metrics['f1']:.3f}, BACC={train_metrics['bacc']:.3f} | "
                  f"Val: Loss={val_metrics['loss']:.4f}, AUC={val_metrics['auc']:.3f}, "
                  f"F1={val_metrics['f1']:.3f}, BACC={val_metrics['bacc']:.3f}")
            
            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                self.save_model(fold_idx, model_name, epoch, val_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics['auc'], self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Training completed. Best validation AUC: {best_val_auc:.3f}")
        return self.train_history, self.val_history
    
    def save_model(self, fold_idx, model_name, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'fold': fold_idx,
            'model_name': model_name,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        filename = f"{model_name}_fold_{fold_idx}_best.pth"
        filepath = os.path.join(self.results_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    # Enhanced inverse frequency weighting for severe imbalance
    weights = total_samples / (len(unique_labels) * counts)
    
    # Apply additional balancing for severe imbalance (4:1 ratio)
    if len(unique_labels) == 2 and counts[1] < counts[0] * 0.3:  # Minority class < 30%
        # Boost minority class weight
        weights[1] *= 1.5  # Additional 50% boost for AD class
    
    return torch.FloatTensor(weights)


def get_optimizer(model, optimizer_name='adamw', lr=1e-4, weight_decay=1e-5):
    """Get optimizer"""
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='cosine', num_epochs=100):
    """Get learning rate scheduler"""
    if scheduler_name.lower() == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name.lower() == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif scheduler_name.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


if __name__ == "__main__":
    # Test trainer
    from models.models import get_model
    
    # Create dummy model and data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_model('resnet18', enhance_small_features=True).to(device)
    
    # Test metrics tracker
    tracker = MetricsTracker()
    
    # Dummy predictions
    dummy_preds = torch.tensor([0, 1, 0, 1])
    dummy_targets = torch.tensor([0, 1, 1, 0])
    dummy_probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
    
    tracker.update(dummy_preds, dummy_targets, dummy_probs)
    metrics = tracker.compute_metrics()
    
    print("Test metrics:", metrics)
    print("Trainer testing completed successfully!") 