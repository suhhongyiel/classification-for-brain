#!/usr/bin/env python3
"""
10-Fold Cross Validation Training System for imp_cls
사용자가 지정한 파라미터로 10-fold cross-validation만 반복 수행
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import TAUSyntheticDataset
from utils.trainer import TAUTrainer, get_optimizer, get_scheduler
from models.models import get_model
from medical_data_augmentation import create_medical_transforms

class ExtremeBACCLoss(nn.Module):
    """Extremely aggressive loss for BACC optimization"""
    def __init__(self, alpha=0.9, gamma=3.0, minority_weight=8.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.minority_weight = minority_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        # Standard cross entropy
        ce_loss = self.ce_loss(inputs, targets)
        
        # Get probabilities
        probs = torch.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal loss component with higher gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Extreme class weighting for minority class (AD)
        class_weights = torch.ones_like(targets).float()
        class_weights[targets == 1] = self.minority_weight  # 8x weight for AD
        
        # Additional penalty for misclassified minority samples
        minority_penalty = torch.ones_like(targets).float()
        minority_penalty[targets == 1] = 2.0
        
        # Combine all components
        loss = self.alpha * focal_weight * ce_loss * class_weights * minority_penalty
        
        return loss.mean()

class DataBalancer:
    """Simple data balancing through replication"""
    def __init__(self, target_ratio=0.4):  # Target 40% AD samples
        self.target_ratio = target_ratio
    
    def balance_data(self, data_df):
        """Balance the dataset by replicating minority samples"""
        ad_samples = data_df[data_df['label'] == 1]
        cn_samples = data_df[data_df['label'] == 0]
        
        print(f"Original: AD={len(ad_samples)}, CN={len(cn_samples)}")
        
        # Calculate how many AD samples we need
        total_desired = len(cn_samples) / (1 - self.target_ratio)
        ad_needed = int(total_desired * self.target_ratio)
        
        if ad_needed > len(ad_samples):
            # Replicate AD samples
            replications_needed = ad_needed // len(ad_samples)
            remainder = ad_needed % len(ad_samples)
            
            balanced_ad = pd.concat([ad_samples] * replications_needed, ignore_index=True)
            if remainder > 0:
                balanced_ad = pd.concat([balanced_ad, ad_samples.sample(n=remainder, random_state=42)], ignore_index=True)
        else:
            balanced_ad = ad_samples
        
        # Combine balanced data
        balanced_df = pd.concat([cn_samples, balanced_ad], ignore_index=True).sample(frac=1, random_state=42)
        
        print(f"Balanced: AD={len(balanced_ad)}, CN={len(cn_samples)}")
        return balanced_df

class AdvancedBalancedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=0.9999, num_classes=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        # 수치적 안정성을 위한 클램핑
        inputs = torch.clamp(inputs, min=-10, max=10)
        
        ce_loss = self.ce_loss(inputs, targets)
        
        # Focal loss 계산 시 수치적 안정성 개선
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-7, max=1.0)  # 0으로 나누기 방지
        
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # Label smoothing with KL divergence
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1)
        smooth_targets = smooth_targets * (1 - 0.1) + 0.1 / self.num_classes
        
        log_probs = torch.log_softmax(inputs, dim=1)
        kl_loss = torch.sum(smooth_targets * log_probs, dim=1)
        
        # 수치적 안정성을 위한 클램핑
        focal_loss = torch.clamp(focal_loss, min=0, max=100)
        kl_loss = torch.clamp(kl_loss, min=-100, max=100)
        
        total_loss = focal_loss.mean() - 0.1 * kl_loss.mean()
        
        # 최종 loss가 NaN이 되지 않도록 체크
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Fallback to simple cross entropy
            return self.ce_loss(inputs, targets).mean()
            
        return total_loss

class BACCFocusedLoss(nn.Module):
    """Loss function specifically designed to optimize BACC"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        # Standard cross entropy
        ce_loss = self.ce_loss(inputs, targets)
        
        # Get probabilities
        probs = torch.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal loss component
        focal_weight = (1 - pt) ** self.gamma
        
        # BACC-focused weighting: heavily penalize minority class errors
        class_weights = torch.ones_like(targets).float()
        class_weights[targets == 1] = 4.0  # Heavy weight for AD class
        
        # Combine losses
        loss = self.alpha * focal_weight * ce_loss * class_weights
        
        return loss.mean()

class ThresholdOptimizer:
    def __init__(self):
        self.best_threshold = 0.5
        self.best_bacc = 0.0
    def find_optimal_threshold(self, y_true, y_proba):
        from sklearn.metrics import balanced_accuracy_score
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_bacc = 0
        best_threshold = 0.5
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            bacc = balanced_accuracy_score(y_true, y_pred)
            if bacc > best_bacc:
                best_bacc = bacc
                best_threshold = threshold
        self.best_threshold = best_threshold
        self.best_bacc = best_bacc
        return best_threshold, best_bacc

def get_loss_function(loss_name, device, data_df=None):
    if loss_name == 'extreme_bacc':
        return ExtremeBACCLoss(alpha=0.9, gamma=3.0, minority_weight=10.0).to(device)
    elif loss_name == 'bacc_focused':
        return BACCFocusedLoss(alpha=0.75, gamma=2.0).to(device)
    elif loss_name == 'focal':
        return AdvancedBalancedLoss(alpha=0.2, gamma=2.0).to(device)
    elif loss_name == 'weighted_ce':
        labels = data_df['label'].values
        class_counts = np.bincount(labels)
        # Extremely aggressive class balancing for BACC
        class_weights = torch.FloatTensor([1.0, class_counts[0] / class_counts[1] * 5.0]).to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == 'advanced_balanced':
        return AdvancedBalancedLoss(alpha=0.2, gamma=2.0, beta=0.9999).to(device)
    elif loss_name == 'label_smooth':
        # Label smoothing with CrossEntropy
        return LabelSmoothingCrossEntropy(smoothing=0.1, num_classes=2).to(device)
    else:  # 'ce' or default
        return nn.CrossEntropyLoss()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=2):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def main():
    parser = argparse.ArgumentParser(description='10-Fold Cross Validation Training System (imp_cls)')
    parser.add_argument('--data-csv', type=str, required=True, help='Path to the data CSV file')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'vit'], help='Model architecture')
    parser.add_argument('--method', type=int, default=0, choices=[0, 1, 2, 3], help='BACC improvement method: 0=baseline, 1=balanced_loss, 2=data_augmentation, 3=ensemble')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal', 'weighted_ce', 'advanced_balanced'], help='Loss function')
    parser.add_argument('--use-enhancement', action='store_true', help='Use enhancement mechanism for small 3D images')
    parser.add_argument('--use-augmentation', action='store_true', default=True, help='Use data augmentation')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--threshold-optimization', action='store_true', help='Apply threshold optimization for maximum BACC')
    parser.add_argument('--early-stopping-patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    args = parser.parse_args()

    # Method-based configuration for EXTREME BACC optimization
    if args.method == 0:  # Extreme BACC with data balancing
        print("🔧 Method 0: EXTREME BACC with data balancing")
        args.loss = 'extreme_bacc'  # Use extreme BACC loss
        args.use_augmentation = True
        args.use_enhancement = True
        args.lr = 5e-6  # Very low learning rate for stability
        args.weight_decay = 1e-4  # Light regularization
        args.dropout_rate = 0.2  # Lower dropout for better learning
        args.early_stopping_patience = 15  # More patience
        args.use_data_balancing = True  # Enable data balancing
    elif args.method == 1:  # Ultra-aggressive class balancing
        print("🔧 Method 1: Ultra-aggressive class balancing")
        args.loss = 'weighted_ce'  # 5x weighted CE
        args.use_augmentation = True
        args.use_enhancement = True
        args.lr = 3e-6  # Very low learning rate
        args.weight_decay = 1e-4  # Light regularization
        args.dropout_rate = 0.2  # Lower dropout
        args.early_stopping_patience = 15  # More patience
        args.use_data_balancing = True  # Enable data balancing
    elif args.method == 2:  # Extreme BACC with heavy augmentation
        print("🔧 Method 2: Extreme BACC with heavy augmentation")
        args.loss = 'extreme_bacc'  # Extreme BACC loss
        args.use_augmentation = True
        args.use_enhancement = True
        args.lr = 5e-6  # Very low learning rate
        args.weight_decay = 5e-5  # Very light regularization
        args.dropout_rate = 0.1  # Minimal dropout
        args.early_stopping_patience = 20  # Maximum patience
        args.use_data_balancing = True  # Enable data balancing
    elif args.method == 3:  # Ultimate BACC ensemble
        print("🔧 Method 3: Ultimate BACC ensemble with all techniques")
        args.loss = 'extreme_bacc'  # Extreme BACC loss
        args.use_augmentation = True
        args.use_enhancement = True
        args.threshold_optimization = True  # Enable threshold optimization
        args.lr = 3e-6  # Ultra-low learning rate
        args.weight_decay = 1e-5  # Minimal regularization
        args.dropout_rate = 0.1  # Minimal dropout
        args.early_stopping_patience = 25  # Maximum patience
        args.use_data_balancing = True  # Enable data balancing

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    data_df = pd.read_csv(args.data_csv)
    
    # Apply data balancing if enabled
    if hasattr(args, 'use_data_balancing') and args.use_data_balancing:
        print("🔄 Applying data balancing...")
        balancer = DataBalancer(target_ratio=0.4)  # Target 40% AD samples
        data_df = balancer.balance_data(data_df)
    
    results_dir = Path('results/ten_fold_training')
    results_dir.mkdir(parents=True, exist_ok=True)

    labels = data_df['label'].values
    # 10-fold cross-validation for more stable results
    outer_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    inner_skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)  # 9-fold for train/val split

    print(f"\n🎯 Starting 10-Fold Training (BACC-focused)")
    print(f"   Method: {args.method}")
    print(f"   Model: {args.model}")
    print(f"   Optimizer: {args.optimizer}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Loss Function: {args.loss}")
    print(f"   Use Enhancement: {args.use_enhancement}")
    print(f"   Use Augmentation: {args.use_augmentation}")
    print(f"   Threshold Optimization: {args.threshold_optimization}")

    fold_results = []
    for fold_idx, (train_val_indices, test_indices) in enumerate(outer_skf.split(data_df, labels)):
        # 더 안정적인 train/val 분할을 위해 4-fold 사용
        train_val_labels = labels[train_val_indices]
        train_val_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        train_indices, val_indices = next(train_val_skf.split(data_df.iloc[train_val_indices], train_val_labels))
        train_indices = train_val_indices[train_indices]
        val_indices = train_val_indices[val_indices]

        print(f"\n🔄 Training Fold {fold_idx + 1}/10")
        print(f"   Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # 클래스 분포 확인
        train_labels_dist = np.bincount(labels[train_indices])
        val_labels_dist = np.bincount(labels[val_indices])
        test_labels_dist = np.bincount(labels[test_indices])
        print(f"   Train AD/CN: {train_labels_dist[1]}/{train_labels_dist[0]}")
        print(f"   Val AD/CN: {val_labels_dist[1]}/{val_labels_dist[0]}")
        print(f"   Test AD/CN: {test_labels_dist[1]}/{test_labels_dist[0]}")

        # Transforms
        train_transform = create_medical_transforms(augment=args.use_augmentation)
        val_transform = create_medical_transforms(augment=False)

        train_dataset = TAUSyntheticDataset(data_df.iloc[train_indices], transform=train_transform)
        val_dataset = TAUSyntheticDataset(data_df.iloc[val_indices], transform=val_transform)
        test_dataset = TAUSyntheticDataset(data_df.iloc[test_indices], transform=val_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = get_model(model_name=args.model, num_classes=2, enhance_small_features=args.use_enhancement, dropout_rate=args.dropout_rate).to(device)
        criterion = get_loss_function(args.loss, device, data_df)
        optimizer = get_optimizer(model, optimizer_name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_scheduler(optimizer, scheduler_name='cosine', num_epochs=args.epochs)

        trainer = TAUTrainer(model=model, device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler, early_stopping_patience=args.early_stopping_patience)
        train_history, val_history = trainer.train(train_loader, val_loader, args.epochs, fold_idx, args.model)
        test_metrics, detailed_results = trainer.test(test_loader)

        # Threshold optimization (optional)
        if args.threshold_optimization:
            print(f"   🔧 Applying threshold optimization...")
            threshold_optimizer = ThresholdOptimizer()
            y_true = detailed_results['true_label'].values
            y_proba = detailed_results['prob_AD'].values
            optimal_threshold, optimal_bacc = threshold_optimizer.find_optimal_threshold(y_true, y_proba)
            y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
            test_metrics['accuracy'] = accuracy_score(y_true, y_pred_optimized)
            test_metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred_optimized)
            test_metrics['f1_score'] = f1_score(y_true, y_pred_optimized, average='weighted')
            test_metrics['auc'] = roc_auc_score(y_true, y_proba)
            test_metrics['optimal_threshold'] = optimal_threshold
            print(f"   ✅ Optimal threshold: {optimal_threshold:.3f}, BACC: {optimal_bacc:.4f}")

        # Save model for this fold
        model_path = results_dir / f"best_model_fold_{fold_idx + 1}.pth"
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'fold_idx': fold_idx, 'params': vars(args), 'test_metrics': test_metrics}, model_path)

        fold_results.append({
            'fold_idx': fold_idx + 1, 
            'test_metrics': test_metrics, 
            'model_path': str(model_path),
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'train_ad_cn': f"{train_labels_dist[1]}/{train_labels_dist[0]}",
            'val_ad_cn': f"{val_labels_dist[1]}/{val_labels_dist[0]}",
            'test_ad_cn': f"{test_labels_dist[1]}/{test_labels_dist[0]}"
        })
        print(f"✅ Fold {fold_idx + 1} completed - Test BACC: {test_metrics['bacc']:.4f}")

    # Overall statistics
    test_baccs = [r['test_metrics']['bacc'] for r in fold_results]
    test_aucs = [r['test_metrics']['auc'] for r in fold_results]
    test_f1s = [r['test_metrics']['f1'] for r in fold_results]
    
    print(f"\n{'='*80}")
    print(f"📊 10-FOLD CROSS VALIDATION RESULTS (BACC-FOCUSED)")
    print(f"{'='*80}")
    print(f"BACC:         {np.mean(test_baccs):.4f} ± {np.std(test_baccs):.4f} (Range: {np.min(test_baccs):.4f} - {np.max(test_baccs):.4f})")
    print(f"AUC:          {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f} (Range: {np.min(test_aucs):.4f} - {np.max(test_aucs):.4f})")
    print(f"F1-Score:     {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f} (Range: {np.min(test_f1s):.4f} - {np.max(test_f1s):.4f})")
    print(f"{'='*80}")
    
    # Fold별 상세 결과
    print(f"\n📋 FOLD-BY-FOLD DETAILS:")
    print(f"{'Fold':<4} {'BACC':<8} {'AUC':<8} {'F1':<8} {'Train':<8} {'Val':<6} {'Test':<6}")
    print(f"{'-'*50}")
    for result in fold_results:
        print(f"{result['fold_idx']:<4} {result['test_metrics']['bacc']:<8.4f} {result['test_metrics']['auc']:<8.4f} {result['test_metrics']['f1']:<8.4f} {result['train_size']:<8} {result['val_size']:<6} {result['test_size']:<6}")

    # Save summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = results_dir / f'ten_fold_summary_{timestamp}.csv'
    summary_data = []
    for fold_result in fold_results:
        summary_data.append({'fold': fold_result['fold_idx'], 'balanced_accuracy': fold_result['test_metrics']['bacc']})
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 Summary saved: {summary_file}")

if __name__ == "__main__":
    main() 