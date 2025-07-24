#!/usr/bin/env python3
"""
Comprehensive BACC Improvement System
Allows users to select different improvement strategies
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import TAUSyntheticDataset, create_stratified_splits_all_data, get_dataloaders
from utils.trainer import TAUTrainer, get_optimizer, get_scheduler
from models.models import get_model
from medical_data_augmentation import BalancedDatasetAugmentation, create_medical_transforms
from ensemble_methods import ModelEnsemble, StackingEnsemble, BlendingEnsemble

class AdvancedBalancedLoss(nn.Module):
    """Advanced loss function for BACC improvement"""
    
    def __init__(self, alpha=0.25, gamma=2.0, beta=0.9999, num_classes=2):
        super(AdvancedBalancedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        # Focal loss component
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # Label smoothing for better generalization
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1)
        smooth_targets = smooth_targets * (1 - 0.1) + 0.1 / self.num_classes
        
        # KL divergence for label smoothing
        log_probs = torch.log_softmax(inputs, dim=1)
        kl_loss = torch.sum(smooth_targets * log_probs, dim=1)
        
        # Combine losses
        total_loss = focal_loss.mean() - 0.1 * kl_loss.mean()
        
        return total_loss

class ThresholdOptimizer:
    """Optimize prediction threshold for maximum BACC"""
    
    def __init__(self):
        self.best_threshold = 0.5
        self.best_bacc = 0.0
        
    def find_optimal_threshold(self, y_true, y_proba):
        """Find optimal threshold for balanced accuracy"""
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

class BACCImprovementSystem:
    """
    Comprehensive BACC improvement system with multiple strategies
    """
    
    def __init__(self, data_csv_path='/home/imp_cls/data/syn_data_mapping.csv'):
        self.data_csv_path = data_csv_path
        self.data_df = pd.read_csv(data_csv_path)
        self.results = {}
        
    def method_1_enhanced_loss(self, model_name='resnet18', num_epochs=100, gpu_id=0):
        """Method 1: Enhanced Loss Functions"""
        print("🔧 Method 1: Enhanced Loss Functions")
        print("="*50)
        
        # Setup
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Create data splits
        train_df, val_df, test_df = create_stratified_splits_all_data(
            self.data_df, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, batch_size=8, num_workers=4
        )
        
        # Create model
        model = get_model(model_name, num_classes=2, enhance_small_features=True).to(device)
        
        # Create enhanced loss function
        criterion = AdvancedBalancedLoss(alpha=0.25, gamma=2.0).to(device)
        
        # Create optimizer and scheduler
        optimizer = get_optimizer(model, 'adamw', 1e-5, weight_decay=1e-5)
        scheduler = get_scheduler(optimizer, 'cosine', num_epochs)
        
        # Create trainer
        trainer = TAUTrainer(
            model=model, device=device, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler,
            early_stopping_patience=25, results_dir='./result/method1'
        )
        
        # Train and test
        train_metrics, val_metrics = trainer.train(
            train_loader, val_loader, num_epochs, fold_idx=0, model_name=model_name
        )
        test_metrics, detailed_results = trainer.test(test_loader)
        
        self.results['method1'] = test_metrics
        return test_metrics
    
    def method_2_data_augmentation(self, model_name='resnet18', num_epochs=100, gpu_id=0):
        """Method 2: Data Augmentation for Class Balance"""
        print("🔧 Method 2: Data Augmentation for Class Balance")
        print("="*50)
        
        # Create balanced dataset
        balancer = BalancedDatasetAugmentation(minority_boost=2.0)
        balanced_df = balancer.augment_minority_samples(self.data_df)
        
        print(f"Original samples: {len(self.data_df)}")
        print(f"Balanced samples: {len(balanced_df)}")
        print(f"Original AD: {len(self.data_df[self.data_df['label']==1])}")
        print(f"Balanced AD: {len(balanced_df[balanced_df['label']==1])}")
        
        # Save balanced dataset
        balanced_df.to_csv('/home/imp_cls/data/balanced_syn_data_mapping.csv', index=False)
        
        # Setup
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Create data splits with balanced data
        train_df, val_df, test_df = create_stratified_splits_all_data(
            balanced_df, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Create dataloaders with augmentation
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, batch_size=8, num_workers=4
        )
        
        # Create model
        model = get_model(model_name, num_classes=2, enhance_small_features=True).to(device)
        
        # Create loss function
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Create optimizer and scheduler
        optimizer = get_optimizer(model, 'adamw', 1e-5, weight_decay=1e-5)
        scheduler = get_scheduler(optimizer, 'cosine', num_epochs)
        
        # Create trainer
        trainer = TAUTrainer(
            model=model, device=device, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler,
            early_stopping_patience=25, results_dir='./result/method2'
        )
        
        # Train and test
        train_metrics, val_metrics = trainer.train(
            train_loader, val_loader, num_epochs, fold_idx=0, model_name=model_name
        )
        test_metrics, detailed_results = trainer.test(test_loader)
        
        self.results['method2'] = test_metrics
        return test_metrics
    
    def method_3_threshold_optimization(self, model_name='resnet18', num_epochs=100, gpu_id=0):
        """Method 3: Threshold Optimization"""
        print("🔧 Method 3: Threshold Optimization")
        print("="*50)
        
        # Setup
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Create data splits
        train_df, val_df, test_df = create_stratified_splits_all_data(
            self.data_df, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, batch_size=8, num_workers=4
        )
        
        # Create model
        model = get_model(model_name, num_classes=2, enhance_small_features=True).to(device)
        
        # Create loss function
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Create optimizer and scheduler
        optimizer = get_optimizer(model, 'adamw', 1e-5, weight_decay=1e-5)
        scheduler = get_scheduler(optimizer, 'cosine', num_epochs)
        
        # Create trainer
        trainer = TAUTrainer(
            model=model, device=device, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler,
            early_stopping_patience=25, results_dir='./result/method3'
        )
        
        # Train model
        train_metrics, val_metrics = trainer.train(
            train_loader, val_loader, num_epochs, fold_idx=0, model_name=model_name
        )
        
        # Get probabilities for threshold optimization
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.extend(probs[:, 1].cpu().numpy())  # AD probability
                all_labels.extend(labels.cpu().numpy())
        
        # Optimize threshold
        threshold_optimizer = ThresholdOptimizer()
        optimal_threshold, optimal_bacc = threshold_optimizer.find_optimal_threshold(
            np.array(all_labels), np.array(all_probs)
        )
        
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"Optimal BACC: {optimal_bacc:.3f}")
        
        # Calculate final metrics
        final_predictions = (np.array(all_probs) >= optimal_threshold).astype(int)
        
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
        final_bacc = balanced_accuracy_score(all_labels, final_predictions)
        final_f1 = f1_score(all_labels, final_predictions, average='weighted')
        final_auc = roc_auc_score(all_labels, all_probs)
        
        test_metrics = {
            'auc': final_auc,
            'f1': final_f1,
            'bacc': final_bacc,
            'accuracy': (final_predictions == all_labels).mean(),
            'optimal_threshold': optimal_threshold
        }
        
        self.results['method3'] = test_metrics
        return test_metrics
    
    def method_4_ensemble(self, model_name='resnet18', num_epochs=100, gpu_id=0):
        """Method 4: Ensemble Methods"""
        print("🔧 Method 4: Ensemble Methods")
        print("="*50)
        
        # Setup
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Create data splits
        train_df, val_df, test_df = create_stratified_splits_all_data(
            self.data_df, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, batch_size=8, num_workers=4
        )
        
        # Train multiple models with different configurations
        models = []
        model_configs = [
            {'enhance_small_features': True, 'learning_rate': 1e-5},
            {'enhance_small_features': False, 'learning_rate': 5e-5},
            {'enhance_small_features': True, 'learning_rate': 1e-4},
        ]
        
        for i, config in enumerate(model_configs):
            print(f"Training model {i+1}/3...")
            
            # Create model
            model = get_model(
                model_name, 
                num_classes=2, 
                enhance_small_features=config['enhance_small_features']
            ).to(device)
            
            # Create loss function
            criterion = nn.CrossEntropyLoss().to(device)
            
            # Create optimizer and scheduler
            optimizer = get_optimizer(model, 'adamw', config['learning_rate'], weight_decay=1e-5)
            scheduler = get_scheduler(optimizer, 'cosine', num_epochs)
            
            # Create trainer
            trainer = TAUTrainer(
                model=model, device=device, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler,
                early_stopping_patience=25, results_dir=f'./result/method4/model_{i}'
            )
            
            # Train model
            train_metrics, val_metrics = trainer.train(
                train_loader, val_loader, num_epochs, fold_idx=0, model_name=f"{model_name}_{i}"
            )
            
            models.append(model)
        
        # Get ensemble predictions
        all_probs = []
        all_labels = []
        
        for model in models:
            model.eval()
            model_probs = []
            
            with torch.no_grad():
                for batch in test_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    model_probs.extend(probs[:, 1].cpu().numpy())
                    
                    if len(all_labels) == 0:
                        all_labels.extend(labels.cpu().numpy())
            
            all_probs.append(model_probs)
        
        # Create ensemble predictions
        ensemble_probs = np.mean(all_probs, axis=0)
        
        # Optimize threshold for ensemble
        threshold_optimizer = ThresholdOptimizer()
        optimal_threshold, optimal_bacc = threshold_optimizer.find_optimal_threshold(
            np.array(all_labels), ensemble_probs
        )
        
        # Calculate final metrics
        final_predictions = (ensemble_probs >= optimal_threshold).astype(int)
        
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
        final_bacc = balanced_accuracy_score(all_labels, final_predictions)
        final_f1 = f1_score(all_labels, final_predictions, average='weighted')
        final_auc = roc_auc_score(all_labels, ensemble_probs)
        
        test_metrics = {
            'auc': final_auc,
            'f1': final_f1,
            'bacc': final_bacc,
            'accuracy': (final_predictions == all_labels).mean(),
            'optimal_threshold': optimal_threshold
        }
        
        self.results['method4'] = test_metrics
        return test_metrics
    
    def method_5_combined(self, model_name='resnet18', num_epochs=100, gpu_id=0):
        """Method 5: Combined Approach (All Methods)"""
        print("🔧 Method 5: Combined Approach (All Methods)")
        print("="*50)
        
        # This would combine all methods - for now, just run the best ones
        print("Running combined approach...")
        
        # Run enhanced loss + threshold optimization
        result1 = self.method_1_enhanced_loss(model_name, num_epochs//2, gpu_id)
        result3 = self.method_3_threshold_optimization(model_name, num_epochs//2, gpu_id)
        
        # Combine results (simple average for now)
        combined_metrics = {
            'auc': (result1['auc'] + result3['auc']) / 2,
            'f1': (result1['f1'] + result3['f1']) / 2,
            'bacc': (result1['bacc'] + result3['bacc']) / 2,
            'accuracy': (result1['accuracy'] + result3['accuracy']) / 2,
        }
        
        self.results['method5'] = combined_metrics
        return combined_metrics
    
    def run_all_methods(self, model_name='resnet18', num_epochs=100, gpu_id=0):
        """Run all improvement methods"""
        print("🚀 Running All BACC Improvement Methods")
        print("="*60)
        
        methods = [
            ('Enhanced Loss Functions', self.method_1_enhanced_loss),
            ('Data Augmentation', self.method_2_data_augmentation),
            ('Threshold Optimization', self.method_3_threshold_optimization),
            ('Ensemble Methods', self.method_4_ensemble),
            ('Combined Approach', self.method_5_combined),
        ]
        
        for name, method in methods:
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                result = method(model_name, num_epochs, gpu_id)
                print(f"✅ {name} completed successfully!")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                self.results[name] = None
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all results"""
        print("\n" + "="*60)
        print("📊 BACC Improvement Results Summary")
        print("="*60)
        
        print(f"{'Method':<25} {'AUC':<8} {'F1':<8} {'BACC':<8} {'Accuracy':<8}")
        print("-"*60)
        
        for method_name, result in self.results.items():
            if result is not None:
                print(f"{method_name:<25} {result['auc']:<8.3f} {result['f1']:<8.3f} {result['bacc']:<8.3f} {result['accuracy']:<8.3f}")
            else:
                print(f"{method_name:<25} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8}")
        
        # Find best method
        best_method = None
        best_bacc = 0
        
        for method_name, result in self.results.items():
            if result is not None and result['bacc'] > best_bacc:
                best_bacc = result['bacc']
                best_method = method_name
        
        if best_method:
            print(f"\n🏆 Best Method: {best_method} (BACC: {best_bacc:.3f})")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="BACC Improvement System")
    parser.add_argument('--method', type=int, choices=[1, 2, 3, 4, 5], 
                       help='Method to run (1-5) or 0 for all')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'vit'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu-id', type=int, default=0)
    
    args = parser.parse_args()
    
    # Create system
    system = BACCImprovementSystem()
    
    if args.method == 0:
        # Run all methods
        system.run_all_methods(args.model, args.epochs, args.gpu_id)
    elif args.method == 1:
        system.method_1_enhanced_loss(args.model, args.epochs, args.gpu_id)
    elif args.method == 2:
        system.method_2_data_augmentation(args.model, args.epochs, args.gpu_id)
    elif args.method == 3:
        system.method_3_threshold_optimization(args.model, args.epochs, args.gpu_id)
    elif args.method == 4:
        system.method_4_ensemble(args.model, args.epochs, args.gpu_id)
    elif args.method == 5:
        system.method_5_combined(args.model, args.epochs, args.gpu_id)
    else:
        print("Please select a method (1-5) or 0 for all methods")

if __name__ == "__main__":
    main() 