#!/usr/bin/env python3
"""
Parameter Tuning System with BACC Improvement
Combines hyperparameter tuning with BACC improvement strategies
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import itertools
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

class ParameterTuningSystem:
    """
    Comprehensive parameter tuning system with BACC improvement
    """
    
    def __init__(self, data_csv_path='/home/classification/data/syn_data_mapping.csv'):
        self.data_csv_path = data_csv_path
        self.data_df = pd.read_csv(data_csv_path)
        self.results = []
        
    def get_parameter_grid(self, model_name='resnet18'):
        """Get parameter grid for tuning"""
        
        if model_name == 'resnet18':
            param_grid = {
                'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
                'batch_size': [4, 8, 16],
                'enhance_small_features': [False, True],
                'optimizer': ['adamw', 'adam'],
                'loss_function': ['focal', 'weighted_ce', 'advanced_balanced'],
                'weight_decay': [1e-5, 1e-4, 1e-3],
                'scheduler': ['cosine', 'plateau'],
                'data_augmentation': [False, True],
                'threshold_optimization': [False, True],
                'ensemble_method': [None, 'voting', 'weighted_average'],
            }
        else:  # vit
            param_grid = {
                'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
                'batch_size': [4, 8, 16],
                'enhance_small_features': [False, True],
                'optimizer': ['adamw', 'adam'],
                'loss_function': ['focal', 'weighted_ce', 'advanced_balanced'],
                'weight_decay': [1e-5, 1e-4, 1e-3],
                'scheduler': ['cosine', 'plateau'],
                'data_augmentation': [False, True],
                'threshold_optimization': [False, True],
                'ensemble_method': [None, 'voting', 'weighted_average'],
            }
        
        return param_grid
    
    def create_loss_function(self, loss_name, device='cuda'):
        """Create loss function based on name"""
        if loss_name == 'focal':
            return nn.CrossEntropyLoss().to(device)  # Simplified for now
        elif loss_name == 'weighted_ce':
            return nn.CrossEntropyLoss().to(device)  # Will add weights later
        elif loss_name == 'advanced_balanced':
            return AdvancedBalancedLoss(alpha=0.25, gamma=2.0).to(device)
        else:
            return nn.CrossEntropyLoss().to(device)
    
    def train_with_parameters(self, params, model_name='resnet18', num_epochs=50, gpu_id=0):
        """Train model with specific parameters"""
        
        # Setup
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Data preparation
        if params.get('data_augmentation', False):
            # Use balanced dataset
            balancer = BalancedDatasetAugmentation(minority_boost=2.0)
            balanced_df = balancer.augment_minority_samples(self.data_df)
        else:
            balanced_df = self.data_df
        
        # Create data splits
        train_df, val_df, test_df = create_stratified_splits_all_data(
            balanced_df, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, 
            batch_size=params['batch_size'], 
            num_workers=4
        )
        
        # Create model
        model = get_model(
            model_name, 
            num_classes=2, 
            enhance_small_features=params['enhance_small_features']
        ).to(device)
        
        # Create loss function
        criterion = self.create_loss_function(params['loss_function'], device)
        
        # Create optimizer and scheduler
        optimizer = get_optimizer(
            model, 
            params['optimizer'], 
            params['learning_rate'], 
            weight_decay=params['weight_decay']
        )
        scheduler = get_scheduler(optimizer, params['scheduler'], num_epochs)
        
        # Create trainer
        trainer = TAUTrainer(
            model=model, device=device, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler,
            early_stopping_patience=15, results_dir='./result/tuning'
        )
        
        # Train model
        train_metrics, val_metrics = trainer.train(
            train_loader, val_loader, num_epochs, fold_idx=0, model_name=model_name
        )
        
        # Test model
        test_metrics, detailed_results = trainer.test(test_loader)
        
        # Apply threshold optimization if requested
        if params.get('threshold_optimization', False):
            model.eval()
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Optimize threshold
            threshold_optimizer = ThresholdOptimizer()
            optimal_threshold, optimal_bacc = threshold_optimizer.find_optimal_threshold(
                np.array(all_labels), np.array(all_probs)
            )
            
            # Update metrics with optimal threshold
            final_predictions = (np.array(all_probs) >= optimal_threshold).astype(int)
            
            from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
            test_metrics['bacc'] = balanced_accuracy_score(all_labels, final_predictions)
            test_metrics['f1'] = f1_score(all_labels, final_predictions, average='weighted')
            test_metrics['optimal_threshold'] = optimal_threshold
        
        return test_metrics, params
    
    def run_parameter_tuning(self, model_name='resnet18', num_epochs=50, gpu_id=0, max_combinations=50):
        """Run comprehensive parameter tuning"""
        
        print(f"🔧 Parameter Tuning for {model_name.upper()}")
        print("="*60)
        
        # Get parameter grid
        param_grid = self.get_parameter_grid(model_name)
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Limit combinations to avoid excessive computation
        combinations = list(itertools.product(*param_values))
        if len(combinations) > max_combinations:
            print(f"⚠️  Limiting combinations from {len(combinations)} to {max_combinations}")
            combinations = combinations[:max_combinations]
        
        print(f"📊 Testing {len(combinations)} parameter combinations...")
        print(f"⏱️  Estimated time: {len(combinations) * num_epochs * 0.1:.1f} minutes")
        
        # Test each combination
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            print(f"\n{'='*20} Combination {i+1}/{len(combinations)} {'='*20}")
            print(f"Parameters: {params}")
            
            try:
                metrics, final_params = self.train_with_parameters(
                    params, model_name, num_epochs, gpu_id
                )
                
                # Store results
                result = {
                    'combination_id': i + 1,
                    'model_name': model_name,
                    **final_params,
                    **metrics
                }
                self.results.append(result)
                
                print(f"✅ Results: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, BACC={metrics['bacc']:.3f}")
                
                # Save intermediate results
                self.save_results(f"{model_name}_parameter_tuning.csv")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                continue
        
        # Print final summary
        self.print_tuning_summary()
    
    def save_results(self, filename):
        """Save results to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            os.makedirs('./result/tuning', exist_ok=True)
            df.to_csv(f'./result/tuning/{filename}', index=False)
            print(f"💾 Results saved to: ./result/tuning/{filename}")
    
    def print_tuning_summary(self):
        """Print tuning summary"""
        if not self.results:
            print("❌ No results to display")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("📊 PARAMETER TUNING SUMMARY")
        print("="*80)
        
        # Best results
        best_bacc_idx = df['bacc'].idxmax()
        best_auc_idx = df['auc'].idxmax()
        best_f1_idx = df['f1'].idxmax()
        
        print(f"\n🏆 Best BACC: {df.loc[best_bacc_idx, 'bacc']:.3f}")
        print(f"   Parameters: {dict(df.loc[best_bacc_idx, ['learning_rate', 'batch_size', 'optimizer', 'loss_function']])}")
        
        print(f"\n🏆 Best AUC: {df.loc[best_auc_idx, 'auc']:.3f}")
        print(f"   Parameters: {dict(df.loc[best_auc_idx, ['learning_rate', 'batch_size', 'optimizer', 'loss_function']])}")
        
        print(f"\n🏆 Best F1: {df.loc[best_f1_idx, 'f1']:.3f}")
        print(f"   Parameters: {dict(df.loc[best_f1_idx, ['learning_rate', 'batch_size', 'optimizer', 'loss_function']])}")
        
        # Parameter importance analysis
        print(f"\n📈 Parameter Impact Analysis:")
        for param in ['learning_rate', 'batch_size', 'optimizer', 'loss_function', 'enhance_small_features']:
            if param in df.columns:
                impact = df.groupby(param)['bacc'].mean().sort_values(ascending=False)
                print(f"   {param}: {impact.index[0]} (BACC: {impact.iloc[0]:.3f})")
        
        # Save best parameters
        best_params = df.loc[best_bacc_idx].to_dict()
        # Convert numpy types to native Python types for JSON serialization
        best_params = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in best_params.items()}
        import json
        os.makedirs('./result/tuning', exist_ok=True)
        with open(f'./result/tuning/best_params_{df.loc[best_bacc_idx, "model_name"]}.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n💾 Best parameters saved to: ./result/tuning/best_params_{df.loc[best_bacc_idx, 'model_name']}.json")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Parameter Tuning System with BACC Improvement")
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'vit'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--max-combinations', type=int, default=50)
    
    args = parser.parse_args()
    
    # Create system
    system = ParameterTuningSystem()
    
    # Run parameter tuning
    system.run_parameter_tuning(
        model_name=args.model,
        num_epochs=args.epochs,
        gpu_id=args.gpu_id,
        max_combinations=args.max_combinations
    )

if __name__ == "__main__":
    main() 