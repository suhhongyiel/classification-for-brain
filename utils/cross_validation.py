#!/usr/bin/env python3
"""
Cross-validation and result logging utilities for TAU synthetic data classification
"""

import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import json
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils.dataset import create_stratified_splits, create_stratified_splits_all_data, get_dataloaders
from utils.trainer import TAUTrainer, calculate_class_weights, get_optimizer, get_scheduler
from models.models import get_model, get_loss_function


def get_device(gpu_id=None):
    """
    Get the appropriate device for training
    
    Args:
        gpu_id: Specific GPU ID to use (0-5). If None, uses first available GPU
    
    Returns:
        torch.device: Device to use for training
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    if gpu_id is not None:
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU ID {gpu_id} is invalid. Available GPUs: 0-{torch.cuda.device_count()-1}")
            print("Using first available GPU")
            device = torch.device('cuda:0')
        else:
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cuda:0')
        print(f"Using first available GPU: {torch.cuda.get_device_name(0)}")
    
    # Print GPU memory info
    if device.type == 'cuda':
        gpu_id = device.index
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print(f"GPU {gpu_id} memory: {total_memory:.1f} GB")
    
    return device


class CrossValidationManager:
    """
    Manages cross-validation experiments and result logging
    """
    
    def __init__(
        self,
        data_csv_path,
        results_dir='./results',
        config=None
    ):
        self.data_csv_path = data_csv_path
        self.results_dir = results_dir
        self.config = config or self._get_default_config()
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load data
        self.data_df = pd.read_csv(data_csv_path)
        
        # Results storage
        self.all_results = []
        self.fold_results = {}
        
    def _get_default_config(self):
        """Default configuration for experiments"""
        return {
            'models': ['resnet18', 'vit'],
            'enhance_small_features': [False, True],
            'batch_size': 8,
            'num_epochs': 100,  # Increased from 50 to 100
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'loss_function': 'focal',
            'early_stopping_patience': 20,  # Increased from 15 to 20
            'num_workers': 4,
            'use_class_weights': True,
            'random_state': 42
        }
    
    def run_single_fold(
        self,
        fold_idx,
        model_name,
        enhance_small_features=False,
        quick_tune=False
    ):
        """
        Run experiment on a single fold (useful for parameter tuning)
        
        Args:
            fold_idx: Fold index (1-10)
            model_name: Model name ('resnet18' or 'vit')
            enhance_small_features: Whether to use enhancement mechanisms
            quick_tune: If True, use fewer epochs for quick tuning
        """
        print(f"\n{'='*60}")
        print(f"Running Single Fold Experiment - Fold {fold_idx}")
        print(f"Model: {model_name}, Enhanced: {enhance_small_features}")
        print(f"{'='*60}")
        
        # Adjust config for quick tuning
        config = self.config.copy()
        if quick_tune:
            config['num_epochs'] = 30  # Increased from 10 to 30
            config['early_stopping_patience'] = 8  # Increased from 5 to 8
            print("Quick tuning mode: Using moderate epochs and patience")
        
        device = get_device(config.get('gpu_id'))
        
        # Create data splits
        train_df, val_df, test_df = create_stratified_splits(
            self.data_df, 
            fold_idx, 
            random_state=config['random_state']
        )
        
        # Calculate class weights if needed
        class_weights = None
        if config['use_class_weights']:
            class_weights = calculate_class_weights(train_df['label'].values)
            class_weights = class_weights.to(device)
            print(f"Class weights: {class_weights}")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            enhance_small_features=enhance_small_features
        )
        
        # Create model
        model = get_model(
            model_name, 
            num_classes=2, 
            enhance_small_features=enhance_small_features
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Create optimizer and scheduler
        optimizer = get_optimizer(
            model,
            optimizer_name=config['optimizer'],
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        scheduler = get_scheduler(
            optimizer,
            scheduler_name=config['scheduler'],
            num_epochs=config['num_epochs']
        )
        
        # Create loss function
        criterion = get_loss_function(
            loss_type=config['loss_function'],
            class_weights=class_weights
        )
        
        # Create trainer
        trainer = TAUTrainer(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping_patience=config['early_stopping_patience'],
            class_weights=class_weights,
            results_dir=self.results_dir
        )
        
        # Train model
        train_history, val_history = trainer.train(
            train_loader, val_loader,
            num_epochs=config['num_epochs'],
            fold_idx=fold_idx,
            model_name=f"{model_name}_enhanced_{enhance_small_features}"
        )
        
        # Test model
        test_metrics, detailed_results = trainer.test(test_loader)
        
        # Compile results
        experiment_result = {
            'fold': fold_idx,
            'model_name': model_name,
            'enhance_small_features': enhance_small_features,
            'config': config,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'test_metrics': test_metrics,
            'train_history': train_history,
            'val_history': val_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save detailed results
        detailed_filename = f"{model_name}_enhanced_{enhance_small_features}_fold_{fold_idx}_detailed.csv"
        detailed_results.to_csv(
            os.path.join(self.results_dir, detailed_filename), 
            index=False
        )
        
        print(f"\nFold {fold_idx} Results:")
        print(f"Test AUC: {test_metrics['auc']:.3f}")
        print(f"Test F1: {test_metrics['f1']:.3f}")
        print(f"Test BACC: {test_metrics['bacc']:.3f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
        
        return experiment_result
    
    def run_cross_validation(
        self,
        model_names=None,
        enhance_options=None,
        folds_to_run=None
    ):
        """
        Run full cross-validation experiment
        
        Args:
            model_names: List of model names to test
            enhance_options: List of enhancement options [True, False]
            folds_to_run: List of fold indices to run (1-10)
        """
        model_names = model_names or self.config['models']
        enhance_options = enhance_options or self.config['enhance_small_features']
        folds_to_run = folds_to_run or list(range(1, 11))
        
        print(f"\n{'='*80}")
        print(f"Starting Cross-Validation Experiment")
        print(f"Models: {model_names}")
        print(f"Enhancement options: {enhance_options}")
        print(f"Folds: {folds_to_run}")
        print(f"{'='*80}")
        
        all_results = []
        
        for model_name in model_names:
            for enhance_small_features in enhance_options:
                print(f"\n{'-'*60}")
                print(f"Model: {model_name}, Enhanced: {enhance_small_features}")
                print(f"{'-'*60}")
                
                model_results = []
                
                for fold_idx in folds_to_run:
                    try:
                        result = self.run_single_fold(
                            fold_idx=fold_idx,
                            model_name=model_name,
                            enhance_small_features=enhance_small_features,
                            quick_tune=False
                        )
                        model_results.append(result)
                        all_results.append(result)
                        
                        # Save intermediate results
                        self.save_results(all_results)
                        
                    except Exception as e:
                        print(f"Error in fold {fold_idx}: {str(e)}")
                        continue
                
                # Print summary for this model configuration
                if model_results:
                    self.print_model_summary(model_results, model_name, enhance_small_features)
        
        # Save final results
        self.all_results = all_results
        self.save_results(all_results)
        self.create_summary_report()
        
        print(f"\n{'='*80}")
        print("Cross-validation completed!")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*80}")
        
        return all_results
    
    def print_model_summary(self, model_results, model_name, enhance_small_features):
        """Print summary statistics for a model configuration"""
        if not model_results:
            return
            
        test_aucs = [r['test_metrics']['auc'] for r in model_results]
        test_f1s = [r['test_metrics']['f1'] for r in model_results]
        test_baccs = [r['test_metrics']['bacc'] for r in model_results]
        
        print(f"\nSummary for {model_name} (Enhanced: {enhance_small_features}):")
        print(f"AUC  - Mean: {np.mean(test_aucs):.3f} ± {np.std(test_aucs):.3f}")
        print(f"F1   - Mean: {np.mean(test_f1s):.3f} ± {np.std(test_f1s):.3f}")
        print(f"BACC - Mean: {np.mean(test_baccs):.3f} ± {np.std(test_baccs):.3f}")
    
    def save_results(self, results):
        """Save results to CSV and JSON files"""
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                'fold': result['fold'],
                'model_name': result['model_name'],
                'enhance_small_features': result['enhance_small_features'],
                'train_samples': result['train_samples'],
                'val_samples': result['val_samples'],
                'test_samples': result['test_samples'],
                'total_params': result['total_params'],
                'trainable_params': result['trainable_params'],
                'test_auc': result['test_metrics']['auc'],
                'test_f1': result['test_metrics']['f1'],
                'test_bacc': result['test_metrics']['bacc'],
                'test_accuracy': result['test_metrics']['accuracy'],
                'test_loss': result['test_metrics']['loss'],
                'timestamp': result['timestamp']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_path = os.path.join(self.results_dir, 'cross_validation_results.csv')
        summary_df.to_csv(csv_path, index=False)
        
        # Save detailed JSON
        json_path = os.path.join(self.results_dir, 'cross_validation_results_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save pickle for complete data
        pickle_path = os.path.join(self.results_dir, 'cross_validation_results.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        if not self.all_results:
            return
        
        # Load results if needed
        if not hasattr(self, 'all_results') or not self.all_results:
            try:
                pickle_path = os.path.join(self.results_dir, 'cross_validation_results.pkl')
                with open(pickle_path, 'rb') as f:
                    self.all_results = pickle.load(f)
            except:
                print("No results found for summary report")
                return
        
        # Group results by model configuration
        grouped_results = {}
        for result in self.all_results:
            key = (result['model_name'], result['enhance_small_features'])
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Create summary statistics
        summary_stats = []
        
        for (model_name, enhance_small_features), results in grouped_results.items():
            test_aucs = [r['test_metrics']['auc'] for r in results]
            test_f1s = [r['test_metrics']['f1'] for r in results]
            test_baccs = [r['test_metrics']['bacc'] for r in results]
            test_accs = [r['test_metrics']['accuracy'] for r in results]
            
            summary_stats.append({
                'model_name': model_name,
                'enhance_small_features': enhance_small_features,
                'n_folds': len(results),
                'auc_mean': np.mean(test_aucs),
                'auc_std': np.std(test_aucs),
                'auc_min': np.min(test_aucs),
                'auc_max': np.max(test_aucs),
                'f1_mean': np.mean(test_f1s),
                'f1_std': np.std(test_f1s),
                'f1_min': np.min(test_f1s),
                'f1_max': np.max(test_f1s),
                'bacc_mean': np.mean(test_baccs),
                'bacc_std': np.std(test_baccs),
                'bacc_min': np.min(test_baccs),
                'bacc_max': np.max(test_baccs),
                'acc_mean': np.mean(test_accs),
                'acc_std': np.std(test_accs),
                'avg_total_params': np.mean([r['total_params'] for r in results]),
                'avg_trainable_params': np.mean([r['trainable_params'] for r in results])
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_stats)
        summary_path = os.path.join(self.results_dir, 'experiment_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary
        print("\n" + "="*100)
        print("EXPERIMENT SUMMARY")
        print("="*100)
        
        for _, row in summary_df.iterrows():
            print(f"\nModel: {row['model_name']} | Enhanced: {row['enhance_small_features']} | Folds: {row['n_folds']}")
            print(f"AUC:  {row['auc_mean']:.3f} ± {row['auc_std']:.3f} (range: {row['auc_min']:.3f} - {row['auc_max']:.3f})")
            print(f"F1:   {row['f1_mean']:.3f} ± {row['f1_std']:.3f} (range: {row['f1_min']:.3f} - {row['f1_max']:.3f})")
            print(f"BACC: {row['bacc_mean']:.3f} ± {row['bacc_std']:.3f} (range: {row['bacc_min']:.3f} - {row['bacc_max']:.3f})")
            print(f"Params: {row['avg_total_params']:.0f}")
        
        print("="*100)
    
    def load_results(self):
        """Load previously saved results"""
        try:
            pickle_path = os.path.join(self.results_dir, 'cross_validation_results.pkl')
            with open(pickle_path, 'rb') as f:
                self.all_results = pickle.load(f)
            print(f"Loaded {len(self.all_results)} results from {pickle_path}")
            return self.all_results
        except FileNotFoundError:
            print("No saved results found")
            return []


def run_parameter_tuning(
    data_csv_path,
    results_dir='./tuning_results',
    tuning_fold=1,
    model_name='resnet18'
):
    """
    Run parameter tuning on a single fold for quick experiments
    
    Args:
        data_csv_path: Path to data CSV
        results_dir: Directory to save tuning results
        tuning_fold: Fold to use for tuning (default: 1)
        model_name: Model to tune
    """
    
    print(f"\n{'='*60}")
    print(f"Parameter Tuning Mode")
    print(f"Model: {model_name}, Fold: {tuning_fold}")
    print(f"{'='*60}")
    
    # Parameter grid for tuning (temporarily reduced for testing)
    param_grid = {
        'learning_rate': [1e-4, 5e-4],  # Reduced from 4 to 2
        'batch_size': [8],              # Reduced from 3 to 1 
        'enhance_small_features': [False, True],
        'optimizer': ['adamw'],         # Reduced from 2 to 1
        'loss_function': ['focal']      # Reduced from 2 to 1
    }
    
    best_score = 0
    best_params = None
    tuning_results = []
    
    # Create manager
    manager = CrossValidationManager(data_csv_path, results_dir)
    
    # Grid search
    from itertools import product
    
    param_combinations = list(product(
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['enhance_small_features'],
        param_grid['optimizer'],
        param_grid['loss_function']
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, (lr, batch_size, enhance_small_features, optimizer, loss_function) in enumerate(param_combinations):
        print(f"\nTuning {i+1}/{len(param_combinations)}")
        print(f"LR: {lr}, Batch: {batch_size}, Enhanced: {enhance_small_features}")
        print(f"Optimizer: {optimizer}, Loss: {loss_function}")
        
        # Update config
        manager.config.update({
            'learning_rate': lr,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'loss_function': loss_function
        })
        
        try:
            result = manager.run_single_fold(
                fold_idx=tuning_fold,
                model_name=model_name,
                enhance_small_features=enhance_small_features,
                quick_tune=True
            )
            
            score = result['test_metrics']['auc']
            result['tuning_params'] = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'enhance_small_features': enhance_small_features,
                'optimizer': optimizer,
                'loss_function': loss_function
            }
            
            tuning_results.append(result)
            
            if score > best_score:
                best_score = score
                best_params = result['tuning_params'].copy()
                print(f"New best score: {best_score:.3f}")
            
        except Exception as e:
            print(f"Error with parameters: {e}")
            continue
    
    # Save tuning results
    tuning_df = pd.DataFrame([
        {
            **r['tuning_params'],
            'test_auc': r['test_metrics']['auc'],
            'test_f1': r['test_metrics']['f1'],
            'test_bacc': r['test_metrics']['bacc']
        }
        for r in tuning_results
    ])
    
    tuning_csv_path = os.path.join(results_dir, f'{model_name}_parameter_tuning.csv')
    tuning_df.to_csv(tuning_csv_path, index=False)
    
    print(f"\n{'='*60}")
    print("Parameter Tuning Results")
    print(f"{'='*60}")
    print(f"Best Score (AUC): {best_score:.3f}")
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Results saved to: {tuning_csv_path}")
    print(f"{'='*60}")
    
    return best_params, tuning_results


def run_parameter_tuning_all_data(
    data_csv_path,
    results_dir='./tuning_results',
    model_name='resnet18'
):
    """
    Run parameter tuning using ALL data (not fold-specific) for more robust tuning
    
    Args:
        data_csv_path: Path to data CSV
        results_dir: Directory to save tuning results
        model_name: Model to tune
    """
    
    print(f"\n{'='*70}")
    print(f"Parameter Tuning Mode - Using ALL DATA")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    # Load data
    data_df = pd.read_csv(data_csv_path)
    print(f"Total data loaded: {len(data_df)} samples")
    
    # Parameter grid for tuning (model-specific)
    if model_name == 'vit':
        param_grid = {
            'learning_rate': [1e-5, 1e-4],
            'batch_size': [2, 4, 8],
            'enhance_small_features': [False, True],
            'optimizer': ['adamw'],
            'loss_function': ['focal', 'weighted_ce'],
            'weight_decay': [1e-5, 1e-4],
            'scheduler': ['cosine', 'plateau']
        }
    else:  # resnet18
        param_grid = {
            'learning_rate': [1e-5, 1e-4],
            'batch_size': [2, 4, 8],
            'enhance_small_features': [False, True],
            'optimizer': ['adamw'],
            'loss_function': ['focal', 'weighted_ce', 'ce'],
            'weight_decay': [1e-5, 1e-4],
            'scheduler': ['cosine', 'plateau']
        }
    
    best_score = 0
    best_params = None
    tuning_results = []
    
    # Default config
    config = {
        'batch_size': 8,
        'num_epochs': 100,  # Increased for meaningful results
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'loss_function': 'focal',
        'early_stopping_patience': 10,  # Increased patience
        'num_workers': 4,  # Increased workers
        'use_class_weights': True,
        'random_state': 42
    }
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Grid search
    from itertools import product
    
    # Create parameter combinations based on model type
    if model_name == 'vit':
        param_combinations = list(product(
            param_grid['learning_rate'],
            param_grid['batch_size'],
            param_grid['enhance_small_features'],
            param_grid['optimizer'],
            param_grid['loss_function'],
            param_grid['weight_decay'],
            param_grid['scheduler']
        ))
    else:  # resnet18
        param_combinations = list(product(
            param_grid['learning_rate'],
            param_grid['batch_size'],
            param_grid['enhance_small_features'],
            param_grid['optimizer'],
            param_grid['loss_function'],
            param_grid['weight_decay'],
            param_grid['scheduler']
        ))
    
    print(f"Testing {len(param_combinations)} parameter combinations for {model_name}...")
    print(f"\n📊 Parameter Grid for {model_name.upper()}:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"🔢 Total combinations: {len(param_combinations)}")
    print(f"⏱️  Estimated time: ~{len(param_combinations) * 50 / 60:.1f} hours (50 epochs each)")
    print(f"{'='*70}")
    
    for i, params in enumerate(param_combinations):
        if model_name == 'vit':
            lr, batch_size, enhance_small_features, optimizer, loss_function, weight_decay, scheduler = params
        else:  # resnet18
            lr, batch_size, enhance_small_features, optimizer, loss_function, weight_decay, scheduler = params
            
        print(f"\n{'-'*50}")
        print(f"🔄 Tuning {i+1}/{len(param_combinations)} ({(i+1)/len(param_combinations)*100:.1f}%)")
        print(f"📈 Progress: {i+1}/{len(param_combinations)} combinations completed")
        print(f"🔧 Parameters:")
        print(f"   Learning Rate: {lr}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Enhanced Features: {enhance_small_features}")
        print(f"   Optimizer: {optimizer}")
        print(f"   Loss Function: {loss_function}")
        print(f"   Weight Decay: {weight_decay}")
        print(f"   Scheduler: {scheduler}")
        print(f"{'-'*50}")
        
        # Update config
        current_config = config.copy()
        current_config.update({
            'learning_rate': lr,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'loss_function': loss_function,
            'weight_decay': weight_decay,
            'scheduler': scheduler
        })
        
        try:
            device = get_device(current_config.get('gpu_id'))
            
            # Create data splits using ALL data
            train_df, val_df, test_df = create_stratified_splits_all_data(
                data_df, 
                test_size=0.2, 
                val_size=0.2,
                random_state=current_config['random_state']
            )
            
            # Calculate class weights if needed
            class_weights = None
            if current_config['use_class_weights']:
                class_weights = calculate_class_weights(train_df['label'].values)
                class_weights = class_weights.to(device)
                print(f"Class weights: {class_weights}")
            
            # Create dataloaders
            train_loader, val_loader, test_loader = get_dataloaders(
                train_df, val_df, test_df,
                batch_size=current_config['batch_size'],
                num_workers=current_config['num_workers'],
                enhance_small_features=enhance_small_features
            )
            
            # Create model
            model = get_model(
                model_name, 
                num_classes=2, 
                enhance_small_features=enhance_small_features
            ).to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            # Create optimizer and scheduler
            optimizer_obj = get_optimizer(
                model,
                optimizer_name=current_config['optimizer'],
                lr=current_config['learning_rate'],
                weight_decay=current_config['weight_decay']
            )
            
            scheduler = get_scheduler(
                optimizer_obj,
                scheduler_name=current_config['scheduler'],
                num_epochs=current_config['num_epochs']
            )
            
            # Create loss function
            criterion = get_loss_function(
                loss_type=current_config['loss_function'],
                class_weights=class_weights
            )
            
            # Create trainer
            trainer = TAUTrainer(
                model=model,
                device=device,
                criterion=criterion,
                optimizer=optimizer_obj,
                scheduler=scheduler,
                early_stopping_patience=current_config['early_stopping_patience'],
                class_weights=class_weights,
                results_dir=results_dir
            )
            
            # Train model
            train_history, val_history = trainer.train(
                train_loader, val_loader,
                num_epochs=current_config['num_epochs'],
                fold_idx='ALL',
                model_name=f"{model_name}_enhanced_{enhance_small_features}_alldata"
            )
            
            # Test model
            test_metrics, detailed_results = trainer.test(test_loader)
            
            score = test_metrics['auc']
            
            # Store results
            result = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'enhance_small_features': enhance_small_features,
                'optimizer': optimizer,
                'loss_function': loss_function,
                'test_auc': test_metrics['auc'],
                'test_f1': test_metrics['f1'],
                'test_bacc': test_metrics['bacc'],
                'test_accuracy': test_metrics['accuracy'],
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            tuning_results.append(result)
            
            print(f"Results - AUC: {score:.3f}, F1: {test_metrics['f1']:.3f}, BACC: {test_metrics['bacc']:.3f}")
            
            if score > best_score:
                best_score = score
                best_params = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'enhance_small_features': enhance_small_features,
                    'optimizer': optimizer,
                    'loss_function': loss_function,
                    'weight_decay': weight_decay,
                    'scheduler': scheduler
                }
                print(f"🎉 New best score: {best_score:.3f}")
                
                # Save best parameters immediately (convert to serializable format)
                best_params_file = os.path.join(results_dir, f'best_params_{model_name}_all_data.json')
                # Convert scheduler object to string for JSON serialization
                serializable_best_params = best_params.copy()
                if 'scheduler' in serializable_best_params:
                    serializable_best_params['scheduler'] = str(serializable_best_params['scheduler'])
                
                with open(best_params_file, 'w') as f:
                    json.dump(serializable_best_params, f, indent=2)
                print(f"💾 Best parameters saved to: {best_params_file}")
            
            # Save intermediate results every 10 combinations
            if len(tuning_results) % 10 == 0:
                tuning_df = pd.DataFrame(tuning_results)
                tuning_csv_path = os.path.join(results_dir, f'{model_name}_parameter_tuning_all_data.csv')
                tuning_df.to_csv(tuning_csv_path, index=False)
                print(f"💾 Intermediate results saved ({len(tuning_results)} combinations completed)")
            
        except Exception as e:
            print(f"❌ Error with parameters: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save tuning results (save after each combination for safety)
    tuning_df = pd.DataFrame(tuning_results)
    tuning_csv_path = os.path.join(results_dir, f'{model_name}_parameter_tuning_all_data.csv')
    tuning_df.to_csv(tuning_csv_path, index=False)
    
    # Save best parameters separately (convert to serializable format)
    best_params_file = os.path.join(results_dir, f'best_params_{model_name}_all_data.json')
    # Convert scheduler object to string for JSON serialization
    serializable_best_params = best_params.copy()
    if 'scheduler' in serializable_best_params:
        serializable_best_params['scheduler'] = str(serializable_best_params['scheduler'])
    
    with open(best_params_file, 'w') as f:
        json.dump(serializable_best_params, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Parameter Tuning Results - ALL DATA")
    print(f"{'='*70}")
    print(f"🏆 Best Score (AUC): {best_score:.3f}")
    print("🔧 Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"📊 Total combinations tested: {len(tuning_results)}")
    print(f"💾 Results saved to: {tuning_csv_path}")
    print(f"{'='*70}")
    
    return best_params, tuning_results


if __name__ == "__main__":
    # Test cross-validation manager
    data_csv_path = "/home/classification/data/syn_data_mapping.csv"
    results_dir = "/home/classification/result"
    
    # Create manager
    manager = CrossValidationManager(data_csv_path, results_dir)
    
    # Test single fold
    result = manager.run_single_fold(
        fold_idx=1,
        model_name='resnet18',
        enhance_small_features=False,
        quick_tune=True
    )
    
    print("Cross-validation manager testing completed successfully!") 