#!/usr/bin/env python3
"""
Test script for all-data parameter tuning with reduced parameter grid
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cross_validation import run_parameter_tuning_all_data

# Quick test with minimal parameter grid
def test_tune_all_data():
    """Test the all-data tuning with minimal parameters"""
    
    # Temporarily modify the parameter grid for quick testing
    import utils.cross_validation as cv_module
    
    # Store original function
    original_func = cv_module.run_parameter_tuning_all_data
    
    def quick_tune_all_data(data_csv_path, results_dir='./tuning_results', model_name='resnet18'):
        """Quick version with reduced parameter grid"""
        print(f"\n{'='*70}")
        print(f"QUICK Parameter Tuning Test - Using ALL DATA")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        
        # Load data
        import pandas as pd
        data_df = pd.read_csv(data_csv_path)
        print(f"Total data loaded: {len(data_df)} samples")
        
        # REDUCED Parameter grid for quick testing
        param_grid = {
            'learning_rate': [1e-4, 5e-4],  # Only 2 options
            'batch_size': [8],              # Only 1 option
            'enhance_small_features': [False, True],  # 2 options
            'optimizer': ['adamw'],         # Only 1 option
            'loss_function': ['focal']      # Only 1 option
        }
        
        best_score = 0
        best_params = None
        tuning_results = []
        
        # Default config with REDUCED epochs
        config = {
            'batch_size': 8,
            'num_epochs': 3,  # Very short for testing
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'loss_function': 'focal',
            'early_stopping_patience': 2,  # Very short patience
            'num_workers': 1,  # Reduced workers
            'use_class_weights': True,
            'random_state': 42
        }
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Grid search
        from itertools import product
        import torch
        from utils.dataset import create_stratified_splits_all_data, get_dataloaders
        from utils.trainer import TAUTrainer, calculate_class_weights, get_optimizer, get_scheduler
        from models.models import get_model, get_loss_function
        
        param_combinations = list(product(
            param_grid['learning_rate'],
            param_grid['batch_size'],
            param_grid['enhance_small_features'],
            param_grid['optimizer'],
            param_grid['loss_function']
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, (lr, batch_size, enhance_small_features, optimizer, loss_function) in enumerate(param_combinations):
            print(f"\n{'-'*50}")
            print(f"Quick Test {i+1}/{len(param_combinations)}")
            print(f"LR: {lr}, Enhanced: {enhance_small_features}")
            print(f"{'-'*50}")
            
            # Update config
            current_config = config.copy()
            current_config.update({
                'learning_rate': lr,
                'batch_size': batch_size,
                'optimizer': optimizer,
                'loss_function': loss_function
            })
            
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {device}")
                
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
                    model_name=f"{model_name}_enhanced_{enhance_small_features}_alldata_test"
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
                        'loss_function': loss_function
                    }
                    print(f"🎉 New best score: {best_score:.3f}")
                
            except Exception as e:
                print(f"❌ Error with parameters: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save tuning results
        import pandas as pd
        tuning_df = pd.DataFrame(tuning_results)
        tuning_csv_path = os.path.join(results_dir, f'{model_name}_quick_test_all_data.csv')
        tuning_df.to_csv(tuning_csv_path, index=False)
        
        print(f"\n{'='*70}")
        print("QUICK TEST Results - ALL DATA")
        print(f"{'='*70}")
        print(f"🏆 Best Score (AUC): {best_score:.3f}")
        print("🔧 Best Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"📊 Total combinations tested: {len(tuning_results)}")
        print(f"💾 Results saved to: {tuning_csv_path}")
        print(f"{'='*70}")
        
        return best_params, tuning_results
    
    # Run the quick test
    return quick_tune_all_data(
        data_csv_path='data/syn_data_mapping.csv',
        results_dir='result/quick_test',
        model_name='resnet18'
    )

if __name__ == "__main__":
    test_tune_all_data() 