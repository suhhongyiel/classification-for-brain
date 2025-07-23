#!/usr/bin/env python3
"""
Main script for TAU synthetic data classification
Supports various execution modes: full cross-validation, single fold, parameter tuning
"""

import argparse
import os
import sys
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cross_validation import CrossValidationManager, run_parameter_tuning, run_parameter_tuning_all_data
from utils.data_mapper import create_data_mapping


def setup_argparser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="TAU Synthetic Data Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full cross-validation with all models and configurations
  python main.py --mode full --models resnet18 vit --enhance-options True False

  # Single fold experiment (useful for testing)
  python main.py --mode single --fold 1 --model resnet18 --enhance-features

  # Parameter tuning on fold 1
  python main.py --mode tune --fold 1 --model resnet18
  
  # Parameter tuning using ALL data (not fold-specific)
  python main.py --mode tune --model resnet18 --use-all-data

  # Quick test with 3 folds
  python main.py --mode full --folds 1 2 3 --epochs 20

  # Resume from existing results
  python main.py --mode resume --results-dir ./results
        """
    )
    
    # Execution mode
    parser.add_argument(
        '--mode', 
        choices=['full', 'single', 'tune', 'resume', 'data-mapping'],
        default='full',
        help='Execution mode (default: full)'
    )
    
    # Data paths
    parser.add_argument(
        '--data-csv',
        default='/home/classification/data/syn_data_mapping.csv',
        help='Path to data mapping CSV file'
    )
    
    parser.add_argument(
        '--results-dir',
        default='/home/classification/result',
        help='Directory to save results'
    )
    
    # Model configuration
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['resnet18', 'vit'],
        default=['resnet18', 'vit'],
        help='Models to train (default: resnet18 vit)'
    )
    
    parser.add_argument(
        '--enhance-options',
        nargs='+',
        type=lambda x: x.lower() == 'true',
        default=[False, True],
        help='Enhancement options as True/False (default: False True)'
    )
    
    # Single fold mode
    parser.add_argument(
        '--fold',
        type=int,
        choices=range(1, 11),
        default=1,
        help='Fold number for single fold mode (1-10, default: 1)'
    )
    
    parser.add_argument(
        '--model',
        choices=['resnet18', 'vit'],
        default='resnet18',
        help='Model for single fold mode (default: resnet18)'
    )
    
    parser.add_argument(
        '--enhance-features',
        action='store_true',
        help='Use feature enhancement in single fold mode'
    )
    
    parser.add_argument(
        '--use-all-data',
        action='store_true',
        help='Use all data for parameter tuning (not fold-specific)'
    )
    
    # Cross-validation configuration
    parser.add_argument(
        '--folds',
        nargs='+',
        type=int,
        choices=range(1, 11),
        default=list(range(1, 11)),
        help='Folds to run (default: all folds 1-10)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size (default: 8)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--optimizer',
        choices=['adamw', 'adam', 'sgd'],
        default='adamw', 
        help='Optimizer (default: adamw)'
    )
    
    parser.add_argument(
        '--scheduler',
        choices=['cosine', 'plateau', 'none'],
        default='cosine',
        help='Learning rate scheduler (default: cosine)'
    )
    
    parser.add_argument(
        '--loss-function',
        choices=['focal', 'ce', 'weighted_ce'],
        default='focal',
        help='Loss function (default: focal)'
    )
    
    # Other options
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers (default: 4)'
    )
    
    parser.add_argument(
        '--quick-tune',
        action='store_true',
        help='Use reduced epochs for quick tuning/testing'
    )
    
    parser.add_argument(
        '--config-file',
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--no-class-weights',
        action='store_true',
        help='Disable class weighting for imbalanced data'
    )
    
    return parser


def load_config_from_file(config_file):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        return {}


def create_config_from_args(args):
    """Create configuration dictionary from command line arguments"""
    config = {
        'models': args.models,
        'enhance_small_features': args.enhance_options,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'loss_function': args.loss_function,
        'early_stopping_patience': 15,
        'num_workers': args.num_workers,
        'use_class_weights': not args.no_class_weights,
        'random_state': 42
    }
    
    # Override with config file if provided
    if args.config_file:
        file_config = load_config_from_file(args.config_file)
        config.update(file_config)
    
    return config


def run_data_mapping_mode():
    """Run data mapping creation"""
    print("="*60)
    print("Data Mapping Mode")
    print("="*60)
    
    syn_data_path = "/nas/research/save_10fold_experience/pods/hysuh2/home/03-inference-file/TAU-DDPM-proposed/TAU3-1/"
    tau_data_csv_path = "/home/data/TAU-data.csv"
    output_csv_path = "/home/classification/data/syn_data_mapping.csv"
    
    print(f"Creating data mapping...")
    print(f"Synthetic data path: {syn_data_path}")
    print(f"TAU data CSV: {tau_data_csv_path}")
    print(f"Output CSV: {output_csv_path}")
    
    try:
        mapping_df = create_data_mapping(syn_data_path, tau_data_csv_path, output_csv_path)
        print(f"\nData mapping completed successfully!")
        print(f"Total samples: {len(mapping_df)}")
        print(f"Output saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error creating data mapping: {e}")
        sys.exit(1)


def run_full_mode(args, config):
    """Run full cross-validation experiment"""
    print("="*60)
    print("Full Cross-Validation Mode")
    print("="*60)
    
    # Create manager
    manager = CrossValidationManager(
        data_csv_path=args.data_csv,
        results_dir=args.results_dir,
        config=config
    )
    
    # Print configuration
    print("\nExperiment Configuration:")
    print(f"Models: {config['models']}")
    print(f"Enhancement options: {config['enhance_small_features']}")
    print(f"Folds: {args.folds}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Loss function: {config['loss_function']}")
    print(f"Results directory: {args.results_dir}")
    
    # Run cross-validation
    try:
        results = manager.run_cross_validation(
            model_names=config['models'],
            enhance_options=config['enhance_small_features'],
            folds_to_run=args.folds
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"Total experiments: {len(results)}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        print("Partial results may be saved in the results directory")
    except Exception as e:
        print(f"Error during experiment: {e}")
        sys.exit(1)


def run_single_mode(args, config):
    """Run single fold experiment"""
    print("="*60)
    print(f"Single Fold Mode - Fold {args.fold}")
    print("="*60)
    
    # Create manager
    manager = CrossValidationManager(
        data_csv_path=args.data_csv,
        results_dir=args.results_dir,
        config=config
    )
    
    print(f"\nSingle Fold Configuration:")
    print(f"Fold: {args.fold}")
    print(f"Model: {args.model}")
    print(f"Enhanced features: {args.enhance_features}")
    print(f"Quick tune: {args.quick_tune}")
    
    try:
        result = manager.run_single_fold(
            fold_idx=args.fold,
            model_name=args.model,
            enhance_small_features=args.enhance_features,
            quick_tune=args.quick_tune
        )
        
        print(f"\nSingle fold experiment completed!")
        print(f"Final results:")
        print(f"  AUC: {result['test_metrics']['auc']:.3f}")
        print(f"  F1: {result['test_metrics']['f1']:.3f}")
        print(f"  BACC: {result['test_metrics']['bacc']:.3f}")
        
    except Exception as e:
        print(f"Error during single fold experiment: {e}")
        sys.exit(1)


def run_tune_mode(args):
    """Run parameter tuning mode"""
    tuning_results_dir = os.path.join(args.results_dir, 'tuning')
    
    if args.use_all_data:
        print("="*70)
        print(f"Parameter Tuning Mode - Using ALL DATA")
        print(f"Model: {args.model}")
        print("="*70)
        
        try:
            best_params, tuning_results = run_parameter_tuning_all_data(
                data_csv_path=args.data_csv,
                results_dir=tuning_results_dir,
                model_name=args.model
            )
            
            print(f"\nParameter tuning completed!")
            print(f"Best parameters found:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
                
            # Save best parameters to JSON
            best_params_file = os.path.join(tuning_results_dir, f'best_params_{args.model}_all_data.json')
            with open(best_params_file, 'w') as f:
                json.dump(best_params, f, indent=2)
            
            print(f"Best parameters saved to: {best_params_file}")
            
        except Exception as e:
            print(f"Error during parameter tuning: {e}")
            sys.exit(1)
    else:
        print("="*60)
        print(f"Parameter Tuning Mode - Fold {args.fold}")
        print("="*60)
        
        try:
            best_params, tuning_results = run_parameter_tuning(
                data_csv_path=args.data_csv,
                results_dir=tuning_results_dir,
                tuning_fold=args.fold,
                model_name=args.model
            )
            
            print(f"\nParameter tuning completed!")
            print(f"Best parameters found:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
                
            # Save best parameters to JSON
            best_params_file = os.path.join(tuning_results_dir, f'best_params_{args.model}.json')
            with open(best_params_file, 'w') as f:
                json.dump(best_params, f, indent=2)
            
            print(f"Best parameters saved to: {best_params_file}")
            
        except Exception as e:
            print(f"Error during parameter tuning: {e}")
            sys.exit(1)


def run_resume_mode(args):
    """Resume and analyze existing results"""
    print("="*60)
    print("Resume Mode - Analyzing Existing Results")
    print("="*60)
    
    try:
        # Create manager
        manager = CrossValidationManager(
            data_csv_path=args.data_csv,
            results_dir=args.results_dir
        )
        
        # Load existing results
        results = manager.load_results()
        
        if results:
            # Create summary report
            manager.create_summary_report()
            print(f"\nAnalyzed {len(results)} existing results")
        else:
            print("No existing results found to resume")
            
    except Exception as e:
        print(f"Error during resume: {e}")
        sys.exit(1)


def main():
    """Main function"""
    print("TAU Synthetic Data Classification System")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Check if data CSV exists (except for data-mapping mode)
    if args.mode != 'data-mapping' and not os.path.exists(args.data_csv):
        print(f"Error: Data CSV file not found: {args.data_csv}")
        print("Run with --mode data-mapping first to create the mapping file")
        sys.exit(1)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save command line arguments
    args_file = os.path.join(args.results_dir, 'command_args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Save configuration
    config_file = os.path.join(args.results_dir, 'experiment_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run based on mode
    try:
        if args.mode == 'data-mapping':
            run_data_mapping_mode()
        elif args.mode == 'full':
            run_full_mode(args, config)
        elif args.mode == 'single':
            run_single_mode(args, config)
        elif args.mode == 'tune':
            run_tune_mode(args)
        elif args.mode == 'resume':
            run_resume_mode(args)
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main() 