#!/usr/bin/env python3
"""
Dataset classes for TAU synthetic data classification
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as transforms
from monai import transforms as monai_transforms
from monai.data import PersistentDataset
import warnings
warnings.filterwarnings('ignore')


class TAUSyntheticDataset(Dataset):
    """
    Dataset class for TAU synthetic data
    Handles 3D brain images (64x64x64) for AD/CN classification
    """
    
    def __init__(self, data_df, transform=None, cache_data=False):
        """
        Args:
            data_df: DataFrame with columns ['file_path', 'label', 'diagnosis']
            transform: Transform to apply to the data
            cache_data: Whether to cache data in memory for faster access
        """
        self.data_df = data_df.reset_index(drop=True)
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {} if cache_data else None
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
            
        # Get file info
        row = self.data_df.iloc[idx]
        file_path = row['file_path']
        label = row['label']
        
        # Load nii.gz file
        try:
            nii_img = nib.load(file_path)
            image = nii_img.get_fdata().astype(np.float32)
            
            # Ensure correct shape (64, 64, 64)
            if image.shape != (64, 64, 64):
                print(f"Warning: Image {file_path} has shape {image.shape}, expected (64, 64, 64)")
                
            # Add channel dimension for MONAI transforms (1, 64, 64, 64)
            image = np.expand_dims(image, axis=0)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor if loading fails
            image = np.zeros((1, 64, 64, 64), dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Ensure shape is correct (C, H, W, D)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add channel dimension
            
        label = torch.tensor(label).long()
        
        sample = {
            'image': image,
            'label': label,
            'subject_id': row['subject_id'],
            'file_path': file_path
        }
        
        if self.cache_data:
            self.cache[idx] = sample
            
        return sample


def get_transforms(mode='train', enhance_small_features=False):
    """
    Get transforms for data augmentation and preprocessing
    
    Args:
        mode: 'train', 'val', or 'test'
        enhance_small_features: Whether to apply enhancement for small image sizes
    """
    
    if mode == 'train':
        transforms_list = [
            monai_transforms.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            monai_transforms.RandRotated(keys=['image'], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
            monai_transforms.RandFlipd(keys=['image'], spatial_axis=0, prob=0.5),
            monai_transforms.RandFlipd(keys=['image'], spatial_axis=1, prob=0.5),
            monai_transforms.RandFlipd(keys=['image'], spatial_axis=2, prob=0.5),
            monai_transforms.RandGaussianNoised(keys=['image'], prob=0.3, std=0.1),
            monai_transforms.RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
            monai_transforms.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.3),
        ]
        
        if enhance_small_features:
            # Add enhancements for small image features
            transforms_list.extend([
                monai_transforms.RandGaussianSmoothd(keys=['image'], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0), prob=0.2),
                monai_transforms.RandHistogramShiftd(keys=['image'], num_control_points=10, prob=0.2),
            ])
            
    else:  # val or test
        transforms_list = [
            monai_transforms.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
        ]
    
    # Convert to Compose
    transform = monai_transforms.Compose(transforms_list)
    
    return transform


def get_numpy_transforms(mode='train', enhance_small_features=False):
    """
    Get numpy-based transforms (for direct numpy arrays)
    """
    if mode == 'train':
        transform = monai_transforms.Compose([
            monai_transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
            monai_transforms.RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
            monai_transforms.RandFlip(spatial_axis=0, prob=0.5),
            monai_transforms.RandFlip(spatial_axis=1, prob=0.5),
            monai_transforms.RandFlip(spatial_axis=2, prob=0.5),
            monai_transforms.RandGaussianNoise(prob=0.3, std=0.1),
            monai_transforms.RandScaleIntensity(factors=0.1, prob=0.3),
            monai_transforms.RandShiftIntensity(offsets=0.1, prob=0.3),
        ])
    else:
        transform = monai_transforms.Compose([
            monai_transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
        ])
    
    return transform


def create_stratified_splits(data_df, fold_idx, n_splits=10, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create stratified train/val/test splits for a specific fold
    
    Args:
        data_df: DataFrame with data
        fold_idx: Current fold index (1-10)
        n_splits: Number of folds for cross-validation
        test_size: Size of test set (fraction)
        val_size: Size of validation set (fraction of remaining data after test split)
        random_state: Random seed
    
    Returns:
        train_df, val_df, test_df
    """
    # Filter data for current fold
    fold_data = data_df[data_df['fold'] == fold_idx].copy()
    
    # First split: separate test set
    skf1 = StratifiedKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_val_idx, test_idx = next(skf1.split(fold_data, fold_data['label']))
    
    train_val_df = fold_data.iloc[train_val_idx]
    test_df = fold_data.iloc[test_idx]
    
    # Second split: separate train and validation from remaining data
    skf2 = StratifiedKFold(n_splits=int(1/val_size), shuffle=True, random_state=random_state+1)
    train_idx, val_idx = next(skf2.split(train_val_df, train_val_df['label']))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    
    print(f"Fold {fold_idx} - Train: {len(train_df)} (AD: {sum(train_df['label'])}, CN: {len(train_df)-sum(train_df['label'])})")
    print(f"Fold {fold_idx} - Val: {len(val_df)} (AD: {sum(val_df['label'])}, CN: {len(val_df)-sum(val_df['label'])})")
    print(f"Fold {fold_idx} - Test: {len(test_df)} (AD: {sum(test_df['label'])}, CN: {len(test_df)-sum(test_df['label'])})")
    
    return train_df, val_df, test_df


def create_stratified_splits_all_data(data_df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create stratified train/val/test splits using ALL data (not fold-specific)
    
    Args:
        data_df: DataFrame with all data from all folds
        test_size: Size of test set (fraction)
        val_size: Size of validation set (fraction of remaining data after test split)
        random_state: Random seed
    
    Returns:
        train_df, val_df, test_df
    """
    print("Using ALL data for stratified splits (not fold-specific)")
    
    # Use all data
    all_data = data_df.copy()
    
    # First split: separate test set
    skf1 = StratifiedKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_val_idx, test_idx = next(skf1.split(all_data, all_data['label']))
    
    train_val_df = all_data.iloc[train_val_idx]
    test_df = all_data.iloc[test_idx]
    
    # Second split: separate train and validation from remaining data
    skf2 = StratifiedKFold(n_splits=int(1/val_size), shuffle=True, random_state=random_state+1)
    train_idx, val_idx = next(skf2.split(train_val_df, train_val_df['label']))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    
    print(f"ALL DATA - Train: {len(train_df)} (AD: {sum(train_df['label'])}, CN: {len(train_df)-sum(train_df['label'])})")
    print(f"ALL DATA - Val: {len(val_df)} (AD: {sum(val_df['label'])}, CN: {len(val_df)-sum(val_df['label'])})")
    print(f"ALL DATA - Test: {len(test_df)} (AD: {sum(test_df['label'])}, CN: {len(test_df)-sum(test_df['label'])})")
    print(f"Total samples used: {len(all_data)} (AD: {sum(all_data['label'])}, CN: {len(all_data)-sum(all_data['label'])})")
    
    return train_df, val_df, test_df


def get_dataloaders(train_df, val_df, test_df, batch_size=8, num_workers=4, enhance_small_features=False):
    """
    Create DataLoaders for train, validation, and test sets
    """
    
    # Create transforms
    train_transform = get_numpy_transforms('train', enhance_small_features)
    val_transform = get_numpy_transforms('val', enhance_small_features)
    
    # Create datasets
    train_dataset = TAUSyntheticDataset(train_df, transform=train_transform, cache_data=True)
    val_dataset = TAUSyntheticDataset(val_df, transform=val_transform, cache_data=True)
    test_dataset = TAUSyntheticDataset(test_df, transform=val_transform, cache_data=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    data_path = "/home/classification/data/syn_data_mapping.csv"
    data_df = pd.read_csv(data_path)
    
    # Test with fold 1
    train_df, val_df, test_df = create_stratified_splits(data_df, fold_idx=1)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, batch_size=4, num_workers=2
    )
    
    # Test loading a batch
    print("Testing data loading...")
    for batch in train_loader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch labels: {batch['label']}")
        print(f"Sample subject IDs: {batch['subject_id'][:3]}")
        break
    
    print("Dataset testing completed successfully!") 