#!/usr/bin/env python3
"""
Medical Image Data Augmentation for 3D Brain Images
Suitable for fixed-position medical imaging (no rotation/translation)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import monai
from packaging import version
from monai import transforms as monai_transforms
import warnings
warnings.filterwarnings('ignore')

class MedicalImageAugmentation:
    """
    Medical image augmentation suitable for 3D brain images
    No rotation/translation to preserve spatial relationships
    """
    
    def __init__(self, 
                 noise_prob=0.3,
                 brightness_prob=0.3,
                 contrast_prob=0.3,
                 gamma_prob=0.3,
                 blur_prob=0.2,
                 sharpen_prob=0.2,
                 mixup_prob=0.1):  # Add mixup probability
        
        self.noise_prob = noise_prob
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.gamma_prob = gamma_prob
        self.blur_prob = blur_prob
        self.sharpen_prob = sharpen_prob
        self.mixup_prob = mixup_prob
        
    def add_gaussian_noise(self, image, mean=0.0, std=0.01):
        """Add Gaussian noise to image"""
        noise = torch.randn_like(image) * std + mean
        return torch.clamp(image + noise, 0, 1)
    
    def adjust_brightness(self, image, factor=0.1):
        """Adjust brightness (additive)"""
        factor = random.uniform(-factor, factor)
        return torch.clamp(image + factor, 0, 1)
    
    def adjust_contrast(self, image, factor=0.1):
        """Adjust contrast (multiplicative)"""
        factor = random.uniform(1 - factor, 1 + factor)
        mean = torch.mean(image)
        return torch.clamp((image - mean) * factor + mean, 0, 1)
    
    def gamma_correction(self, image, gamma=0.1):
        """Apply gamma correction"""
        gamma = random.uniform(1 - gamma, 1 + gamma)
        return torch.clamp(torch.pow(image, gamma), 0, 1)
    
    def gaussian_blur(self, image, sigma=0.5):
        """Apply Gaussian blur"""
        from scipy.ndimage import gaussian_filter
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
        else:
            image_np = image
            
        # Apply 3D Gaussian blur
        blurred = gaussian_filter(image_np, sigma=sigma)
        return torch.from_numpy(blurred).float()
    
    def sharpen(self, image, strength=0.5):
        """Sharpen image using unsharp masking"""
        # Create a blurred version
        blurred = self.gaussian_blur(image, sigma=1.0)
        
        # Unsharp masking
        sharpened = image + strength * (image - blurred)
        return torch.clamp(sharpened, 0, 1)
    
    def mixup(self, image, alpha=0.2):
        """Mixup augmentation with random noise"""
        if random.random() < self.mixup_prob:
            # Create random noise with same shape
            noise = torch.randn_like(image) * 0.1
            # Mix with original image
            mixed = image * (1 - alpha) + noise * alpha
            return torch.clamp(mixed, 0, 1)
        return image
    
    def __call__(self, image):
        """Apply augmentations with probabilities"""
        if random.random() < self.noise_prob:
            image = self.add_gaussian_noise(image)
            
        if random.random() < self.brightness_prob:
            image = self.adjust_brightness(image)
            
        if random.random() < self.contrast_prob:
            image = self.adjust_contrast(image)
            
        if random.random() < self.gamma_prob:
            image = self.gamma_correction(image)
            
        if random.random() < self.blur_prob:
            image = self.gaussian_blur(image)
            
        if random.random() < self.sharpen_prob:
            image = self.sharpen(image)
            
        # Apply mixup at the end
        image = self.mixup(image)
        
        return image

class BalancedDatasetAugmentation:
    """
    Balanced dataset augmentation for class imbalance
    Applies more augmentation to minority class (AD)
    """
    
    def __init__(self, minority_boost=2.0):
        self.minority_boost = minority_boost
        self.augmentation = MedicalImageAugmentation()
        
    def augment_minority_samples(self, data_df, target_class=1):
        """Augment minority class samples"""
        minority_samples = data_df[data_df['label'] == target_class].copy()
        majority_samples = data_df[data_df['label'] != target_class]
        
        # Calculate how many augmented samples to create
        minority_count = len(minority_samples)
        majority_count = len(majority_samples)
        target_count = int(majority_count / self.minority_boost)
        augment_count = target_count - minority_count
        
        if augment_count <= 0:
            return data_df
        
        # Create augmented samples
        augmented_samples = []
        for _ in range(augment_count):
            # Randomly select a minority sample
            sample = minority_samples.sample(n=1).iloc[0]
            
            # Create augmented version
            augmented_sample = sample.copy()
            augmented_sample['subject_id'] = f"{sample['subject_id']}_aug_{_}"
            augmented_sample['filename'] = f"{sample['filename']}_aug_{_}"
            augmented_sample['augmented'] = True
            
            augmented_samples.append(augmented_sample)
        
        # Combine original and augmented data
        augmented_df = pd.concat([
            data_df,
            pd.DataFrame(augmented_samples)
        ], ignore_index=True)
        
        return augmented_df

def create_medical_transforms(augment=True, intensity=0.9):
    """
    Create simple medical transforms without MONAI LoadImage
    Since we load images manually in dataset, just return augmentation wrapper
    """
    class SimpleTransform:
        def __init__(self, augment=True, intensity=0.9):
            self.augment = augment
            self.augmenter = MedicalImageAugmentation(
                noise_prob=intensity,
                brightness_prob=intensity,
                contrast_prob=intensity,
                gamma_prob=intensity,
                blur_prob=intensity*0.9,  # Very high blur probability
                sharpen_prob=intensity*0.8,  # Very high sharpen probability
                mixup_prob=intensity*0.5  # High mixup probability
            ) if augment else None
            
        def __call__(self, image):
            # Image is already loaded as tensor in dataset
            if self.augment and self.augmenter:
                return self.augmenter(image)
            return image
    
    return SimpleTransform(augment=augment, intensity=intensity)

if __name__ == "__main__":
    # Test augmentation
    import pandas as pd
    
    # Load data
    data_df = pd.read_csv('/home/imp_cls/data/syn_data_mapping.csv')
    
    # Test balanced augmentation
    balancer = BalancedDatasetAugmentation(minority_boost=2.0)
    balanced_df = balancer.augment_minority_samples(data_df)
    
    print(f"Original samples: {len(data_df)}")
    print(f"Balanced samples: {len(balanced_df)}")
    print(f"Original AD: {len(data_df[data_df['label']==1])}")
    print(f"Balanced AD: {len(balanced_df[balanced_df['label']==1])}")
    
    # Save balanced dataset
    balanced_df.to_csv('/home/imp_cls/data/balanced_syn_data_mapping.csv', index=False)
    print("Balanced dataset saved!") 