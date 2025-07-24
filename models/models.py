#!/usr/bin/env python3
"""
3D models for TAU synthetic data classification
Implements 3D ViT and 3D ResNet18 using MONAI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT, ResNet
from monai.networks.blocks import Warp
import numpy as np


class Enhanced3DViT(nn.Module):
    """
    Enhanced 3D Vision Transformer for small brain images (64x64x64)
    Includes mechanisms to handle small image sizes effectively
    """
    
    def __init__(
        self,
        img_size=(64, 64, 64),
        patch_size=(8, 8, 8),
        in_channels=1,
        num_classes=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout_rate=0.1,
        enhance_small_features=False
    ):
        super().__init__()
        
        self.enhance_small_features = enhance_small_features
        self.img_size = img_size
        
        # Feature enhancement for small images
        if enhance_small_features:
            self.feature_enhancer = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, in_channels, kernel_size=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            )
            
            # Multi-scale feature extraction
            self.multiscale_conv = nn.ModuleList([
                nn.Conv3d(in_channels, in_channels, kernel_size=k, padding=k//2)
                for k in [3, 5, 7]
            ])
            
            # Attention for feature fusion
            self.feature_attention = nn.Sequential(
                nn.Conv3d(in_channels * 4, in_channels, kernel_size=1),  # 4 = original + 3 scales
                nn.Sigmoid()
            )
        
        # 3D Vision Transformer
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            num_layers=depth,
            num_heads=num_heads,
            classification=False,  # We'll add our own classifier
            dropout_rate=dropout_rate,
            spatial_dims=3
        )
        
        # Custom classifier with additional regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Input shape: (B, C, H, W, D)
        original_x = x
        
        if self.enhance_small_features:
            # Enhance features for small images
            enhanced_x = self.feature_enhancer(x)
            
            # Multi-scale feature extraction
            multiscale_features = [enhanced_x]
            for conv in self.multiscale_conv:
                multiscale_features.append(conv(enhanced_x))
            
            # Concatenate multi-scale features
            concat_features = torch.cat(multiscale_features, dim=1)
            
            # Apply attention to fuse features
            attention_weights = self.feature_attention(concat_features)
            x = enhanced_x * attention_weights + original_x
        
        # Pass through ViT
        features, _ = self.vit(x)  # Returns (B, num_patches, embed_dim)
        
        # Global average pooling over patches
        features = features.mean(dim=1)  # (B, embed_dim)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class Enhanced3DResNet(nn.Module):
    """
    Enhanced 3D ResNet18 for small brain images (64x64x64)
    Includes mechanisms to handle small image sizes effectively
    """
    
    def __init__(
        self,
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2,
        block='basic',
        layers=[2, 2, 2, 2],  # ResNet18 structure
        block_inplanes=[64, 128, 256, 512],
        enhance_small_features=False,
        dropout_prob=0.2
    ):
        super().__init__()
        
        self.enhance_small_features = enhance_small_features
        
        # Feature enhancement for small images
        if enhance_small_features:
            self.feature_enhancer = nn.Sequential(
                nn.Conv3d(n_input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, n_input_channels, kernel_size=1),
                nn.BatchNorm3d(n_input_channels)
            )
            
            # Residual connection with attention
            mid_channels = max(1, n_input_channels // 4)  # At least 1 channel
            self.residual_attention = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(n_input_channels, mid_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, n_input_channels, 1),
                nn.Sigmoid()
            )
        
        # 3D ResNet using MONAI
        self.resnet = ResNet(
            block=block,
            layers=layers,
            block_inplanes=block_inplanes,
            spatial_dims=spatial_dims,
            n_input_channels=n_input_channels,
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
            shortcut_type='B',
            widen_factor=1.0,
            num_classes=num_classes,  # Direct classification
            feed_forward=True,
            bias_downsample=True
        )
        
        # No additional classifier needed - using ResNet's built-in classifier
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: (B, C, H, W, D)
        original_x = x
        
        if self.enhance_small_features:
            # Enhance features for small images
            enhanced_x = self.feature_enhancer(x)
            
            # Apply attention mechanism
            attention_weights = self.residual_attention(enhanced_x)
            x = enhanced_x * attention_weights + original_x
        
        # Use ResNet directly for classification
        logits = self.resnet(x)
        
        return logits


def get_model(model_name, num_classes=2, enhance_small_features=False, **kwargs):
    """
    Factory function to get models
    
    Args:
        model_name: 'vit' or 'resnet18'
        num_classes: Number of output classes
        enhance_small_features: Whether to apply enhancement mechanisms
        **kwargs: Additional model parameters
    """
    
    if model_name.lower() == 'vit':
        model = Enhanced3DViT(
            num_classes=num_classes,
            enhance_small_features=enhance_small_features,
            **kwargs
        )
    elif model_name.lower() == 'resnet18':
        model = Enhanced3DResNet(
            num_classes=num_classes,
            enhance_small_features=enhance_small_features,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'vit' or 'resnet18'")
    
    return model


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(loss_type='focal', class_weights=None):
    """
    Get loss function for training
    
    Args:
        loss_type: 'focal', 'ce' (cross entropy), or 'weighted_ce'
        class_weights: Weights for classes (for handling imbalance)
    """
    
    if loss_type == 'focal':
        return FocalLoss(alpha=1, gamma=2, weight=class_weights)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Testing models on device: {device}")
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    
    # Test ViT
    print("Testing Enhanced 3D ViT...")
    vit_model = get_model('vit', enhance_small_features=True).to(device)
    vit_output = vit_model(x)
    print(f"ViT output shape: {vit_output.shape}")
    
    # Test ResNet18
    print("Testing Enhanced 3D ResNet18...")
    resnet_model = get_model('resnet18', enhance_small_features=True).to(device)
    resnet_output = resnet_model(x)
    print(f"ResNet18 output shape: {resnet_output.shape}")
    
    # Test loss function
    print("Testing Focal Loss...")
    targets = torch.tensor([0, 1]).to(device)
    focal_loss = get_loss_function('focal')
    loss_value = focal_loss(vit_output, targets)
    print(f"Focal loss value: {loss_value.item()}")
    
    print("Model testing completed successfully!") 