#!/usr/bin/env python3
"""
Ensemble Methods for BACC Improvement
Combines multiple models and predictions for better performance
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class ModelEnsemble:
    """
    Ensemble of multiple models for improved BACC
    """
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        
    def predict_proba(self, X):
        """Get ensemble predictions"""
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                # For PyTorch models
                model.eval()
                with torch.no_grad():
                    outputs = model(X)
                    pred = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred / sum(self.weights)
    
    def predict(self, X, threshold=0.5):
        """Get ensemble predictions with threshold"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

class StackingEnsemble:
    """
    Stacking ensemble with meta-learner
    """
    
    def __init__(self, base_models, meta_learner=None):
        self.base_models = base_models
        self.meta_learner = meta_learner if meta_learner is not None else LogisticRegression()
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit stacking ensemble"""
        # Get base model predictions
        base_predictions = []
        
        for model in self.base_models:
            if hasattr(model, 'fit'):
                model.fit(X, y)
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                else:
                    pred = model.predict(X).reshape(-1, 1)
            else:
                # For pre-trained PyTorch models
                model.eval()
                with torch.no_grad():
                    outputs = model(X)
                    pred = torch.softmax(outputs, dim=1).cpu().numpy()
            
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_X = np.hstack(base_predictions)
        
        # Fit meta-learner
        self.meta_learner.fit(stacked_X, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X):
        """Get stacking predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get base model predictions
        base_predictions = []
        
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            elif hasattr(model, 'predict'):
                pred = model.predict(X).reshape(-1, 1)
            else:
                # For PyTorch models
                model.eval()
                with torch.no_grad():
                    outputs = model(X)
                    pred = torch.softmax(outputs, dim=1).cpu().numpy()
            
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_X = np.hstack(base_predictions)
        
        # Get meta-learner predictions
        return self.meta_learner.predict_proba(stacked_X)
    
    def predict(self, X, threshold=0.5):
        """Get stacking predictions with threshold"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

class BlendingEnsemble:
    """
    Blending ensemble with validation set
    """
    
    def __init__(self, base_models, meta_learner=None, val_size=0.2):
        self.base_models = base_models
        self.meta_learner = meta_learner if meta_learner is not None else LogisticRegression()
        self.val_size = val_size
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit blending ensemble"""
        from sklearn.model_selection import train_test_split
        
        # Split data for blending
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, stratify=y, random_state=42
        )
        
        # Train base models on training data
        base_predictions_train = []
        base_predictions_val = []
        
        for model in self.base_models:
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
                
                # Get predictions on training and validation sets
                if hasattr(model, 'predict_proba'):
                    pred_train = model.predict_proba(X_train)
                    pred_val = model.predict_proba(X_val)
                else:
                    pred_train = model.predict(X_train).reshape(-1, 1)
                    pred_val = model.predict(X_val).reshape(-1, 1)
            else:
                # For pre-trained PyTorch models
                model.eval()
                with torch.no_grad():
                    outputs_train = model(X_train)
                    outputs_val = model(X_val)
                    pred_train = torch.softmax(outputs_train, dim=1).cpu().numpy()
                    pred_val = torch.softmax(outputs_val, dim=1).cpu().numpy()
            
            base_predictions_train.append(pred_train)
            base_predictions_val.append(pred_val)
        
        # Stack validation predictions for meta-learner
        stacked_X_val = np.hstack(base_predictions_val)
        
        # Fit meta-learner on validation predictions
        self.meta_learner.fit(stacked_X_val, y_val)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X):
        """Get blending predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get base model predictions
        base_predictions = []
        
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            elif hasattr(model, 'predict'):
                pred = model.predict(X).reshape(-1, 1)
            else:
                # For PyTorch models
                model.eval()
                with torch.no_grad():
                    outputs = model(X)
                    pred = torch.softmax(outputs, dim=1).cpu().numpy()
            
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_X = np.hstack(base_predictions)
        
        # Get meta-learner predictions
        return self.meta_learner.predict_proba(stacked_X)
    
    def predict(self, X, threshold=0.5):
        """Get blending predictions with threshold"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

def create_ensemble_predictions(model_predictions, method='voting', weights=None):
    """
    Create ensemble predictions from multiple model outputs
    
    Args:
        model_predictions: List of (predictions, probabilities) tuples
        method: 'voting', 'weighted_average', 'stacking'
        weights: Weights for weighted average
    """
    
    if method == 'voting':
        # Majority voting
        predictions = [pred for pred, _ in model_predictions]
        ensemble_pred = np.mean(predictions, axis=0)
        return (ensemble_pred >= 0.5).astype(int)
    
    elif method == 'weighted_average':
        # Weighted average of probabilities
        if weights is None:
            weights = [1.0] * len(model_predictions)
        
        probabilities = [prob[:, 1] for _, prob in model_predictions]  # AD probabilities
        ensemble_prob = np.zeros_like(probabilities[0])
        
        for prob, weight in zip(probabilities, weights):
            ensemble_prob += prob * weight
        
        ensemble_prob /= sum(weights)
        return (ensemble_prob >= 0.5).astype(int)
    
    elif method == 'stacking':
        # Stacking with logistic regression
        probabilities = [prob for _, prob in model_predictions]
        stacked_features = np.hstack(probabilities)
        
        # Simple meta-learner (logistic regression)
        from sklearn.linear_model import LogisticRegression
        meta_learner = LogisticRegression(random_state=42)
        
        # For simplicity, use the first model's predictions as target
        # In practice, you'd use cross-validation
        y_target = model_predictions[0][0]
        meta_learner.fit(stacked_features, y_target)
        
        ensemble_prob = meta_learner.predict_proba(stacked_features)[:, 1]
        return (ensemble_prob >= 0.5).astype(int)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

def evaluate_ensemble(predictions, true_labels, probabilities=None):
    """
    Evaluate ensemble performance
    """
    bacc = balanced_accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    if probabilities is not None:
        auc = roc_auc_score(true_labels, probabilities)
    else:
        auc = 0.5  # Default if no probabilities
    
    return {
        'bacc': bacc,
        'f1': f1,
        'auc': auc,
        'accuracy': (predictions == true_labels).mean()
    }

if __name__ == "__main__":
    # Test ensemble methods
    print("Ensemble methods ready for use!")
    print("Available methods:")
    print("1. ModelEnsemble - Simple weighted averaging")
    print("2. StackingEnsemble - Stacking with meta-learner")
    print("3. BlendingEnsemble - Blending with validation set")
    print("4. create_ensemble_predictions - Utility function") 