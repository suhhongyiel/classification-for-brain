# BACC Improvement Classification System for 3D Medical Images

## Overview

This repository contains a comprehensive classification system designed specifically for improving Balanced Accuracy (BACC) in 3D medical image classification tasks. The system is optimized for TAU-PET synthetic data classification (AD vs CN) and includes multiple advanced techniques for handling class imbalance and improving model performance.

## Features

### 🎯 BACC-Focused Optimization
- **Extreme BACC Loss**: Custom loss function with aggressive minority class weighting (8x-10x)
- **Data Balancing**: Automatic replication of minority samples to achieve target class ratios
- **Threshold Optimization**: Post-processing optimization for maximum BACC
- **Multiple Methods**: 4 different optimization strategies (Methods 0-3)

### 🧠 Model Architectures
- **3D ResNet18**: Enhanced with dropout and feature enhancement
- **3D Vision Transformer (ViT)**: For advanced feature learning
- **MONAI Integration**: Leveraging medical imaging libraries

### 📊 Advanced Training Techniques
- **10-Fold Cross-Validation**: For stable and reliable results
- **Stratified Sampling**: Maintaining class distribution across folds
- **Early Stopping**: Preventing overfitting with configurable patience
- **Learning Rate Scheduling**: Cosine annealing for optimal convergence

### 🔄 Data Augmentation (Medical-Safe)
- **Gaussian Noise**: Controlled noise addition
- **Brightness/Contrast Adjustment**: Intensity modifications
- **Gamma Correction**: Non-linear intensity transformations
- **Gaussian Blur & Sharpen**: Spatial filtering operations
- **Mixup**: Interpolation-based augmentation
- **No Position Shifts/Rotations**: Medical imaging safety compliance

### 📈 Performance Monitoring
- **Real-time Metrics**: AUC, F1-Score, BACC during training
- **Comprehensive Logging**: Detailed fold-by-fold results
- **Confusion Matrix Analysis**: Detailed error analysis
- **Threshold Optimization**: Automatic BACC maximization

## Installation

```bash
# Clone the repository
git clone https://github.com/suhhongyiel/imp_cv.git
cd imp_cv

# Install dependencies
pip install torch torchvision
pip install monai
pip install nibabel
pip install scikit-learn
pip install pandas numpy matplotlib
pip install scipy
```

## Usage

### Basic Training

```bash
python ten_fold_training.py \
    --data-csv /path/to/your/data.csv \
    --model resnet18 \
    --method 0 \
    --epochs 15 \
    --gpu-id 0
```

### Method Selection

The system provides 4 different optimization methods:

- **Method 0**: Extreme BACC with data balancing
  - Extreme BACC loss (10x minority weight)
  - Data balancing (40% AD target ratio)
  - Heavy augmentation (90% intensity)

- **Method 1**: Ultra-aggressive class balancing
  - 5x weighted CrossEntropy loss
  - Data balancing + augmentation
  - Conservative learning rate

- **Method 2**: Extreme BACC with heavy augmentation
  - Extreme BACC loss + maximum augmentation
  - Minimal regularization for better learning
  - Extended patience (20 epochs)

- **Method 3**: Ultimate BACC ensemble
  - All techniques combined
  - Threshold optimization enabled
  - Maximum patience (25 epochs)

### Advanced Configuration

```bash
python ten_fold_training.py \
    --data-csv /path/to/data.csv \
    --model resnet18 \
    --method 3 \
    --epochs 20 \
    --batch-size 4 \
    --lr 3e-6 \
    --weight-decay 1e-5 \
    --dropout-rate 0.1 \
    --early-stopping-patience 25 \
    --threshold-optimization \
    --gpu-id 0
```

## Data Format

The system expects a CSV file with the following columns:
- `subject_id`: Unique identifier for each subject
- `file_path`: Path to the NIfTI file (.nii.gz)
- `label`: Binary label (0=CN, 1=AD)

Example:
```csv
subject_id,file_path,label
sub_001,/path/to/sub_001.nii.gz,0
sub_002,/path/to/sub_002.nii.gz,1
...
```

## Results

The system generates comprehensive results including:

- **Fold-by-fold metrics**: Detailed performance for each fold
- **Overall statistics**: Mean, standard deviation, and range
- **Model checkpoints**: Best models for each fold
- **Summary CSV**: Complete results in tabular format

Example output:
```
📊 10-FOLD CROSS VALIDATION RESULTS (BACC-FOCUSED)
================================================================================
BACC:         0.8464 ± 0.0423 (Range: 0.7858 - 0.9044)
AUC:          0.9122 ± 0.0332 (Range: 0.8826 - 0.9566)
F1-Score:     0.8373 ± 0.0408 (Range: 0.7821 - 0.8998)
================================================================================
```

## Key Innovations

### 1. Extreme BACC Loss
```python
class ExtremeBACCLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=3.0, minority_weight=10.0):
        # Combines focal loss with extreme class weighting
        # 10x weight for minority class (AD)
```

### 2. Data Balancing
```python
class DataBalancer:
    def balance_data(self, data_df):
        # Replicates minority samples to achieve 40% AD ratio
        # Maintains data integrity while improving balance
```

### 3. Medical-Safe Augmentation
- **No geometric transformations**: Preserves spatial relationships
- **Intensity-based modifications**: Safe for medical interpretation
- **Controlled randomness**: Reproducible results

## Performance Comparison

| Method | Average BACC | Best Fold BACC | Stability |
|--------|-------------|----------------|-----------|
| Method 0 | 0.8464 ± 0.0423 | 0.9044 | High |
| Method 1 | TBD | TBD | TBD |
| Method 2 | TBD | TBD | TBD |
| Method 3 | TBD | TBD | TBD |

## File Structure

```
imp_cls_cv/
├── ten_fold_training.py          # Main training script
├── utils/
│   ├── dataset.py               # Data loading and preprocessing
│   └── trainer.py               # Training and evaluation logic
├── models/
│   └── models.py                # Model architectures
├── medical_data_augmentation.py # Medical-safe augmentation
├── results/                     # Training results (excluded from repo)
└── README.md                    # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bacc_improvement_classification,
  title={BACC Improvement Classification System for 3D Medical Images},
  author={Your Name},
  year={2025},
  url={https://github.com/suhhongyiel/imp_cv}
}
```

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This system is specifically designed for medical imaging tasks and includes safety measures to ensure clinical relevance and interpretability. 