#!/usr/bin/env python3
"""
Data mapping utility for TAU synthetic data classification
Maps synthetic data paths to diagnosis labels from TAU-data.csv
"""

import os
import pandas as pd
import glob
from pathlib import Path
import re


def extract_subject_id(filename):
    """
    Extract subject ID from synthetic data filename
    Example: fold_1_subj_sub-ADNI002S4213_sess_ses-M072_pred.nii.gz -> sub-ADNI002S4213
    """
    pattern = r'fold_\d+_subj_(sub-ADNI\d+S\d+)_sess_'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


def create_data_mapping(syn_data_path, tau_data_csv_path, output_csv_path):
    """
    Create mapping CSV file with synthetic data paths and diagnosis labels
    
    Args:
        syn_data_path: Path to synthetic data directory
        tau_data_csv_path: Path to TAU-data.csv
        output_csv_path: Path to save the mapping CSV
    """
    
    # Load TAU data with diagnosis information
    tau_data = pd.read_csv(tau_data_csv_path)
    
    # Create diagnosis mapping (SMC -> CN, Dementia -> AD)
    diagnosis_mapping = {
        'SMC': 'CN',
        'CN': 'CN', 
        'Dementia': 'AD'
    }
    
    # Filter data to only include CN, SMC, Dementia subjects
    valid_dx = ['CN', 'SMC', 'Dementia']
    tau_data_filtered = tau_data[tau_data['DX'].isin(valid_dx)].copy()
    
    # Apply diagnosis mapping
    tau_data_filtered['diagnosis'] = tau_data_filtered['DX'].map(diagnosis_mapping)
    
    # Create subject to diagnosis mapping
    subject_diagnosis = dict(zip(tau_data_filtered['subject'], tau_data_filtered['diagnosis']))
    
    # Find all synthetic data files
    all_files = []
    for fold in range(1, 11):  # 10 folds
        fold_dir = os.path.join(syn_data_path, f'fold_{fold}_test_nii_results')
        if os.path.exists(fold_dir):
            nii_files = glob.glob(os.path.join(fold_dir, '*.nii.gz'))
            for file_path in nii_files:
                filename = os.path.basename(file_path)
                subject_id = extract_subject_id(filename)
                
                if subject_id and subject_id in subject_diagnosis:
                    all_files.append({
                        'fold': fold,
                        'subject_id': subject_id,
                        'file_path': file_path,
                        'filename': filename,
                        'diagnosis': subject_diagnosis[subject_id],
                        'label': 1 if subject_diagnosis[subject_id] == 'AD' else 0  # AD=1, CN=0
                    })
    
    # Create DataFrame
    mapping_df = pd.DataFrame(all_files)
    
    # Save to CSV
    mapping_df.to_csv(output_csv_path, index=False)
    
    # Print statistics
    print(f"Total samples: {len(mapping_df)}")
    print(f"AD samples: {len(mapping_df[mapping_df['diagnosis'] == 'AD'])}")
    print(f"CN samples: {len(mapping_df[mapping_df['diagnosis'] == 'CN'])}")
    print(f"Samples per fold:")
    print(mapping_df.groupby(['fold', 'diagnosis']).size().unstack(fill_value=0))
    print(f"Mapping saved to: {output_csv_path}")
    
    return mapping_df


if __name__ == "__main__":
    # Paths
    syn_data_path = "/nas/research/save_10fold_experience/pods/hysuh2/home/03-inference-file/TAU-DDPM-proposed/TAU3-1/"
    tau_data_csv_path = "/home/data/TAU-data.csv"
    output_csv_path = "/home/classification/data/syn_data_mapping.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Create mapping
    mapping_df = create_data_mapping(syn_data_path, tau_data_csv_path, output_csv_path) 