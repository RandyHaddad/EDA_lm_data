"""
Comprehensive EDA for Cosmopedia-100k Dataset
Focus: Training efficiency optimization for MoE models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

print("Loading Cosmopedia-100k dataset...")
# Load the dataset
try:
    ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
    print(f"Dataset loaded successfully!")
    print(f"Dataset size: {len(ds)} samples")
    
    # Convert to pandas for easier analysis
    df = ds.to_pandas()
    print(f"Converted to pandas DataFrame with shape: {df.shape}")
    
    # Basic dataset info
    print("\n" + "="*50)
    print("BASIC DATASET INFORMATION")
    print("="*50)
    print(f"Dataset columns: {list(df.columns)}")
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    print(f"\nMissing values per column:")
    missing_info = df.isnull().sum()
    for col, missing in missing_info.items():
        print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")
    
    # Data types
    print(f"\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # First few rows for inspection
    print(f"\nFirst 3 samples (truncated):")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        for col in df.columns:
            if col == 'text':
                print(f"  {col}: {str(df.iloc[i][col])[:200]}...")
            elif col == 'prompt':
                print(f"  {col}: {str(df.iloc[i][col])[:150]}...")
            else:
                print(f"  {col}: {df.iloc[i][col]}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have internet connection and the datasets library installed.")

