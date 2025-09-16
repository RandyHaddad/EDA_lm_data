"""
Comprehensive Statistical Analysis of Cosmopedia-100k Dataset
Focus: MoE Training Efficiency Optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textstat
import re
from collections import Counter
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Create output directory
import os
os.makedirs('eda_output', exist_ok=True)

print("="*60)
print("COSMOPEDIA-100K COMPREHENSIVE EDA")
print("="*60)

# Load dataset
print("Loading dataset...")
ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
df = ds.to_pandas()

print(f"Dataset loaded: {len(df)} samples")

# ============================================================================
# BASIC STATISTICS AND DISTRIBUTIONS
# ============================================================================

print("\n" + "="*50)
print("1. COMPREHENSIVE DATASET STATISTICS")
print("="*50)

# Token length statistics
token_stats = df['text_token_length'].describe()
print("Token Length Statistics:")
print(token_stats)

# Calculate additional percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
token_percentiles = df['text_token_length'].quantile([p/100 for p in percentiles])
print("\nDetailed Percentiles:")
for p, val in zip(percentiles, token_percentiles):
    print(f"  {p}th percentile: {val:.0f} tokens")

# ============================================================================
# CONTENT TYPE ANALYSIS
# ============================================================================

print("\n" + "="*50)
print("2. CONTENT TYPE ANALYSIS")
print("="*50)

# Format distribution
format_counts = df['format'].value_counts()
print("Format Distribution:")
for fmt, count in format_counts.items():
    print(f"  {fmt}: {count} ({count/len(df)*100:.1f}%)")

# Audience distribution
audience_counts = df['audience'].value_counts()
print("\nAudience Distribution:")
for aud, count in audience_counts.items():
    print(f"  {aud}: {count} ({count/len(df)*100:.1f}%)")

# Seed data distribution
seed_counts = df['seed_data'].value_counts()
print("\nSeed Data Distribution:")
for seed, count in seed_counts.items():
    print(f"  {seed}: {count} ({count/len(df)*100:.1f}%)")

# Cross-tabulation analysis
print("\nFormat vs Audience Cross-tabulation:")
crosstab = pd.crosstab(df['format'], df['audience'])
print(crosstab)

# ============================================================================
# TEXT QUALITY METRICS
# ============================================================================

print("\n" + "="*50)
print("3. TEXT QUALITY ANALYSIS")
print("="*50)

print("Computing text quality metrics for sample (first 1000 samples)...")
sample_size = min(1000, len(df))
sample_df = df.head(sample_size).copy()

# Calculate readability metrics
readability_metrics = []
for i, text in enumerate(sample_df['text']):
    if i % 100 == 0:
        print(f"  Processing sample {i+1}/{sample_size}")
    
    try:
        metrics = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'smog_index': textstat.smog_index(text),
            'difficult_words': textstat.difficult_words(text),
            'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
            'sentence_count': textstat.sentence_count(text),
            'word_count': len(text.split()),
            'char_count': len(text),
            'avg_sentence_length': len(text.split()) / max(1, textstat.sentence_count(text)),
            'avg_word_length': len(text.replace(' ', '')) / max(1, len(text.split()))
        }
        readability_metrics.append(metrics)
    except:
        # Handle any text processing errors
        readability_metrics.append({k: np.nan for k in ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'automated_readability_index', 'coleman_liau_index', 'smog_index', 'difficult_words', 'dale_chall_readability_score', 'sentence_count', 'word_count', 'char_count', 'avg_sentence_length', 'avg_word_length']})

readability_df = pd.DataFrame(readability_metrics)
sample_df = pd.concat([sample_df.reset_index(drop=True), readability_df], axis=1)

print("\nReadability Metrics Summary (sample):")
for col in ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'word_count', 'sentence_count']:
    stats = readability_df[col].describe()
    print(f"\n{col}:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  Max: {stats['max']:.2f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*50)
print("4. CREATING VISUALIZATIONS")
print("="*50)

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 24))

# 1. Token length distribution
plt.subplot(4, 3, 1)
plt.hist(df['text_token_length'], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(df['text_token_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["text_token_length"].mean():.0f}')
plt.axvline(df['text_token_length'].median(), color='orange', linestyle='--', label=f'Median: {df["text_token_length"].median():.0f}')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.title('Distribution of Token Lengths')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Token length by format
plt.subplot(4, 3, 2)
df.boxplot(column='text_token_length', by='format', ax=plt.gca())
plt.xticks(rotation=45)
plt.title('Token Length by Format')
plt.suptitle('')

# 3. Format distribution
plt.subplot(4, 3, 3)
format_counts.plot(kind='bar')
plt.title('Distribution of Formats')
plt.xlabel('Format')
plt.ylabel('Count')
plt.xticks(rotation=45)

# 4. Audience distribution
plt.subplot(4, 3, 4)
audience_counts.plot(kind='bar')
plt.title('Distribution of Audiences')
plt.xlabel('Audience')
plt.ylabel('Count')
plt.xticks(rotation=45)

# 5. Seed data distribution
plt.subplot(4, 3, 5)
seed_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Seed Data Sources')

# 6. Token length log scale
plt.subplot(4, 3, 6)
plt.hist(df['text_token_length'], bins=50, alpha=0.7, edgecolor='black')
plt.yscale('log')
plt.xlabel('Token Length')
plt.ylabel('Frequency (log scale)')
plt.title('Token Length Distribution (Log Scale)')
plt.grid(True, alpha=0.3)

# 7. Readability metrics (if available)
if len(readability_df) > 0:
    plt.subplot(4, 3, 7)
    plt.hist(readability_df['flesch_reading_ease'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Flesch Reading Ease Score')
    plt.ylabel('Frequency')
    plt.title('Reading Ease Distribution (Sample)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 8)
    plt.hist(readability_df['word_count'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution (Sample)')
    plt.grid(True, alpha=0.3)

# 9. Format vs Audience heatmap
plt.subplot(4, 3, 9)
sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
plt.title('Format vs Audience Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 10. Token length percentiles
plt.subplot(4, 3, 10)
plt.plot(percentiles, token_percentiles.values, marker='o')
plt.xlabel('Percentile')
plt.ylabel('Token Length')
plt.title('Token Length Percentiles')
plt.grid(True, alpha=0.3)

# 11. Token length by seed data
plt.subplot(4, 3, 11)
df.boxplot(column='text_token_length', by='seed_data', ax=plt.gca())
plt.xticks(rotation=45)
plt.title('Token Length by Seed Data')
plt.suptitle('')

# 12. Token efficiency (tokens per character ratio for sample)
if len(sample_df) > 0 and 'char_count' in sample_df.columns:
    plt.subplot(4, 3, 12)
    token_efficiency = sample_df['text_token_length'] / sample_df['char_count'].replace(0, 1)
    plt.hist(token_efficiency.dropna(), bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Tokens per Character')
    plt.ylabel('Frequency')
    plt.title('Token Efficiency (Sample)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_output/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Comprehensive analysis plot saved to eda_output/comprehensive_analysis.png")

# Save statistical summaries
with open('eda_output/basic_statistics.txt', 'w') as f:
    f.write("COSMOPEDIA-100K BASIC STATISTICS\n")
    f.write("="*50 + "\n\n")
    
    f.write("Dataset Overview:\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
    
    f.write("Token Length Statistics:\n")
    f.write(str(token_stats) + "\n\n")
    
    f.write("Detailed Percentiles:\n")
    for p, val in zip(percentiles, token_percentiles):
        f.write(f"  {p}th percentile: {val:.0f} tokens\n")
    
    f.write("\nFormat Distribution:\n")
    for fmt, count in format_counts.items():
        f.write(f"  {fmt}: {count} ({count/len(df)*100:.1f}%)\n")
    
    f.write("\nAudience Distribution:\n")
    for aud, count in audience_counts.items():
        f.write(f"  {aud}: {count} ({count/len(df)*100:.1f}%)\n")
    
    f.write("\nSeed Data Distribution:\n")
    for seed, count in seed_counts.items():
        f.write(f"  {seed}: {count} ({count/len(df)*100:.1f}%)\n")

print("Basic statistics saved to eda_output/basic_statistics.txt")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print("Next: Running advanced topic analysis and MoE optimization strategies...")

