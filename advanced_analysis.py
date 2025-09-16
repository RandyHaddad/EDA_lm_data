"""
Advanced Analysis: Topic Diversity, Clustering, and MoE Optimization
Focus: Identifying optimal subset selection strategies for MoE training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
from wordcloud import WordCloud
from collections import Counter
import re
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED ANALYSIS: TOPIC DIVERSITY & MOE OPTIMIZATION")
print("="*70)

# Load dataset
print("Loading dataset...")
ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
df = ds.to_pandas()

# ============================================================================
# CONTENT ANALYSIS AND KEYWORD EXTRACTION
# ============================================================================

print("\n" + "="*50)
print("1. CONTENT ANALYSIS & KEYWORD EXTRACTION")
print("="*50)

def extract_keywords(text, top_n=10):
    """Extract important keywords from text"""
    # Remove common words and extract meaningful terms
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Filter out very common words
    stop_words = {'the', 'and', 'are', 'for', 'with', 'this', 'that', 'you', 'can', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'what', 'about', 'than', 'into', 'some', 'very', 'when', 'much', 'use', 'your', 'how', 'our', 'out', 'just', 'way', 'too', 'any', 'may', 'say'}
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    return Counter(filtered_words).most_common(top_n)

# Analyze content characteristics by format
print("Analyzing content characteristics by format...")

format_analysis = {}
for fmt in df['format'].unique():
    format_subset = df[df['format'] == fmt]
    
    # Sample for analysis (to manage computation time)
    sample_size = min(100, len(format_subset))
    sample = format_subset.sample(sample_size, random_state=42)
    
    # Extract keywords
    all_text = ' '.join(sample['text'].astype(str))
    keywords = extract_keywords(all_text, 20)
    
    # Calculate metrics
    avg_token_length = format_subset['text_token_length'].mean()
    std_token_length = format_subset['text_token_length'].std()
    
    format_analysis[fmt] = {
        'count': len(format_subset),
        'avg_tokens': avg_token_length,
        'std_tokens': std_token_length,
        'keywords': keywords,
        'sample_size': sample_size
    }

print("\nContent Analysis by Format:")
for fmt, analysis in format_analysis.items():
    print(f"\n{fmt}:")
    print(f"  Count: {analysis['count']}")
    print(f"  Avg tokens: {analysis['avg_tokens']:.1f} ± {analysis['std_tokens']:.1f}")
    print(f"  Top keywords: {', '.join([kw[0] for kw in analysis['keywords'][:5]])}")

# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================

print("\n" + "="*50)
print("2. TOPIC CLUSTERING ANALYSIS")
print("="*50)

# Sample for clustering analysis (computational efficiency)
clustering_sample_size = 2000
print(f"Using sample of {clustering_sample_size} documents for clustering...")

sample_indices = np.random.choice(len(df), clustering_sample_size, replace=False)
sample_df = df.iloc[sample_indices].copy()

# Prepare text for vectorization
print("Preparing text data for clustering...")
texts = sample_df['text'].astype(str).tolist()

# TF-IDF Vectorization
print("Computing TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Determine optimal number of clusters
print("Finding optimal number of clusters...")
k_range = range(5, 21)
silhouette_scores = []
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    sil_score = silhouette_score(tfidf_matrix, cluster_labels)
    silhouette_scores.append(sil_score)
    inertias.append(kmeans.inertia_)
    print(f"  k={k}: silhouette={sil_score:.3f}")

# Select optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")

# Perform final clustering
print(f"Performing final clustering with k={optimal_k}...")
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(tfidf_matrix)

# Add cluster labels to sample dataframe
sample_df['cluster'] = cluster_labels

# Analyze clusters
print("\nCluster Analysis:")
cluster_analysis = {}
for cluster_id in range(optimal_k):
    cluster_docs = sample_df[sample_df['cluster'] == cluster_id]
    
    # Get top terms for this cluster
    cluster_center = final_kmeans.cluster_centers_[cluster_id]
    top_indices = cluster_center.argsort()[-10:][::-1]
    top_terms = [feature_names[i] for i in top_indices]
    
    # Analyze cluster characteristics
    format_dist = cluster_docs['format'].value_counts()
    audience_dist = cluster_docs['audience'].value_counts()
    avg_tokens = cluster_docs['text_token_length'].mean()
    
    cluster_analysis[cluster_id] = {
        'size': len(cluster_docs),
        'top_terms': top_terms,
        'main_format': format_dist.index[0] if len(format_dist) > 0 else 'N/A',
        'main_audience': audience_dist.index[0] if len(audience_dist) > 0 else 'N/A',
        'avg_tokens': avg_tokens,
        'format_diversity': len(format_dist),
        'audience_diversity': len(audience_dist)
    }
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {len(cluster_docs)} docs")
    print(f"  Top terms: {', '.join(top_terms[:5])}")
    print(f"  Main format: {format_dist.index[0] if len(format_dist) > 0 else 'N/A'}")
    print(f"  Main audience: {audience_dist.index[0] if len(audience_dist) > 0 else 'N/A'}")
    print(f"  Avg tokens: {avg_tokens:.1f}")

# ============================================================================
# DIMENSIONALITY REDUCTION AND VISUALIZATION
# ============================================================================

print("\n" + "="*50)
print("3. DIMENSIONALITY REDUCTION & VISUALIZATION")
print("="*50)

print("Computing PCA...")
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(tfidf_matrix.toarray())

print("Computing UMAP...")
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_coords = umap_reducer.fit_transform(tfidf_matrix.toarray())

# Create comprehensive visualizations
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Advanced Topic Analysis and Clustering', fontsize=16)

# 1. Cluster elbow curve
axes[0, 0].plot(k_range, inertias, 'bo-')
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].set_title('Elbow Method for Optimal k')
axes[0, 0].grid(True, alpha=0.3)

# 2. Silhouette scores
axes[0, 1].plot(k_range, silhouette_scores, 'ro-')
axes[0, 1].axvline(optimal_k, color='green', linestyle='--', label=f'Optimal k={optimal_k}')
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Analysis')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. PCA visualization
scatter = axes[0, 2].scatter(pca_coords[:, 0], pca_coords[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
axes[0, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
axes[0, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
axes[0, 2].set_title('PCA Visualization of Clusters')
plt.colorbar(scatter, ax=axes[0, 2])

# 4. UMAP visualization
scatter2 = axes[1, 0].scatter(umap_coords[:, 0], umap_coords[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
axes[1, 0].set_xlabel('UMAP 1')
axes[1, 0].set_ylabel('UMAP 2')
axes[1, 0].set_title('UMAP Visualization of Clusters')
plt.colorbar(scatter2, ax=axes[1, 0])

# 5. Format distribution by cluster
cluster_format_matrix = pd.crosstab(sample_df['cluster'], sample_df['format'])
sns.heatmap(cluster_format_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('Format Distribution by Cluster')
axes[1, 1].set_xlabel('Format')
axes[1, 1].set_ylabel('Cluster')

# 6. Token length by cluster
sample_df.boxplot(column='text_token_length', by='cluster', ax=axes[1, 2])
axes[1, 2].set_title('Token Length Distribution by Cluster')
axes[1, 2].set_xlabel('Cluster')
axes[1, 2].set_ylabel('Token Length')

# 7. Cluster sizes
cluster_sizes = sample_df['cluster'].value_counts().sort_index()
axes[2, 0].bar(range(len(cluster_sizes)), cluster_sizes.values)
axes[2, 0].set_xlabel('Cluster ID')
axes[2, 0].set_ylabel('Number of Documents')
axes[2, 0].set_title('Cluster Sizes')
axes[2, 0].set_xticks(range(len(cluster_sizes)))

# 8. Format vs Audience by token efficiency
format_audience_tokens = sample_df.groupby(['format', 'audience'])['text_token_length'].mean().reset_index()
pivot_table = format_audience_tokens.pivot(index='format', columns='audience', values='text_token_length')
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[2, 1])
axes[2, 1].set_title('Avg Token Length: Format vs Audience')

# 9. Diversity metrics by cluster
diversity_metrics = []
for cluster_id in range(optimal_k):
    cluster_docs = sample_df[sample_df['cluster'] == cluster_id]
    format_entropy = -sum((cluster_docs['format'].value_counts(normalize=True) * 
                          np.log(cluster_docs['format'].value_counts(normalize=True) + 1e-10)))
    diversity_metrics.append(format_entropy)

axes[2, 2].bar(range(len(diversity_metrics)), diversity_metrics)
axes[2, 2].set_xlabel('Cluster ID')
axes[2, 2].set_ylabel('Format Diversity (Entropy)')
axes[2, 2].set_title('Format Diversity by Cluster')
axes[2, 2].set_xticks(range(len(diversity_metrics)))

plt.tight_layout()
plt.savefig('eda_output/advanced_clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Advanced clustering visualization saved to eda_output/advanced_clustering_analysis.png")

# ============================================================================
# MOE OPTIMIZATION ANALYSIS
# ============================================================================

print("\n" + "="*50)
print("4. MOE TRAINING OPTIMIZATION ANALYSIS")
print("="*50)

print("Analyzing optimal subset selection strategies for MoE training...")

# Strategy 1: Diversity-based selection
print("\nStrategy 1: Maximum Diversity Selection")
diversity_scores = []
for i, row in sample_df.iterrows():
    # Calculate diversity score based on multiple factors
    format_rarity = 1 / format_analysis[row['format']]['count']
    cluster_balance = 1 / len(sample_df[sample_df['cluster'] == row['cluster']])
    token_efficiency = row['text_token_length'] / 800  # normalized to average
    
    diversity_score = format_rarity + cluster_balance + min(token_efficiency, 2)
    diversity_scores.append(diversity_score)

sample_df['diversity_score'] = diversity_scores

# Strategy 2: Quality-based selection
print("Strategy 2: Quality-based Selection")
quality_scores = []
for i, row in sample_df.iterrows():
    # Quality factors
    optimal_length_score = 1 - abs(row['text_token_length'] - 800) / 800
    format_quality = {'textbook_academic_tone': 1.0, 'educational_piece': 0.9, 
                     'blogpost': 0.7, 'story_reddit': 0.5}.get(row['format'], 0.6)
    audience_quality = {'college_students': 1.0, 'general': 0.8, 
                       'young_children': 0.6}.get(row['audience'], 0.7)
    
    quality_score = optimal_length_score * format_quality * audience_quality
    quality_scores.append(quality_score)

sample_df['quality_score'] = quality_scores

# Strategy 3: Balanced selection
print("Strategy 3: Balanced Selection")
sample_df['balanced_score'] = (sample_df['diversity_score'] + sample_df['quality_score']) / 2

# Generate recommendations for 1000-sample selection
strategies = {
    'diversity': sample_df.nlargest(1000, 'diversity_score'),
    'quality': sample_df.nlargest(1000, 'quality_score'),
    'balanced': sample_df.nlargest(1000, 'balanced_score'),
    'random': sample_df.sample(1000, random_state=42)
}

print("\nSelection Strategy Comparison (1000 samples):")
strategy_analysis = {}

for strategy_name, selected_df in strategies.items():
    analysis = {
        'avg_tokens': selected_df['text_token_length'].mean(),
        'std_tokens': selected_df['text_token_length'].std(),
        'format_diversity': len(selected_df['format'].unique()),
        'audience_diversity': len(selected_df['audience'].unique()),
        'cluster_coverage': len(selected_df['cluster'].unique()),
        'format_distribution': selected_df['format'].value_counts().to_dict(),
        'token_efficiency': selected_df['text_token_length'].sum()
    }
    strategy_analysis[strategy_name] = analysis
    
    print(f"\n{strategy_name.upper()} Strategy:")
    print(f"  Avg tokens: {analysis['avg_tokens']:.1f} ± {analysis['std_tokens']:.1f}")
    print(f"  Format diversity: {analysis['format_diversity']} types")
    print(f"  Audience diversity: {analysis['audience_diversity']} types")
    print(f"  Cluster coverage: {analysis['cluster_coverage']}/{optimal_k} clusters")
    print(f"  Total tokens: {analysis['token_efficiency']:,}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*50)
print("5. SAVING ANALYSIS RESULTS")
print("="*50)

# Save clustering results
with open('eda_output/clustering_analysis.txt', 'w') as f:
    f.write("COSMOPEDIA-100K CLUSTERING ANALYSIS\n")
    f.write("="*50 + "\n\n")
    
    f.write(f"Sample size: {clustering_sample_size}\n")
    f.write(f"Optimal clusters: {optimal_k}\n")
    f.write(f"Silhouette score: {max(silhouette_scores):.3f}\n\n")
    
    for cluster_id, analysis in cluster_analysis.items():
        f.write(f"Cluster {cluster_id}:\n")
        f.write(f"  Size: {analysis['size']} documents\n")
        f.write(f"  Top terms: {', '.join(analysis['top_terms'][:10])}\n")
        f.write(f"  Main format: {analysis['main_format']}\n")
        f.write(f"  Main audience: {analysis['main_audience']}\n")
        f.write(f"  Avg tokens: {analysis['avg_tokens']:.1f}\n")
        f.write(f"  Format diversity: {analysis['format_diversity']}\n")
        f.write(f"  Audience diversity: {analysis['audience_diversity']}\n\n")

# Save MoE optimization results
with open('eda_output/moe_optimization.txt', 'w') as f:
    f.write("MOE TRAINING OPTIMIZATION RECOMMENDATIONS\n")
    f.write("="*50 + "\n\n")
    
    f.write("Selection Strategy Comparison (1000 samples):\n\n")
    
    for strategy_name, analysis in strategy_analysis.items():
        f.write(f"{strategy_name.upper()} Strategy:\n")
        f.write(f"  Average tokens: {analysis['avg_tokens']:.1f} ± {analysis['std_tokens']:.1f}\n")
        f.write(f"  Format diversity: {analysis['format_diversity']} types\n")
        f.write(f"  Audience diversity: {analysis['audience_diversity']} types\n")
        f.write(f"  Cluster coverage: {analysis['cluster_coverage']}/{optimal_k} clusters\n")
        f.write(f"  Total tokens: {analysis['token_efficiency']:,}\n")
        f.write(f"  Top formats: {dict(list(selected_df['format'].value_counts().head().items()))}\n\n")

print("Clustering analysis saved to eda_output/clustering_analysis.txt")
print("MoE optimization saved to eda_output/moe_optimization.txt")

print("\n" + "="*50)
print("ADVANCED ANALYSIS COMPLETE")
print("="*50)

