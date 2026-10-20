import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

def perform_clustering_and_tsne(df, n_clusters=3, random_state=42, cache_path="data/students_clustered_cache.csv"):
    """
    Performs KMeans clustering and t-SNE 2D dimensionality reduction on student usage metrics.
    Caches calculated results to ensure instantaneous application launches.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path)
            if len(cached_df) == len(df) and 'TSNE1' in cached_df.columns:
                return cached_df
        except Exception:
            pass

    df = df.copy()
    
    feature_cols = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Addicted_Score', 'Mental_Health_Score']
    X = df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster_Label'] = df['Cluster'].apply(lambda c: f'Cluster {c+1}')
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    tsne_results = tsne.fit_transform(X_scaled)
    
    df['TSNE1'] = tsne_results[:, 0]
    df['TSNE2'] = tsne_results[:, 1]
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        
    return df
