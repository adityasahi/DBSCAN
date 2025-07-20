import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import (AgglomerativeClustering, KMeans, DBSCAN)
import hdbscan
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, rand_score, 
                           davies_bouldin_score, adjusted_rand_score)
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings("ignore")

# Loading data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Feature selection: removing highly correlated features
corr = X.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i + 1, corr.shape[0]):
        if corr.iloc[i, j] >= 0.9 and columns[j]:
            columns[j] = False
selected_columns = X.columns[columns]
X = X[selected_columns] # selects around 20 features

# Feature scaling
scaler = StandardScaler()
X_Scale = scaler.fit_transform(X)

# Dimensionality reduction for visualization
reduction = {
    'PCA': PCA(n_components=2, random_state=42),
    't-SNE': TSNE(n_components=2, random_state=42),
    'UMAP': umap.UMAP(random_state=42)
}

X_reduced = {}
for name, reducer in reduction.items():
    X_reduced[name] = reducer.fit_transform(X_Scale)

# Visualization plots
def plot_clusters(X_reduced, y, title, ax):
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
                        cmap='viridis', alpha=0.6, edgecolor='k', s=50)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Component 1', fontsize=10)
    ax.set_ylabel('Component 2', fontsize=10)
    ax.grid(alpha=0.3)
    legend = ax.legend(*scatter.legend_elements(), 
                      title="Clusters", fontsize=8)
    legend.get_title().set_fontsize(9)

def plot_pie_chart(y, title, ax):
    counts = pd.Series(y).value_counts()
    wedges, texts, autotexts = ax.pie(counts, labels=[f'Cluster {i}' for i in counts.index], 
                                    autopct='%1.1f%%', colors=sns.color_palette('viridis'),
                                    startangle=90, textprops={'fontsize': 8})
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis('equal')

def plot_metric_comparison(all_results_df, metric, ax):
    algorithms = all_results_df.index.unique()
    x = np.arange(len(algorithms))
    width = 0.35
    
    for i, param in enumerate(all_results_df['param'].unique()):
        values = all_results_df[all_results_df['param'] == param][metric]
        ax.bar(x + i*width, values, width, label=f'Param: {param}')
    
    ax.set_title(f'{metric} Score Comparison', fontsize=12, pad=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Data Visualization Before Clustering', fontsize=14, y=1.02)

for i, (name, X_red) in enumerate(X_reduced.items()):
    plot_clusters(X_red, y, f"{name} Projection", axes[i//3, i%3])

plot_pie_chart(y, "True Class Distribution", axes[1, 2])
plt.tight_layout()
plt.show()

# Clustering algorithms using parameter ranges
param_grids = {
    'KMeans': {'n_clusters': [2, 3, 4, 5]},
    'Agglomerative': {'n_clusters': [2, 3, 4, 5], 'linkage': ['ward', 'complete', 'average']},
    'DBSCAN': {'eps': [0.5, 1.0, 1.5, 2.0], 'min_samples': [3, 5, 7]},
    'HDBSCAN': {'min_cluster_size': [3, 5, 7], 'min_samples': [1, 3, 5]}
}

# Evaluating all algorithms using parameter ranges
all_results = []

for name in param_grids.keys():
    if name == 'KMeans':
        algorithm = KMeans
    elif name == 'Agglomerative':
        algorithm = AgglomerativeClustering
    elif name == 'DBSCAN':
        algorithm = DBSCAN
    elif name == 'HDBSCAN':
        algorithm = hdbscan.HDBSCAN
    
    for params in ParameterGrid(param_grids[name]):
        if name == 'KMeans':
            params['random_state'] = 42
            params['n_init'] = 'auto'
        
        model = algorithm(**params)
        y_pred = model.fit_predict(X_Scale)
        
        # Handle noise points (-1 label)
        if -1 in y_pred:
            y_pred[y_pred == -1] = max(y_pred) + 1
        
        # Skip if only one cluster found
        if len(np.unique(y_pred)) < 2:
            continue
            
        metrics = {
            'Algorithm': name,
            'param': str(params),
            'Silhouette': silhouette_score(X_Scale, y_pred),
            'Calinski-Harabasz': calinski_harabasz_score(X_Scale, y_pred),
            'Davies-Bouldin': davies_bouldin_score(X_Scale, y_pred),
            'Rand': rand_score(y, y_pred),
            'Adjusted Rand': adjusted_rand_score(y, y_pred),
            'Clusters': len(np.unique(y_pred))
        }
        all_results.append(metrics)

# Creating a DataFrame for the results

all_results_df = pd.DataFrame(all_results)

# Plot for the  metric comparisons
plot_metrics = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Adjusted Rand']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Clustering Algorithm Performance Comparison', fontsize=14, y=1.02)

for i, metric in enumerate(plot_metrics):
    ax = axes[i//2, i%2]
    plot_metric_comparison(all_results_df, metric, ax)

plt.tight_layout()
plt.show()

# Using the best parameters for each algorithm
best_params = all_results_df.loc[all_results_df.groupby('Algorithm')['Adjusted Rand'].idxmax()]
print("Best parameters for each algorithm:")
print(best_params[['Algorithm', 'param', 'Adjusted Rand', 'Clusters']].to_string(index=False))

# Visualizing the results
best_models = {}
for _, row in best_params.iterrows():
    if row['Algorithm'] == 'KMeans':
        model = KMeans(n_clusters=eval(row['param'])['n_clusters'], random_state=42, n_init='auto')
    elif row['Algorithm'] == 'Agglomerative':
        params = eval(row['param'])
        model = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params['linkage'])
    elif row['Algorithm'] == 'DBSCAN':
        params = eval(row['param'])
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    elif row['Algorithm'] == 'HDBSCAN':
        params = eval(row['param'])
        model = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'], 
                               min_samples=params['min_samples'])
    
    y_pred = model.fit_predict(X_Scale)
    if -1 in y_pred:
        y_pred[y_pred == -1] = max(y_pred) + 1
    best_models[row['Algorithm']] = y_pred

# Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Best Clustering Results (UMAP Projection)', fontsize=14, y=1.02)

for i, (name, y_pred) in enumerate(best_models.items()):
    ax = axes[i//2, i%2]
    plot_clusters(X_reduced['UMAP'], y_pred, f"{name} Clustering", ax)
    ari = adjusted_rand_score(y, y_pred)
    ax.text(0.05, 0.95, f"ARI: {ari:.3f}\nClusters: {len(np.unique(y_pred))}", 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# pie charts 
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Cluster Distributions for Best Parameters', fontsize=14, y=1.02)

for i, (name, y_pred) in enumerate(best_models.items()):
    ax = axes[i//2, i%2]
    plot_pie_chart(y_pred, f"{name} Cluster Distribution", ax)

plt.tight_layout()
plt.show()

# generating a summary report in Excel
with pd.ExcelWriter('clustering_results.xlsx') as writer:
    all_results_df.to_excel(writer, sheet_name='All Results')
    best_params.to_excel(writer, sheet_name='Best Parameters')
    
    # Adding metric summaries
    for metric in plot_metrics:
        summary = all_results_df.groupby('Algorithm')[metric].agg(['mean', 'max', 'min', 'std'])
        summary.to_excel(writer, sheet_name=f'{metric} Summary') 