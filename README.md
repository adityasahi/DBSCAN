# 📊 Clustering Analysis using Dimensionality Reduction and Unsupervised Learning
# 🔍 Overview
This pipeline applies unsupervised clustering algorithms to the Breast Cancer Wisconsin dataset from sklearn.datasets, using dimensionality reduction (PCA, t-SNE, UMAP) and extensive metric evaluation to find the best-performing models.

Clustering algorithms included:
- K-Means

- Agglomerative Clustering

- DBSCAN

- HDBSCAN

Dimensionality reduction techniques used:

+ Principal Component Analysis (PCA)

+ t-Distributed Stochastic Neighbor Embedding (t-SNE)

+ Uniform Manifold Approximation and Projection (UMAP)
# 🧪 Dataset
Source: sklearn.datasets.load_breast_cancer()

Type: Diagnostic features of breast cancer biopsies (malignant vs benign)

Shape: (569 samples, 30 features)

Output: Binary labels (0 = malignant, 1 = benign)

# 🧰 Main Features
1. Data Preprocessing
Correlation-based feature selection (removing features with > 0.9 correlation)

Standard scaling of selected features

2. Dimensionality Reduction
Visualizes the dataset using PCA, t-SNE, and UMAP

2D projections help understand cluster behavior in reduced space

3. Clustering Models
Applies four algorithms:

KMeans (with varying n_clusters)

Agglomerative Clustering (with n_clusters and linkage)

DBSCAN (with eps and min_samples)

HDBSCAN (with min_cluster_size and min_samples)

4. Model Evaluation Metrics
Each configuration is scored using:

Silhouette Score

Calinski-Harabasz Index

Davies-Bouldin Score

Adjusted Rand Index (compared with ground-truth)

Number of clusters identified

5. Best Model Selection
Selects top-performing parameters based on Adjusted Rand Index

Visualizes clusters and class distributions for each best model

6. Reporting
Exports a full performance summary to clustering_results.xlsx

Includes individual run results and per-algorithm metric summaries

# 📈 Output Visualizations
2D scatter plots of clusters in PCA, t-SNE, and UMAP spaces

Pie charts for true class vs predicted cluster distribution

Bar plots comparing performance across metrics and algorithms