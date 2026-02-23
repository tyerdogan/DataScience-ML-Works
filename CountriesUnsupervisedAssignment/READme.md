# Country Clustering (Unsupervised Learning)

This notebook groups countries into **similar clusters** using **unsupervised learning**.

## What the notebook does
- Loads and quickly explores the dataset (basic checks + simple plots)
- Prepares features (scaling / preprocessing)
- Creates country clusters in two setups:
  - **With PCA** (dimensionality reduction)
  - **Without PCA** (use original features directly)

## Clustering methods tested
- K-Means (baseline)
- Agglomerative (Hierarchical) Clustering
- Gaussian Mixture Model (GMM)
- DBSCAN
- HDBSCAN (only if the library is available)

## How results are compared
- Reports basic clustering quality scores (e.g., silhouette and other standard clustering metrics)
- Visualizes final clusters on a **world map** to check if patterns look reasonable

## Outputs
- Simple EDA figures (distributions / correlations)
- Cluster comparison results (with vs. without PCA)
- World map showing country clusters
