"""
## Idea process

    1. PCA and visualise

    2. Take the resulting pca and run it through tree method

    3. Using the fact that we know the redshift values of the classes, classify and visualise marking classes and outliers (this will be a yes no question)

    4. Take the ensemble of tree method and apply dbscan visualising the full picture
### Data Processing
"""
import os, warnings 
warnings.filterwarnings("ignore") 
# ── Third-party ─────────────────────────────────────────────────────────────── 
import numpy as np 
import pandas as pd 
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import ( 
    classification_report, confusion_matrix, accuracy_score, f1_score, 
    adjusted_rand_score, normalized_mutual_info_score ) 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from umap import UMAP 
from hdbscan import HDBSCAN
import gc

file_path = os.path.join(os.getcwd(),'SkyObjects_FITS.csv')
print(file_path)
data = pd.read_csv(file_path)
data.columns
useful_data = data[['u', 'g', 'r', 'i', 'z','petroRad_r', 'psfMag_r', 'modelMag_r', 'extinction_r','redshift','class']]
useful_data['one_hot'],unique = pd.factorize(useful_data['class'])
useful_data.columns
scaler = StandardScaler()
X = useful_data[['u', 'g', 'r', 'i', 'z', 'petroRad_r', 'psfMag_r', 'modelMag_r',
       'extinction_r']].values
X_scaled = scaler.fit_transform(X)
y = useful_data['class'].values
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_scaled,y)
importances = rf.feature_importances_
feature_indexes = np.argsort(importances)[::-1][:3]
del rf
gc.collect()
X_selected = X_scaled[:, feature_indexes]

best_ari_A, best_nmi_A = None, None
best_score_A = -float('inf')
hdbscan_A = None

for min_cluster_size in [10, 20, 50, 75, 100, 200, 300, 400, 800, 1200]:

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
    ).fit(X_selected)

    labels = clusterer.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    ari_A = adjusted_rand_score(y, labels)
    nmi_A = normalized_mutual_info_score(y, labels)

    score = (ari_A + nmi_A) / 2

    if score > best_score_A:
        best_score_A = score
        best_ari_A = ari_A
        best_nmi_A = nmi_A
        hdbscan_A = clusterer

    print(
        f"SIZE={min_cluster_size}, "
        f"clusters={n_clusters}, "
        f"ARI={ari_A:.3f}, "
        f"NMI={nmi_A:.3f}, "
        f"SCORE={score:.3f}"
    )

labels_A = hdbscan_A.labels_
umapp = UMAP(n_components=3, random_state=42)
X_reduced_UMAP = umapp.fit_transform(X_scaled)

best_ari_B, best_nmi_B = None, None
best_score_B = -float('inf')
hdbscan_B= None

for min_cluster_size in [10, 20, 50, 75, 100, 200, 300, 400, 800, 1200]:

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
    ).fit(X_reduced_UMAP)

    labels = clusterer.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    ari_B = adjusted_rand_score(y, labels)
    nmi_B = normalized_mutual_info_score(y, labels)

    score = (ari_B + nmi_B) / 2

    if score > best_score_B:
        best_score_B = score
        best_ari_B = ari_B
        best_nmi_B = nmi_B
        hdbscan_B = clusterer

    print(
        f"SIZE={min_cluster_size}, "
        f"clusters={n_clusters}, "
        f"ARI={ari_B:.3f}, "
        f"NMI={nmi_B:.3f}, "
        f"SCORE={score:.3f}"
    )
labels_B = hdbscan_B.labels_
print(f"Pipeline A: ARI = {best_ari_A:.3f}, NMI = {best_nmi_A:.3f}")
print(f"Pipeline B: ARI = {best_ari_B:.3f}, NMI = {best_nmi_B:.3f}")
gc.collect()
# Visualization

"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Top 2 features, colored by true class
class_colors = {'STAR': 0, 'GALAXY': 1, 'QSO': 2}
y_numeric = np.array([class_colors[c] for c in y])
scatter_A = axes[0].scatter(X_selected[:, 0], X_selected[:, 1], c=y_numeric, cmap='viridis', alpha=0.6, s=20)
axes[0].set_title(f'Pipeline A: Top 2 Features (ARI={adjusted_rand_score(y, labels_A):.3f})')
axes[0].set_xlabel(f'Feature {feature_indexes[0]}')
axes[0].set_ylabel(f'Feature {feature_indexes[1]}')
plt.colorbar(scatter_A, ax=axes[0], label='Class')

# Plot B: UMAP, colored by true class
scatter_B = axes[1].scatter(X_reduced_UMAP[:, 0], X_reduced_UMAP[:, 1], c=y_numeric, cmap='viridis', alpha=0.6, s=20)
axes[1].set_title(f'Pipeline B: UMAP 2D (ARI={adjusted_rand_score(y, labels_B):.3f})')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')
plt.colorbar(scatter_B, ax=axes[1], label='Class')

plt.tight_layout()
plt.savefig('class_separation.png', dpi=300)
plt.show()

# Optional: Show HDBSCAN cluster assignments
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_selected[:, 0], X_selected[:, 1], c=labels_A, cmap='tab10', alpha=0.6, s=20)
axes[0].set_title('Pipeline A: HDBSCAN Clusters')
axes[1].scatter(X_reduced_UMAP[:, 0], X_reduced_UMAP[:, 1], c=labels_B, cmap='tab10', alpha=0.6, s=20)
axes[1].set_title('Pipeline B: HDBSCAN Clusters')
plt.tight_layout()
plt.show()
"""

# Plot A: Top 2 features, colored by true class
class_colors = {'STAR': 0, 'GALAXY': 1, 'QSO': 2}
y_numeric = np.array([class_colors[c] for c in y])

# 3D Visualization: Ground truth classes
fig = plt.figure(figsize=(16, 6))

# Pipeline A: Top 3 features colored by class
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
scatter1 = ax1.scatter(X_selected[:, 0], X_selected[:, 1], X_selected[:, 2], 
                       c=y_numeric, cmap='viridis', alpha=0.6, s=10)
ax1.set_title(f'Pipeline A: Top 3 Features by Class\n(ARI={ari_A:.3f})', fontsize=12)
ax1.set_xlabel(f'Feature {feature_indexes[0]}')
ax1.set_ylabel(f'Feature {feature_indexes[1]}')
ax1.set_zlabel(f'Feature {feature_indexes[2]}')
cbar1 = plt.colorbar(scatter1, ax=ax1, label='Class', pad=0.1, shrink=0.8)
cbar1.set_ticks([0, 1, 2])
cbar1.set_ticklabels(['star', 'galaxy', 'qso'])

# Pipeline B: UMAP colored by class
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
scatter2 = ax2.scatter(X_reduced_UMAP[:, 0], X_reduced_UMAP[:, 1], X_reduced_UMAP[:, 2], 
                       c=y_numeric, cmap='viridis', alpha=0.6, s=10)
ax2.set_title(f'Pipeline B: UMAP 3D by Class\n(ARI={ari_B:.3f})', fontsize=12)
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.set_zlabel('UMAP 3')
cbar2 = plt.colorbar(scatter2, ax=ax2, label='Class', pad=0.1, shrink=0.8)
cbar2.set_ticks([0, 1, 2])
cbar2.set_ticklabels(['star', 'galaxy', 'qso'])

plt.tight_layout()
plt.savefig('3d_class_separation.png', dpi=600)
plt.show()

# 3D Visualization: HDBSCAN clusters
fig = plt.figure(figsize=(16, 6))

# Pipeline A: HDBSCAN clusters
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
scatter1 = ax1.scatter(X_selected[:, 0], X_selected[:, 1], X_selected[:, 2], 
                       c=labels_A, cmap='tab20', alpha=0.6, s=10)
ax1.set_title(f'Pipeline A: HDBSCAN Clusters (Top 3 Features)', fontsize=12)
ax1.set_xlabel(f'Feature {feature_indexes[0]}')
ax1.set_ylabel(f'Feature {feature_indexes[1]}')
ax1.set_zlabel(f'Feature {feature_indexes[2]}')
plt.colorbar(scatter1, ax=ax1, label='Cluster', pad=0.1, shrink=0.8)

# Pipeline B: HDBSCAN clusters
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
scatter2 = ax2.scatter(X_reduced_UMAP[:, 0], X_reduced_UMAP[:, 1], X_reduced_UMAP[:, 2], 
                       c=labels_B, cmap='tab20', alpha=0.6, s=10)
ax2.set_title(f'Pipeline B: HDBSCAN Clusters (UMAP 3D)', fontsize=12)
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.set_zlabel('UMAP 3')
plt.colorbar(scatter2, ax=ax2, label='Cluster', pad=0.1, shrink=0.8)

plt.tight_layout()
plt.savefig('3d_hdbscan_clusters.png', dpi=600)
plt.show()
gc.collect()