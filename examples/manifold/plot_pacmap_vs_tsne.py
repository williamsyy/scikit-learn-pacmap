"""
=======================================================
PaCMAP and t-SNE on clusters of clusters
=======================================================

This example compares :class:`~sklearn.manifold.PaCMAP` and
:class:`~sklearn.manifold.TSNE` on data with two levels of structure: five
groups ("macro clusters") placed at different distances from each other,
each made of four tighter sub-clusters.

Both methods separate the sub-clusters, which is local structure. They differ
in how faithfully they keep the *arrangement* of the groups, which is global
structure. PaCMAP adds mid-near pairs to its objective for that purpose, and
it does not need a perplexity calibration or a Barnes-Hut tree.

We quantify the trade-off with three scores:

- **trustworthiness** (:func:`~sklearn.manifold.trustworthiness`): are the
  embedding's nearest neighbors also neighbors in the input space?
- **random-triplet accuracy**: for random triplets :math:`(i, j, k)`, how
  often does the embedding agree with the input space on whether :math:`j`
  or :math:`k` is closer to :math:`i`?
- **sub-cluster rank correlation**: Spearman correlation between the pairwise
  distances of the 20 sub-cluster centroids in the input space and in the
  embedding.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Generate clusters of clusters
# -----------------------------
# The five macro clusters are spread at increasing scales, so that some are
# much closer to each other than others.
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, PaCMAP, trustworthiness

rng = np.random.RandomState(0)
n_macro, n_sub, n_per_sub, n_features = 5, 4, 100, 50

macro_centers = rng.randn(n_macro, n_features) * np.linspace(6, 30, n_macro)[:, None]
X, macro_label, sub_label = [], [], []
for m in range(n_macro):
    for s in range(n_sub):
        center = macro_centers[m] + 2.5 * rng.randn(n_features)
        X.append(center + 0.6 * rng.randn(n_per_sub, n_features))
        macro_label += [m] * n_per_sub
        sub_label += [m * n_sub + s] * n_per_sub
X = np.vstack(X)
macro_label = np.array(macro_label)
sub_label = np.array(sub_label)
print(f"X has shape {X.shape}")


# %%
# Scores
# ------
def random_triplet_accuracy(X, X_embedded, n_triplets=20_000, seed=0):
    triplets = np.random.RandomState(seed).randint(0, len(X), size=(n_triplets, 3))
    i, j, k = triplets.T

    def closer(Z):
        return np.linalg.norm(Z[i] - Z[j], axis=1) < np.linalg.norm(Z[i] - Z[k], axis=1)

    return np.mean(closer(X) == closer(X_embedded))


def centroid_rank_correlation(X, X_embedded, labels):
    groups = np.unique(labels)
    high = np.array([X[labels == g].mean(axis=0) for g in groups])
    low = np.array([X_embedded[labels == g].mean(axis=0) for g in groups])
    return spearmanr(pdist(high), pdist(low)).statistic


# %%
# Fit the embeddings
# ------------------
# PCA is included as a linear reference: it keeps the global arrangement but
# cannot separate the sub-clusters.
methods = {
    "PCA": PCA(n_components=2),
    "t-SNE": TSNE(random_state=0),
    "PaCMAP": PaCMAP(random_state=0),
}
embeddings, scores = {}, {}
for name, method in methods.items():
    t0 = time()
    embeddings[name] = method.fit_transform(X)
    duration = time() - t0
    scores[name] = (
        duration,
        trustworthiness(X, embeddings[name], n_neighbors=10),
        random_triplet_accuracy(X, embeddings[name]),
        centroid_rank_correlation(X, embeddings[name], sub_label),
    )

print(f"{'method':8s}{'time [s]':>10s}{'trust.':>9s}{'triplet':>9s}{'rank':>7s}")
for name, (duration, trust, triplet, rank) in scores.items():
    print(f"{name:8s}{duration:10.2f}{trust:9.3f}{triplet:9.3f}{rank:7.3f}")

# %%
# Plot
# ----
# Points are colored by macro cluster.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, Y) in zip(axes, embeddings.items()):
    ax.scatter(Y[:, 0], Y[:, 1], c=macro_label, cmap="tab10", s=4, vmax=9)
    duration, trust, triplet, rank = scores[name]
    ax.set_title(f"{name}  ({duration:.2f} s)")
    ax.set_xlabel(f"trust. {trust:.3f} | triplet {triplet:.3f} | rank {rank:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

# %%
# Interpretation
# --------------
# t-SNE has the highest trustworthiness: locally it is the most faithful.
# PaCMAP is close on that score, keeps more of the arrangement of the
# sub-clusters (higher triplet accuracy and rank correlation) and is an order
# of magnitude faster on this dataset. PCA keeps the global arrangement best
# but merges the sub-clusters.
#
# The differences depend on the data. On data without a hierarchy of
# clusters, such as the digits dataset or the S-curve, t-SNE with its default
# PCA initialization preserves global structure about as well as PaCMAP.
