# tsne_plot.py
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(embeddings, labels, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter, label="Class ID")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()
