import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity


def kmeans_cluster(embeddings: np.ndarray, n_clusters: int = 5, random_state: int = 42):
    """Run KMeans clustering on embeddings."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km


def find_representative_samples(sentences, embeddings, labels, n_reps: int = 3):
    """Return representative samples for each cluster."""
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    results = []
    for label, indices in clusters.items():
        centroid = embeddings[indices].mean(axis=0)
        sims = cosine_similarity([centroid], embeddings[indices])[0]
        sorted_idx = np.argsort(sims)[::-1]
        reps = [sentences[indices[i]] for i in sorted_idx[:n_reps]]
        results.append({"cluster_id": label, "size": len(indices), "representative_samples": reps})
    results.sort(key=lambda x: x["cluster_id"])
    return results


def build_conditional_markov(selected_ids, sequences, n_clusters):
    """Build a Markov transition matrix for the given subset of criminals."""
    matrix = np.zeros((n_clusters, n_clusters))
    for cid in selected_ids:
        seq = sequences.get(cid, [])
        for s1, s2 in zip(seq[:-1], seq[1:]):
            matrix[s1, s2] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)
        for i in range(n_clusters):
            if row_sums[i] == 0:
                matrix[i] = 1.0 / n_clusters
    return matrix


def compute_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stat = eigvecs[:, idx].real
    stat /= stat.sum()
    return stat


def plot_state_transition_diagram(transition_matrix: np.ndarray, out_path: str = "state_transition.png"):
    """Plot a state transition diagram using networkx."""
    G = nx.DiGraph()
    num_states = transition_matrix.shape[0]
    for i in range(num_states):
        G.add_node(i)
    for i in range(num_states):
        for j in range(num_states):
            w = transition_matrix[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue")
    edge_labels = nx.get_edge_attributes(G, "weight")
    edge_labels = {(i, j): f"{w:.2f}" for (i, j), w in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title("State Transition Diagram")
    plt.savefig(out_path)
    plt.close()


def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Return common clustering validation metrics."""
    metrics = {
        "silhouette": float(silhouette_score(embeddings, labels)),
        "davies_bouldin": float(davies_bouldin_score(embeddings, labels)),
    }
    return metrics


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str = "tsne.png",
    random_state: int = 42,
):
    """Visualize embeddings using t-SNE."""
    tsne = TSNE(n_components=2, random_state=random_state)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", legend=False)
    plt.title("t-SNE of Event Embeddings")
    plt.savefig(out_path)
    plt.close()


def transition_entropy(matrix: np.ndarray) -> float:
    """Compute the Shannon entropy of a Markov transition matrix."""
    with np.errstate(divide="ignore"):
        logp = np.where(matrix > 0, np.log2(matrix), 0)
    return float(-(matrix * logp).sum())
