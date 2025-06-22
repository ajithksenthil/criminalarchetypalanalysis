import numpy as np
import torch
from torch import nn


def compute_prototypes(embeddings, labels):
    """Compute mean prototype vectors for each label."""
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    else:
        embeddings = embeddings.float()
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = labels.long()
    unique_labels = torch.unique(labels)
    prototypes = []
    for lbl in unique_labels:
        mask = labels == lbl
        prototypes.append(embeddings[mask].mean(dim=0))
    return torch.stack(prototypes, dim=0)


class PrototypicalNetwork(nn.Module):
    """Simple feed-forward encoder used in a prototypical network."""

    def __init__(self, input_dim, hidden_dim=128, rep_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )

    def forward(self, x, prototypes):
        x = self.encoder(x)
        if not isinstance(prototypes, torch.Tensor):
            prototypes = torch.tensor(prototypes, dtype=x.dtype, device=x.device)
        dists = torch.cdist(x, prototypes)
        return -dists


def train_prototypical_network(embeddings, labels, epochs=20, lr=1e-3):
    """Train a prototypical network with simple episodic sampling."""
    embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    model = PrototypicalNetwork(embeddings.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(labels)
    val_acc = 0.0

    for _ in range(epochs):
        perm = torch.randperm(n)
        split = max(1, int(0.8 * n))
        sup_idx = perm[:split]
        qry_idx = perm[split:]
        sup_emb, sup_labels = embeddings[sup_idx], labels[sup_idx]
        qry_emb, qry_labels = embeddings[qry_idx], labels[qry_idx]

        model.train()
        sup_encoded = model.encoder(sup_emb)
        prototypes = compute_prototypes(sup_encoded, sup_labels)
        qry_encoded = model.encoder(qry_emb)
        dists = torch.cdist(qry_encoded, prototypes)
        log_p_y = torch.log_softmax(-dists, dim=1)
        loss = nn.functional.nll_loss(log_p_y, qry_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = log_p_y.argmax(dim=1)
            val_acc = (preds == qry_labels).float().mean().item()

    with torch.no_grad():
        all_encoded = model.encoder(embeddings)
        final_prototypes = compute_prototypes(all_encoded, labels).cpu().numpy()
    return model, final_prototypes, val_acc
