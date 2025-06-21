import numpy as np
from prototype_network import compute_prototypes


def test_compute_prototypes_shape():
    embeddings = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    labels = np.array([0, 0, 1, 1])
    protos = compute_prototypes(embeddings, labels)
    assert protos.shape == (2, 2)
