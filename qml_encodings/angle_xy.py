import pennylane as qml
import numpy as np

def encoding(features, wires):
    n = len(wires)
    needed = 2 * n

    # pad or truncate features
    if len(features) < needed:
        features = np.pad(features, (0, needed - len(features)))
    else:
        features = features[:needed]

    # split into X and Y components
    x_feats = features[:n]
    y_feats = features[n:]

    # apply X then Y rotations
    qml.AngleEmbedding(x_feats, wires=wires, rotation="X")
    qml.AngleEmbedding(y_feats, wires=wires, rotation="Y")
