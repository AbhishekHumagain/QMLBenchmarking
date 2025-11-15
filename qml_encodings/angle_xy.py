# encodings/angle_xy.py
import pennylane as qml
import numpy as np

def encoding(features, wires):
    n = len(wires)
    needed = 2 * n
    if len(features) < needed:
        features = np.pad(features, (0, needed - len(features)))
    else:
        features = features[:needed]
    qml.AngleEmbedding(features, wires=wires, rotation="XY")