# encodings/iqp.py
import pennylane as qml
import numpy as np

def encoding(features, wires):
    if len(features) < len(wires):
        features = np.pad(features, (0, len(wires) - len(features)))
    else:
        features = features[:len(wires)]
    qml.IQPEmbedding(features, wires=wires, n_repeats=3)