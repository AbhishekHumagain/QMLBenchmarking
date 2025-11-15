# encodings/basis.py
import pennylane as qml
import numpy as np

def encoding(features, wires):
    binary = (features > np.pi / 2).astype(int)[:len(wires)]
    qml.BasisEmbedding(binary, wires=wires)