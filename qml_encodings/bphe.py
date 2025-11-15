# encodings/bphe.py – Bit-Partition Hybrid Encoding
import pennylane as qml
import numpy as np

def bphe_encoding(features, wires):
    n_wires = len(wires)
    split = n_wires // 2
    n_amp = split
    n_basis = n_wires - split

    # Amplitude partition (use as much as available, pad the rest)
    amp_raw = features[:2**n_amp] if len(features) >= 2**n_amp else features
    norm = np.linalg.norm(amp_raw) + 1e-12
    amp_normalized = amp_raw / norm
    amp_padded = np.pad(amp_normalized, (0, 2**n_amp - len(amp_normalized)))
    qml.AmplitudeEmbedding(amp_padded, wires=wires[:n_amp], normalize=False, pad_with=0.0)

    # Basis partition (threshold at π/2, repeat/tile if needed)
    basis_raw = features[-n_basis:] if len(features) >= n_basis else features
    binary = (basis_raw > np.pi / 2).astype(int)
    repeated = np.tile(binary, (n_basis // len(binary)) + 1)[:n_basis]
    qml.BasisEmbedding(repeated, wires=wires[n_amp:])