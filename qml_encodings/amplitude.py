import pennylane as qml

def amplitude_encoding(features, wires):
    qml.AmplitudeEmbedding(features, wires=wires, normalize=True, pad_with=0.)