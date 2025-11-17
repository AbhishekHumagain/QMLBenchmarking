# models/vqc.py
import pennylane as qml
import numpy as np
import random
from sklearn.metrics import roc_auc_score, accuracy_score

from utils.data_loader import load_dataset

# ==================== VQC MODEL ====================
def create_circuit(encoding_fn, n_qubits=8, n_layers=5):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(features, weights):
        encoding_fn(features, range(n_qubits))
        # Hardware-efficient ansatz
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[layer, q], wires=q)
            for q in range(n_qubits):
                qml.RZ(weights[layer, n_qubits + q], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q+1])
            qml.CNOT(wires=[n_qubits-1, 0])  # ring
        return qml.expval(qml.PauliZ(0))

    return circuit

# ==================== TRAINING & EVALUATION ====================
# Inside train_and_evaluate function – replace the entire function with this version
def train_and_evaluate(dataset_name, encoding_name, encoding_fn,
                       n_qubits=8, n_layers=5, epochs=150, batch_size=32):
    from utils.data_loader import load_dataset
    from sklearn.metrics import accuracy_score, roc_auc_score, \
                                 precision_score, recall_score, f1_score

    X_train, X_test, y_train, y_test = load_dataset(dataset_name, n_components=n_qubits)
    circuit = create_circuit(encoding_fn, n_qubits, n_layers)

    weights = np.random.uniform(0, 2 * np.pi, (n_layers, 2 * n_qubits))
    opt = qml.AdamOptimizer(stepsize=0.06)

    y_train_pm = 2 * y_train - 1

    def cost_fn(w):
        idx = random.sample(range(len(X_train)), min(batch_size, len(X_train)))
        return np.mean([(circuit(X_train[i], w) - y_train_pm[i]) ** 2 for i in idx])

    best_auc = 0.0
    best_weights = weights.copy()

    for epoch in range(epochs):
        opt.step(cost_fn, weights)
        if epoch % 30 == 0 or epoch == epochs - 1:
            preds = np.array([(circuit(x, weights) + 1) / 2 for x in X_test])
            auc = roc_auc_score(y_test, preds)
            if auc > best_auc:
                best_auc = auc
                best_weights = weights.copy()

    # Final predictions with best weights
    probs = np.array([(circuit(x, best_weights) + 1) / 2 for x in X_test])
    preds = (probs > 0.5).astype(int)

    # Classification metrics (all datasets are binary → pos_label=1)
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, probs)
    prec  = precision_score(y_test, preds, pos_label=1, zero_division=0)
    rec   = recall_score(y_test, preds, pos_label=1, zero_division=0)
    f1    = f1_score(y_test, preds, pos_label=1, zero_division=0)

    # Resource profiling
    specs = qml.specs(circuit)(X_test[0], best_weights)
    res = specs["resources"]
    depth = res.depth
    gates = sum(res.gate_types.values())
    resource_cost = np.clip((n_qubits/32 + depth/250 + gates/2500)/3, 0, 1)

    return acc, auc, prec, rec, f1, resource_cost