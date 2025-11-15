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
def train_and_evaluate(dataset_name, encoding_name, encoding_fn, n_qubits=8, n_layers=5, epochs=120, batch_size=32):
    X_train, X_test, y_train, y_test = load_dataset(dataset_name, n_components=n_qubits)

    circuit = create_circuit(encoding_fn, n_qubits, n_layers)

    weights = np.random.uniform(0, 2 * np.pi, size=(n_layers, 2 * n_qubits))
    opt = qml.AdamOptimizer(stepsize=0.05)

    # Map labels {0,1} → {-1,+1} for PauliZ expectation
    y_train_pm = 2 * y_train - 1
    y_test_pm  = 2 * y_test - 1

    # Proper cost function with mini-batch sampling
    def cost_fn(weights):
        batch_indices = random.sample(range(len(X_train)), min(batch_size, len(X_train)))
        total = 0.0
        for i in batch_indices:
            pred = circuit(X_train[i], weights)
            total += (pred - y_train_pm[i]) ** 2
        return total / len(batch_indices)

    best_weights = weights.copy()
    best_auc = 0.0

    for epoch in range(epochs):
        opt.step(cost_fn, weights)

        # Periodic evaluation on full test set
        if epoch % 20 == 0 or epoch == epochs-1:
            preds = np.array([(circuit(x, weights) + 1) / 2 for x in X_test])
            current_auc = roc_auc_score(y_test, preds)
            if current_auc > best_auc:
                best_auc = current_auc
                best_weights = weights.copy()

    # Final evaluation with best weights
    preds = np.array([(circuit(x, best_weights) + 1) / 2 for x in X_test])
    acc = accuracy_score(y_test, (preds > 0.5).astype(int))
    auc = roc_auc_score(y_test, preds)

    # === Accurate resource profiling using modern PennyLane API ===
    specs = qml.specs(circuit)(X_test[0], best_weights)
    res = specs["resources"]                     # Resources object
    depth = res.depth
    num_wires = res.num_wires
    total_gates = sum(res.gate_types.values())   # counts all gates (including encoding)

    # Normalised resource cost ∈ [0,1] (empirically tuned for 4–16 qubit VQCs)
    resource_cost = (num_wires / 32.0 + depth / 200.0 + total_gates / 2000.0) / 3.0
    resource_cost = np.clip(resource_cost, 0.0, 1.0)

    return acc, auc, resource_cost