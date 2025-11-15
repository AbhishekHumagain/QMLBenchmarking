# benchmark.py – execute this file
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import load_dataset
from utils.metrics import composite_efficiency_score
from models.vqc import create_circuit, train_and_evaluate
from qml_encodings import angle_xy, iqp, basis, bphe, amplitude

os.makedirs("results", exist_ok=True)

ENCODINGS = {
    # "angle_xy": angle_xy.encoding,
    "iqp":      iqp.encoding,
    "basis":    basis.encoding,
    "bphe":     bphe.bphe_encoding,
    "amplitude": amplitude.amplitude_encoding
}

results = []
n_qubits = 8
datasets = ["iris", "cancer", "mnist_4vs9"]   # mnist uses n_components=16 → change n_qubits=16 if desired

for ds in datasets:
    n_q = 16 if "mnist" in ds else n_qubits
    print(f"\n=== Dataset: {ds} (qubits = {n_q}) ===")
    for name, enc_fn in ENCODINGS.items():
        print(f"  Running {name}...")
        acc, auc, res_cost = train_and_evaluate(ds, name, enc_fn, n_q)
        ces = composite_efficiency_score(auc, res_cost)
        results.append({
            "Dataset": ds,
            "Encoding": name,
            "Qubits": n_q,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Resource Cost": round(res_cost, 4),
            "CES": round(ces, 4)
        })

# ==================== PREVIOUS BENCHMARK LOOP (FOR REFERENCE) ====================
# results = []
# datasets = ["iris", "cancer", "mnist_4vs9"]

# for ds in datasets:
#     n_qubits = 16 if "mnist" in ds else 8
#     print(f"\n=== {ds.upper()} – {n_qubits} qubits ===")
#     for name, enc_fn in ENCODINGS.items():
#         print(f"  → {name:12}", end="\r")
#         acc, auc, rc = train_and_evaluate(dataset_name=ds,
#                                           encoding_name=name,
#                                           encoding_fn=enc_fn,
#                                           n_qubits=n_qubits)
#         ces = composite_efficiency_score(float(auc), float(rc))
#         results.append({
#             "Dataset": ds,
#             "Encoding": name,
#             "Qubits": n_qubits,
#             "Accuracy": round(acc, 4),
#             "AUC": round(auc, 4),
#             "ResourceCost": round(rc, 4),
#             "CES": round(ces, 4)
#         })
#         print(f"  → {name:12} AUC={auc:.4f} CES={ces:.4f}")

df = pd.DataFrame(results)
print("\n=== FINAL QEBS BENCHMARK TABLE ===")
print(df)
df.to_csv("results/qebs_nov2025_modular.csv", index=False)

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="Dataset", y="CES", hue="Encoding", palette="viridis")
plt.title("Composite Efficiency Score across Datasets")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("results/CES_comparison_nov2025.png")
plt.show()