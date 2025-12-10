import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import load_dataset
# from utils.metrics import composite_efficiency_score
from models.vqc import create_circuit, train_and_evaluate
from qml_encodings import angle_xy, iqp, basis, bphe, amplitude
from models.classical_models import CLASSICAL_MODELS, classical_train_and_evaluate
from models.classical_categorical import CLASSICAL_CAT_ENCODINGS, classical_categorical_train_and_evaluate

os.makedirs("results", exist_ok=True)

ENCODINGS = {
    "angle_xy": angle_xy.encoding,
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
        acc, auc, prec, rec, f1, rc = train_and_evaluate(ds, name, enc_fn, n_q)
        # ces = composite_efficiency_score(auc, rc)
        results.append({
            "Category": "Quantum",
            "EncodingMethods": name,
            "Dataset": ds,
            "Qubits/Features": n_q,
            "Accuracy": round(acc,4),
            "AUC": round(auc,4),
            "Precision": round(prec,4),
            "Recall": round(rec,4),
            "F1-Score": round(f1,4),
            "ResourceCost": round(rc,4),
            # "CES": round(ces,4)
        })
        print(f"  → {name:12} AUC={auc:.4f} F1={f1:.4f}")

print("\n=== CLASSICAL BASELINES ===")
for ds in datasets:
    n_f = 16 if "mnist" in ds else 8
    print(f"\nDataset: {ds} – {n_f} features")
    for name, model in CLASSICAL_MODELS.items():
        print(f"  → {name:12}", end="\r")
        acc, auc,prec, rec, f1, rc = classical_train_and_evaluate(ds, name, model, n_f)
        # ces = composite_efficiency_score(auc, rc) 
        results.append({
            "Category": "Classical",
            "EncodingMethods": name,  
            "Dataset": ds,
            "Qubits/Features": n_f,
            "Accuracy": round(acc,4),
            "AUC": round(auc,4),
            "Precision": round(prec,4),
            "Recall": round(rec,4),
            "F1-Score": round(f1,4),
            "ResourceCost": round(rc,4),
            # "CES": round(ces,4)
        })
        print(f"  → {name:12} AUC={auc:.4f} F1={f1:.4f}")

print("\n=== CLASSICAL CATEGORICAL ENCODINGS ===")
for ds in datasets:
    n_f = 16 if "mnist" in ds else 8
    print(f"\nDataset: {ds} – {n_f} original features → 8 bins each")
    for enc_name, encoder in CLASSICAL_CAT_ENCODINGS.items():
        print(f"  → {enc_name:12}", end="\r")
        cat_results = classical_categorical_train_and_evaluate(ds, enc_name, encoder, n_f)
        results.extend(cat_results)   # append all 4 classifier results per encoding


df = pd.DataFrame(results)
print("\n=== FINAL QEBS BENCHMARK TABLE ===")
print(df)
df.to_csv("results/qebs_quantum_vs_classical_nov2025-2.csv", index=False)

# g = sns.catplot(
#     data=df,
#     kind="bar",
#     x="Dataset",
#     y="CES",
#     hue="EncodingMethods",
#     col="Category",
#     palette="viridis",
#     height=6,
#     aspect=1.1,
#     legend_out=True,
#     edgecolor="0.3"
# )

g = sns.catplot(
    data=df,
    kind="bar",
    x="Dataset",
    # y="CES",
    hue="EncodingMethods" ,
    # if "Category" == "Quantum" else "Encoding",   # simplified hue logic
    col="Category",
    palette="viridis",
    height=7, aspect=1.2 # different y-scales OK because classical dominates
)
g.set_titles("{col_name}")
g.fig.suptitle("Quantum vs Classical (Continuous & Categorical) Encodings", y=1.02)
plt.savefig("results/Full_comparison_nov2025-2.png", dpi=300)

# g.set_axis_labels("Dataset", "Composite Efficiency Score (CES)")
# g.set_titles("{col_name} Methods")
# g.fig.suptitle("Quantum vs Classical Encoding Methods – Composite Efficiency Score", 
#                 fontsize=16, y=1.05)
# g.add_legend(title="EncodingMethods")

# Optional: annotate best performer per panel
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9)

plt.tight_layout()
plt.savefig("results/Quantum_vs_classical_nov2025-2.png", dpi=300)
plt.show()