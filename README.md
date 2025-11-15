```markdown
# QEBS – Quantum Encoding Benchmark Suite

**A clean, modular, reproducible benchmark for classical-to-quantum data encoding strategies in Quantum Machine Learning.**

This repository implements the **Quantum Encoding Benchmark Suite (QEBS)** used in the paper  
*“Benchmarking Hybrid Quantum Data Encoding Methods”*  
Abhishek Humagain¹⋆ & Wilson Patterson¹⋆ – Mississippi State University (November 2025).

![Composite Efficiency Score (Noise-Free)](results/CES_nov2025_modular.png)

## Key Features

- Standardized preprocessing pipeline (Challenge 1)
- Four encoding strategies compared under identical qubit budget:
  - Angle-XY (dense angle encoding)
  - IQP (dense instantaneous quantum polynomial)
  - Basis encoding
  - **BPHE** – Bit-Partition Hybrid Encoding (focus of the study)
- Hardware-efficient Variational Quantum Classifier (VQC)
- Modern PennyLane resource profiling (`qml.specs`)
- Composite Efficiency Score (CES) with weights [0.5, 0.3, 0.2]
- Ready for noisy simulation & real hardware (Week 5 extensions)

**Main result (noise-free simulator, Nov 14 2025):**  
BPHE achieves the highest CES on the MNIST 4-vs-9 task (16 qubits) and ranks in the top-2 on Iris & Breast Cancer – first empirical validation of BPHE superiority under resource constraints.

## Repository Structure

```
quantum-encoding-benchmark/
├── benchmark.py                  # Main runner – execute this
├── utils/
│   ├── data_loader.py            # Datasets + standardized preprocessing
│   └── metrics.py                # Composite Efficiency Score
├── qml_encodings/                # All encodings (linear in qubits)
│   ├── __init__.py
│   ├── angle_xy.py
│   ├── iqp.py
│   ├── basis.py
│   └── bphe.py
├── models/
│   ├── __init__.py
│   └── vqc.py                    # VQC ansatz + training loop
├── results/                      # Auto-generated CSV + plots
└── README.md
```

## Quick Start

```bash
git clone https://github.com/yourusername/quantum-encoding-benchmark.git
cd quantum-encoding-benchmark

python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# .\venv\Scripts\activate         # Windows

pip install pennylane scikit-learn pandas matplotlib seaborn
```

Run the full benchmark:

```bash
python benchmark.py
```

Runtime ≈ 8–12 minutes on Apple Silicon M3 / Intel i9.

Outputs appear in `./results/`:
- `qebs_nov2025_modular.csv`
- `CES_nov2025_modular.png`

## Example Results (Noise-Free, Nov 14 2025)

| Dataset       | Encoding   | Qubits | Accuracy | AUC    | Resource Cost | CES    |
|---------------|------------|--------|----------|--------|---------------|--------|
| iris          | bphe       | 8      | 1.0000   | 1.0000 | 0.33          | **0.928** |
| cancer        | angle_xy   | 8      | 0.9825   | 0.997  | 0.32          | **0.919** |
| mnist_4vs9    | bphe       | 16     | 0.969    | 0.994  | 0.43          | **0.889** |

## Extending the Suite

- **Noisy simulation** → edit `models/vqc.py` and change device to `default.mixed` with depolarizing noise.
- **Real hardware** → replace device with an IBM Quantum backend (free tier works for 8-qubit jobs).
- **Add new encoding** → drop a new file in `qml_encodings/` and import it in `__init__.py`.

## Citation (forthcoming)

```bibtex
@misc{humagain2025qebs,
  title = {Benchmarking Hybrid Quantum Data Encoding Methods},
  author = {Abhishek Humagain and Wilson Patterson},
  year = {2025},
  note = {Preprint, code available at https://github.com/yourusername/quantum-encoding-benchmark}
}
```

## License

MIT License – use, modify, and redistribute freely.
