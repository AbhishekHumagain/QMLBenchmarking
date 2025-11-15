
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

# 1. Create an isolated environment (highly recommended)
python3 -m venv qml-bench
source qml-bench/bin/activate

# 2. Upgrade pip and install the core stack
pip install --upgrade pip

# Core quantum frameworks
pip install pennylane==0.38.\* qiskit==1.2.\* qiskit-aer

# Additional useful tools
pip install scikit-learn pandas matplotlib seaborn jupyterlab torch torchvision

# Optional: IBM Quantum access (free account is enough for 5–7 qubit jobs)
pip install qiskit-ibm-provider

#If you get any error related to autoray then run this command
pip uninstall pennylane autoray -y
pip install pennylane
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

## License

MIT License – use, modify, and redistribute freely.
