
## Repository Structure

```
quantum-encoding-benchmark/
├── benchmark.py                  # Main benchmark runner
├── utils/
│   ├── data_loader.py            # Dataset loading & preprocessing
│   └── metrics.py                # Composite Efficiency Score
├── qml_encodings/                # Quantum encoding implementations
│   ├── __init__.py
│   ├── angle_xy.py
│   ├── iqp.py
│   ├── basis.py
│   └── bphe.py
├── models/
│   ├── __init__.py
│   └── vqc.py                    # VQC ansatz & training
└── results/                      # Generated outputs
```

## Quick Start

```bash
git clone https://github.com/yourusername/quantum-encoding-benchmark.git
cd quantum-encoding-benchmark

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install pennylane==0.38.* qiskit==1.2.* qiskit-aer
pip install scikit-learn pandas matplotlib seaborn
pip install qiskit-ibm-provider

#If you get any error related to autoray then run this command
pip uninstall pennylane autoray -y
pip install pennylane
```

Run the benchmark:

```bash
python benchmark.py
```

Estimated runtime: 8–12 minutes on Apple Silicon M3 or Intel i9.

Output files in `./results/`:
- `qebs_nov2025_modular.csv`
- `CES_nov2025_modular.png`

## Extending

- **Add noise simulation** → Modify `models/vqc.py` to use `default.mixed` with depolarizing channels
- **Use real hardware** → Replace device with IBM Quantum backend
- **New encoding** → Create file in `qml_encodings/` and update `__init__.py`

## License

MIT License

