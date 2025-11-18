# utils/data_loader.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml

def load_dataset(name: str, n_components: int = 8):
    if name == "iris":
        data = load_iris()
        X_raw = data.data
        y = (data.target == 0).astype(int)  # binary: setosa vs non-setosa
    elif name == "cancer":
        data = load_breast_cancer()
        X_raw = data.data
        y = data.target
    elif name == "mnist_4vs9":
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X_raw = mnist.data.astype(np.float64) / 255.0
        y = mnist.target.astype(int)
        mask = np.isin(y, [4, 9])
        X_raw, y = X_raw[mask], y[mask]
        y = (y == 9).astype(int)  # binary 4 vs 9
    else:
        raise ValueError("Unknown dataset")

    # 1. Standard-scale (zero mean, unit variance – ideal for PCA)
    X = StandardScaler().fit_transform(X_raw)

    # 2. Dimensionality adjustment to exactly n_components
    if X.shape[1] > n_components:
        X = PCA(n_components=n_components, random_state=42).fit_transform(X)
    
    if X.shape[1] < n_components:
        pad_width = n_components - X.shape[1]
        padding = np.zeros((X.shape[0], pad_width))
        X = np.concatenate([X, padding], axis=1)

    # 3. Final MinMax scaling to [0, π] (optimal for angle/IQP; safe for amplitude + BPHE)
    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def load_dataset_classical(name: str, n_components: int = 8):
    """Returns X_train, X_test, y_train, y_test with same preprocessing as quantum"""
    X_train, X_test, y_train, y_test = load_dataset(name, n_components=n_components)
    # Quantum pipeline scales to [0,π] for rotations → undo to [0,1] for classical fairness
    X_train = X_train / np.pi
    X_test  = X_test  / np.pi
    return X_train, X_test, y_train, y_test