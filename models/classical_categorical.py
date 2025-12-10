# 2. New file: classical_categorical.py (next to classical_models.py)
# classical_categorical.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, \
                             precision_score, recall_score, f1_score

# from utils.metrics import composite_efficiency_score

# Discretize all features into 8 uniform bins → categorical
def discretize(X_train, X_test):
    kb = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform', subsample=None)
    X_train_b = kb.fit_transform(X_train)
    X_test_b  = kb.transform(X_test)
    return X_train_b, X_test_b

CLASSICAL_CAT_ENCODINGS = {
    "label":      ce.OrdinalEncoder(),              # Label encoding (ordinal)
    "onehot":     OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
    "dummy":      OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
    "binary":     ce.BinaryEncoder(),                # Binary (log2) encoding
    "target":     ce.TargetEncoder()                 # Mean target encoding (supervised)
}

def classical_categorical_train_and_evaluate(dataset_name, enc_name, encoder, n_features):
    from utils.data_loader import load_dataset_classical   # [0,1] scaled data (no [0,π])

    X_train, X_test, y_train, y_test = load_dataset_classical(dataset_name, n_components=n_features)
    
    # Discretize → categorical
    X_train_b, X_test_b = discretize(X_train, X_test)
    
    # Convert to DataFrame for category_encoders compatibility
    cols = [f"f{i}" for i in range(X_train_b.shape[1])]
    X_train_df = pd.DataFrame(X_train_b, columns=cols)
    X_test_df  = pd.DataFrame(X_test_b,  columns=cols)
    
    # Fit encoder on training data only
    X_train_enc = encoder.fit_transform(X_train_df, y_train if enc_name == "target" else None)
    X_test_enc  = encoder.transform(X_test_df)
    
    # Use same four classifiers as before
    from models.classical_models import CLASSICAL_MODELS
    results = []
    for model_name, model in CLASSICAL_MODELS.items():
        model.fit(X_train_enc, y_train)
        preds = model.predict(X_test_enc)
        probs = model.predict_proba(X_test_enc)[:, 1]
        
        acc  = accuracy_score(y_test, preds)
        auc  = roc_auc_score(y_test, probs)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        f1   = f1_score(y_test, preds, zero_division=0)
        
        # Resource cost = log2( dimensionality after encoding )
        dim = X_train_enc.shape[1]
        rc = np.clip(np.log2(dim + 1) / 10, 0.05, 1.0)
        
        results.append({
            "Category": "Classical-Categorical",
            "EncodingMethods": f"{enc_name}+{model_name}",
            "Dataset": dataset_name,
            "Qubits/Features": dim,
            "Accuracy": round(acc,4),
            "AUC": round(auc,4),
            "Precision": round(prec,4),
            "Recall": round(rec,4),
            "F1-Score": round(f1,4),
            "ResourceCost": round(rc,4)
            #"CES": round(composite_efficiency_score(auc, rc),4)
        })
    return results