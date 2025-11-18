import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, \
                             precision_score, recall_score, f1_score

CLASSICAL_MODELS = {
    "LogReg": LogisticRegression(max_iter=1000, random_state=42),
    "LinearSVM": SVC(kernel="linear", probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                         random_state=42, early_stopping=True)
}

def classical_train_and_evaluate(dataset_name, model_name, model, n_features):
    from utils.data_loader import load_dataset_classical

    X_train, X_test, y_train, y_test = load_dataset_classical(dataset_name,
                                                             n_components=n_features)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, preds)
    auc  = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)

    # Very rough parameter count as classical "resource cost"
    if hasattr(model, "coef_"):
        params = np.prod(model.coef_.shape) + model.intercept_.size
    elif hasattr(model, "n_estimators"):
        params = sum(tree.tree_.node_count for tree in model.estimators_)
    else:
        params = sum([p.size for p in model.coefs_] + [p.size for p in model.intercepts_])
    resource_cost = np.clip(np.log2(params + 1) / 20, 0.05, 1.0)  # normalised ~[0.05,1]

    return acc, auc, prec, rec, f1, resource_cost