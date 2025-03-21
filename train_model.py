import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from processor import ClickDataPreprocessor

# === STEP 1: Load Data ===
df = pd.read_csv("C:\\Users\\a.montinaro\\OneDrive - Fortop S.r.l\\Desktop\\ip_model\\labeled_data_click_test.csv")
df = df.drop(columns=['engaged_users', 'total_sessions', 'total_conversions'])

# === STEP 2: Separate Target ===
y = df["is_bot"].astype(int)
X = df.drop(columns=["is_bot"])

# === STEP 3: Fit Preprocessor ===
preprocessor = ClickDataPreprocessor()
preprocessor.fit(X, y)
X_transformed = preprocessor.transform(X)

# use SMOTE to balance the dataset
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_transformed, y = smote.fit_resample(X_transformed, y)

# Estrazione robusta dei nomi delle feature dal preprocessor
feature_names = []
for name, transformer, cols in preprocessor.pipeline.transformers_:
    try:
        feature_names.extend(transformer.get_feature_names_out(cols))
    except Exception:
        if isinstance(cols, list):
            feature_names.extend(cols)
        else:
            feature_names.extend([f"{name}_{i}" for i in range(len(cols))])

# === STEP 4: Feature Selection con Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_transformed, y)
importances = rf.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]  # Top 10 features
X_transformed = X_transformed[:, top_indices]
feature_names = [feature_names[i] for i in top_indices]

# === STEP 5: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42, stratify=y
)

# === STEP 6: Train Sparse Model using Logistic Regression with L1 penalty ===
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 1, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['saga'],
    'class_weight': ['balanced', {0: 0.33, 1: 0.67}]
}

grid = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), param_grid, scoring='roc_auc', cv=5)
grid.fit(X_train, y_train)
print(f"Best parameters: {grid.best_params_}")

model = grid.best_estimator_
model.fit(X_train, y_train)

# Estrazione delle feature con coefficienti non nulli (cioè quelle predittive)
coef = model.coef_[0]
nonzero_indices = (coef != 0)
sparse_feature_names = [fname for i, fname in enumerate(feature_names) if nonzero_indices[i]]
model.sparse_features_ = sparse_feature_names

# === STEP 7: Evaluate Model ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"🚀 ROC AUC Score: {roc_auc:.4f}")

# === STEP 8: Save Model and Preprocessor ===
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(model, "model.pkl")

print("✅ Preprocessor and sparse model saved to disk.")
