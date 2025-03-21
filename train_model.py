import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

from processor import ClickDataPreprocessor

# -------------------------------
# Custom transformer for feature selection
# -------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.indices]

# === STEP 1: Load Data ===
df = pd.read_csv("C:\\Users\\a.montinaro\\OneDrive - Fortop S.r.l\\Desktop\\ip_model\\labeled_data_click_test.csv")
df = df.drop(columns=['engaged_users', 'total_sessions', 'total_conversions'])

# === STEP 2: Separate Target ===
y = df["is_bot"].astype(int)
X = df.drop(columns=["is_bot"])

# === STEP 3: Fit Preprocessor on full data ===
preprocessor = ClickDataPreprocessor()
preprocessor.fit(X, y)
X_transformed = preprocessor.transform(X)

# === STEP 4: Feature Selection using Random Forest ===
# Train on full data for importance (could be swapped with resampled later if desired)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_transformed, y)
importances = rf.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]  # Top 10 features

# Get feature names from preprocessor
feature_names = []
for name, transformer, cols in preprocessor.pipeline.transformers_:
    try:
        feature_names.extend(list(transformer.get_feature_names_out(cols)))
    except Exception:
        if isinstance(cols, list):
            feature_names.extend(cols)
        else:
            feature_names.extend([f"{name}_{i}" for i in range(len(cols))])
selected_feature_names = [feature_names[i] for i in top_indices]

# === STEP 5: Create Final Preprocessing Pipeline (with feature selection) ===
final_preprocessor = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', FeatureSelector(top_indices))
])

# Transform the entire dataset using the final pipeline
X_final = final_preprocessor.transform(X)

# === STEP 6: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

# === STEP 7: Balance the training set using SMOTE ===
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# === STEP 8: Train Logistic Regression Model with GridSearchCV ===
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2'],  # Consider 'l1' as well if you want sparsity
    'solver': ['saga'],
    'class_weight': ['balanced']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    param_grid,
    scoring='recall',  # Could also try 'f1' or 'roc_auc'
    cv=10,
    n_jobs=-1
)
grid.fit(X_res, y_res)  # ✅ Train on resampled data

print(f"Best parameters: {grid.best_params_}")

model = grid.best_estimator_
model.fit(X_res, y_res)  # ✅ Refit on resampled training data

# === STEP 9: Interpret Model — Non-zero coefficients ===
coef = model.coef_[0]
nonzero_indices = (coef != 0)
sparse_feature_names = [selected_feature_names[i] for i, flag in enumerate(nonzero_indices) if flag]
model.sparse_features_ = sparse_feature_names

# === STEP 10: Evaluate Model ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"🚀 ROC AUC Score: {roc_auc:.4f}")

# === STEP 11: Save Preprocessor and Model ===
joblib.dump(final_preprocessor, "preprocessor.pkl")
joblib.dump(model, "model.pkl")

print("✅ Preprocessor and sparse model saved to disk.")
