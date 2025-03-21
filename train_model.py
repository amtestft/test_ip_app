import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

# Estrazione robusta dei nomi delle feature dal preprocessor
feature_names = []
for name, transformer, cols in preprocessor.pipeline.transformers_:
    try:
        feature_names.extend(transformer.get_feature_names_out(cols))
    except Exception:
        # Se il transformer non espone get_feature_names_out, usa i nomi originali o un fallback generico
        if isinstance(cols, list):
            feature_names.extend(cols)
        else:
            feature_names.extend([f"{name}_{i}" for i in range(len(cols))])

# === STEP 4: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42, stratify=y
)

# === STEP 5: Train Sparse Model using Logistic Regression with L1 penalty ===
model = LogisticRegression(penalty='l1', solver='saga', C=1.0, random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Estrazione delle feature con coefficienti non nulli (cioè quelle predittive)
coef = model.coef_[0]
nonzero_indices = (coef != 0)
sparse_feature_names = [fname for i, fname in enumerate(feature_names) if nonzero_indices[i]]
model.sparse_features_ = sparse_feature_names

# === STEP 6: Evaluate Model ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"🚀 ROC AUC Score: {roc_auc:.4f}")

# === STEP 7: Save Model and Preprocessor ===
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(model, "model.pkl")

print("✅ Preprocessor and sparse model saved to disk.")
