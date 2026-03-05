# =====================================================
# COMPLETE EVALUATION & VISUALIZATION SCRIPT
# AI-Based Disease Prediction (XGBoost)
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.linear_model import LogisticRegression

# Create output folder for figures
OUTPUT_DIR = "research_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------

dataset_df = pd.read_csv('../data/dataset.csv')
dataset_df = dataset_df.apply(lambda col: col.str.strip())

# One-hot encode symptoms
symptoms_df = pd.get_dummies(
    dataset_df.filter(regex='Symptom'),
    prefix='',
    prefix_sep=''
)

symptoms_df = symptoms_df.groupby(symptoms_df.columns, axis=1).max()

clean_df = pd.merge(
    symptoms_df,
    dataset_df['Disease'],
    left_index=True,
    right_index=True
)

X = clean_df.iloc[:, :-1]
y = clean_df.iloc[:, -1]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# -----------------------------
# Train XGBoost
# -----------------------------

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss'
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

preds = model.predict(X_test)
xgb_acc = accuracy_score(y_test, preds)

print("\nXGBoost Accuracy:", round(xgb_acc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, preds, target_names=le.classes_))

# =====================================================
# 1️⃣ Confusion Matrix
# =====================================================

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================================================
# 2️⃣ Model Comparison
# =====================================================

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_preds)

plt.figure(figsize=(6,5))
plt.bar(['Logistic Regression', 'XGBoost'], [log_acc, xgb_acc])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0,1)
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================================================
# 3️⃣ ROC Curve
# =====================================================

y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
y_score = model.predict_proba(X_test)

plt.figure(figsize=(8,6))

for i in range(len(le.classes_)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{le.classes_[i]} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC Curve")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================================================
# 4️⃣ Feature Importance
# =====================================================

plt.figure(figsize=(8,6))
xgb.plot_importance(model, max_num_features=15)
plt.title("Top 15 Important Symptoms")
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================================================
# 5️⃣ Training vs Validation Loss
# =====================================================

results = model.evals_result()

plt.figure(figsize=(8,6))
plt.plot(results['validation_0']['mlogloss'], label="Train")
plt.plot(results['validation_1']['mlogloss'], label="Validation")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/training_validation_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================================================
# 6️⃣ Disease Distribution
# =====================================================

plt.figure(figsize=(10,6))
pd.Series(y).value_counts().plot(kind='bar')
plt.title("Disease Distribution")
plt.xlabel("Disease")
plt.ylabel("Number of Cases")
plt.xticks(rotation=90)
plt.savefig(f"{OUTPUT_DIR}/disease_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================================================
# 7️⃣ Sample Prediction Probability
# =====================================================

sample_probs = model.predict_proba(X_test.iloc[:1])[0]

plt.figure(figsize=(10,6))
plt.bar(le.classes_, sample_probs)
plt.xticks(rotation=90)
plt.title("Predicted Disease Probabilities (Sample Patient)")
plt.ylabel("Probability")
plt.savefig(f"{OUTPUT_DIR}/sample_prediction_probability.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nAll evaluation graphs saved inside 'research_figures' folder.\n")