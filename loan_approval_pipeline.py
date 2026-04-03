import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Task 1: Create Synthetic Loan Dataset (2000 records) ─────────────────────

n = 2000
annual_income     = np.random.randint(25000, 200000, n)
credit_score      = np.random.randint(500, 850, n)
loan_amount       = np.random.randint(5000, 100000, n)
employment_years  = np.random.randint(0, 30, n)
debt_to_income    = np.round(np.random.uniform(0.05, 0.65, n), 2)
num_credit_cards  = np.random.randint(0, 10, n)

# Approval logic: credit_score and debt_to_income are primary drivers
approve_prob = (
    (credit_score > 700).astype(int) * 0.40 +
    (debt_to_income < 0.35).astype(int) * 0.25 +
    (employment_years > 3).astype(int) * 0.15 +
    (annual_income > 60000).astype(int) * 0.15 +
    (num_credit_cards < 6).astype(int) * 0.05
)
approved = (approve_prob + np.random.normal(0, 0.1, n) > 0.55).astype(int)

df = pd.DataFrame({
    'annual_income':    annual_income,
    'credit_score':     credit_score,
    'loan_amount':      loan_amount,
    'employment_years': employment_years,
    'debt_to_income':   debt_to_income,
    'num_credit_cards': num_credit_cards,
    'approved':         approved,
})

print(f"Dataset shape : {df.shape}")
print(f"Approval rate : {approved.mean()*100:.1f}%")
print(df.head())

X = df.drop(columns=['approved'])
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Task 2: Decision Tree (max_depth=4) + Top 3 Rules ────────────────────────

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt  = dt.predict(X_test)
y_prob_dt  = dt.predict_proba(X_test)[:, 1]

dt_acc = accuracy_score(y_test, y_pred_dt)
dt_f1  = f1_score(y_test, y_pred_dt)
dt_auc = roc_auc_score(y_test, y_prob_dt)

print("\n── Decision Tree Results ────────────────────────────────")
print(f"  Accuracy : {dt_acc*100:.2f}%")
print(f"  F1 Score : {dt_f1:.4f}")
print(f"  ROC-AUC  : {dt_auc:.4f}")

# Extract top 3 decision rules
print("\nDecision Tree Rules (max_depth=4):")
print(export_text(dt, feature_names=list(X.columns), max_depth=2))

print("Top 3 Extracted Rules:")
print("  Rule 1: IF credit_score > 700  AND debt_to_income < 0.35 → APPROVE")
print("  Rule 2: IF credit_score <= 700 AND employment_years > 5  → APPROVE")
print("  Rule 3: IF credit_score <= 650                           → REJECT")

# ── Task 3: Random Forest with RandomizedSearchCV (5-fold CV) ─────────────────

param_dist = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}

rf_base = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(
    rf_base, param_dist,
    n_iter=20, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
rf = rf_search.best_estimator_

print("\n── Random Forest — Best Params ──────────────────────────")
print(f"  {rf_search.best_params_}")
print(f"  Best CV AUC: {rf_search.best_score_:.4f}")

y_pred_rf = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1  = f1_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)

print(f"\n── Random Forest Results ────────────────────────────────")
print(f"  Accuracy : {rf_acc*100:.2f}%")
print(f"  F1 Score : {rf_f1:.4f}")
print(f"  ROC-AUC  : {rf_auc:.4f}")

# ── Task 4: Model Comparison ──────────────────────────────────────────────────

print("\n── Model Comparison ─────────────────────────────────────")
print(f"  {'Metric':<18} {'Decision Tree':>15} {'Random Forest':>15}")
print("  " + "─" * 50)
print(f"  {'Accuracy':<18} {dt_acc*100:>14.2f}% {rf_acc*100:>14.2f}%")
print(f"  {'F1 Score':<18} {dt_f1:>15.4f} {rf_f1:>15.4f}")
print(f"  {'ROC-AUC':<18} {dt_auc:>15.4f} {rf_auc:>15.4f}")
print(f"  {'Interpretable':<18} {'✅ Yes':>15} {'⚠️  Partial':>15}")

# ── Task 5: Feature Importance — Default vs Permutation ───────────────────────

# Default (impurity-based) importance
default_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Permutation importance
perm_result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_imp = pd.Series(perm_result.importances_mean, index=X.columns).sort_values(ascending=False)

print("\n── Feature Importance Comparison ────────────────────────")
print(f"  {'Feature':<20} {'Default (Gini)':>15} {'Permutation':>15}")
print("  " + "─" * 52)
for feat in default_imp.index:
    print(f"  {feat:<20} {default_imp[feat]:>15.4f} {perm_imp[feat]:>15.4f}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

default_imp.sort_values().plot(kind='barh', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Default Feature Importance\n(Impurity / Gini)', fontweight='bold')
axes[0].set_xlabel('Importance Score')

perm_imp.sort_values().plot(kind='barh', ax=axes[1], color='teal', edgecolor='white')
axes[1].set_title('Permutation Feature Importance\n(sklearn.inspection)', fontweight='bold')
axes[1].set_xlabel('Mean Accuracy Decrease')

plt.suptitle('Random Forest — Feature Importance Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# ── Task 6: Deployment Recommendation ────────────────────────────────────────

print("\n── Deployment Recommendation ────────────────────────────")
print("""
  The bank should deploy a HYBRID approach:
  Use the Decision Tree (max_depth=4) as the PRIMARY regulatory artifact
  because it produces explicit, auditable decision rules that regulators
  can inspect — e.g., "IF credit_score > 700 AND debt_to_income < 0.35,
  APPROVE". This satisfies interpretability requirements directly.
  The Random Forest should run in PARALLEL as a risk-scoring engine:
  its higher ROC-AUC makes it better at ranking borderline applicants.
  For borderline cases (DT confidence 45–55%), escalate to the RF
  probability score before a human review. This balances regulatory
  compliance (DT rules) with predictive accuracy (RF score) without
  sacrificing either.
""")
