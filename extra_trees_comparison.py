import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Recreate the loan dataset
n = 2000
annual_income     = np.random.randint(25000, 200000, n)
credit_score      = np.random.randint(500, 850, n)
loan_amount       = np.random.randint(5000, 100000, n)
employment_years  = np.random.randint(0, 30, n)
debt_to_income    = np.round(np.random.uniform(0.05, 0.65, n), 2)
num_credit_cards  = np.random.randint(0, 10, n)

approve_prob = (
    (credit_score > 700).astype(int) * 0.40 +
    (debt_to_income < 0.35).astype(int) * 0.25 +
    (employment_years > 3).astype(int) * 0.15 +
    (annual_income > 60000).astype(int) * 0.15 +
    (num_credit_cards < 6).astype(int) * 0.05
)
approved = (approve_prob + np.random.normal(0, 0.1, n) > 0.55).astype(int)

X = pd.DataFrame({
    'annual_income': annual_income, 'credit_score': credit_score,
    'loan_amount': loan_amount, 'employment_years': employment_years,
    'debt_to_income': debt_to_income, 'num_credit_cards': num_credit_cards,
})
y = approved

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── (a) How does splitting differ? ───────────────────────────────────────────
print("── (a) Splitting Difference ─────────────────────────────────────────")
print("""
  Random Forest:
    At each node, selects the BEST split among a random subset of features
    (evaluated by Gini/entropy). The threshold is optimised — the algorithm
    searches all possible values for the chosen feature and picks the one
    that maximises information gain.

  Extra Trees (Extremely Randomized Trees):
    At each node, selects the BEST feature among a random subset BUT uses
    a RANDOM threshold (drawn uniformly from the feature's range) rather
    than the optimal one. This means splits are not locally optimal.
    The randomness is increased further: in the default setting, it
    evaluates ALL features (not a subset) and picks the best split among
    randomly drawn thresholds.
    Result: more randomisation → lower variance but potentially higher bias.
""")

# ── (b) Speed comparison ──────────────────────────────────────────────────────
print("── (b) Speed Comparison ─────────────────────────────────────────────")

rf = RandomForestClassifier(n_estimators=200, random_state=42)
et = ExtraTreesClassifier(n_estimators=200, random_state=42)

start = time.time()
rf.fit(X_train, y_train)
rf_time = time.time() - start

start = time.time()
et.fit(X_train, y_train)
et_time = time.time() - start

print(f"  Random Forest training time : {rf_time:.4f}s")
print(f"  Extra Trees training time   : {et_time:.4f}s")
print(f"  Extra Trees is {rf_time/et_time:.1f}x faster than Random Forest")
print("  (Extra Trees skips threshold optimisation → faster node splits)")

# ── (c) Performance comparison ────────────────────────────────────────────────
print("\n── (c) Performance Comparison ───────────────────────────────────────")

for name, model in [("Random Forest", rf), ("Extra Trees", et)]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv  = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    print(f"\n  {name}")
    print(f"    Accuracy    : {acc*100:.2f}%")
    print(f"    F1 Score    : {f1:.4f}")
    print(f"    ROC-AUC     : {auc:.4f}")
    print(f"    CV AUC (5f) : {cv:.4f}")
    print(f"    Train time  : {rf_time if name == 'Random Forest' else et_time:.4f}s")
