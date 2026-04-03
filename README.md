# Week-06-Day-32-AM

**Take-Home Assignment: Decision Trees & Random Forest**
**Day 32 | AM Session | Week 6 — Machine Learning & AI**

**Scenario:** Build a loan approval system at a bank using Decision Tree and Random Forest. The system must be both interpretable (for regulators) and accurate.

---

# Part A — Concept Application (40%)

*   Create synthetic loan dataset (2000 records): `annual_income`, `credit_score`, `loan_amount`, `employment_years`, `debt_to_income`, `num_credit_cards` → target: `approved`
*   Train Decision Tree (`max_depth=4`), extract top 3 decision rules
*   Train Random Forest (tuned with `RandomizedSearchCV`, 5-fold CV)
*   Compare models: Accuracy, F1, ROC-AUC, Interpretability
*   Compare default vs permutation feature importance
*   Write deployment recommendation <br>
[Solution](loan_approval_pipeline.py)

### Decision Rules Extracted (DT, max_depth=4)

```
Rule 1: IF credit_score > 700  AND debt_to_income < 0.35 → APPROVE
Rule 2: IF credit_score <= 700 AND employment_years > 5  → APPROVE
Rule 3: IF credit_score <= 650                           → REJECT
```

### Model Comparison Summary

| Metric | Decision Tree | Random Forest |
| :--- | :--- | :--- |
| Accuracy | (see script) | (see script) |
| F1 Score | (see script) | (see script) |
| ROC-AUC | (see script) | (see script) |
| Interpretable | ✅ Yes | ⚠️ Partial |

### Deployment Recommendation

Use a **hybrid approach**: the Decision Tree (`max_depth=4`) serves as the primary regulatory artifact — it produces explicit, auditable rules that regulators can inspect. The Random Forest runs in parallel as a risk-scoring engine for borderline cases. For applications where the DT confidence is 45–55%, escalate to the RF probability score before human review. This balances regulatory compliance (DT rules) with predictive accuracy (RF score).

---

## Part B — Stretch Problem (30%)

*   Research `ExtraTreesClassifier` from sklearn
*   Compare ExtraTrees vs RandomForest: splitting strategy, speed, performance <br>
[Code](extra_trees_comparison.py) | [Findings](extra_trees_findings.md)

---

## Part C — Interview Ready (20%)

**Q1 — Bias-Variance tradeoff using DT (high variance) and RF (lower variance). Bagging diagram.**

**Q2 (Coding) — `plot_overfitting_curve(X, y, max_depths)`**

**Q3 — Debug: Identical train/test accuracy (0.95) — is this a problem?** <br>
[Answers](interview_questions.md) | [Q2 Code](overfitting_curve.py)

---

## Part D — AI-Augmented Task (10%)

**Prompt:** *"Generate a side-by-side matplotlib visualization comparing Decision Tree, Random Forest, and Logistic Regression for a non-technical audience."* <br>
[Visualization Code](model_comparison_infographic.py) | [Evaluation](AI_output.md)

---

## File Index

| File | Purpose |
| :--- | :--- |
| `loan_approval_pipeline.py` | Part A — Full pipeline: data creation → DT → RF → comparison → feature importance |
| `extra_trees_comparison.py` | Part B — ExtraTrees vs RF: splitting, speed, performance |
| `extra_trees_findings.md` | Part B — Written findings and analysis |
| `overfitting_curve.py` | Part C Q2 — `plot_overfitting_curve()` function |
| `interview_questions.md` | Part C Q1 + Q3 — Written answers with ASCII diagram |
| `model_comparison_infographic.py` | Part D — matplotlib infographic |
| `AI_output.md` | Part D — AI output documented and evaluated |
