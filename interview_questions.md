# Part C — Interview Ready

## Q1 — Bias-Variance Tradeoff: Decision Tree vs Random Forest

### The Tradeoff

Every ML model's error can be decomposed as:

**Total Error = Bias² + Variance + Irreducible Noise**

*   **Bias** — how far the model's average prediction is from the true value (underfitting)
*   **Variance** — how much the model's predictions vary across different training sets (overfitting)

### Decision Tree = High Variance

A fully grown Decision Tree memorises the training data — it learns every split down to individual data points. Change the training set slightly and the tree structure changes dramatically. This is **high variance**:

*   Train accuracy: near 100%
*   Test accuracy: much lower (overfitting)
*   Small changes in training data → very different tree structures

Pruning (`max_depth`) reduces variance at the cost of some bias.

### Random Forest = Lower Variance

Random Forest applies **bagging (Bootstrap Aggregating)**:
1. Draw `B` bootstrap samples (random samples with replacement) from training data
2. Train one Decision Tree on each bootstrap sample
3. Average (or majority vote) the predictions of all B trees

Each tree sees a different subset of data AND a random subset of features at each split. The trees are **decorrelated** — their errors are less correlated — so averaging cancels out much of the individual variance.

```
Variance(average of B trees) ≈ σ²/B  (if trees were independent)
```

In practice: Random Forest variance is significantly lower than a single tree.

### ASCII Diagram — How Bagging Reduces Variance

```
Training Data (N rows)
         │
    ┌────┴──────────────────────────────┐
    │  Bootstrap Sampling (with replacement) │
    └────┬──────┬──────┬──────┬──────┬──┘
         │      │      │      │      │
       D₁    D₂    D₃    D₄    D₅     ← Different bootstrap datasets
         │      │      │      │      │
       T₁    T₂    T₃    T₄    T₅     ← Different Decision Trees
    (high var)(high var)(high var)...
         │      │      │      │      │
         └──────┴──────┴──────┴──────┘
                       │
                  Average / Vote
                       │
              Final Prediction
           (LOW variance, stable)

Single Tree:  Variance = σ²
Random Forest: Variance ≈ ρσ² + (1-ρ)σ²/B
              where ρ = correlation between trees (kept low by feature randomness)
```

Key insight: The more trees, and the lower their correlation (achieved by feature randomness), the more variance is reduced.

---

## Q2 — Coding: `plot_overfitting_curve(X, y, max_depths)`
[Solution](overfitting_curve.py)

---

## Q3 — Debug: Identical Train and Test Accuracy (0.95)

**The code:**
```python
rf = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
print(f'Train: {rf.score(X_train, y_train):.2f}')  # 0.95
print(f'Test: {rf.score(X_test, y_test):.2f}')      # 0.95
```

**Is this a problem? No — this is the expected and ideal outcome.**

**Why it is NOT a problem:**

*   Identical or very close train and test accuracy is the **goal** of a well-generalising model.
*   A Random Forest with `max_depth=3` is already quite constrained (shallow trees with low variance). It will not overfit severely.
*   If both scores are 0.95, the model has learned genuine patterns in the data without memorising noise.

**What would actually be a problem:**
*   Train = 0.99, Test = 0.72 → **Overfitting** (model memorised training data)
*   Train = 0.62, Test = 0.60 → **Underfitting** (model too simple, not learning patterns)
*   Train = Test = 0.52 on a balanced binary dataset → **Barely better than random guessing**

**Potential concern to investigate (not a flaw, but worth checking):**
*   Is the dataset leaking target information into the features? If a feature is a direct proxy for the label, 0.95 accuracy is easy to achieve artificially.
*   Is the test set representative of real-world data? If train and test come from the same time period or distribution, performance may degrade on future data.

**Conclusion:** Identical train and test accuracy at 0.95 with `max_depth=3` is a sign of a well-regularised, well-generalising model. No debugging action needed — this is the desired outcome.
