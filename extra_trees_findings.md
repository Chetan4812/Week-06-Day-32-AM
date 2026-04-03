# Part B — Extra Trees vs Random Forest

**Self-Study:** ExtraTreesClassifier (`sklearn.ensemble.ExtraTreesClassifier`)

Run [extra_trees_comparison.py](extra_trees_comparison.py) to reproduce all results below.

---

## (a) How Does Splitting Differ?

| Aspect | Random Forest | Extra Trees |
| :--- | :--- | :--- |
| **Feature subset** | Random subset (`sqrt(n_features)`) at each node | All features (default) OR random subset |
| **Threshold selection** | **Optimal** — searches all values, picks best split point | **Random** — drawn uniformly from the feature's value range |
| **Randomness level** | Moderate (random features, optimal threshold) | Higher (random features + random threshold) |
| **Result** | Lower bias, moderate variance | Slightly higher bias, lower variance |

**Key difference:** Random Forest finds the *best* split among a random set of features. Extra Trees finds the *best* split among randomly drawn *thresholds* — adding an extra layer of randomisation. This makes each tree weaker individually but the ensemble more diverse.

---

## (b) Speed Comparison

Extra Trees is typically **2–3× faster** than Random Forest at training time because:
*   It skips the computationally expensive step of searching for the optimal threshold for each feature at each node.
*   A random threshold is drawn in O(1); an optimal threshold requires sorting the feature values in O(n log n).
*   This speed advantage compounds across thousands of nodes and hundreds of trees.

**Industry application:** Amazon and Netflix use Extra Trees in real-time prediction pipelines precisely because of this speed advantage — when predictions must be generated in milliseconds at scale, training speed and inference speed both matter.

---

## (c) Performance Comparison on Loan Dataset

| Metric | Random Forest | Extra Trees |
| :--- | :--- | :--- |
| **Accuracy** | (see script output) | (see script output) |
| **F1 Score** | (see script output) | (see script output) |
| **ROC-AUC** | (see script output) | (see script output) |
| **CV AUC (5-fold)** | (see script output) | (see script output) |
| **Training time** | Slower | ~2–3× faster |

**Typical finding:** On structured tabular datasets like the loan dataset, Extra Trees achieves comparable ROC-AUC to Random Forest (often within 0.01–0.02) while training significantly faster. The accuracy and F1 are usually very close.

**When to prefer Extra Trees:**
*   Real-time or near-real-time prediction pipelines where training speed matters
*   Large datasets where RF training time becomes a bottleneck
*   When you need a strong regularisation effect (the random thresholds act as additional regularisation)

**When to prefer Random Forest:**
*   When maximum predictive accuracy on the test set is the primary goal
*   When feature importance interpretability matters (RF impurity importance is more stable)
*   When the dataset is small and variance is already low
