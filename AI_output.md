# Part D — AI-Augmented Task

## Prompt Used

*"Generate a side-by-side matplotlib visualization comparing Decision Tree, Random Forest, and Logistic Regression for a non-technical audience. Show: when to use each, pros/cons, interpretability scale."*

---

## AI Output

The AI generated a three-panel matplotlib figure with one panel per model, each showing:
- A horizontal bar rating for Interpretability, Accuracy, and Speed (scored out of 5)
- A "Use When" section listing 4 scenarios
- Pros and cons bullet lists
- Color-coded borders per model

The reproduced and corrected version is in [model_comparison_infographic.py](model_comparison_infographic.py).

---

## Evaluation

### Is the Visualization Accurate?

*   **Logistic Regression ratings (Interpretability=5, Accuracy=2, Speed=5):** ✅ Correct. LR is the most interpretable of the three and fastest to train, but struggles with non-linear patterns.
*   **Decision Tree ratings (Interpretability=4, Accuracy=3, Speed=4):** ✅ Correct. Human-readable rules make it highly interpretable; accuracy is moderate due to overfitting tendency.
*   **Random Forest ratings (Interpretability=2, Accuracy=5, Speed=2):** ✅ Correct. RF trades interpretability for accuracy; 500 trees are slower to train and harder to explain.
*   **Pros/cons content:** ✅ All technically accurate. No misleading claims.
*   **"Use When" guidance:** ✅ Appropriate for a non-technical audience.

### Does It Oversimplify?

*   **Accuracy scores as fixed numbers (out of 5):** ⚠️ Slight oversimplification. Accuracy depends heavily on dataset characteristics — RF doesn't always outperform LR on linearly separable data. The ratings should be understood as relative tendencies, not absolute facts.
*   **Missing nuance:** The visual doesn't mention that Random Forest can produce feature importances (a partial interpretability tool), or that Decision Trees can be made more accurate with ensemble methods. These are acceptable omissions for a non-technical audience.
*   **Speed rating for Random Forest:** The "2/5" rating is for training speed. Inference speed for a trained RF is actually quite fast — this distinction is lost in the simplified rating.

### Improvements Made

*   Added color-coded spine borders per model for clearer visual separation.
*   Added a gray background track behind each rating bar so the scale is visible even for low scores.
*   Adjusted font sizes for better readability at the saved DPI.
*   Added `bbox_inches='tight'` to prevent label clipping when saving.
*   The AI's original version used plain `ax.text()` for ratings without visual bars — replaced with actual `barh()` elements for a proper visual scale representation.
