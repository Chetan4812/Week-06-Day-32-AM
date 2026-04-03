import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── AI Prompt Used ────────────────────────────────────────────────────────────
# "Generate a side-by-side matplotlib visualization comparing Decision Tree,
# Random Forest, and Logistic Regression for a non-technical audience.
# Show: when to use each, pros/cons, interpretability scale."

# ── AI-generated visualization (reproduced and corrected below) ───────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 9))
fig.patch.set_facecolor('#f8f9fa')

models = [
    {
        "name": "Logistic\nRegression",
        "color": "#3498db",
        "icon": "📈",
        "when_to_use": [
            "Binary classification",
            "Need probability scores",
            "Linearly separable data",
            "Fast training needed",
        ],
        "pros": ["Very interpretable", "Fast & lightweight", "Outputs probabilities", "Low variance"],
        "cons": ["Assumes linearity", "Struggles with complex patterns", "Sensitive to outliers"],
        "interpretability": 5,
        "accuracy": 2,
        "speed": 5,
    },
    {
        "name": "Decision\nTree",
        "color": "#2ecc71",
        "icon": "🌳",
        "when_to_use": [
            "Need explainable rules",
            "Regulatory compliance",
            "Mixed feature types",
            "Non-linear patterns",
        ],
        "pros": ["Human-readable rules", "No scaling needed", "Handles non-linearity", "Fast inference"],
        "cons": ["Prone to overfitting", "High variance", "Unstable (small data changes)"],
        "interpretability": 4,
        "accuracy": 3,
        "speed": 4,
    },
    {
        "name": "Random\nForest",
        "color": "#e74c3c",
        "icon": "🌲🌲🌲",
        "when_to_use": [
            "High accuracy needed",
            "Tabular/structured data",
            "Feature importance needed",
            "Noisy datasets",
        ],
        "pros": ["High accuracy", "Robust to overfitting", "Handles missing values", "Feature importance"],
        "cons": ["Less interpretable", "Slower training", "Larger memory footprint"],
        "interpretability": 2,
        "accuracy": 5,
        "speed": 2,
    },
]

bar_labels  = ["Interpretability", "Accuracy", "Speed"]
bar_colors  = ["#9b59b6", "#e67e22", "#1abc9c"]

for ax, m in zip(axes, models):
    ax.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, m["name"], ha='center', va='top', fontsize=16,
            fontweight='bold', color=m["color"])

    # Rating bars
    y_start = 8.6
    for label, score, bc in zip(bar_labels, [m["interpretability"], m["accuracy"], m["speed"]], bar_colors):
        ax.barh(y_start, score * 2, height=0.35, left=0, color=bc, alpha=0.8)
        ax.barh(y_start, 10, height=0.35, left=0, color='#ecf0f1', zorder=0)
        ax.barh(y_start, score * 2, height=0.35, left=0, color=bc, alpha=0.85, zorder=1)
        ax.text(-0.1, y_start, label, ha='right', va='center', fontsize=7.5, color='#2c3e50')
        ax.text(score * 2 + 0.15, y_start, f'{score}/5', ha='left', va='center',
                fontsize=7.5, fontweight='bold', color=bc)
        y_start -= 0.55

    # When to use
    ax.text(5, 6.1, "✅ Use When", ha='center', fontsize=9,
            fontweight='bold', color='#27ae60')
    for j, tip in enumerate(m["when_to_use"]):
        ax.text(0.3, 5.6 - j * 0.45, f"• {tip}", ha='left', fontsize=8, color='#2c3e50')

    # Pros
    ax.text(5, 3.6, "👍 Pros", ha='center', fontsize=9, fontweight='bold', color='#2980b9')
    for j, pro in enumerate(m["pros"]):
        ax.text(0.3, 3.1 - j * 0.42, f"• {pro}", ha='left', fontsize=7.8, color='#2c3e50')

    # Cons
    ax.text(5, 1.3, "⚠️ Cons", ha='center', fontsize=9, fontweight='bold', color='#e74c3c')
    for j, con in enumerate(m["cons"]):
        ax.text(0.3, 0.85 - j * 0.42, f"• {con}", ha='left', fontsize=7.8, color='#7f8c8d')

    # Border
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(m["color"])
        ax.spines[spine].set_linewidth(2.5)

plt.suptitle('ML Model Comparison: When to Use Each?\n'
             'Decision Tree  •  Random Forest  •  Logistic Regression',
             fontsize=14, fontweight='bold', color='#2c3e50', y=1.01)

plt.tight_layout()
plt.savefig('model_comparison_infographic.png', dpi=150, bbox_inches='tight',
            facecolor='#f8f9fa')
plt.show()
print("Infographic saved as model_comparison_infographic.png")
