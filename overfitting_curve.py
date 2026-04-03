import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


def plot_overfitting_curve(X_train, y_train, X_test, y_test, max_depths):
    """
    Train Decision Trees at each max_depth, plot train vs test accuracy,
    and identify the optimal depth.

    Parameters:
        X_train, y_train : training data
        X_test,  y_test  : test data
        max_depths       : list of int depths to evaluate
    """
    train_accs = []
    test_accs  = []

    for depth in max_depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
        test_accs.append(accuracy_score(y_test,  dt.predict(X_test)))

    optimal_depth = max_depths[np.argmax(test_accs)]

    # Print table
    print(f"{'Depth':<8} {'Train Acc':>10} {'Test Acc':>10}")
    print("─" * 30)
    for d, tr, te in zip(max_depths, train_accs, test_accs):
        marker = " ← optimal" if d == optimal_depth else ""
        print(f"{d:<8} {tr*100:>9.2f}% {te*100:>9.2f}%{marker}")

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(max_depths, [a*100 for a in train_accs], 'o-', color='steelblue',
             linewidth=2, markersize=6, label='Training Accuracy')
    plt.plot(max_depths, [a*100 for a in test_accs], 's-', color='crimson',
             linewidth=2, markersize=6, label='Test Accuracy')
    plt.axvline(optimal_depth, color='green', linestyle='--', linewidth=1.5,
                label=f'Optimal depth = {optimal_depth}')

    # Shade regions
    plt.axvspan(1, optimal_depth - 0.5, alpha=0.06, color='red', label='Underfitting')
    plt.axvspan(optimal_depth + 0.5, max(max_depths), alpha=0.06,
                color='orange', label='Overfitting')

    plt.title('Decision Tree — Overfitting Curve\n(Train vs Test Accuracy by max_depth)',
              fontweight='bold')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('overfitting_curve.png', dpi=150)
    plt.show()

    print(f"\nOptimal max_depth = {optimal_depth}  (highest test accuracy)")
    return optimal_depth


# Run the function
depths = list(range(1, 21))
optimal = plot_overfitting_curve(X_train, y_train, X_test, y_test, depths)
