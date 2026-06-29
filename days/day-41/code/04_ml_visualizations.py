"""
Day 41 — Matplotlib: Data Visualization
Topic: ML-specific Visualizations
Date: 28 June 2026
Author: Bala Ravi

These are the charts every ML engineer makes!
Training curves, confusion matrix, ROC curve,
feature importance — standard ML toolkit!
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_confusion_matrix() -> None:
    """
    Confusion matrix heatmap.
    Used to evaluate classification models!
    """
    np.random.seed(42)
    classes = ['Data Scientist', 'ML Engineer',
               'Data Analyst', 'AI Engineer']
    n = len(classes)

    cm = np.array([[45, 3, 2, 1],
                   [4, 38, 1, 2],
                   [2, 1, 52, 0],
                   [1, 3, 0, 41]])

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    im = ax.imshow(cm, cmap='Greens')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, color='#94a3b8',
                       rotation=30, ha='right')
    ax.set_yticklabels(classes, color='#94a3b8')

    for i in range(n):
        for j in range(n):
            text_color = ('black' if cm[i, j] > cm.max()/2
                          else 'white')
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color=text_color, fontsize=14,
                    fontweight='bold')

    accuracy = cm.diagonal().sum() / cm.sum() * 100
    ax.set_title(
        f'Confusion Matrix — Accuracy: {accuracy:.1f}%',
        color='white', fontsize=13, pad=15)
    ax.set_xlabel('Predicted', color='#94a3b8')
    ax.set_ylabel('Actual', color='#94a3b8')

    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


def plot_feature_importance() -> None:
    """
    Feature importance bar chart.
    Used after training tree-based models!
    """
    features = ['Experience', 'City', 'Skills Count',
                'Company Size', 'Remote', 'Rating',
                'Degree', 'Age']
    importance = [0.35, 0.22, 0.15, 0.10,
                  0.08, 0.05, 0.03, 0.02]

    sorted_idx = np.argsort(importance)
    features_sorted = [features[i] for i in sorted_idx]
    importance_sorted = [importance[i]
                         for i in sorted_idx]

    colors = ['#10b981' if imp >= 0.20 else
              '#3b82f6' if imp >= 0.10 else
              '#64748b'
              for imp in importance_sorted]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.barh(features_sorted, importance_sorted,
                   color=colors, edgecolor='#0f172a',
                   height=0.6)

    for bar, imp in zip(bars, importance_sorted):
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height()/2,
                f'{imp:.0%}', va='center',
                color='white', fontweight='bold',
                fontsize=10)

    ax.set_title('Feature Importance — Salary Predictor',
                 color='white', fontsize=13, pad=15)
    ax.set_xlabel('Importance Score',
                  color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.set_xlim(0, 0.45)
    ax.grid(True, axis='x', alpha=0.2,
            color='#64748b')

    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


def plot_roc_curve() -> None:
    """
    ROC Curve — model evaluation visualization!
    Used for binary classification models.
    """
    np.random.seed(42)

    def generate_roc(auc_target):
        n = 100
        x = np.linspace(0, 1, n)
        y = np.power(x, 1 / (auc_target / (1 - auc_target)))
        y = np.clip(y + np.random.normal(
            0, 0.02, n), 0, 1)
        y = np.sort(y)
        return x, y

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    models = [
        ('Random Forest', 0.94, '#10b981'),
        ('Logistic Reg', 0.82, '#3b82f6'),
        ('Decision Tree', 0.78, '#f59e0b')
    ]

    for name, auc, color in models:
        fpr, tpr = generate_roc(auc)
        ax.plot(fpr, tpr, color=color,
                linewidth=2.5,
                label=f'{name} (AUC={auc:.2f})')

    ax.plot([0, 1], [0, 1], 'white',
            linewidth=1.5, linestyle='--',
            alpha=0.5, label='Random (AUC=0.50)')

    ax.fill_between([0, 1], [0, 1], alpha=0.05,
                    color='white')

    ax.set_title('ROC Curve — Salary Prediction Models',
                 color='white', fontsize=13, pad=15)
    ax.set_xlabel('False Positive Rate',
                  color='#94a3b8')
    ax.set_ylabel('True Positive Rate',
                  color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.legend(facecolor='#334155',
              labelcolor='white', fontsize=10)
    ax.grid(True, alpha=0.2, color='#64748b')

    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/roc_curve.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    print("=== ML Visualization Charts ===\n")
    plot_confusion_matrix()
    plot_feature_importance()
    plot_roc_curve()
    print(f"\n🤖 All ML charts saved to 'charts/'!")
    print(f"   These are standard charts used in")
    print(f"   EVERY ML project! 🔥")
