"""
Day 41 — Matplotlib: Data Visualization
Topic: Histograms and Scatter Plots
Date: 28 June 2026
Author: Bala Ravi

Histograms → distribution of one feature
Scatter → relationship between two features
Both CRITICAL for ML data exploration!
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def histogram_salary_distribution() -> None:
    """
    Histogram of salary distribution.
    ALWAYS check distributions before ML!
    """
    np.random.seed(42)
    salaries = np.concatenate([
        np.random.normal(18, 5, 150),
        np.random.normal(32, 4, 50)
    ])
    salaries = np.clip(salaries, 5, 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')

    for ax in axes:
        ax.set_facecolor('#1e293b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    # Left — raw distribution
    axes[0].hist(salaries, bins=25,
                 color='#10b981', edgecolor='#0f172a',
                 alpha=0.8)
    axes[0].axvline(np.mean(salaries), color='#ef4444',
                    linestyle='--', linewidth=2,
                    label=f'Mean: ₹{np.mean(salaries):.1f}L')
    axes[0].axvline(np.median(salaries),
                    color='#f59e0b',
                    linestyle='--', linewidth=2,
                    label=f'Median: ₹{np.median(salaries):.1f}L')
    axes[0].set_title('Salary Distribution',
                      color='white', fontsize=13)
    axes[0].set_xlabel('Salary (₹ LPA)',
                       color='#94a3b8')
    axes[0].set_ylabel('Count', color='#94a3b8')
    axes[0].tick_params(colors='#94a3b8')
    axes[0].legend(facecolor='#334155',
                   labelcolor='white')
    axes[0].grid(True, alpha=0.2, color='#64748b')

    # Right — by experience level
    fresher = salaries[:80]
    mid = salaries[80:150]
    senior = salaries[150:]

    axes[1].hist(fresher, bins=15, alpha=0.6,
                 color='#10b981', label='Fresher (0-2yr)')
    axes[1].hist(mid, bins=15, alpha=0.6,
                 color='#3b82f6', label='Mid (2-5yr)')
    axes[1].hist(senior, bins=15, alpha=0.6,
                 color='#f59e0b', label='Senior (5+yr)')
    axes[1].set_title('Distribution by Experience',
                      color='white', fontsize=13)
    axes[1].set_xlabel('Salary (₹ LPA)',
                       color='#94a3b8')
    axes[1].set_ylabel('Count', color='#94a3b8')
    axes[1].tick_params(colors='#94a3b8')
    axes[1].legend(facecolor='#334155',
                   labelcolor='white')
    axes[1].grid(True, alpha=0.2, color='#64748b')

    fig.suptitle('Salary Distribution Analysis',
                 color='white', fontsize=15, y=1.02)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/salary_distribution.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


def scatter_experience_salary() -> None:
    """
    Scatter plot of experience vs salary.
    Shows correlation — key for feature selection!
    """
    np.random.seed(42)
    n = 200
    experience = np.random.randint(0, 12, n)
    salary = (8 + experience * 1.8 +
               np.random.normal(0, 3, n))
    salary = np.clip(salary, 5, 55)

    cities = np.random.choice(
        ['Bangalore', 'Mumbai', 'Delhi',
         'Hyderabad', 'Pune'], n)

    city_colors = {
        'Bangalore': '#10b981',
        'Mumbai': '#3b82f6',
        'Delhi': '#f59e0b',
        'Hyderabad': '#ef4444',
        'Pune': '#8b5cf6'
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    for city, color in city_colors.items():
        mask = cities == city
        ax.scatter(experience[mask], salary[mask],
                   color=color, label=city,
                   alpha=0.7, s=60,
                   edgecolors='white',
                   linewidths=0.3)

    # Add trend line
    z = np.polyfit(experience, salary, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 12, 100)
    ax.plot(x_line, p(x_line), color='white',
            linewidth=2, linestyle='--',
            alpha=0.8, label='Trend')

    corr = np.corrcoef(experience, salary)[0, 1]
    ax.text(0.05, 0.95,
            f'Correlation: {corr:.2f}',
            transform=ax.transAxes,
            color='white', fontsize=12,
            bbox=dict(boxstyle='round',
                      facecolor='#334155',
                      alpha=0.8))

    ax.set_title('Experience vs Salary by City',
                 color='white', fontsize=15, pad=15)
    ax.set_xlabel('Experience (years)',
                  color='#94a3b8')
    ax.set_ylabel('Salary (₹ LPA)', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.legend(facecolor='#334155',
              labelcolor='white', fontsize=9)
    ax.grid(True, alpha=0.2, color='#64748b')

    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/experience_vs_salary.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


def ml_training_curve() -> None:
    """
    Plot ML model training curves.
    Used in EVERY deep learning project!
    """
    np.random.seed(42)
    epochs = np.arange(1, 51)

    train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.15
    val_loss = (2.5 * np.exp(-0.07 * epochs) +
                0.25 + np.random.normal(0, 0.05, 50))
    train_acc = (1 - np.exp(-0.1 * epochs)) * 95
    val_acc = ((1 - np.exp(-0.09 * epochs)) * 92 +
                np.random.normal(0, 0.5, 50))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')

    for ax in axes:
        ax.set_facecolor('#1e293b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    # Loss curves
    axes[0].plot(epochs, train_loss,
                 color='#10b981', linewidth=2,
                 label='Training Loss')
    axes[0].plot(epochs, val_loss,
                 color='#ef4444', linewidth=2,
                 linestyle='--', label='Validation Loss')
    axes[0].set_title('Model Loss',
                      color='white', fontsize=13)
    axes[0].set_xlabel('Epoch', color='#94a3b8')
    axes[0].set_ylabel('Loss', color='#94a3b8')
    axes[0].tick_params(colors='#94a3b8')
    axes[0].legend(facecolor='#334155',
                   labelcolor='white')
    axes[0].grid(True, alpha=0.2, color='#64748b')

    # Accuracy curves
    axes[1].plot(epochs, train_acc,
                 color='#10b981', linewidth=2,
                 label='Training Accuracy')
    axes[1].plot(epochs, val_acc,
                 color='#3b82f6', linewidth=2,
                 linestyle='--',
                 label='Validation Accuracy')
    axes[1].set_title('Model Accuracy',
                      color='white', fontsize=13)
    axes[1].set_xlabel('Epoch', color='#94a3b8')
    axes[1].set_ylabel('Accuracy (%)',
                       color='#94a3b8')
    axes[1].tick_params(colors='#94a3b8')
    axes[1].legend(facecolor='#334155',
                   labelcolor='white')
    axes[1].grid(True, alpha=0.2, color='#64748b')
    axes[1].set_ylim(0, 100)

    fig.suptitle('ML Model Training Progress',
                 color='white', fontsize=15, y=1.02)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/training_curves.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    print("=== Histogram & Scatter Charts ===\n")
    histogram_salary_distribution()
    scatter_experience_salary()
    ml_training_curve()
    print("\nAll charts saved to 'charts/' folder!")
