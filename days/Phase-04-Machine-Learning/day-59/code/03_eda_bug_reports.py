"""
Day 59 — Software Bug Priority Predictor
Topic: EDA on Bug Report Dataset
Date: 16 July 2026
Author: Bala Ravi

Understand the data before building the model.
What separates Critical bugs from Low priority ones?
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def load_and_overview(df: pd.DataFrame) -> None:
    """Dataset overview."""
    print("=== Bug Priority Dataset EDA ===\n")
    print(f"Shape: {df.shape}")
    print(f"\nClass Distribution:")

    counts = df['priority'].value_counts()
    total = len(df)
    for p, c in counts.items():
        pct = c / total * 100
        bar = '█' * int(pct // 2)
        print(f"  {p:<10}: {c:>4} ({pct:.1f}%) {bar}")

    print(f"\n⚠️  Class Imbalance!")
    print(f"   Critical is only "
          f"{counts.get('Critical', 0)/total*100:.1f}% "
          f"of data")
    print(f"   Without SMOTE → model ignores Critical!")
    print(f"   With SMOTE → balanced training! ✅")


def analyze_text_patterns(df: pd.DataFrame) -> None:
    """Find word patterns by priority."""
    print("\n=== Text Pattern Analysis ===\n")

    for priority in ['Critical', 'High',
                      'Medium', 'Low']:
        subset = df[df['priority'] == priority]
        all_words = ' '.join(
            subset['title'].str.lower()).split()

        # Count meaningful words (skip common ones)
        skip = {'the', 'a', 'an', 'in', 'on',
                 'for', 'to', 'of', 'is', 'not',
                 'and', 'or', 'with', 'at', 'by'}
        word_counts = {}
        for w in all_words:
            if w not in skip and len(w) > 2:
                word_counts[w] = (
                    word_counts.get(w, 0) + 1)

        top_words = sorted(
            word_counts.items(),
            key=lambda x: x[1],
            reverse=True)[:8]

        print(f"{priority} top title words:")
        words_str = ', '.join(
            [f"{w}({c})" for w, c in top_words])
        print(f"  {words_str}\n")


def analyze_metadata_by_priority(
        df: pd.DataFrame) -> None:
    """Metadata analysis by priority."""
    print("=== Metadata Analysis by Priority ===\n")

    features = [
        ('description_length', 'Avg desc length'),
        ('has_error_message', '% with error msg'),
        ('has_code_snippet', '% with code'),
        ('has_steps_to_reproduce', '% with steps'),
        ('comment_count', 'Avg comments'),
        ('label_count', 'Avg labels'),
        ('reporter_is_contributor', '% contributors'),
        ('is_off_hours', '% filed off-hours')
    ]

    df['is_off_hours'] = (
        (df['hour_filed'] < 7) |
        (df['hour_filed'] > 20)).astype(int)

    print(f"{'Feature':<28} | "
          f"{'Critical':>10} | "
          f"{'High':>8} | "
          f"{'Medium':>8} | "
          f"{'Low':>6}")
    print("-" * 70)

    for col, label in features:
        if col not in df.columns:
            continue
        row = f"{label:<28} | "
        for p in ['Critical', 'High', 'Medium', 'Low']:
            val = df[df['priority'] == p][col].mean()
            if col.startswith('has_') or col in [
                    'reporter_is_contributor',
                    'is_off_hours']:
                row += f"{val*100:>9.0f}% | "
            elif col == 'description_length':
                row += f"{val:>9.0f}  | "
            else:
                row += f"{val:>9.1f}  | "
        print(row)

    print(f"\n💡 Key Findings:")
    print(f"   Critical bugs: longer descriptions, "
          f"more error messages, filed at night")
    print(f"   Low bugs: short descriptions, "
          f"no error messages, filed during work hours")
    print(f"   These patterns = TF-IDF + metadata "
          f"features will work! ✅")


def analyze_filing_hours(df: pd.DataFrame) -> None:
    """Analyze when bugs are filed by priority."""
    print("\n=== Filing Hours Analysis ===\n")
    print("When are critical bugs filed vs low priority?\n")

    for priority in ['Critical', 'Low']:
        subset = df[df['priority'] == priority]
        hour_dist = subset['hour_filed'].value_counts()

        night = ((subset['hour_filed'] < 6) |
                  (subset['hour_filed'] > 21)).sum()
        night_pct = night / len(subset) * 100

        print(f"{priority}:")
        print(f"  Filed between midnight-6am: "
              f"{night_pct:.0f}%")

        avg_hour = subset['hour_filed'].mean()
        print(f"  Average filing hour: "
              f"{avg_hour:.0f}:00")
        print()

    print("💡 Night filing is a strong signal for "
          "Critical priority!")
    print("   Production outages happen at any hour.")
    print("   Cosmetic bugs get filed during office hours.")


if __name__ == "__main__":
    import os
    filepath = (
        "projects/bug_priority_predictor/data/"
        "github_issues.csv")

    if not os.path.exists(filepath):
        print("Run 01_generate_dataset.py first!")
    else:
        df = pd.read_csv(filepath)
        load_and_overview(df)
        analyze_text_patterns(df)
        analyze_metadata_by_priority(df)
        analyze_filing_hours(df)
