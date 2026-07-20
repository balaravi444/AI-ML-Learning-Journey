# Day 59 — Software Bug Priority Predictor 🚀
# Project Day 1 of 4 — Data + Feature Engineering

**Date:** 16 July 2026
**Time Spent:** ( 2 hours)
**Project:** Software Bug Priority Predictor
**Problem:** Predict GitHub issue priority before engineers triage it

---

## 📚 What I'm Building Today

- Generate realistic GitHub issues dataset
- Understand the features that determine bug priority
- Engineer text + metadata features
- EDA on the dataset
- Prepare ML-ready feature matrix

---

## 🔑 The Real Problem
Every software company gets 50-200 GitHub issues per day.
Someone has to read each one and decide:

Critical → production down, fix NOW
High → major feature broken, fix this sprint
Medium → annoying bug, fix next sprint
Low → minor issue, fix when possible

Manual triage = 2-4 hours of senior engineer time DAILY.
Wrong triage = critical bug sits ignored = outage.

Our ML model reads the issue and predicts priority
in under 100ms. Engineers triage 10x faster.

This is used at Microsoft, Google, Atlassian.
---

## 🔑 What Features Predict Bug Priority?
Text features (from title + description):
→ Words like "crash", "down", "error" → Critical
→ Words like "slow", "broken", "fails" → High
→ Words like "wrong", "incorrect" → Medium
→ Words like "typo", "cosmetic" → Low

Metadata features:
→ description_length (longer = more detail = higher priority often)
→ has_code_snippet (engineers who paste code are serious)
→ has_error_message (stack traces = real bug)
→ has_steps_to_reproduce (clear reporter = real issue)
→ label_count (more labels = more context)
→ reporter_is_contributor (insiders know priorities)
→ hour_filed (2am filing = production incident!)
→ word_count (too short = low quality report)
→ exclamation_count (urgency signals)
→ has_version_number (reproducible = higher priority)
---

## 🔑 Class Imbalance Problem
Real GitHub repos have imbalanced priorities:
→ Critical: 5-8% of issues
→ High: 20-25%
→ Medium: 40-45%
→ Low: 25-30%

If we ignore this:
→ Model predicts "Medium" for everything
→ Gets 42% accuracy (useless!)
→ Misses ALL critical bugs!

Solution: SMOTE (Synthetic Minority Oversampling)
→ Creates synthetic Critical + High examples
→ Balanced training → model learns all classes!

This is WHY class imbalance matters in real ML!
---

## 💎 Important Realizations

1. **Text IS the most important feature**
   The words in a bug report tell you everything.
   TF-IDF converts words → numbers → ML can learn!

2. **Metadata amplifies text signals**
   "crash" in title + filed at 2am + reporter is contributor
   = almost certainly Critical!
   Features work TOGETHER not independently.

3. **Class imbalance kills models silently**
   Without SMOTE: 42% accuracy but misses critical bugs.
   With SMOTE: learns all classes properly.
   Always check class distribution first!

4. **This is NLP + ML combined**
   Text features (TF-IDF) + metadata features (numerical)
   combined in one feature matrix.
   Preview of Day 63-66 NLP week!

---

## 🎯 Next Goal (Day 60)

- Build the full ML pipeline
- TF-IDF + Random Forest
- Handle class imbalance with SMOTE
- Evaluate with classification report

---

*Day 59 complete — Data + Features ready! 🔥*



Now we're building it. 🔥
