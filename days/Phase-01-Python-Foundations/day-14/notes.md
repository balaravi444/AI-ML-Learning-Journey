# Day 14 — Modules, Packages & Virtual Environments 🚀

**Date:** 02 June 2026
**Time Spent:** (4 hours)
**Resource Used:** [GeeksforGeeks Python Modules](https://www.geeksforgeeks.org/python-modules/)

---

## 📚 Topics Covered

- What are Modules
- Built-in Modules
- Creating Your Own Modules
- Importing Modules (3 ways)
- Module level variables
- Packages — folders of modules
- Virtual Environments

---

## 🔑 Core Concepts

| Concept | What It Means |
|---------|--------------|
| Module | A single Python file with reusable code |
| Package | A folder containing multiple modules |
| Library | A collection of packages |
| `import` | Brings a module into your program |
| `from x import y` | Imports specific thing from module |
| `import x as y` | Imports with a nickname |
| Virtual Environment | Isolated Python workspace per project |

---

## 3 Ways to Import

```python
# Way 1 — import whole module
import math
print(math.pi)

# Way 2 — import specific thing
from math import pi, sqrt
print(pi)

# Way 3 — import with nickname
import numpy as np    # used in ALL AI/ML code!
import pandas as pd
```

---

## 💻 Programs Practiced

| # | Program |
|---|---------|
| 1 | Built-in math module |
| 2 | random module — randint, choice, shuffle |
| 3 | ai_utils.py — custom module |
| 4 | main.py — importing custom module |

---

## 💡 Key Lessons

- Modules split big projects into manageable files
- Every `.py` file is a module
- `import numpy as np` — `np` saves typing time
- You can create your own modules just like NumPy!
- Virtual environments keep project dependencies separate

---

## 🔥 Real AI/ML Functions Built Today

### calculate_accuracy()
```python
def calculate_accuracy(correct, total):
    return (correct / total) * 100
# Same formula used in every ML model evaluation!
```

### normalize()
```python
def normalize(number, min_val, max_val):
    return (number - min_val) / (max_val - min_val)
# Min-Max Normalization — used in EVERY ML project!
```

### is_outlier()
```python
def is_outlier(value, numbers):
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std = math.sqrt(variance)
    return abs(value - mean) > 2 * std
# Outlier detection — used before training every ML model!
```

---

## 🗂️ How Real AI/ML Projects Are Structured

my_ml_project/
├── main.py              ← runs everything
├── data_loader.py       ← loads datasets
├── preprocessor.py      ← cleans data
├── model.py             ← ML model code
├── evaluator.py         ← accuracy metrics
└── utils.py             ← helper functions

This is exactly how Scikit-learn is structured internally!

---

## 🔗 How This Connects to AI/ML

```python
# When you write this:
from sklearn.linear_model import LinearRegression

# You're importing from sklearn's module structure:
# sklearn/
# ├── linear_model.py  ← LinearRegression lives here!
# ├── tree.py          ← DecisionTree lives here!
# └── metrics.py       ← accuracy_score lives here!
```

---

## 💎 Important Realization

Every AI/ML library is just a collection of Python modules!
NumPy, Pandas, Scikit-learn — all built exactly like our
`ai_utils.py` — just much bigger!

---

## 🤔 Challenges Faced

- Understanding difference between module and package
- Remembering to match parameter names consistently
- Understanding normalization formula

---

## ✅ Day 14 Wins

- ✅ Used built-in modules — math, random
- ✅ Created own module — ai_utils.py
- ✅ Built real ML utility functions from scratch
- ✅ Understood how Scikit-learn is structured
- ✅ Learned Min-Max Normalization used in real ML!

---

## 🎯 Next Goal

- Regular Expressions (Regex)
- Pattern matching in text
- Used heavily in NLP and data cleaning!

---

*Day 14 complete* 🔥
