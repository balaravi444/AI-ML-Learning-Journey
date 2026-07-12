# Day 05 — Advanced Functions, Lambda & Recursion 🚀

**Date:** 23 May 2026
**Time Spent:** (3 hours)
**Resource Used:** [CS50P Harvard](https://cs50.harvard.edu/python/) | [GeeksforGeeks](https://www.geeksforgeeks.org/python-programming-language/)

---

## 📚 Topics Covered

- Advanced Functions & Default Parameters
- Lambda Functions
- Recursion
- Base Case — why it's critical
- Introduction to Decorators

---

## 🔑 Core Concepts

### Default Parameters
```python
def greet(name="Bala"):
    print(f"Hello, {name}")

greet()        # Hello, Bala
greet("Ravi")  # Hello, Ravi
```

**So what? Why does this matter?**
Default parameters are used in EVERY ML library!
```python
# Scikit-learn uses default parameters everywhere!
model = RandomForestClassifier(
    n_estimators=100,    # default!
    max_depth=None,      # default!
    random_state=42      # you override this!
)
```
When you don't pass a value — the default is used.
This is why ML models work even without specifying every parameter! 🔥

---

### Lambda Functions
```python
square = lambda x: x * x
print(square(5))  # 25
```

**So what? Why does this matter?**
Lambda functions are used heavily in data processing!
```python
# Pandas uses lambda for applying functions to columns!
import pandas as pd
df['marks_normalized'] = df['marks'].apply(lambda x: x / 100)

# Sorting with custom logic
students = [{"name": "Bala", "marks": 85}, {"name": "Ravi", "marks": 92}]
sorted_students = sorted(students, key=lambda x: x['marks'])
```
Without understanding lambda — you can't read most data science code! 🤯

---

### Recursion
```python
def factorial(n):
    if n == 1:        # base case — CRITICAL!
        return 1
    return n * factorial(n - 1)
```

**So what? Why does this matter?**
Base case is essential to stop recursion — without it Python
hits its default recursion limit of ~1000 calls and raises
a `RecursionError`.

```python
# Without base case:
def factorial(n):
    return n * factorial(n - 1)  # infinite loop!
# → RecursionError: maximum recursion depth exceeded

# This is why iterative solutions are sometimes preferred
# in production ML code over recursive ones!
```

Real ML use — Decision Trees use recursion internally:

Split data → Split again → Split again → Stop (base case!)
The "stop" condition is the base case! 🔥

---

## 💻 Programs Practiced

| # | Problem | Key Concept |
|---|---------|-------------|
| 1 | Function with Multiple Parameters | Parameter handling |
| 2 | Default Parameter Function | Default values |
| 3 | Lambda Square Function | Anonymous functions |
| 4 | Recursive Countdown | Base case |
| 5 | Recursive Sum of First N Numbers | Recursion pattern |
| 6 | Recursive Factorial | Classic recursion |

---

## 💡 Key Lessons — With Real Depth

**1. Functions make code reusable**
→ So what? In ML, you write `model.fit()` once — it works on
any dataset. That's reusability in action!

**2. Recursion is a function calling itself**
→ So what? Every Decision Tree, file system traversal, and
JSON parsing uses recursion. Understanding it = understanding
how these systems work internally!

**3. Base case is essential to stop recursion**
→ So what? Without base case → RecursionError after ~1000 calls.
Python's default recursion limit is 1000. You can change it with
`sys.setrecursionlimit()` but it's better to use iteration for
deep recursion in production code!

**4. Lambda functions simplify small operations**
→ So what? Pandas `.apply()`, `.map()`, `.filter()` all use
lambda functions. You can't do data preprocessing without them!

---

## 🔗 How This Connects to AI/ML

```python
# 1. Lambda in Pandas data preprocessing
df['normalized'] = df['marks'].apply(lambda x: (x - df['marks'].min()) /
                                     (df['marks'].max() - df['marks'].min()))

# 2. Default parameters in ML models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()  # all defaults!
model = RandomForestClassifier(n_estimators=200)  # override one!

# 3. Recursion in Decision Trees
def split_node(data, depth=0):
    if depth == max_depth:  # base case!
        return leaf_node
    left, right = best_split(data)
    return split_node(left, depth+1), split_node(right, depth+1)
```

---

## ❌ Mistakes I Made & How I Fixed Them

**Mistake 1:** Forgot the base case in recursion
```python
# What I wrote first:
def countdown(n):
    print(n)
    countdown(n - 1)  # no base case → infinite loop!

# Fixed:
def countdown(n):
    if n == 0:        # base case added!
        return
    print(n)
    countdown(n - 1)
```

**Mistake 2:** Confused lambda syntax
```python
# Wrong:
square = lambda(x): x * x

# Correct:
square = lambda x: x * x  # no parentheses around parameter!
```

---

## 💎 Important Realization

Recursion and lambda aren't just Python concepts —
they are fundamental computer science ideas used in:
- Decision Trees (recursion)
- Data preprocessing (lambda)
- Neural network architectures (recursion)
- Functional programming in ML pipelines (lambda)

---

## 🎯 Next Goal

- Strings, Lists, Tuples, Dictionaries
- Data Structures — the foundation of all data manipulation
- Direct connection to Pandas DataFrames!

---

*Day 05 complete* 🔥

