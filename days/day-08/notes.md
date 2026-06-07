# Day 08 — Python Collections Module 🚀

**Date:** 26 May 2026
**Time Spent:** (2 hours)
**Resource Used:** [CS50P Harvard](https://cs50.harvard.edu/python/) | [GeeksforGeeks](https://www.geeksforgeeks.org/python-collections-module/)

---

## 📚 Topics Covered

- Counter
- defaultdict
- deque
- namedtuple
- When to use each one

---

## 🔑 Core Concepts

### Counter
```python
from collections import Counter

text = "machine learning"
freq = Counter(text)
print(freq)
# Counter({'a': 2, 'n': 2, 'i': 2, ...})

# Most common elements
print(freq.most_common(3))
# [('a', 2), ('n', 2), ('i', 2)]
```

**So what? Why does this matter?**
Counter is used in NLP for word frequency analysis!
```python
# Real NLP use case:
words = ["AI", "ML", "AI", "Python", "AI", "ML"]
word_freq = Counter(words)
print(word_freq.most_common(2))
# [('AI', 3), ('ML', 2)]

# This is EXACTLY how search engines rank keywords!
# This is how spam filters count suspicious words!
# This is how sentiment analyzers find common words!
```
Without Counter you'd need loops — Counter does it in one line! 🔥

---

### defaultdict
```python
from collections import defaultdict

# Normal dict crashes on missing key:
normal = {}
normal['missing_key'] += 1  # KeyError! ❌

# defaultdict handles it automatically:
d = defaultdict(int)
d['missing_key'] += 1  # works! ✅
print(d['missing_key'])  # 1
```

**So what? Why does this matter?**
defaultdict prevents KeyError crashes in data processing!
```python
# Real ML use case — grouping data:
from collections import defaultdict

student_marks = defaultdict(list)
data = [("Bala", 85), ("Ravi", 92), ("Bala", 78), ("Ravi", 88)]

for name, mark in data:
    student_marks[name].append(mark)

print(dict(student_marks))
# {'Bala': [85, 78], 'Ravi': [92, 88]}

# This pattern is used when building:
# - Word indexes for NLP
# - Group-by operations before Pandas
# - Building adjacency lists for graph algorithms
```

---

### deque
```python
from collections import deque

# Normal list — slow for left operations!
lst = [1, 2, 3]
lst.insert(0, 0)    # O(n) — slow! ❌

# deque — fast for both ends!
d = deque([1, 2, 3])
d.appendleft(0)     # O(1) — fast! ✅
d.popleft()         # O(1) — fast! ✅
```

**So what? Why does this matter?**
deque is used for efficient queue/sliding window operations!
```python
# Real ML use case — sliding window for time series:
from collections import deque

# Keep last 5 stock prices
window = deque(maxlen=5)
prices = [100, 102, 98, 105, 103, 107, 99]

for price in prices:
    window.append(price)
    if len(window) == 5:
        print(f"Window average: {sum(window)/5:.1f}")

# This is how moving averages work in stock prediction!
# This is how sliding window works in CNN image processing!
```

---

### namedtuple
```python
from collections import namedtuple

# Normal tuple — confusing!
student = (85, 20, "AI")
print(student[0])   # what is index 0? 😕

# namedtuple — clear!
Student = namedtuple("Student", ["marks", "age", "course"])
bala = Student(marks=85, age=20, course="AI")
print(bala.marks)   # clear! ✅
print(bala.age)     # clear! ✅
```

**So what? Why does this matter?**
namedtuple makes data structures readable and memory efficient!
```python
# Real ML use case — storing model results:
ModelResult = namedtuple("ModelResult", ["accuracy", "precision", "recall", "f1"])

result = ModelResult(
    accuracy=0.95,
    precision=0.93,
    recall=0.91,
    f1=0.92
)

print(f"Model Accuracy: {result.accuracy}")
print(f"F1 Score: {result.f1}")

# namedtuple uses LESS memory than a class
# Perfect for storing large numbers of records!
```

---

## 💻 Programs Practiced

| # | Problem | Key Concept |
|---|---------|-------------|
| 1 | Character frequency Counter | Counter.most_common() |
| 2 | Auto default values | defaultdict(int) |
| 3 | Queue operations | deque appendleft/popleft |
| 4 | Structured data | namedtuple |

---

## 💡 Key Lessons — With Real Depth

**1. Counter makes frequency counting simple**
→ So what? Word frequency is the foundation of NLP!
TF-IDF (used in search engines) starts with word counts.
Bag of Words model (used in spam detection) uses Counter logic!

**2. defaultdict avoids KeyError**
→ So what? Real datasets have missing values and unexpected keys.
defaultdict prevents crashes when processing messy real-world data.
This is especially important when building word indexes for NLP!

**3. deque is better than lists for queue operations**
→ So what? `list.insert(0, x)` is O(n) — slow for large data!
`deque.appendleft(x)` is O(1) — always fast!
In ML sliding window operations on time series data — speed matters!

**4. namedtuple makes data more readable**
→ So what? `result[0]` vs `result.accuracy` — which is clearer?
When storing thousands of model evaluation results — namedtuple
uses less memory than a class and is more readable than a tuple!

---

## 🔗 How This Connects to AI/ML

```python
# 1. Counter — NLP word frequency
from collections import Counter
words = text.split()
word_freq = Counter(words)
top_10_words = word_freq.most_common(10)
# Used in: search engines, spam filters, sentiment analysis!

# 2. defaultdict — building word index
from collections import defaultdict
word_index = defaultdict(list)
for i, word in enumerate(words):
    word_index[word].append(i)
# Used in: NLP tokenization, inverted indexes!

# 3. deque — sliding window
from collections import deque
window = deque(maxlen=window_size)
# Used in: time series analysis, CNN feature maps!

# 4. namedtuple — model metrics
from collections import namedtuple
Metrics = namedtuple("Metrics", ["accuracy", "f1", "recall"])
# Used in: storing and comparing model results!
```

---

## ❌ Mistakes I Made & How I Fixed Them

**Mistake 1:** Used normal dict instead of defaultdict
```python
# Caused KeyError:
word_count = {}
word_count['new_word'] += 1  # ❌ KeyError!

# Fixed with defaultdict:
word_count = defaultdict(int)
word_count['new_word'] += 1  # ✅ works!
```

**Mistake 2:** Used list for queue operations
```python
# Slow for large data:
queue = []
queue.insert(0, item)  # ❌ O(n) — slow!

# Fixed with deque:
queue = deque()
queue.appendleft(item)  # ✅ O(1) — fast!
```

---

## 📊 When to Use Which Collection

| Situation | Use |
|-----------|-----|
| Count frequency of items | Counter |
| Need default values for missing keys | defaultdict |
| Need fast operations on both ends | deque |
| Need readable tuple with field names | namedtuple |
| Simple key-value storage | dict |
| Ordered list of items | list |

---

## 💎 Important Realization

Python's collections module exists because the built-in
data structures (list, dict, tuple) are not always the
most efficient choice!

Choosing the RIGHT data structure:
- Makes code faster ⚡
- Makes code cleaner 📝
- Prevents bugs 🐛
- Handles edge cases automatically ✅

This is why senior developers think about DATA STRUCTURES
before writing code — not after! 🔥

---

## 🎯 Next Goal

- Exception Handling
- Writing code that handles unexpected situations gracefully
- Direct connection to production ML code reliability!

---

*Day 08 complete* 🔥
