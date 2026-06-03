# Day 15 — Regular Expressions (Regex) 🚀

**Date:** 03 June 2026
**Time Spent:** (3 hours)
**Resource Used:** [GeeksforGeeks Regex](https://www.geeksforgeeks.org/regular-expression-python-examples/)

---

## 📚 Topics Covered

- What is Regex
- The `re` module
- Basic patterns
- Common regex functions
- Real world regex examples
- Regex in NLP and data cleaning

---

## 🔑 Core Concepts

| Pattern | Meaning | Example |
|---------|---------|---------|
| `\d` | Any digit 0-9 | `\d+` matches "123" |
| `\w` | Any word character | `\w+` matches "hello" |
| `\s` | Any whitespace | `\s+` matches spaces |
| `.` | Any character | `h.t` matches "hat", "hot" |
| `*` | 0 or more | `ab*` matches "a", "ab", "abb" |
| `+` | 1 or more | `ab+` matches "ab", "abb" |
| `?` | 0 or 1 | `ab?` matches "a", "ab" |
| `^` | Start of string | `^Hello` |
| `$` | End of string | `world$` |
| `[]` | Character set | `[aeiou]` matches vowels |

---

## 4 Main Regex Functions

```python
import re

# 1. re.match() — checks at beginning of string
# 2. re.search() — searches anywhere in string
# 3. re.findall() — finds ALL matches
# 4. re.sub() — replaces matches
```

---

## 💻 Programs Practiced

| # | Program |
|---|---------|
| 1 | Basic regex patterns |
| 2 | Email validator |
| 3 | Phone number extractor |
| 4 | Data cleaner |
| 5 | NLP text preprocessor |

---

## 💡 Key Lessons

- Regex finds patterns in text without loops
- `re.findall()` is most commonly used in AI/ML
- Regex is used to clean data before ML training
- Email, phone, URL validation all use regex
- NLP preprocessing heavily uses regex

---

## 🔗 How This Connects to AI/ML

```python
import re

# Real NLP preprocessing!
text = "Hello World! This is Day 15. #Python #AI"

# Remove special characters
clean = re.sub(r'[^a-zA-Z\s]', '', text)
print(clean)  # "Hello World This is Day  Python AI"

# Extract hashtags
hashtags = re.findall(r'#\w+', text)
print(hashtags)  # ['#Python', '#AI']

# This is EXACTLY what happens before training NLP models!
```

---

## 💎 Important Realization

Before training ANY NLP model:
1. Remove special characters — regex
2. Remove numbers — regex
3. Extract specific patterns — regex
4. Clean messy text data — regex

Regex is the foundation of NLP data cleaning! 🔥

---

## 🤔 Challenges Faced

- Remembering regex patterns
- Understanding when to use match vs search
- Building complex patterns

---

## ✅ Day 15 Wins

- ✅ Understood regex patterns
- ✅ Built email validator
- ✅ Built phone number extractor
- ✅ Built NLP text preprocessor
- ✅ Connected regex to real NLP preprocessing

---

## 🎯 Next Goal

- Working with JSON and CSV files
- Loading real datasets
- Preparing for Data Science phase!

---

*Day 15 complete* 🔥
