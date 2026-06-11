# Day 24 — DSA: Hash Maps & Sets 🚀

**Date:** 11 June 2026
**Time Spent:** (4 hours)
**Resource Used:** [LeetCode](https://leetcode.com/) | [CS50P](https://cs50.harvard.edu/python/)

---

## 📚 Topics Covered

- What is a Hash Map
- What is a Hash Function
- Collision Handling
- Hash Set vs Hash Map
- Python dict and set internals
- Common Hash Map patterns
- Real applications in ML

---

## 🔑 What is a Hash Map?

A Hash Map stores key-value pairs with O(1) average lookup!

```python
# Python dict IS a hash map!
student = {
    "name": "Bala",    # key → value
    "age": 20,
    "course": "AI"
}

print(student["name"])  # O(1) lookup!
```

**How it works internally:**
key → hash function → index → value
"name" → hash("name") → 4829 % 8 → index 5 → "Bala"

**So what? Why does this matter?**
Hash maps give O(1) average lookup vs O(n) for arrays!
This is why Two Sum goes from O(n²) to O(n) with a hash map!
In ML — feature lookup, vocabulary indexing, caching
predictions all use hash maps internally! 🔥

---

## 🔑 Hash Function

A hash function converts a key to an array index:

```python
# Python's built-in hash function
print(hash("hello"))    # some large integer
print(hash(42))         # 42
print(hash(3.14))       # some integer

# Index = hash(key) % array_size
index = hash("name") % 8
```

Properties of a good hash function:
- Fast to compute — O(1)
- Uniform distribution — no clustering
- Deterministic — same key → same hash always

---

## 🔑 Collision Handling

Two keys can hash to the same index — called a collision!
"name" → index 5
"age"  → index 5  ← collision!
Solutions:
1. **Chaining** — store list at each index
2. **Open addressing** — find next empty slot

Python uses open addressing with pseudo-random probing!

---

## 🔑 Hash Set

A set is a hash map with only keys — no values!

```python
# Set — only unique values!
seen = set()
seen.add(1)      # O(1)
seen.add(2)      # O(1)
1 in seen        # O(1) lookup!
seen.remove(1)   # O(1)
```

**When to use Set vs Map:**
Hash Map (dict) → need key-value pairs
Hash Set (set)  → need uniqueness check only
---

## ⏱️ Time Complexity

| Operation | Average | Worst Case |
|-----------|---------|------------|
| Insert | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Lookup | O(1) | O(n) |
| Search | O(1) | O(n) |

Worst case O(n) happens when ALL keys collide!
In practice — Python's dict is almost always O(1)!

---

## 🔑 Common Hash Map Patterns

### Pattern 1 — Frequency Count
```python
def count_frequency(arr: list) -> dict:
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    return freq
```

### Pattern 2 — Two Sum
```python
def two_sum(nums: list, target: int) -> list:
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
    return []
```

### Pattern 3 — Group Anagrams
```python
def group_anagrams(words: list) -> list:
    groups = {}
    for word in words:
        key = tuple(sorted(word))
        groups.setdefault(key, []).append(word)
    return list(groups.values())
```

### Pattern 4 — Sliding Window with Hash Map
```python
def longest_unique_substring(s: str) -> int:
    seen = {}
    left = 0
    max_len = 0
    for right, char in enumerate(s):
        if char in seen and seen[char] >= left:
            left = seen[char] + 1
        seen[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len
```

---

## 💻 Programs Practiced

| # | Problem | Pattern | Complexity |
|---|---------|---------|------------|
| 1 | Frequency Counter | Hash Map | O(n) |
| 2 | Two Sum | Hash Map | O(n) |
| 3 | Valid Anagram | Hash Map | O(n) |
| 4 | Group Anagrams | Hash Map | O(n*k) |
| 5 | Longest Substring Without Repeating | Sliding Window + Map | O(n) |
| 6 | Top K Frequent Elements | Hash Map + Sort | O(n log k) |
| 7 | Word Frequency Counter | Hash Map | O(n) |

---

## 🎯 LeetCode Problems Solved

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 1 | Two Sum | Easy | Hash Map |
| 242 | Valid Anagram | Easy | Hash Map |
| 49 | Group Anagrams | Medium | Hash Map |
| 3 | Longest Substring Without Repeating | Medium | Sliding Window |
| 347 | Top K Frequent Elements | Medium | Hash Map + Heap |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Vocabulary building in NLP — hash map!
vocabulary = {}
for word in corpus:
    if word not in vocabulary:
        vocabulary[word] = len(vocabulary)
# word → index mapping used in every NLP model!

# 2. Feature lookup in ML
feature_map = {
    "age": [20, 21, 19],
    "marks": [85, 92, 78],
    "course": ["AI", "ML", "DS"]
}
# Pandas DataFrame is essentially a hash map!

# 3. Caching ML predictions
prediction_cache = {}
def predict(input_data: str) -> float:
    if input_data in prediction_cache:
        return prediction_cache[input_data]  # O(1)!
    result = model.predict(input_data)
    prediction_cache[input_data] = result
    return result

# 4. Word embeddings — hash map internally!
embeddings = {
    "king":  [0.2, 0.8, 0.3],
    "queen": [0.1, 0.9, 0.2],
    "AI":    [0.9, 0.1, 0.7]
}
# Word2Vec stores embeddings in a hash map!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Using list for lookup:**
```python
# Wrong — O(n) lookup!
seen = []
if target in seen:   # ❌ O(n) every time!
    return True
seen.append(num)

# Correct — O(1) lookup!
seen = set()
if target in seen:   # ✅ O(1) every time!
    return True
seen.add(num)
```

**Mistake 2 — KeyError on missing key:**
```python
# Wrong — crashes if key missing!
freq[word] += 1   # ❌ KeyError!

# Correct — use .get() with default!
freq[word] = freq.get(word, 0) + 1  # ✅

# Or use defaultdict!
from collections import defaultdict
freq = defaultdict(int)
freq[word] += 1   # ✅ no KeyError!
```

**Mistake 3 — Mutable key in hash map:**
```python
# Wrong — lists can't be keys!
d[[1, 2, 3]] = "value"   # ❌ TypeError!

# Correct — use tuple (immutable)!
d[(1, 2, 3)] = "value"   # ✅
```

---

## 💎 Important Realizations

1. **Hash maps are the most powerful tool in DSA interviews!**
   Most O(n²) brute force problems become O(n) with a hash map!

2. **Python's dict is one of the most optimized data structures
   ever built** — almost always O(1), handles collisions beautifully!

3. **Every NLP vocabulary is a hash map** — word → index!
   Without hash maps, NLP would be impossibly slow!

4. **The pattern is always the same:**
5. 
   - Need O(1) lookup? → Use hash map/set
   - Need frequency count? → Use hash map
   - Need uniqueness? → Use hash set
   - Need grouping? → Use hash map with list values

---

## 🎯 Next Goal

- Recursion & Backtracking
- Solving complex problems by breaking into subproblems
- Used in Decision Trees and search algorithms in ML!

---

*Day 24 complete* 🔥





