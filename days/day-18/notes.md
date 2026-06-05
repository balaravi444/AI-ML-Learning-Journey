# Day 18 — Magic Methods & Operator Overloading 🪄

**Date:** 05 June 2026
**Time Spent:** (add your hours)
**Resource Used:** Python Official Docs — Data Model

---

## 📚 Topics Covered

- What are Magic Methods (Dunder Methods)
- `__init__` — object creation
- `__str__` — print control
- `__add__`, `__sub__`, `__mul__` — arithmetic operators
- `__gt__`, `__lt__`, `__eq__` — comparison operators
- `__len__` — len() support
- Operator Overloading concept
- How ML libraries use magic methods

---

## 🔑 Core Concept

Magic Methods = special methods with **double underscores** on both sides

Also called **Dunder Methods** — **d**ouble **under**score!

When you write `2 + 3` — Python secretly runs `(2).__add__(3)`

**You can define these methods in YOUR classes too!**
That's called **Operator Overloading!** 🪄

---

## 📋 Magic Methods Table

| Magic Method | Operator | When Called |
|-------------|----------|-------------|
| `__init__` | — | object created |
| `__str__` | — | print(object) |
| `__add__` | `+` | obj1 + obj2 |
| `__sub__` | `-` | obj1 - obj2 |
| `__mul__` | `*` | obj1 * obj2 |
| `__gt__` | `>` | obj1 > obj2 |
| `__lt__` | `<` | obj1 < obj2 |
| `__eq__` | `==` | obj1 == obj2 |
| `__len__` | — | len(obj) |

---

## 💡 Easy Memory Tricks

```
gt  →  Greater Than  →  >
lt  →  Less Than     →  <
eq  →  EQual         →  ==
```

```python
s1  +  s2
↓       ↓
self   other       # left = self, right = other — always!
```

---

## 🔥 Key Code Patterns

### Without __str__ — ugly output:
```python
print(s1)
# <__main__.Student object at 0x000001A2>  ❌
```

### With __str__ — clean output:
```python
def __str__(self):
    return f"Student: {self.name}, Marks: {self.marks}"

print(s1)
# Student: Bala, Marks: 85  ✅
```

### Operator Overloading pattern:
```python
def __add__(self, other):
    return self.price + other.price
#          ↑ left side    ↑ right side
#          s1 + s2
```

---

## 🔗 How This Connects to ML

```python
# NumPy uses magic methods internally!
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)    # [5, 7, 9]
# secretly calls a.__add__(b)!

# Same with Pandas, PyTorch, TensorFlow!
# All built using magic methods 🤯
```

---

## ✅ Day 18 Wins

- ✅ Understood magic methods concept
- ✅ Built Student class with operators
- ✅ Built complete Phone class from memory
- ✅ Used arithmetic + comparison magic methods
- ✅ Zero logic mistakes in challenges!

---

## 🎯 Next Goal

- Day 19 — Iterators & Generators
- `__iter__`, `__next__` magic methods
- `yield` keyword
- Memory efficient loops!

---

*Day 18 complete* 🔥
