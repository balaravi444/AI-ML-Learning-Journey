# Day 19 — Decorators & Generators ⚡

**Date:** 06 June 2026
**Time Spent:** (1 hour)
**Resource Used:** Python Official Docs — Decorators & Generators

---

## 📚 Topics Covered

- Functions as arguments
- What is a Decorator
- `wrapper` function inside decorator
- `@` symbol shortcut
- Real world decorators — timer, logger
- What is a Generator
- `yield` vs `return`
- Generator with for loop
- Infinite generator
- Memory efficiency concept
- How ML uses generators

---

## 🔑 Core Concepts

### Decorator
A decorator **adds extra behaviour** to a function — without changing the original function!

Think of it like gift wrapping 🎁
- Box = original function
- Wrapping = decorator
- Wrapped box = decorated function

### Generator
A generator **gives one value at a time** — only when needed!

Think of it like borrowing books 📖
- Normal list = buying 1000 books at once 😫
- Generator = borrowing 1 book at a time ✅

---

## 🔥 Key Code Patterns

### Decorator — manual way:
```python
def my_decorator(func):
    def wrapper():
        print("before")
        func()             # original function
        print("after")
    return wrapper

greet = my_decorator(greet)   # manual apply
```

### Decorator — @ shortcut:
```python
@my_decorator          # same as greet = my_decorator(greet)
def greet():
    print("Hello!")
```

### Generator:
```python
def my_generator():
    yield 1        # pause and give 1
    yield 2        # pause and give 2
    yield 3        # pause and give 3

for num in my_generator():
    print(num)
```

---

## 💡 Key Differences

| | Normal Function | Generator |
|--|----------------|-----------|
| Keyword | `return` | `yield` |
| Values | all at once | one at a time |
| Memory | high | very low |
| Resumable | ❌ | ✅ |

---

## 💡 Easy Memory Tricks

```
@ symbol    =  shortcut for decorator
yield       =  return + bookmark 📖
wrapper     =  function inside decorator
```

```python
# These two are EXACTLY the same!
greet = my_decorator(greet)   # manual
@my_decorator                  # shortcut
def greet(): ...
```

---

## 🔗 How This Connects to ML

```python
# Timer decorator — measure model training time!
import time

def timer(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        print(f"Time: {end - start:.4f}s")
    return wrapper

@timer
def train_model():
    print("training...")


# Generator — load huge datasets efficiently!
def load_data(filename):
    with open(filename) as f:
        for line in f:
            yield line      # one line at a time!

# PyTorch DataLoader works exactly like this! 🤯
```

---

## ✅ Day 19 Wins

- ✅ Understood decorator concept
- ✅ Built logger decorator from memory
- ✅ Used @ symbol correctly
- ✅ Built even numbers generator
- ✅ Understood yield vs return
- ✅ Zero logic mistakes in challenges!

---

## 🎯 Next Goal

- Day 20 — Context Managers & `with` statement
- `__enter__` and `__exit__` magic methods
- Writing custom context managers!

---

*Day 19 complete* 🔥
