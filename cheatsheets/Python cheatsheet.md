# 🐍 Python Cheatsheet

Quick reference for core Python — syntax you'll use every day.

---

## Variables & Data Types

```python
x = 10                  # int
y = 3.14                # float
name = "Bala"           # str
is_valid = True         # bool
nums = [1, 2, 3]        # list
point = (1, 2)          # tuple
person = {"name": "Bala", "age": 21}  # dict
unique = {1, 2, 3}      # set
```

## Control Flow

```python
if x > 5:
    print("big")
elif x == 5:
    print("equal")
else:
    print("small")

for i in range(5):
    print(i)

while x > 0:
    x -= 1

# Comprehensions
squares = [n**2 for n in range(10)]
evens = [n for n in range(20) if n % 2 == 0]
d = {k: v for k, v in zip(["a","b"], [1,2])}
```

## Functions

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

def total(*args, **kwargs):
    return sum(args), kwargs

square = lambda x: x**2

# Decorator
def timer(func):
    def wrapper(*args, **kwargs):
        # timing logic
        return func(*args, **kwargs)
    return wrapper
```

## OOP

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError

    def __str__(self):
        return f"Animal({self.name})"

class Dog(Animal):
    def speak(self):
        return "Woof!"
```

## File Handling

```python
with open("file.txt", "r") as f:
    data = f.read()

with open("file.txt", "w") as f:
    f.write("hello")

import json
with open("data.json") as f:
    data = json.load(f)
```

## Exception Handling

```python
try:
    risky()
except ValueError as e:
    print(f"Error: {e}")
except (TypeError, KeyError):
    print("Multiple types")
else:
    print("No error")
finally:
    print("Always runs")
```

## Useful Built-ins

| Function | Use |
|---|---|
| `zip(a, b)` | pair up two iterables |
| `enumerate(lst)` | index + value |
| `map(fn, lst)` | apply fn to all |
| `filter(fn, lst)` | keep matching items |
| `sorted(lst, key=fn)` | sort with custom key |
| `any(iter)` / `all(iter)` | boolean checks |
| `Counter(lst)` | frequency count (`collections`) |
| `defaultdict(int)` | dict with default value |

## String Formatting

```python
f"{name} is {age} years old"
f"{price:.2f}"          # 2 decimal places
f"{num:,}"               # thousands separator
f"{value:>10}"           # right-align width 10
```

## Common Gotchas

- Mutable default args: `def f(lst=[])` → bug! Use `def f(lst=None): lst = lst or []`
- `is` checks identity, `==` checks value
- List slicing: `lst[start:stop:step]`
- `range(n)` is `[0, n)`, not inclusive of `n`
