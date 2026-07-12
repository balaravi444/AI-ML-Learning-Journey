# 🐍 Python Basics Cheatsheet

> Quick reference for Python fundamentals

---

## 📦 Variables & Data Types

```python
# Numbers
x = 10          # int
y = 3.14        # float

# String
name = "Bala"

# Boolean
is_ready = True

# Check type
print(type(x))  # <class 'int'>
```

---

## 🔄 Conditions

```python
if marks >= 90:
    print("A Grade")
elif marks >= 75:
    print("B Grade")
elif marks >= 35:
    print("Pass")
else:
    print("Fail")
```

---

## 🔁 Loops

```python
# For loop
for i in range(5):
    print(i)

# While loop
i = 0
while i < 5:
    print(i)
    i += 1

# Loop through list
names = ["Bala", "Ravi", "Kumar"]
for name in names:
    print(name)
```

---

## 🔧 Functions

```python
# Basic function
def greet(name):
    return f"Hello {name}"

# Default parameter
def greet(name="Bala"):
    return f"Hello {name}"

# Lambda
square = lambda x: x * x
```

---

## 📚 Data Structures

```python
# List — ordered, changeable
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers.remove(1)

# Tuple — ordered, unchangeable
coords = (10, 20)

# Dictionary — key-value pairs
student = {"name": "Bala", "age": 20}
print(student["name"])

# Set — unique values only
unique = {1, 2, 3, 3, 4}  # {1, 2, 3, 4}
```

---

## ⚠️ Exception Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
except ValueError:
    print("Invalid value")
else:
    print("Success!")
finally:
    print("Always runs")
```

---

## 📁 File Handling

```python
# Write
with open("file.txt", "w") as f:
    f.write("Hello!")

# Read
with open("file.txt", "r") as f:
    content = f.read()

# Append
with open("file.txt", "a") as f:
    f.write("More text")
```

---

## 🏛️ OOP

```python
class Student:
    school = "AI Academy"      # Class variable

    def __init__(self, name, age):
        self.name = name       # Instance variable
        self.age = age

    def introduce(self):
        print(f"Hi I am {self.name}")

# Inheritance
class AIStudent(Student):
    def __init__(self, name, age, course):
        super().__init__(name, age)
        self.course = course

    def introduce(self):
        print(f"Hi I am {self.name} studying {self.course}")

# Creating objects
bala = AIStudent("Bala", 20, "AI/ML")
bala.introduce()
```

---

*Last updated: 30 May 2026*
