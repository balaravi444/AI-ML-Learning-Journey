# Day 11 — OOP: Classes & Objects 🚀

**Date:** 30 May 2026
**Time Spent:** (2 hours)
**Resource Used:** [GeeksforGeeks OOP](https://www.geeksforgeeks.org/python-oops-concepts/)

---

## 📚 Topics Covered

- What is OOP and why it exists
- Classes and Objects
- `__init__` constructor
- `self` keyword
- Instance Variables vs Class Variables
- Instance Methods
- Inheritance
- `super()` keyword
- Method Overriding

---

## 🔑 Core Concepts

| Concept | What It Means |
|---------|--------------|
| Class | Blueprint or template |
| Object | Real thing built from blueprint |
| `__init__` | Constructor — runs when object is created |
| `self` | Refers to the specific object itself |
| Class Variable | Shared by ALL objects |
| Instance Variable | Unique to EACH object |
| Inheritance | Child class gets parent's properties |
| `super()` | Calls parent's method from child |
| Method Overriding | Child replaces parent's method |

---

## 💻 Programs Practiced

| # | Problem |
|---|---------|
| 1 | Basic Student Class with introduce() and calculate_grade() |
| 2 | AIStudent Class with class variables and grade calculator |
| 3 | MLModel Inheritance — RandomForest & NeuralNetwork |

---

## 💡 Key Lessons

- Class is a blueprint — objects are real things built from it
- `__init__` runs automatically when object is created
- `self` means "this specific object"
- Class variables are shared by all objects
- Instance variables are unique to each object
- Child class inherits everything from parent
- `super()` calls parent's `__init__` from child
- Method overriding lets child replace parent's behaviour

---

## 💎 Important Realization

OOP is not just Python theory — it is how every real AI/ML library is built.

```python
# This is OOP in action every time you use ML!
model = LinearRegression()   # Creating an object
model.fit(X, y)              # Calling a method
model.predict(X_test)        # Calling another method
```

---

## 🔗 How This Connects to AI/ML

| ML Code | OOP Concept |
|---------|------------|
| `model = LinearRegression()` | Creating an object |
| `model.fit(X, y)` | Calling a method |
| `RandomForest` extends base `MLModel` | Inheritance |
| Each model has unique `predict()` | Method Overriding |

---

## 🤔 Challenges Faced

- Understanding `self` initially
- Remembering indentation inside methods
- Mixing up class variables and instance variables

---

## ✅ Day 11 Wins

- ✅ Wrote Student class from memory
- ✅ Built AIStudent with class variables
- ✅ Built full inheritance system independently
- ✅ Added unique child properties without being taught
- ✅ Connected OOP to real AI/ML libraries

---

## 🎯 Next Goal

- Encapsulation — hiding data safely
- Polymorphism — same method, different behaviour
- Abstract Classes — forcing rules on child classes

---

*Day 11 complete* 🔥
