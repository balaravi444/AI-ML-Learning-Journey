# 🏛️ OOP Cheatsheet

> Quick reference for Object Oriented Programming in Python

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
| Encapsulation | Hiding data using private variables |
| Polymorphism | Same method, different behaviour |

---

## 📝 Class Structure

```python
class ClassName:
    # Class variable
    class_var = "shared"

    # Constructor
    def __init__(self, param1, param2):
        self.param1 = param1    # Instance variable
        self.param2 = param2

    # Method
    def method_name(self):
        print(f"Hello {self.param1}")
```

---

## 👨‍👧 Inheritance

```python
# Parent
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("Some sound")

# Child
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)   # Call parent
        self.breed = breed

    def speak(self):             # Override parent
        print(f"{self.name} says Woof!")

dog = Dog("Bruno", "Labrador")
dog.speak()
```

---

## 🔒 Encapsulation

```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance   # Private variable

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())   # 1500
```

---

## 🔄 Polymorphism

```python
class Dog:
    def speak(self):
        print("Woof!")

class Cat:
    def speak(self):
        print("Meow!")

# Same method name, different behaviour
animals = [Dog(), Cat()]
for animal in animals:
    animal.speak()
```

---

## 🔗 OOP in AI/ML

```python
# Every ML model is OOP!
from sklearn.linear_model import LinearRegression

model = LinearRegression()   # Creating object
model.fit(X, y)              # Calling method
predictions = model.predict(X_test)  # Another method
```

---

*Last updated: 30 May 2026*
