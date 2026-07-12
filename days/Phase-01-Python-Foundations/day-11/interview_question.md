# 🎯 OOP Interview Questions — Day 11

> These are real interview questions asked in AI/ML and Python developer interviews.
> Practice answering these out loud — not just reading them!

---

## 🟢 Basic Level

**Q1. What is a class in Python?**
> A class is a blueprint or template for creating objects.
> It defines what data and actions an object will have.

**Q2. What is an object?**
> An object is a real instance created from a class.
> Example: if `Student` is the class, then `bala = Student("Bala", 20)` is the object.

**Q3. What is `__init__` in Python?**
> `__init__` is the constructor method.
> It runs automatically when an object is created.
> It initializes the object's data.

**Q4. What is `self` in Python?**
> `self` refers to the current object itself.
> It lets the object access its own variables and methods.
> Every method in a class must have `self` as the first parameter.

**Q5. What is the difference between a class variable and an instance variable?**
> Class variable → shared by ALL objects of the class
> Instance variable → unique to EACH object
> Example:
> ```python
> class Student:
>     school = "AI Academy"      # class variable — same for all
>     def __init__(self, name):
>         self.name = name       # instance variable — unique per object
> ```

---

## 🟡 Intermediate Level

**Q6. What is inheritance in Python?**
> Inheritance allows a child class to get all properties
> and methods from a parent class.
> It promotes code reuse — write once, use everywhere.

**Q7. What is `super()` in Python?**
> `super()` calls the parent class method from inside a child class.
> Most commonly used in `__init__` to avoid rewriting parent setup code.
> ```python
> class AIStudent(Student):
>     def __init__(self, name, course):
>         super().__init__(name)  # calls Student's __init__
>         self.course = course
> ```

**Q8. What is method overriding?**
> When a child class defines a method with the same name
> as the parent class — the child's version replaces the parent's.
> ```python
> class Animal:
>     def speak(self): print("Some sound")
>
> class Dog(Animal):
>     def speak(self): print("Woof!")  # overrides parent
> ```

**Q9. Can a class have multiple methods?**
> Yes! A class can have as many methods as needed.
> Each method defines a specific action the object can perform.

**Q10. What happens if a child class doesn't override a parent method?**
> The child class automatically uses the parent's version of the method.
> This is the core benefit of inheritance — reuse without rewriting!

---

## 🔴 Advanced Level

**Q11. What is multiple inheritance?**
> A class inheriting from more than one parent class.
> ```python
> class C(A, B):  # C inherits from both A and B
>     pass
> ```

**Q12. What is the difference between `__init__` and `__new__`?**
> `__new__` creates the object in memory.
> `__init__` initializes the object with data.
> In most cases you only need `__init__`.

**Q13. What is a constructor?**
> A constructor is a special method that runs automatically
> when an object is created. In Python it is `__init__`.

---

## 🤖 AI/ML Related Questions

**Q14. How is OOP used in Scikit-learn?**
> Every ML model in Scikit-learn is a class!
> ```python
> model = LinearRegression()  # creating object from class
> model.fit(X, y)             # calling method on object
> model.predict(X_test)       # calling another method
> ```

**Q15. What happens internally when you write `model = LinearRegression()`?**
> Python calls `LinearRegression.__init__()` automatically
> which sets up all the model's default parameters.
> This is exactly the same as `bala = Student("Bala", 20)`.

**Q16. Why do all Scikit-learn models have `.fit()` and `.predict()`?**
> Because they all inherit from a common base class!
> The base class defines the interface that all models must follow.
> This is inheritance and polymorphism working together!

---

## 💡 Quick Revision

| Concept | One Line Answer |
|---------|----------------|
| Class | Blueprint for creating objects |
| Object | Real instance built from a class |
| `__init__` | Constructor — runs when object is created |
| `self` | Refers to the current object |
| Class variable | Shared by ALL objects |
| Instance variable | Unique to EACH object |
| Inheritance | Child gets parent's properties |
| `super()` | Calls parent's method from child |
| Method overriding | Child replaces parent's method |

---

*Practice these answers out loud — not just reading!
If you can explain it simply, you truly understand it.* 🔥
