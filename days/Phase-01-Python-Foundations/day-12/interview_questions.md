# 🎯 OOP Interview Questions — Day 12

> These are real interview questions asked in AI/ML and Python developer interviews.
> Practice answering these out loud — not just reading them!

---

## 🟢 Basic Level

**Q1. What is encapsulation in Python?**
> Encapsulation is hiding internal data from direct outside access.
> It protects data by making variables private and
> only allowing access through controlled methods.

**Q2. How do you make a variable private in Python?**
> Add double underscore `__` before the variable name.
> ```python
> self.__balance = 1000  # private — cannot access directly outside class
> ```

**Q3. What is a getter method?**
> A getter method safely reads a private variable.
> ```python
> def get_balance(self):
>     return self.__balance
> ```

**Q4. What is a setter method?**
> A setter method safely updates a private variable
> usually with validation to prevent invalid data.
> ```python
> def update_marks(self, marks):
>     if 0 <= marks <= 100:
>         self.__marks = marks
>     else:
>         print("Invalid marks!")
> ```

**Q5. What is polymorphism in Python?**
> Polymorphism means same method name behaves
> differently depending on which class calls it.
> ```python
> dog.speak()   # prints "Woof!"
> cat.speak()   # prints "Meow!"
> # same method name — different behaviour!
> ```

---

## 🟡 Intermediate Level

**Q6. What is an abstract class in Python?**
> An abstract class is a class that cannot be instantiated directly.
> It defines a blueprint that child classes MUST follow.
> It uses `ABC` from the `abc` module.

**Q7. What is an abstract method?**
> A method decorated with `@abstractmethod` that
> every child class MUST implement.
> If a child doesn't implement it — Python throws an error!

**Q8. How do you create an abstract class in Python?**
> ```python
> from abc import ABC, abstractmethod
>
> class MLModel(ABC):
>     @abstractmethod
>     def train(self):
>         pass
>
>     @abstractmethod
>     def predict(self):
>         pass
> ```

**Q9. What is the difference between encapsulation and abstraction?**
> Encapsulation → HIDES data using private variables
> Abstraction → HIDES complexity by showing only what's necessary
> Encapsulation is about DATA protection
> Abstraction is about COMPLEXITY reduction

**Q10. Can you create an object from an abstract class?**
> NO! Abstract classes cannot be instantiated directly.
> ```python
> model = MLModel()  # ❌ TypeError!
> model = LinearRegression()  # ✅ child class works!
> ```

---

## 🔴 Advanced Level

**Q11. What are the 4 pillars of OOP?**
> 1. Encapsulation — hiding data
> 2. Abstraction — hiding complexity
> 3. Inheritance — reusing code
> 4. Polymorphism — same method, different behaviour

**Q12. What is the difference between method overriding and method overloading?**
> Overriding → child class replaces parent's method (Python supports this)
> Overloading → same method with different parameters (Python handles
> this with default parameters)

**Q13. Why can't we access private variables directly?**
> Python name-mangles private variables.
> `self.__balance` becomes `_ClassName__balance` internally.
> This prevents accidental access from outside the class.

---

## 🤖 AI/ML Related Questions

**Q14. How is encapsulation used in ML libraries?**
> ML model internal weights and parameters are private!
> You can't directly change `model.__weights`.
> You access them through proper methods like `model.coef_`
> This prevents accidental corruption of trained model data.

**Q15. How is polymorphism used in Scikit-learn?**
> Every model has the same `.fit()` and `.predict()` interface!
> ```python
> models = [LinearRegression(), RandomForest(), SVM()]
> for model in models:
>     model.fit(X_train, y_train)    # same method — different algorithm!
>     model.predict(X_test)          # same method — different calculation!
> ```
> This lets you swap models easily without changing your code!

**Q16. How are abstract classes used in ML frameworks?**
> Scikit-learn uses abstract base classes internally!
> Every model MUST implement `fit()` and `predict()`.
> If you build a custom model without these — it won't work!
> This enforces a consistent interface across all models.

**Q17. What is the benefit of using OOP in AI/ML projects?**
> - Code reuse through inheritance
> - Data protection through encapsulation
> - Flexible model swapping through polymorphism
> - Enforced interfaces through abstract classes
> - Cleaner and more maintainable code overall

---

## 💡 Quick Revision

| Concept | One Line Answer |
|---------|----------------|
| Encapsulation | Hiding data with private variables |
| Private variable | `self.__variable` — only class can access |
| Getter | Method to safely READ private data |
| Setter | Method to safely UPDATE private data |
| Polymorphism | Same method name, different behaviour |
| Abstract class | Blueprint — cannot create object directly |
| Abstract method | Method child class MUST implement |
| `@abstractmethod` | Decorator that enforces implementation |
| `ABC` | Abstract Base Class — imported from `abc` |

---

## 🎯 Most Asked in Interviews

These 5 questions come up in almost every Python interview:
1. What are the 4 pillars of OOP?
2. What is encapsulation with example?
3. What is polymorphism with example?
4. What is an abstract class?
5. How is OOP used in ML libraries?

**Practice these 5 until you can answer without thinking!** 💪

---

*Practice these answers out loud — not just reading!
If you can explain it simply, you truly understand it.* 🔥
