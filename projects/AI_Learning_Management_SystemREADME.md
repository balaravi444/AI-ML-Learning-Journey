# 🎓 Project 1 — AI Learning Management System

> First real project built during my AI/ML learning journey!
> Built on Day 13 after completing OOP fundamentals.

---

## 📋 What This Project Does

A complete command-line Learning Management System that:
- ➕ Adds students with name, age, course and marks
- 👥 Views all enrolled students with auto-calculated grades
- 🔍 Finds any student by name instantly
- 📝 Updates student marks with validation
- 📊 Shows class statistics — highest, lowest, class average
- 💾 Saves all data to file automatically
- 📂 Loads saved data on every startup

---

## 🛠️ Concepts Used

| Concept | Implementation |
|---------|---------------|
| Abstract Classes | `Person(ABC)` base class |
| Inheritance | `Student` inherits from `Person` |
| Encapsulation | Private `__marks`, `__course`, `__students` |
| Polymorphism | `get_profile()` and `calculate_grade()` |
| File Handling | Save and load student data |
| Exception Handling | Validates all user inputs |

---

## 🚀 How to Run

```bash
python ai_learning_management_system.py
```

Then follow the menu:

1 → Add a student
2 → View all students
3 → Find a student
4 → Update marks
5 → See class statistics
6 → Save data
7 → Exit
---

## 📊 Sample Output
========================================
🎓 AI Learning Management System

➕ Add Student
👥 View All Students
🔍 Find Student
📝 Update Marks
📊 Show Statistics
💾 Save Data
🚪 Exit
========================================

Name: Bala | Age: 20 | Course: AI Engineering | Average: 85.0 | Grade: B
---

## 📁 Files

| File | Purpose |
|------|---------|
| `ai_learning_management_system.py` | Main program |
| `students.txt` | Auto-generated when you save data |

---

## 🔗 Part of My Journey

This project is part of my public AI/ML learning journey!
👉 [View Full Journey](../../README.md)

---

*Built on Day 13 — 01 June 2026* 🔥
