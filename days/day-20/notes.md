# Project 1 — Student Report Card Generator 🏗️

**Built on:** Day 20 of 90
**Difficulty:** Beginner
**Status:** ✅ Complete

---

## 📋 What This Project Does

Takes student name, roll number and subject marks as input — generates a clean formatted report card and saves it to a JSON file!

---

## 🖥️ Sample Output

```
===================================
   STUDENT REPORT CARD
Name     : Bala
Roll No  : 101
-----------------------------------
Subject         Marks      Grade
-----------------------------------
Python          85         A
Mathematics     72         B
English         90         A+
Science         65         C
Tamil           78         B
-----------------------------------
Total      : 390 / 500
Percentage : 78.0%
Grade      : B
Result     : PASS ✅
===================================
✅ Saved to Bala_report.json!
```

---

## 🔑 Concepts Used

| Concept | Where Used |
|---------|-----------|
| OOP & Classes | `Student` class |
| `__init__` | store name, roll_no, marks |
| Dictionary | store subject marks |
| Decorator | `@report_border` for borders |
| JSON | save report to file |
| Ternary Operator | PASS/FAIL logic |
| f-strings | formatted output |
| for loop | print each subject |

---

## 📁 Files

```
project-1/
├── code/
│   ├── student_report.py   ← main program
│   └── Bala_report.json    ← sample output
└── README.md
```

---

## ▶️ How to Run

```bash
python student_report.py
```

---

## 🎯 What I Learned Building This

- How to combine OOP + Decorators + JSON together
- Real world use of dictionary for structured data
- How decorators keep code clean and reusable
- Saving structured data to JSON files

---

*Project 1 of many — Day 20 of 90* 🔥
