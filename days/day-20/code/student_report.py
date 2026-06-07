# Project 1 — Student Report Card Generator
# Concepts Used: OOP, Decorators, JSON, File Handling, Magic Methods

import json

# Decorator — adds border to report automatically!
def report_border(func):
    def wrapper(self):
        print("=" * 35)
        func(self)
        print("=" * 35)
    return wrapper


class Student:
    def __init__(self, name, roll_no):
        self.name    = name
        self.roll_no = roll_no
        self.marks   = {}           # subject: marks — key value pair!

    def add_subject(self, subject, marks):
        self.marks[subject] = marks

    def get_total(self):
        return sum(self.marks.values())

    def get_percentage(self):
        total   = self.get_total()
        maximum = len(self.marks) * 100
        return (total / maximum) * 100

    def get_grade(self, percentage):
        if percentage >= 90:   return "A+"
        elif percentage >= 80: return "A"
        elif percentage >= 70: return "B"
        elif percentage >= 60: return "C"
        elif percentage >= 50: return "D"
        else:                  return "F"

    @report_border
    def generate_report(self):
        print(f"   STUDENT REPORT CARD")
        print(f"Name     : {self.name}")
        print(f"Roll No  : {self.roll_no}")
        print("-" * 35)
        print(f"{'Subject':<15} {'Marks':<10} {'Grade'}")
        print("-" * 35)

        for subject, marks in self.marks.items():
            grade = self.get_grade(marks)
            print(f"{subject:<15} {marks:<10} {grade}")

        print("-" * 35)
        total      = self.get_total()
        percentage = self.get_percentage()
        grade      = self.get_grade(percentage)
        result     = "PASS ✅" if percentage >= 50 else "FAIL ❌"

        print(f"Total      : {total} / {len(self.marks) * 100}")
        print(f"Percentage : {percentage:.1f}%")
        print(f"Grade      : {grade}")
        print(f"Result     : {result}")

    def save_to_json(self):
        data = {
            "name"       : self.name,
            "roll_no"    : self.roll_no,
            "marks"      : self.marks,
            "total"      : self.get_total(),
            "percentage" : self.get_percentage(),
            "grade"      : self.get_grade(self.get_percentage())
        }
        filename = f"{self.name}_report.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Saved to {filename}!")


# Run it!
s1 = Student("Bala", 101)
s1.add_subject("Python",      85)
s1.add_subject("Mathematics", 72)
s1.add_subject("English",     90)
s1.add_subject("Science",     65)
s1.add_subject("Tamil",       78)

s1.generate_report()
s1.save_to_json()
