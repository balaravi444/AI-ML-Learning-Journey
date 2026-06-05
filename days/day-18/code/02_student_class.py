# Program 2 — Student Class with Magic Methods
# Day 18 — Magic Methods & Operator Overloading

class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def __str__(self):
        return f"Student: {self.name}, Marks: {self.marks}"

    def __add__(self, other):        # s1 + s2
        return self.marks + other.marks

    def __sub__(self, other):        # s1 - s2
        return self.marks - other.marks

    def __mul__(self, other):        # s1 * s2
        return self.marks * other.marks

    def __gt__(self, other):         # s1 > s2
        return self.marks > other.marks

    def __lt__(self, other):         # s1 < s2
        return self.marks < other.marks

    def __eq__(self, other):         # s1 == s2
        return self.marks == other.marks

    def __len__(self):               # len(s1)
        return len(self.name)


# Create objects
s1 = Student("Bala", 85)
s2 = Student("Ravi", 90)

# __str__
print(s1)               # Student: Bala, Marks: 85
print(s2)               # Student: Ravi, Marks: 90

print("---")

# Arithmetic operators
print(s1 + s2)          # 175
print(s1 - s2)          # -5
print(s1 * s2)          # 7650

print("---")

# Comparison operators
print(s1 > s2)          # False — 85 > 90
print(s1 < s2)          # True  — 85 < 90
print(s1 == s2)         # False — 85 == 90

print("---")

# len()
print(len(s1))          # 4 — "Bala" has 4 characters
