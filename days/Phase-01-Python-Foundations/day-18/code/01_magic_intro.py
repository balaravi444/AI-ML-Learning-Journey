# Program 1 — Magic Methods Introduction
# Day 18 — Magic Methods & Operator Overloading

# Python secretly calls magic methods behind every operator!

# Normal way
print(2 + 3)            # 5

# Magic method way — same result!
print((2).__add__(3))   # 5

# Same thing! Just hidden! 🪄

print("---")

# Without __str__ — ugly output
class StudentBad:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

s = StudentBad("Bala", 85)
print(s)   # <__main__.StudentBad object at 0x...> ❌

print("---")

# With __str__ — clean output
class StudentGood:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def __str__(self):
        return f"Student: {self.name}, Marks: {self.marks}"

s = StudentGood("Bala", 85)
print(s)   # Student: Bala, Marks: 85 ✅
