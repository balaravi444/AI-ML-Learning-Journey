# Program 1 - Basic Student Class
# Day 11 - OOP: Classes & Objects

class Student:
    def __init__(self, name, age, course):
        self.name = name
        self.age = age
        self.course = course

    def introduce(self):
        print(f"Hi! I am {self.name}, studying {self.course}")

    def calculate_grade(self, marks):
        if marks >= 90:
            return "A"
        elif marks >= 80:
            return "B"
        elif marks >= 70:
            return "C"
        elif marks >= 60:
            return "D"
        else:
            return "F"

# Creating objects
bala = Student("Bala", 20, "BCA")
ravi = Student("Ravi", 21, "BSc")

# Using methods
bala.introduce()
print(f"Bala's grade: {bala.calculate_grade(85)}")

ravi.introduce()
print(f"Ravi's grade: {ravi.calculate_grade(72)}")
