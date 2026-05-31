# Program 1 - Encapsulation with Private Variables
# Day 12 - OOP: Encapsulation, Polymorphism & Abstract Classes

class AIStudentProfile:
    def __init__(self, name, marks):
        self.__name = name       # private variable
        self.__marks = marks     # private variable

    def get_name(self):
        return self.__name

    def get_marks(self):
        return self.__marks

    def update_marks(self, new_marks):
        if 0 <= new_marks <= 100:
            self.__marks = new_marks
            print(f"Success! Marks updated to {self.__marks} for {self.__name}")
        else:
            print("Invalid marks! Marks must be between 0 and 100.")


# Testing
student = AIStudentProfile("Bala", 86)
print(f"Student Name: {student.get_name()}")
print(f"Initial Marks: {student.get_marks()}")

student.update_marks(97)
print(f"Updated Marks: {student.get_marks()}")

student.update_marks(150)   # invalid — too high
student.update_marks(-10)   # invalid — negative
