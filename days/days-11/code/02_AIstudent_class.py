# Program 2 - AIStudent Class with Class Variables
# Day 11 - OOP: Classes & Objects

class AIStudent:
    # Class variables - same for ALL students
    bootcamp = "AI/ML Bootcamp"
    total_student = 0

    def __init__(self, name, course, marks):
        # Instance variables - unique for EACH student
        self.name = name
        self.course = course
        self.marks = marks
        AIStudent.total_student += 1

    def introduce(self):
        print(f"Hi! I am {self.name} enrolled in {AIStudent.bootcamp}")
        print(f"Course: {self.course}")

    def calculate_grade(self):
        if self.marks >= 90:
            return "A"
        elif self.marks >= 80:
            return "B"
        elif self.marks >= 70:
            return "C"
        elif self.marks >= 60:
            return "D"
        else:
            return "F"

# Creating objects
student1 = AIStudent("Bala", "Computer Vision", 92)
student2 = AIStudent("Ravi", "Natural Language", 78)
student3 = AIStudent("Hari", "Deep Learning", 85)

# Loop through all students
print("--- Introduction & Grade ---")
students = [student1, student2, student3]
for student in students:
    student.introduce()
    print(f"Grade: {student.calculate_grade()}\n")

# Class stats
print("--- Class Stats ---")
print(f"Total students enrolled: {AIStudent.total_student}")
