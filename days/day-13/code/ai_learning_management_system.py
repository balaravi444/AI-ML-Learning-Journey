# ============================================
# AI Learning Management System
# Day 13 - OOP Capstone Project
# Author: Balaravi
# Date: 01 June 2026
#
# Concepts Used:
# - Abstract Classes & Methods
# - Inheritance
# - Encapsulation (Private Variables)
# - Polymorphism (Method Overriding)
# - File Handling
# - Exception Handling
# - List Comprehensions
# ============================================

from abc import ABC, abstractmethod


# ============================================
# ABSTRACT BASE CLASS
# Acts as a contract — every child MUST
# implement get_profile() and calculate_grade()
# ============================================
class Person(ABC):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @abstractmethod
    def get_profile(self):
        pass

    @abstractmethod
    def calculate_grade(self):
        pass


# ============================================
# STUDENT CLASS
# Inherits from Person
# Uses encapsulation for marks and course
# ============================================
class Student(Person):
    def __init__(self, name, age, course, marks):
        super().__init__(name, age)        # parent handles name and age
        self.__course = course             # private
        self.__marks = marks               # private — list of marks

    def get_profile(self):
        grade = self.calculate_grade()
        average = self.get_average()
        return (f"Name: {self.name} | "
                f"Age: {self.age} | "
                f"Course: {self.__course} | "
                f"Average: {average:.1f} | "
                f"Grade: {grade}")

    def calculate_grade(self):
        if not self.__marks:
            return "N/A"
        average = sum(self.__marks) / len(self.__marks)
        if average >= 90:
            return "A"
        elif average >= 80:
            return "B"
        elif average >= 70:
            return "C"
        elif average >= 60:
            return "D"
        else:
            return "F"

    def get_average(self):
        if not self.__marks:
            return 0
        return sum(self.__marks) / len(self.__marks)

    def get_marks(self):
        return self.__marks

    def get_course(self):
        return self.__course

    def update_marks(self, new_marks):
        if all(0 <= mark <= 100 for mark in new_marks):
            self.__marks = new_marks
            print(f"✅ Marks for {self.name} updated successfully!")
        else:
            print("❌ Error: All marks must be between 0 and 100!")


# ============================================
# LEARNING MANAGEMENT SYSTEM CLASS
# Manages all students
# Handles file operations
# ============================================
class LearningManagementSystem:
    def __init__(self):
        self.__students = []               # private list of students
        self.__filename = "students.txt"   # file to save data

    def add_student(self, student):
        self.__students.append(student)
        print(f"✅ {student.name} enrolled successfully!")

    def find_student(self, name):
        for student in self.__students:
            if student.name.lower() == name.lower():
                return student
        return None

    def show_all_students(self):
        if not self.__students:
            print("❌ No students enrolled.")
        else:
            print("\n--- Enrolled Students ---")
            for i, student in enumerate(self.__students, 1):
                print(f"{i}. {student.get_profile()}")

    def show_statistics(self):
        total_students = len(self.__students)
        if total_students == 0:
            print("❌ No students enrolled.")
            return
        all_averages = [s.get_average() for s in self.__students]
        class_avg = sum(all_averages) / len(all_averages)
        highest = max(all_averages)
        lowest = min(all_averages)
        print("\n--- Class Statistics ---")
        print(f"Total Students : {total_students}")
        print(f"Highest Average: {highest:.1f}")
        print(f"Lowest Average : {lowest:.1f}")
        print(f"Class Average  : {class_avg:.1f}")

    def save_data(self):
        try:
            with open(self.__filename, "w") as file:
                for student in self.__students:
                    course = student.get_course()
                    marks_str = ",".join(map(str, student.get_marks()))
                    file.write(
                        f"{student.name},{student.age},{course},{marks_str}\n"
                    )
            print(f"✅ Data saved to {self.__filename}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def load_data(self):
        try:
            with open(self.__filename, "r") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        parts = line.split(",")
                        name = parts[0]
                        age = int(parts[1])
                        course = parts[2]
                        marks = [float(x) for x in parts[3:]]
                        student = Student(name, age, course, marks)
                        self.__students.append(student)
            print(f"✅ Data loaded successfully!")
        except FileNotFoundError:
            print("ℹ️ No saved data found. Starting fresh!")
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def display_menu(self):
        print("\n========================================")
        print("    🎓 AI Learning Management System    ")
        print("========================================")
        print("1. ➕ Add Student")
        print("2. 👥 View All Students")
        print("3. 🔍 Find Student")
        print("4. 📝 Update Marks")
        print("5. 📊 Show Statistics")
        print("6. 💾 Save Data")
        print("7. 🚪 Exit")
        print("========================================")


# ============================================
# MAIN PROGRAM
# Entry point — runs the whole system
# ============================================
def main():
    lms = LearningManagementSystem()
    lms.load_data()

    while True:
        lms.display_menu()
        choice = input("\nEnter choice (1-7): ").strip()

        if choice == "1":
            print("\n--- Add New Student ---")
            try:
                name = input("Enter name: ").strip()
                age = int(input("Enter age: "))
                course = input("Enter course: ").strip()
                marks_input = input(
                    "Enter marks separated by commas (e.g. 85,90,78): "
                )
                marks = [float(m.strip()) for m in marks_input.split(",")]
                student = Student(name, age, course, marks)
                lms.add_student(student)
            except ValueError:
                print("❌ Invalid input! Please enter correct values.")

        elif choice == "2":
            lms.show_all_students()

        elif choice == "3":
            print("\n--- Find Student ---")
            name = input("Enter student name: ").strip()
            student = lms.find_student(name)
            if student:
                print("\n✅ Student Found!")
                print(student.get_profile())
            else:
                print(f"❌ Student '{name}' not found!")

        elif choice == "4":
            print("\n--- Update Marks ---")
            name = input("Enter student name: ").strip()
            student = lms.find_student(name)
            if student:
                try:
                    marks_input = input(
                        "Enter new marks separated by commas: "
                    )
                    new_marks = [
                        float(m.strip()) for m in marks_input.split(",")
                    ]
                    student.update_marks(new_marks)
                except ValueError:
                    print("❌ Invalid marks entered!")
            else:
                print(f"❌ Student '{name}' not found!")

        elif choice == "5":
            lms.show_statistics()

        elif choice == "6":
            lms.save_data()

        elif choice == "7":
            lms.save_data()
            print("\n👋 Thank you for using AI LMS!")
            print("Keep learning! 🔥")
            break

        else:
            print("❌ Invalid choice! Please enter 1-7.")


# Run the program
main()
