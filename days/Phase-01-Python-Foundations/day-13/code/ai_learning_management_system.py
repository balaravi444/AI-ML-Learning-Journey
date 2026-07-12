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

class Person(ABC):
    """Abstract base class for all people in the system."""
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @abstractmethod
    def get_profile(self):
        """Return a formatted string of the person's profile."""
        pass

    @abstractmethod
    def calculate_grade(self):
        """Calculate and return the grade based on performance."""
        pass

class Student(Person):
    """Represents a student enrolled in the AI learning path."""
    def __init__(self, name, age, course, marks):
        super().__init__(name, age)
        self._course = course  # Single underscore for 'protected' variable
        self._marks = marks    # Single underscore is more Pythonic

    def get_profile(self):
        """Returns the full professional profile of the student."""
        grade = self.calculate_grade()
        average = self.get_average()
        return (f"Name: {self.name} | Age: {self.age} | "
                f"Course: {self._course} | Avg: {average:.1f} | Grade: {grade}")

    def calculate_grade(self):
        """Determines the letter grade based on the average marks."""
        if not self._marks:
            return "N/A"
        
        avg = self.get_average()
        # Using a more efficient range check
        if avg >= 90: return "A"
        if avg >= 80: return "B"
        if avg >= 70: return "C"
        if avg >= 60: return "D"
        return "F"

    def get_average(self):
        """Helper to calculate the mean of marks."""
        return sum(self._marks) / len(self._marks) if self._marks else 0

    def get_marks(self):
        return self._marks

    def get_course(self):
        return self._course

class LearningManagementSystem:
    """Manages student records and file operations."""
    def __init__(self, filename="students.txt"):
        self._students = []
        self._filename = filename

    def add_student(self, student):
        """Adds a new student object to the system."""
        self._students.append(student)
        print(f" {student.name} enrolled successfully!")

    def show_statistics(self):
        """Calculates and displays class-wide performance metrics."""
        if not self._students:
            print(" No data available.")
            return

        averages = [s.get_average() for s in self._students]
        print(f"\n--- Class Statistics ---")
        print(f"Total Students: {len(self._students)}")
        print(f"Class Average: {sum(averages)/len(averages):.1f}")
        print(f"Top Score: {max(averages):.1f}")
