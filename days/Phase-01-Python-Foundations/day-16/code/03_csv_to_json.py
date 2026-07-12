# Program 3 - CSV to JSON Converter
# Day 16 - JSON & CSV Files
# This is real data transformation used in AI/ML pipelines!

import csv
import json

# Step 1 — Read CSV
students = []
with open("ai_student.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        students.append(row)

print(f"✅ Loaded {len(students)} students from CSV")

# Step 2 — Save as JSON
with open("ai_students.json", "w") as f:
    json.dump(students, f, indent=4)

print("✅ Converted and saved to JSON!")

# Step 3 — Load JSON and print
with open("ai_students.json", "r") as f:
    loaded = json.load(f)

print("\n=== Students from JSON ===")
for student in loaded:
    print(f"Name: {student['name']} | Course: {student['course']} | Marks: {student['marks']} | Grade: {student['grade']}")
