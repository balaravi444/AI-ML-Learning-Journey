# Program 2 - CSV Basics
# Day 16 - JSON & CSV Files

import csv

# Write CSV
with open("ai_student.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "course", "marks", "grade"])
    writer.writerow(["Bala", "AI Engineering", 85, "B"])
    writer.writerow(["Ravi", "Data Science", 92, "A"])
    writer.writerow(["Kumar", "ML", 78, "C"])

print("✅ CSV saved!")

# Read CSV — as list
print("\n=== Reading as List ===")
with open("ai_student.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        print(f"Name: {row[0]} | Course: {row[1]} | Marks: {row[2]} | Grade: {row[3]}")

# Read CSV — as dictionary
print("\n=== Reading as Dictionary ===")
with open("ai_student.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Name: {row['name']} | Course: {row['course']} | Marks: {row['marks']}")
