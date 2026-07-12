# Program 1 - JSON Basics
# Day 16 - JSON & CSV Files

import json

# Python dictionary
student = {
    "name": "Bala",
    "age": 20,
    "college": "The Oxford College of Science",
    "goal": "AI Engineer",
    "skills": ["Python", "OOP", "Regex", "JSON"]
}

# Convert to JSON string
json_string = json.dumps(student, indent=4)
print("=== JSON String ===")
print(json_string)
print(f"Type: {type(json_string)}")

# Convert back to Python dict
python_dict = json.loads(json_string)
print("\n=== Python Dict ===")
print(python_dict)
print(f"Type: {type(python_dict)}")

# Save to file
with open("profile.json", "w") as f:
    json.dump(student, f, indent=4)
print("\n✅ Saved to profile.json!")

# Load from file
with open("profile.json", "r") as f:
    loaded = json.load(f)

print("\n=== Loaded from File ===")
for key, value in loaded.items():
    print(f"{key}: {value}")
