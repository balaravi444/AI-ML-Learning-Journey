import re

dataset = [
    "AI!!!",
    "Machine Learning???",
    "Python@@@",
    "Data Science###"
]

cleaned_data = []

for text in dataset:
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_data.append(cleaned)

print("Cleaned Dataset:")
for item in cleaned_data:
    print(item)
