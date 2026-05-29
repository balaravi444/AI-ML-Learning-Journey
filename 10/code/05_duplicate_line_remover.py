# Problem 5 - Duplicate Line Remover
# Day 10 - File Handling

with open("data.txt", "w") as f:
    f.write("Python\nAI\nPython\nML\nAI\nDeep Learning\n")

with open("data.txt", "r") as f:
    lines = f.readlines()

unique_lines = list(dict.fromkeys(lines))

with open("data_cleaned.txt", "w") as f:
    f.writelines(unique_lines)

print("Duplicates removed. Unique lines:")
for line in unique_lines:
    print(line.strip())
