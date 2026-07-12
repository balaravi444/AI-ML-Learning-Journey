# Problem 2 - Counting Lines and Words in a File
# Day 10 - File Handling

with open("sample.txt", "w") as f:
    f.write("Python is great\nAI is the future\nFile handling is useful")

with open("sample.txt", "r") as f:
    lines = f.readlines()
    line_count = len(lines)
    word_count = sum(len(line.split()) for line in lines)

print("Total lines:", line_count)
print("Total words:", word_count)
