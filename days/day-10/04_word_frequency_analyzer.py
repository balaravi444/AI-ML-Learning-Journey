# Problem 4 - Word Frequency Analyzer
# Day 10 - File Handling

with open("sample.txt", "w") as f:
    f.write("python is great python is easy python is powerful")

with open("sample.txt", "r") as f:
    text = f.read()

words = text.split()
frequency = {}
for word in words:
    frequency[word] = frequency.get(word, 0) + 1

print("Word Frequency:")
for word, count in frequency.items():
    print(f"  {word}: {count}")
