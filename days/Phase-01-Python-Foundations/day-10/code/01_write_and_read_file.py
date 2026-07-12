# problem 1 -Writing and Reading File Content
# Day 10 - File handling

# Writing to file
with open(S"sample.txt", "w") as f:
  f.write("Hello this is my first file!\n")
  f.write("Python file handling is useful.\n")

# Reading from file
with open("sample.txt", "r") as f:
  content = f.read()
  print("File Content:\n", content)
  
