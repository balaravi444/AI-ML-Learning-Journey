# Problem 2 - Counting Lines and Words in a File
# Day 10 - File handling

# Save student marks
with open("marks.txt", "w") as f:
  f.write("Bala: 85\n")
  f.write("Ravi: 92\n")
  f.write("Rohit: 88\n")

#Read and display marks
print("----Student Marks ---")
with open("marks.txt", "r") as f:
  for line in f:
    print(line.strip())
  
