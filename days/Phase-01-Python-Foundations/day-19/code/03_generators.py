# Program 3 — Generators
# Day 19 — Decorators & Generators

# Step 1 — yield vs return
def normal_func():
    return [1, 2, 3]        # gives all at once

def my_generator():
    yield 1                 # pause, give 1
    yield 2                 # pause, give 2
    yield 3                 # pause, give 3

# Normal function
print(normal_func())        # [1, 2, 3] — all at once

print("---")

# Generator — one at a time!
gen = my_generator()
print(next(gen))            # 1
print(next(gen))            # 2
print(next(gen))            # 3

print("---")

# Step 2 — Generator with for loop
def my_generator():
    yield 1
    yield 2
    yield 3

for num in my_generator():  # for loop calls next() automatically!
    print(num)

print("---")

# Step 3 — Challenge solution — even numbers generator!
def even_numbers():
    for num in range(2, 11, 2):
        yield num

for num in even_numbers():
    print(num)

print("---")

# Step 4 — Infinite generator (memory efficient!)
def count_forever():
    num = 1
    while True:
        yield num
        num += 1

gen = count_forever()
print(next(gen))            # 1
print(next(gen))            # 2
print(next(gen))            # 3
# can go forever — uses almost zero memory! 🤯

print("---")

# Step 5 — ML use case — load huge file line by line!
# (simulated — no real file needed)
def load_data():
    data = ["row1", "row2", "row3", "row4", "row5"]
    for row in data:
        yield row           # one row at a time!

for row in load_data():
    print(row)
# PyTorch DataLoader works exactly like this! 🤯
