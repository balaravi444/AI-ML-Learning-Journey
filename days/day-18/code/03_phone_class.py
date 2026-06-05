# Program 3 — Phone Class (Challenge Solution)
# Day 18 — Magic Methods & Operator Overloading
# Written from memory — zero logic mistakes! 💪

class Phone:
    def __init__(self, brand, price):
        self.brand = brand
        self.price = price

    def __str__(self):
        return f"Phone: {self.brand}, Price: {self.price}"

    def __add__(self, other):
        return self.price + other.price

    def __sub__(self, other):
        return self.price - other.price

    def __gt__(self, other):
        return self.price > other.price

    def __lt__(self, other):
        return self.price < other.price

    def __eq__(self, other):
        return self.price == other.price

    def __len__(self):
        return len(self.brand)


# Create objects
p1 = Phone("iPhone", 80000)
p2 = Phone("Samsung", 60000)
p3 = Phone("OnePlus", 80000)

# __str__
print(p1)               # Phone: iPhone, Price: 80000
print(p2)               # Phone: Samsung, Price: 60000
print(p3)               # Phone: OnePlus, Price: 80000

print("---")

# Arithmetic
print(p1 + p2)          # 140000
print(p1 - p2)          # 20000

print("---")

# Comparison
print(p1 > p2)          # True  — 80000 > 60000
print(p1 < p2)          # False — 80000 < 60000
print(p1 == p3)         # True  — 80000 == 80000 (same price!)

print("---")

# len()
print(len(p1))          # 6 — "iPhone" has 6 characters
