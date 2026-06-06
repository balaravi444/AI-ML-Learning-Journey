# Program 1 — Decorator Basics
# Day 19 — Decorators & Generators

# Step 1 — Functions can be passed as arguments!
def greet():
    print("Hello Bala!")

def run(func):
    func()

run(greet)      # Hello Bala! ✅

print("---")

# Step 2 — Build a decorator manually
def my_decorator(func):
    def wrapper():
        print("--- start ---")
        func()                  # original function runs here
        print("--- end ---")
    return wrapper

def greet():
    print("Hello Bala!")

def wish():
    print("Good Morning!")

# Apply manually
greet = my_decorator(greet)
wish  = my_decorator(wish)

greet()
wish()

print("---")

# Step 3 — @ symbol shortcut (same thing!)
def my_decorator(func):
    def wrapper():
        print("--- start ---")
        func()
        print("--- end ---")
    return wrapper

@my_decorator           # same as greet = my_decorator(greet)
def greet():
    print("Hello Bala!")

@my_decorator           # same as wish = my_decorator(wish)
def wish():
    print("Good Morning!")

greet()
wish()
