# Program 2 — Real World Decorators
# Day 19 — Decorators & Generators

import time

# Decorator 1 — Timer (used in ML to measure training time!)
def timer(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        print(f"Time taken: {end - start:.4f} seconds")
    return wrapper

@timer
def slow_task():
    print("doing heavy task...")
    time.sleep(0.5)         # simulate slow task

slow_task()

print("---")

# Decorator 2 — Logger (challenge solution!)
def logger(func):
    def wrapper():
        print("Function starting...")
        func()
        print("Function done!")
    return wrapper

@logger
def calculate():
    print("calculating 100 + 200 =", 100 + 200)

calculate()

print("---")

# Decorator 3 — Apply same decorator to multiple functions!
def logger(func):
    def wrapper():
        print("Function starting...")
        func()
        print("Function done!")
    return wrapper

@logger
def add():
    print("10 + 20 =", 10 + 20)

@logger
def multiply():
    print("10 * 20 =", 10 * 20)

add()
multiply()
