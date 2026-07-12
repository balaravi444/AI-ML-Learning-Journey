# Program 1 - Basic Regex Patterns
# Day 15 - Regular Expressions

import re

text = "My phone is 9876543210 and email is bala@gmail.com"

# Find all digits
digits = re.findall(r'\d', text)
print(f"All digits: {digits}")

# Find all words
words = re.findall(r'\w+', text)
print(f"All words: {words}")

# Find phone number
phone = re.findall(r'\d{10}', text)
print(f"Phone number: {phone}")

# Search for email pattern
email = re.search(r'\w+@\w+\.\w+', text)
if email:
    print(f"Email found: {email.group()}")

# Check if string starts with "My"
match = re.match(r'My', text)
if match:
    print("String starts with 'My'!")
