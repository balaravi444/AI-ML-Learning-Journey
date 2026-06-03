# Program 2 - Email Validator
# Day 15 - Regular Expressions

import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return f"✅ '{email}' is a valid email!"
    else:
        return f"❌ '{email}' is NOT a valid email!"

# Test emails
emails = [
    "bala@gmail.com",
    "balaravi444@yahoo.com",
    "invalid-email",
    "missing@dotcom",
    "test@test.co.in"
]

print("=== Email Validator ===")
for email in emails:
    print(validate_email(email))
