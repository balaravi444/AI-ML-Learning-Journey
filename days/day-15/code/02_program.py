import re

password = "AI2026@ML"

pattern = r'^(?=.*[A-Z])(?=.*[0-9]).{8,}$'

if re.match(pattern, password):
    print("Strong Password")
else:
    print("Weak Password")
