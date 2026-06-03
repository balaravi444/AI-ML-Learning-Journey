import re

text = """
bala@gmail.com
ravi@yahoo.com
test123@outlook.com
"""

emails = re.findall(r'\S+@\S+', text)

print("Emails Found:")
for email in emails:
    print(email)
