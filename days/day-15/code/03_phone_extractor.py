# Program 3 - Phone Number Extractor
# Day 15 - Regular Expressions

import re

def extract_phones(text):
    pattern = r'\b[6-9]\d{9}\b'
    phones = re.findall(pattern, text)
    return phones

text = """
Contact us:
Sales: 9876543210
Support: 8765432109
Invalid: 1234567890
Office: 7654321098
"""

print("=== Phone Number Extractor ===")
phones = extract_phones(text)
print(f"Valid Indian phone numbers found: {phones}")
print(f"Total: {len(phones)}")
