# Program 4 - Data Cleaner using Regex
# Day 15 - Regular Expressions

import re

def clean_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower().strip()
    return text

def extract_hashtags(text):
    return re.findall(r'#\w+', text)

def extract_mentions(text):
    return re.findall(r'@\w+', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)


# Test
post = "Hey @balaravi444! Check out #Python and #AI — Day 15 of my journey! 🔥 #100DaysOfCode"

print("=== Data Cleaner ===")
print(f"Original : {post}")
print(f"Cleaned  : {clean_text(post)}")
print(f"Hashtags : {extract_hashtags(post)}")
print(f"Mentions : {extract_mentions(post)}")
print(f"No numbers: {remove_numbers(post)}")
