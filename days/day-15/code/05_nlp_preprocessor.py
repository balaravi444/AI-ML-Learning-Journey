# Program 5 - NLP Text Preprocessor
# Day 15 - Regular Expressions
# This is REAL preprocessing used before training NLP models!

import re

def preprocess_text(text):
    """
    Complete NLP text preprocessing pipeline.
    This is exactly what happens before training NLP models!
    """
    # Step 1 - Convert to lowercase
    text = text.lower()

    # Step 2 - Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3 - Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Step 4 - Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Step 5 - Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_keywords(text):
    """Extract meaningful words (length > 3)"""
    words = re.findall(r'\b\w{4,}\b', text.lower())
    return list(set(words))


# Test with real social media text
texts = [
    "Check out https://github.com/balaravi444 for my AI journey! #Python #ML",
    "Email me at bala@gmail.com for collaboration 🤝",
    "Day 15 of #100DaysOfCode — Learning Regex for NLP! So excited!! 🔥🔥"
]

print("=== NLP Text Preprocessor ===\n")
for text in texts:
    print(f"Original  : {text}")
    print(f"Processed : {preprocess_text(text)}")
    print(f"Keywords  : {extract_keywords(text)}")
    print("---")
