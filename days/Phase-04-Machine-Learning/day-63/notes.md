# Day 63 — NLP Basics & Text Processing 🚀

**Date:** 20 July 2026
**Time Spent:** (3 hours)
**Resource Used:** [NLTK Docs](https://nltk.org) | [spaCy Docs](https://spacy.io)

---

## 📚 Topics Covered

- What is NLP and why it matters
- Text preprocessing pipeline
- Tokenization — word and sentence
- Stopword removal
- Stemming vs Lemmatization
- Bag of Words (BoW)
- TF-IDF — deep dive
- N-grams
- Text normalization
- Real application — preprocessing job descriptions

---

## 🔑 What is NLP?
NLP = Natural Language Processing

Making computers understand human language.

Before NLP:
Computer sees: "Production is DOWN!!!"
Computer thinks: random characters

After NLP:
Computer understands: urgent negative sentiment,
system failure,
high priority signal

EVERY AI product uses NLP:
→ ChatGPT → generates text
→ Google Search → understands queries
→ Gmail → smart reply, spam filter
→ Alexa → speech to text
→ Our Bug Predictor → understood "crash" = Critical!
---

## 🔑 Text Preprocessing Pipeline
Raw text → Clean text → Tokens → Normalized tokens
→ Remove noise → Feature vector

Step 1: Lowercase
"Production SERVER Down" → "production server down"

Step 2: Remove special characters
"error@#$%!!" → "error"

Step 3: Tokenize
"server is down" → ["server", "is", "down"]

Step 4: Remove stopwords
["server", "is", "down"] → ["server", "down"]
("is" = stopword, carries no meaning)

Step 5: Stem OR Lemmatize
["running", "runs", "ran"] → ["run"]
(all same root word!)

Step 6: Convert to features
Bag of Words OR TF-IDF
---

## 🔑 Tokenization

```python
import nltk
from nltk.tokenize import (
    word_tokenize, sent_tokenize)

text = "Production is down! Users can't login. Fix immediately."

# Word tokenization
words = word_tokenize(text)
# ['Production', 'is', 'down', '!', 'Users',
#  'ca', "n't", 'login', '.', 'Fix', 'immediately', '.']

# Sentence tokenization
sents = sent_tokenize(text)
# ["Production is down!", "Users can't login.", "Fix immediately."]
```

---

## 🔑 Stemming vs Lemmatization
Stemming → chops word endings (fast, crude)
"running" → "run"
"studies" → "studi" ← not a real word!
"caring" → "car" ← WRONG! car ≠ caring

Lemmatization → finds root word (slower, accurate)
"running" → "run"
"studies" → "study" ← real word ✅
"caring" → "care" ← correct! ✅
"better" → "good" ← knows grammar! ✅

For production NLP → use Lemmatization!
For quick prototyping → Stemming is fine.
```python
from nltk.stem import (
    PorterStemmer, WordNetLemmatizer)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "studies", "better", "caring"]

for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, pos='v')
    print(f"{word:<12} → stem: {stem:<10} lemma: {lemma}")
```

---

## 🔑 Bag of Words vs TF-IDF
Bag of Words (BoW):
Count how many times each word appears.
"crash crash error" → crash=2, error=1

Problem: common words dominate!
"the the the error" → the=3, error=1
"the" is useless but ranked highest!

TF-IDF (Term Frequency × Inverse Document Frequency):
TF = how often word appears in THIS document
IDF = how rare the word is ACROSS all documents

"the" appears in EVERY document → low IDF
"crash" rare across docs → high IDF

TF-IDF("the") ≈ 0.001 (common → low score)
TF-IDF("crash") ≈ 0.89 (rare + relevant → high!)

This is why TF-IDF works!
Rare meaningful words → high weight
Common words → near zero weight
---

## 🔑 N-grams
Unigrams (1-gram): single words
["server", "is", "down"]

Bigrams (2-gram): word pairs
["server is", "is down"]

Trigrams (3-gram): word triples
["server is down"]

Why bigrams matter:
"not working" → bigram captures negation!
"data loss" → bigram captures the concept!
"highly recommend" vs "not recommend"

In our Bug Predictor we used ngram_range=(1,2)
That's why it worked so well! ✅
---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Text preprocessing pipeline | Lowercase, clean, tokenize |
| 2 | NLTK tokenization + stopwords | word_tokenize, sent_tokenize |
| 3 | Stemming vs Lemmatization | PorterStemmer vs WordNetLemmatizer |
| 4 | Bag of Words | CountVectorizer |
| 5 | TF-IDF deep dive | Math + sklearn |
| 6 | N-grams | ngram_range |
| 7 | Job description preprocessor | Real application |

---

## 🔗 How This Connects to AI/ML

```python
# NLP is EVERYWHERE in AI products!

# Bug Priority Predictor (Day 59-62):
# TF-IDF on issue titles! Already used NLP!

# AI Hiring Assistant (Day 67-70):
# NLP on resumes + job descriptions
# Match skills, extract experience, rank candidates

# MemoryOS (Day 87-90):
# NLP on all documents
# Understand meaning, link concepts

# ChatGPT itself:
# Transformer = advanced NLP at scale
# Everything we learn now → foundation for Day 81+

# Rule: Can't do modern AI without NLP!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Forgetting to download NLTK data:**
```python
import nltk
nltk.word_tokenize("hello")  # ❌ Error!

# Fix — download first!
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.word_tokenize("hello")  # ✅
```

**Mistake 2 — Not specifying pos in lemmatizer:**
```python
lemmatizer = WordNetLemmatizer()

# Wrong — assumes noun by default!
lemmatizer.lemmatize("running")  # → "running" ❌
lemmatizer.lemmatize("better")   # → "better" ❌

# Correct — specify part of speech!
lemmatizer.lemmatize("running", pos='v')  # → "run" ✅
lemmatizer.lemmatize("better", pos='a')   # → "good" ✅
```

**Mistake 3 — Removing ALL punctuation:**
```python
# Wrong — destroys meaning!
text = "not working"
clean = re.sub(r'[^a-zA-Z]', ' ', text)
# Still "not working" but...

text2 = "u.s.a"
clean2 = re.sub(r'[^a-zA-Z]', ' ', text2)
# "u s a" → context lost ❌

# Better — keep meaningful punctuation!
# Remove only noise: @#$%^&*()
```

---

## 💎 Important Realizations

1. **NLP is just feature engineering for text**
   Same concept as Day 44 (Feature Engineering)!
   We convert unstructured text → structured numbers
   That ML algorithms can understand!

2. **TF-IDF is why our Bug Predictor worked**
   "crash" got high TF-IDF → model learned Critical!
   "typo" got high TF-IDF → model learned Low!
   We used NLP without even knowing it! 🔥

3. **Stopwords depend on domain**
   Generic stopwords: "the", "is", "a"
   But in some domains "not" is critical!
   "not working" vs "working" = opposite!
   Always review stopword list for your domain!

4. **Lemmatization > Stemming for quality**
   For production: always lemmatize
   Stemming creates non-words → confuses models
   Lemmatization keeps real words → better features

---

## 🎯 Next Goal (Day 64)

- Word Embeddings — Word2Vec, GloVe
- Words as vectors in semantic space!
- "king" - "man" + "woman" = "queen"
- Foundation for transformers!

---

*Day 63 complete — NLP Basics mastered! 📝🔥*


