# Day 66 — Named Entity Recognition (NER) 🚀

**Date:** 23 July 2026
**Time Spent:** (2 hours)
**Resource Used:** [spaCy Docs](https://spacy.io) | [NLTK Docs](https://nltk.org)

---

## 📚 Topics Covered

- What is Named Entity Recognition
- Standard NER entity types
- spaCy NER pipeline
- Custom NER for tech domain
- Regex-based entity extraction
- Rule-based vs ML-based NER
- NER on bug reports
- Real application — tech entity extractor

---

## 🔑 What is NER?
NER = Named Entity Recognition

Extract structured information from unstructured text!

Input:
"The auth-service v2.3.1 is failing on AWS us-east-1
with Error 503. Affects 12,000 users since 14:32 UTC."

Output:
SERVICE: auth-service
VERSION: v2.3.1
PLATFORM: AWS us-east-1
ERROR: Error 503
USERS: 12,000
TIME: 14:32 UTC

Unstructured text → structured data!
That's NER. 🔥
---

## 🔑 Standard NER Entity Types
spaCy built-in entities:
PERSON → "Elon Musk", "Satya Nadella"
ORG → "Google", "Microsoft", "Anthropic"
GPE → "India", "Bangalore", "New York"
DATE → "yesterday", "July 2026", "2 weeks ago"
TIME → "14:32 UTC", "3am", "since morning"
MONEY → "$50,000", "₹5 lakh"
PERCENT → "30%", "95% of users"
PRODUCT → "iPhone", "Windows 11"
EVENT → "World Cup", "IPO"

---

## 🔑 Custom NER for Tech Domain
Standard NER misses tech-specific entities:
→ Service names (auth-service, api-gateway)
→ Version numbers (v2.3.1, 1.0.0-beta)
→ Error codes (Error 503, HTTP 404, FATAL)
→ AWS regions (us-east-1, ap-south-1)
→ Environment (production, staging, dev)
→ Databases (PostgreSQL, MongoDB, Redis)

Two approaches:

Regex rules — fast, precise, interpretable
ML-based (spaCy training) — flexible, generalizes

For tech entities → regex works great! ✅
For general text → use spaCy!
---

## 🔑 Regex-Based Entity Extraction

```python
import re

VERSION_PATTERN = r'v?\d+\.\d+(?:\.\d+)?(?:-\w+)?'
ERROR_CODE = r'(?:Error|HTTP|Status|Code)\s*:?\s*\d{3}'
SERVICE_PATTERN = r'\b\w+(?:-\w+)+(?:-service|-api|-gateway)\b'

text = "auth-service v2.3.1 returning HTTP 503"

versions = re.findall(VERSION_PATTERN, text)
errors = re.findall(ERROR_CODE, text)
services = re.findall(SERVICE_PATTERN, text)

# versions: ['v2.3.1']
# errors: ['HTTP 503']
# services: ['auth-service']
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | spaCy NER basics | Built-in entity types |
| 2 | Custom regex NER | Tech entity patterns |
| 3 | NER on bug reports | Extract structured info |
| 4 | Entity-based search | Find similar incidents |
| 5 | Tech entity extractor | Full application |

---

## 🔗 How This Connects to AI/ML

```python
# NER → structured features for ML!
# Bug Predictor v3:
# Extract service name → encode as feature
# Extract version → new release = more bugs
# Extract error code → 500s more critical than 400s
# Extract environment → production > staging

# AI Hiring Assistant (was Day 67 — now AutoDS):
# NER on resumes:
# SKILL: Python, TensorFlow
# COMPANY: Google, Microsoft
# DEGREE: B.Tech, Masters
# YEAR: 2023, 3 years experience

# MemoryOS (Day 87):
# NER on all saved documents
# Build knowledge graph from entities!
# "Python" connected to "scikit-learn"
# connected to "Random Forest"
```

---

## 💎 Important Realizations

1. **NER converts text into structured data**
   Unstructured bug reports → structured incident data
   Enables search, aggregation, analysis!
   "How many incidents involved auth-service?"

2. **Regex is underrated for tech NER**
   Version numbers follow strict patterns
   Regex catches them perfectly every time
   No training data needed!

3. **Entity types are domain-specific**
   General NER misses tech entities
   Always extend with domain rules!

4. **NER + Sentiment = powerful combo**
   "auth-service is COMPLETELY BROKEN"
   Entity: auth-service
   Sentiment: negative (high confidence)
   → Alert: auth-service degradation detected!

---

## 🎯 Next Goal (Day 67)

- Autonomous Data Scientist project starts!
- AutoML + Explainability + Auto Deployment
- 4-day build replacing AI Hiring Assistant

---

*Day 66 complete — NER mastered! 🏷️🔥*

