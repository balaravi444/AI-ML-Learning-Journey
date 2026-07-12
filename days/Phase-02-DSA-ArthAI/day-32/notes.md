# Day 32 — ArthAI: AI Chatbot with LLM 🚀

**Date:** 19 June 2026
**Time Spent:** (add your hours)
**Resource Used:** [Google AI Studio](https://aistudio.google.com/) | [Gemini API Docs](https://ai.google.dev/docs)

---

## 📚 Topics Covered

- LLM API integration basics
- Prompt engineering for financial advice
- System prompts and context injection
- API key security (.env files)
- Building a domain-specific chatbot
- RAG fundamentals (simplified)

---

## 🔑 What is Prompt Engineering?

The art of writing instructions that make an LLM
behave exactly how you want!

```python
# Bad prompt — generic response
prompt = "What should I invest in?"

# Good prompt — specific, contextual, structured
prompt = f"""
You are ArthAI, a financial advisor for Indians.
User profile: Age {age}, Income ₹{income}/month,
Risk appetite: {risk_level}

User question: {question}

Give specific advice in simple English with
actual numbers. Mention Indian investment
options like PPF, ELSS, FD where relevant.
Keep response under 150 words.
"""
```

**So what? Why does this matter?**
A generic ChatGPT wrapper gives generic advice!
A well-engineered prompt with USER CONTEXT
gives advice that feels personal and accurate!
This is what separates toy projects from real products!

---

## 🔑 System Prompt Design

The system prompt defines the AI's PERSONALITY and RULES:

```python
SYSTEM_PROMPT = """
You are ArthAI — a friendly, knowledgeable
financial advisor for Indians.

RULES:
1. Always answer in simple, jargon-free English
2. Use Indian Rupee (₹) for all amounts
3. Reference Indian financial instruments
   (PPF, ELSS, NPS, FD, Mutual Funds)
4. Never give specific stock recommendations
5. Always mention "consult a SEBI registered
   advisor for personalized advice"
6. Keep responses concise (under 200 words)
7. Use simple examples with real numbers
"""
```

**So what? Why does this matter?**
System prompts are how every production AI app
controls LLM behavior! ChatGPT, Claude, Gemini —
all have hidden system prompts shaping responses!

---

## 🔑 Context Injection (Simplified RAG)

Instead of generic answers — inject USER'S data!

```python
def build_context(user_data: dict) -> str:
    """Inject user's financial data into prompt."""
    return f"""
    User Financial Profile:
    - Age: {user_data['age']}
    - Monthly Income: ₹{user_data['income']:,}
    - Current Savings: ₹{user_data['savings']:,}
    - Existing Loans: ₹{user_data['loans']:,}
    - Risk Appetite: {user_data['risk_level']}
    """
```

**This is the foundation of RAG (Retrieval Augmented
Generation)!** Real RAG retrieves from a database —
we're doing a simplified version with user input!

---

## 🔑 API Security — Never Hardcode Keys!

```python
# WRONG — never do this!
API_KEY = "AIzaSyD...actual_key_here"  # ❌ EXPOSED!

# CORRECT — use environment variables!
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")  # ✅ SAFE!
```

**So what? Why does this matter?**
If you push API keys to GitHub — bots scan and
steal them within MINUTES! This costs real money
when stolen keys are used by attackers!
Every professional dev uses .env files!

---

## 💻 Components Built Today

| # | Component | Purpose |
|---|-----------|---------|
| 1 | LLM Service Module | Handles API calls to Gemini |
| 2 | Prompt Builder | Constructs context-aware prompts |
| 3 | Chat Endpoint | FastAPI route for chatbot |
| 4 | Chat UI Widget | Frontend chat interface |
| 5 | Conversation Memory | Tracks chat history |

---

## 🔗 How This Connects to AI/ML

```python
# This chatbot architecture is EXACTLY how
# production LLM apps work at companies!

# 1. System Prompt = Model Behavior Control
# 2. Context Injection = Simplified RAG
# 3. Conversation History = Memory Management
# 4. Temperature/Max Tokens = Generation Control

# Real companies building this RIGHT NOW:
# - Jupiter Money (Indian fintech) uses this exact pattern!
# - Cleartax uses LLM for tax queries!
# - Groww uses AI for investment suggestions!

# I'm building the SAME architecture as funded startups!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — No max_tokens limit:**
```python
# Wrong — can generate huge expensive responses!
response = model.generate_content(prompt)  # ❌

# Correct — limit response length!
response = model.generate_content(
    prompt,
    generation_config={"max_output_tokens": 300}
)  # ✅ controls cost and response time!
```

**Mistake 2 — No error handling for API failures:**
```python
# Wrong — crashes if API is down!
response = model.generate_content(prompt)

# Correct — graceful fallback!
try:
    response = model.generate_content(prompt)
    return response.text
except Exception as e:
    return "Sorry, I'm having trouble right now. Try again!"
```

**Mistake 3 — Sending full chat history every time:**
```python
# Wrong — grows huge, expensive, slow!
full_history = all_messages_ever  # ❌

# Correct — limit context window!
recent_history = all_messages[-5:]  # ✅ last 5 only!
```

---

## 💎 Important Realizations

1. **A chatbot isn't "just calling an API"**
   Real chatbot engineering = prompt design +
   context management + error handling + UX!

2. **System prompts are competitive advantage**
   Two apps using the SAME LLM can feel completely
   different based on system prompt quality!

3. **This is literally a startup feature**
   Jupiter, Groww, Cred — all have AI advisors.
   I built the same core feature on Day 32! 🤯

---

## 🎯 Next Goal

- Add voice input/output to chatbot
- Multi-language support (Hindi + English)
- Save chat history per user

---

*Day 32 complete — ArthAI now has an AI brain! 🧠🔥*
