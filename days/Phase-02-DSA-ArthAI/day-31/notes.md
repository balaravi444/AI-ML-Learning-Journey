# Day 31 — ArthAI: Deployment + Goal Planner 🚀

**Date:** 18 June 2026
**Time Spent:** (2 hours)
**Resource Used:** [Render Docs](https://render.com/docs) | [FastAPI Docs](https://fastapi.tiangolo.com/)

---

## 📚 Topics Covered

- Deploying FastAPI apps to production
- Environment variables and config
- Goal-based financial planning algorithm
- Multiple goal prioritization (Knapsack DP!)
- Git workflow for deployment

---

## 🔑 Why Deployment Matters
Code on GitHub = "I can write code"

Live deployed app = "I can ship products"
Recruiters care about the SECOND one! 🔥

**So what? Why does this matter?**
A working live link in your resume gets 10x more
clicks than just a GitHub link. Recruiters can
test it in 30 seconds without cloning anything!

---
**So what? Why does this matter?**
A working live link in your resume gets 10x more
clicks than just a GitHub link. Recruiters can
test it in 30 seconds without cloning anything!

---

## 🔑 Deployment Steps (Render)
Push code to GitHub
Connect Render to GitHub repo
Set build command: pip install -r requirements.txt
Set start command: uvicorn app:app --host 0.0.0.0 --port $PORT
Deploy!
**Key file needed:** `requirements.txt` with exact versions

---

## 🔑 Goal Planner — The Algorithm

Multiple financial goals compete for limited monthly savings!

Goals:

Emergency Fund (high priority, short term)
House Down Payment (medium priority, medium term)
Child Education (high priority, long term)
Vacation (low priority, short term)

Limited budget: ₹15,000/month savings capacity
Question: How to allocate optimally?

**This is literally the Knapsack Problem from Day 28!**

```python
# Goals = items
# Priority score = value
# Monthly required = weight
# Available savings = capacity

def optimize_goals(goals, available_savings):
    # Same DP pattern as 0/1 Knapsack!
    n = len(goals)
    dp = [[0] * (available_savings + 1)
          for _ in range(n + 1)]
    # ... knapsack logic
```

**So what? Why does this matter?**
Real financial planning isn't just calculation —
it's OPTIMIZATION under constraints!
This is exactly what Knapsack DP solves!

---

## 💻 Modules Built Today

| # | Module | DSA Concept Used |
|---|--------|------------------|
| 1 | Goal Priority Calculator | Sorting + Greedy |
| 2 | Multi-Goal Optimizer | Knapsack DP |
| 3 | Goal Timeline Merger | Merge Intervals |
| 4 | Deployment Config | DevOps basics |

---

## 🔗 How This Connects to AI/ML

```python
# Goal optimization = Multi-objective optimization in ML!
# Same concept used in:
# - Hyperparameter tuning with multiple metrics
# - Resource allocation in distributed ML training
# - Portfolio optimization in quant finance ML models!

def optimize_goals(goals: list, budget: float) -> list:
    """
    This is EXACTLY Pareto optimization used in ML
    when balancing accuracy vs speed vs memory!
    """
    pass
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Wrong start command for Render:**
```bash
# Wrong — Render doesn't know the port!
uvicorn app:app --reload          # ❌ dev mode!

# Correct — production mode with dynamic port!
uvicorn app:app --host 0.0.0.0 --port $PORT  # ✅
```

**Mistake 2 — Forgot to add templates/static to git:**
```bash
# Always check .gitignore doesn't exclude needed folders!
cat .gitignore  # make sure templates/ and static/ aren't ignored!
```

---

## 💎 Important Realizations

1. **Deployment is a skill, not magic**
   Once you deploy one app — every future deployment
   takes 10 minutes instead of hours!

2. **Goal planning IS Knapsack Problem**
   Every "optimize X under constraint Y" problem
   in finance maps directly to a DSA pattern!

3. **A live link > 100 GitHub stars**
   Recruiters trust what they can click and test!

---

## 🎯 Next Goal

- AI Chatbot for ArthAI using LLM
- Natural language financial Q&A
- "Should I invest in FD or mutual fund?"

---

*Day 31 complete — ArthAI is LIVE! 🏆🔥*


## 🔑 Deployment Steps (Render)
