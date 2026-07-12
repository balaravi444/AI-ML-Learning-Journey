# Day 33 — ArthAI: Portfolio Tracker & Investment Analyzer 🚀

**Date:** 20 June 2026
**Time Spent:** (3 hours)
**Resource Used:** [Yahoo Finance API](https://pypi.org/project/yfinance/) | [FastAPI Docs](https://fastapi.tiangolo.com/)

---

## 📚 Topics Covered

- Real-time stock/mutual fund data fetching
- Portfolio performance calculation
- Risk assessment algorithms
- Asset allocation visualization
- Diversification scoring
- Sliding window for trend analysis (Day 29 callback!)

---

## 🔑 Why a Portfolio Tracker?
People invest in multiple places:

→ Stocks

→ Mutual Funds

→ FDs

→ PPF/EPF
But nobody tracks EVERYTHING together!

ArthAI's Portfolio Tracker = single dashboard

for your ENTIRE financial picture.
**So what? Why does this matter?**
This is the feature that turns ArthAI from
"calculator collection" into "financial command center" —
exactly what apps like Kuvera and Groww charge for!

---

## 🔑 Portfolio Diversification Score

A mathematical way to check if you're "putting
all eggs in one basket"!

```python
def diversification_score(allocations: dict) -> float:
    """
    Uses concept similar to entropy in ML!
    Higher score = better diversified
    """
    import math
    total = sum(allocations.values())
    score = 0
    for amount in allocations.values():
        weight = amount / total
        if weight > 0:
            score -= weight * math.log(weight)
    return score
```

**So what? Why does this matter?**
This is literally SHANNON ENTROPY — the same
formula used in:
- Decision Tree splits (information gain!)
- Measuring model uncertainty in ML
- Feature importance calculations!

DSA/Math concepts keep appearing everywhere! 🔥

---

## 🔑 Risk Assessment Algorithm

```python
def calculate_portfolio_risk(holdings: list) -> dict:
    """
    Weighted average risk score across holdings.
    Uses weighted sum — basic but powerful!
    """
    total_value = sum(h['value'] for h in holdings)
    weighted_risk = sum(
        h['value'] * h['risk_score']
        for h in holdings
    ) / total_value
    return weighted_risk
```

**Risk scores by asset type (1-10 scale):**
Fixed Deposit       → 1 (very safe)

PPF/EPF            → 1 (very safe)

Debt Mutual Fund    → 3 (low risk)

Balanced Fund       → 5 (moderate)

Equity Mutual Fund  → 7 (high)

Direct Stocks       → 8 (high)

Crypto             → 10 (very high)
---

## 🔑 Trend Analysis Using Sliding Window

Callback to Day 29 concept — applied to REAL data!

```python
def analyze_trend(prices: list[float],
                   window: int = 7) -> str:
    """
    Determine if investment is trending up/down.
    Uses sliding window moving average!
    """
    if len(prices) < window * 2:
        return "Insufficient data"

    recent_avg = sum(prices[-window:]) / window
    previous_avg = sum(prices[-window*2:-window]) / window

    change = (recent_avg - previous_avg) / previous_avg * 100

    if change > 2:
        return f"📈 Trending UP (+{change:.1f}%)"
    elif change < -2:
        return f"📉 Trending DOWN ({change:.1f}%)"
    return f"➡️ STABLE ({change:.1f}%)"
```

**So what? Why does this matter?**
This EXACT pattern (compare recent window vs
previous window) is used in:
- Anomaly detection in ML monitoring systems
- A/B testing significance checks
- Stock market technical analysis!

---

## 💻 Modules Built Today

| # | Module | DSA/ML Concept Used |
|---|--------|---------------------|
| 1 | Portfolio Aggregator | Hash Map grouping |
| 2 | Diversification Scorer | Shannon Entropy |
| 3 | Risk Calculator | Weighted averages |
| 4 | Trend Analyzer | Sliding Window |
| 5 | Asset Allocation Chart | Data visualization |

---

## 🔗 How This Connects to AI/ML

```python
# Shannon Entropy — used EVERYWHERE in ML!

# 1. Decision Tree splits use entropy!
def information_gain(parent_entropy, children):
    weighted_child_entropy = sum(
        len(c)/len(parent) * entropy(c)
        for c in children
    )
    return parent_entropy - weighted_child_entropy

# 2. Model uncertainty quantification
def prediction_uncertainty(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)

# 3. Feature selection (max information features)
# Same entropy formula picks best splitting features!

# Portfolio diversification and ML decision trees
# use the EXACT SAME MATH! 🤯
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Division by zero in risk calculation:**
```python
# Wrong — crashes if portfolio is empty!
weighted_risk = sum(...) / total_value  # ❌ ZeroDivisionError!

# Correct — guard against empty portfolio!
if total_value == 0:
    return 0
weighted_risk = sum(...) / total_value  # ✅
```

**Mistake 2 — Not handling log(0) in entropy:**
```python
# Wrong — math domain error!
score -= weight * math.log(weight)  # ❌ crashes if weight=0!

# Correct — skip zero weights!
if weight > 0:
    score -= weight * math.log(weight)  # ✅
```

---

## 💎 Important Realizations

1. **Shannon Entropy bridges finance and ML perfectly**
   The same formula measuring "portfolio diversification"
   measures "decision tree split quality" in ML!

2. **Sliding window from Day 29 is reusable everywhere**
   Stock trends, ML monitoring, anomaly detection —
   one pattern, countless applications!

3. **Real apps = combining multiple DSA concepts**
   Today's portfolio tracker uses: hash maps,
   weighted averages, entropy, AND sliding window —
   all working together in ONE feature!

---

## 🎯 Next Goal

- Add data export (PDF reports)
- Email notifications for goal milestones
- Polish UI/UX for final demo

---

*Day 33 complete — ArthAI getting smarter! 📊🔥*
