# Day 17 — Python APIs & requests Library 🌐

**Date:** 04 June 2026
**Time Spent:** (1 hour
)
**Resource Used:** [requests docs](https://docs.python-requests.org)

---

## 📚 Topics Covered

- What is an API
- requests library — get()
- Status codes
- response.json()
- API parameters
- Nested JSON from API
- try/except with APIs
- Saving API data to JSON

---

## 🔑 Core Concepts

| Code | What It Does |
|------|-------------|
| `requests.get(url)` | Fetch data from internet |
| `response.status_code` | Check if request succeeded |
| `response.json()` | Convert response to dictionary |
| `params={}` | Send filters to API |

---

## 💡 Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success ✅ |
| 404 | Not Found ❌ |
| 401 | Wrong API Key ❌ |
| 500 | Server Error ❌ |

---

## 🔥 Key Code Pattern

### Basic API Call:
​```python
import requests

url = "https://api.example.com/data"

try:
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"Error: {response.status_code}")

except requests.exceptions.ConnectionError:
    print("No internet!")
​```

### API with Parameters:
​```python
params = {
    "q": city,
    "appid": API_KEY,
    "units": "metric"
}
response = requests.get(url, params=params)
​```

### Nested JSON:
​```python
data['main']['temp']           # dict inside dict
data['weather'][0]['description']  # list then dict
​```

---

## 🔗 How This Connects to AI/ML

​```python
# Real ML data collection pipeline:

# Step 1 — Fetch from API
response = requests.get(url, params=params)
data = response.json()

# Step 2 — Save to JSON
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

# Step 3 — Load for ML training
with open("data.json", "r") as f:
    training_data = json.load(f)
​```

---

## ✅ Day 17 Wins

- ✅ Hit real GitHub API
- ✅ Built live weather app
- ✅ Used API parameters
- ✅ Handled errors properly
- ✅ Saved API data to JSON file
- ✅ Connected API + JSON together

---

## 🎯 Next Goal

- Day 18 — Pandas Library
- Load real Kaggle datasets
- Analyse data with Python!

---

*Day 17 complete* 🔥
