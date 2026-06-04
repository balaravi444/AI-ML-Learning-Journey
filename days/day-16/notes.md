# Day 16 — Working with JSON & CSV Files 🚀

**Date:** 04 June 2026
**Time Spent:** (1 hour)
**Resource Used:** [GeeksforGeeks JSON](https://www.geeksforgeeks.org/json-python/) | [GeeksforGeeks CSV](https://www.geeksforgeeks.org/working-csv-files-python/)

---

## 📚 Topics Covered

- JSON — what it is and why we use it
- CSV — what it is and why we use it
- json.dumps() and json.loads()
- json.dump() and json.load()
- csv.writer() and csv.reader()
- csv.DictReader()
- Converting CSV to JSON

---

## 🔑 Core Concepts

| Function | What It Does |
|----------|-------------|
| `json.dumps()` | Python dict → JSON string |
| `json.loads()` | JSON string → Python dict |
| `json.dump()` | Python dict → JSON file |
| `json.load()` | JSON file → Python dict |
| `csv.writer()` | Write rows to CSV file |
| `csv.reader()` | Read rows as lists |
| `csv.DictReader()` | Read rows as dictionaries |

---

## 💡 CSV vs JSON

| | CSV | JSON |
|--|-----|------|
| Best for | Simple tables, Excel, datasets | Complex/nested data, APIs |
| Used in | Kaggle datasets, training data | API responses, configs |
| Looks like | Spreadsheet | Python dictionary |

---

## 💻 Programs Practiced

| # | Program |
|---|---------|
| 1 | JSON — save and load profile |
| 2 | CSV — write and read students |
| 3 | CSV DictReader |
| 4 | CSV to JSON converter |

---

## 🔥 Key Code Patterns

### Save to JSON:
```python
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)
```

### Load from JSON:
```python
with open("data.json", "r") as f:
    data = json.load(f)
```

### Write CSV:
```python
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "age"])  # header
    writer.writerow(["Bala", 20])     # data
```

### Read CSV as Dictionary:
```python
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['name'])  # access by column name!
```

---

## 🔗 How This Connects to AI/ML

```python
# Every Kaggle dataset is a CSV file!
# This is how Pandas loads it internally:

import csv
data = []
with open("titanic.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Pandas does exactly this with one line:
# df = pd.read_csv("titanic.csv")
```

---

## 💎 Important Realization

Every ML dataset starts as a CSV or JSON file!
Before training any model:
1. Load CSV/JSON data
2. Clean and preprocess
3. Convert to the right format
4. Train the model

Today I learned Step 1! 🔥

---

## 🤔 Challenges Faced

- Understanding difference between dump/dumps
- Remembering newline="" for CSV writing
- Understanding DictReader vs reader

---

## ✅ Day 16 Wins

- ✅ Saved and loaded JSON files
- ✅ Wrote and read CSV files
- ✅ Used DictReader for clean access
- ✅ Built CSV to JSON converter
- ✅ Connected to real ML data loading

---

## 🎯 Next Goal

- Python APIs — requests library
- Fetch real data from the internet
- Connect to real AI APIs!

---

*Day 16 complete* 🔥
