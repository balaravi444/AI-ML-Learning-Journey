# ml_package/preprocessing.py

def normalize(data):
    return [x / max(data) for x in data]

# ml_package/metrics.py

def accuracy(correct, total):
    return (correct / total) * 100

# main.py

from ml_package.preprocessing import normalize
from ml_package.metrics import accuracy

scores = [20, 40, 60, 80]

print("Normalized:", normalize(scores))
print("Accuracy:", accuracy(96, 100))
