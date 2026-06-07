"""
Day 08 — Collections Module
Topic: deque for efficient queue operations
Date: 26 May 2026
Author: Bala Ravi

Real World Connection:
    deque is O(1) for both ends — list.insert(0) is O(n)!
    Used in:
    - Sliding window for time series analysis
    - BFS (Breadth First Search) graph algorithm
    - Moving average calculations in stock prediction
"""
from collections import deque


def moving_average(prices: list[float], window: int) -> list[float]:
    """
    Calculate moving average using deque sliding window.
    Used in stock price prediction and time series analysis!

    Args:
        prices: List of prices or values
        window: Size of sliding window

    Returns:
        List of moving averages
    """
    result = []
    current_window = deque(maxlen=window)

    for price in prices:
        current_window.append(price)
        if len(current_window) == window:
            avg = sum(current_window) / window
            result.append(round(avg, 2))

    return result


def simulate_queue(items: list[str]) -> None:
    """
    Simulate a queue using deque — FIFO (First In First Out).

    Args:
        items: List of items to add to queue
    """
    queue = deque()

    print("Adding to queue:")
    for item in items:
        queue.append(item)
        print(f"  Added: {item} | Queue: {list(queue)}")

    print("\nRemoving from queue:")
    while queue:
        item = queue.popleft()
        print(f"  Removed: {item} | Queue: {list(queue)}")


if __name__ == "__main__":
    print("=== Queue Simulation ===")
    simulate_queue(["Bala", "Ravi", "Kumar"])

    print("\n=== Moving Average (Stock Analysis) ===")
    stock_prices = [100, 102, 98, 105, 103, 107, 99, 110, 108, 112]
    averages = moving_average(stock_prices, window=3)
    print(f"Prices  : {stock_prices}")
    print(f"3-day MA: {averages}")
