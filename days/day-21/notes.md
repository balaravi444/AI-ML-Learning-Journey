Day 21 — DSA
│🗺️ Today's Roadmap
├── 📦 Arrays
│   ├── What is an Array?
│   ├── Memory Layout
│   └── Operation Complexity
│
├── ⏱️ Big O Notation
│   ├── O(1), O(log n), O(n), O(n²)
│   └── Why it matters (FAANG!)
│
├── 🔍 Searching
│   ├── Linear Search  → O(n)
│   └── Binary Search  → O(log n)
│
├── 📊 Sorting
│   ├── Bubble Sort    → O(n²)
│   ├── Selection Sort → O(n²)
│   ├── Insertion Sort → O(n²)
│   └── Merge Sort     → O(n log n) ⭐
│
└── 💼 Interview Problems (9 problems solved!)
    ├── Easy   → find_max, is_sorted, reverse_array
    ├── Medium → second_largest, find_pair
    └── Hard   → max_subarray (Kadane's Algorithm)

⏱️ Big O Notation
Big O = how many steps your code takes as input grows
NotationName1M elementsFeelO(1)Constant1 step🏆 BestO(log n)Logarithmic~20 steps✅ GreatO(n)Linear1,000,000 steps😊 OKO(n²)Quadratic1 trillion steps😫 BadO(2ⁿ)Exponential∞💀 Avoid
🧠 How to think about it:

O(1) → Same steps no matter the size → arr[0]
O(log n) → Cuts problem in half each step → Binary Search
O(n) → One loop through data → Linear Search
O(n²) → Loop inside loop → Bubble Sort


📦 Arrays
An array = list in Python — stored side by side in memory.
pythonnumbers = [10, 20, 30, 40, 50]
#           ↑   ↑   ↑   ↑   ↑
# index:    0   1   2   3   4
# memory: 100 104 108 112 116
Operation Complexity:
OperationComplexityWhyarr[i] — accessO(1)Direct jump by addressarr.append(x) — endO(1)Just add at endarr.pop() — from endO(1)Just remove lastarr.insert(0, x) — frontO(n)Shifts ALL elements rightarr.pop(0) — from frontO(n)Shifts ALL elements leftx in arr — searchO(n)Check one by one

🔍 Searching
1. Linear Search — O(n)
Check every element one by one.
pythondef linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i       # found! return index
    return -1              # not found

numbers = [10, 20, 30, 40, 50]
print(linear_search(numbers, 30))   # 2
print(linear_search(numbers, 99))   # -1

✅ Works on unsorted arrays
❌ Slow on large data


2. Binary Search — O(log n)
⚠️ Array MUST be sorted first!
Idea: Check the MIDDLE → if not found, eliminate HALF the array → repeat.
Find 70 in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

Step 1: mid=4 → arr[4]=50 → 50 < 70 → search RIGHT → left=5
Step 2: mid=7 → arr[7]=80 → 80 > 70 → search LEFT  → right=6
Step 3: mid=5 → arr[5]=60 → 60 < 70 → search RIGHT → left=6
Step 4: mid=6 → arr[6]=70 → 70 == 70 → FOUND! ✅
10 elements → found in just 4 steps! vs Linear Search = 7 steps 🤯
pythondef binary_search(arr, target):
    left  = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid              # found!
        elif arr[mid] < target:
            left = mid + 1          # search RIGHT half
        else:
            right = mid - 1         # search LEFT half

    return -1                       # not found

numbers = [10, 20, 30, 40, 50]
print(binary_search(numbers, 30))   # 2
print(binary_search(numbers, 99))   # -1
The 3 Conditions (memorise!):
arr[mid] == target  →  FOUND!          → return mid
arr[mid] <  target  →  go RIGHT        → left  = mid + 1
arr[mid] >  target  →  go LEFT         → right = mid - 1
Linear vs Binary Comparison:
FeatureLinear SearchBinary SearchComplexityO(n)O(log n)Sorted needed?❌ No✅ Yes1M elements1,000,000 steps20 stepsUse whenSmall/unsortedLarge sorted

📊 Sorting Algorithms
1. Bubble Sort — O(n²)
Compare neighbours → swap if wrong order → biggest "bubbles" to end.
pythondef bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):    # last i already sorted
            if arr[j] > arr[j + 1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]  # Python swap trick!
    return arr

print(bubble_sort([64, 34, 25, 12, 22]))  # [12, 22, 25, 34, 64]

2. Selection Sort — O(n²)
Find minimum → put at front → repeat for remaining.
pythondef selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

print(selection_sort([64, 25, 12, 22, 11]))  # [11, 12, 22, 25, 64]

3. Insertion Sort — O(n²)
Like sorting playing cards — pick one, insert it in the right place.
pythondef insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]    # shift right
            j -= 1
        arr[j + 1] = key           # insert in right spot
    return arr

print(insertion_sort([12, 11, 13, 5, 6]))  # [5, 6, 11, 12, 13]

4. Merge Sort — O(n log n) ⭐ MOST IMPORTANT
Divide the array in half → sort each half → merge back together.
[38, 27, 43, 3, 9, 82, 10]
         ↓ DIVIDE
[38, 27, 43]    [3, 9, 82, 10]
[38] [27, 43]   [3, 9] [82, 10]
[38] [27] [43]  [3] [9] [82] [10]
         ↓ MERGE (sorted)
[27, 38, 43]    [3, 9, 10, 82]
[3, 9, 10, 27, 38, 43, 82] ✅
pythondef merge_sort(arr):
    if len(arr) <= 1:
        return arr                       # base case!

    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])        # sort left half
    right = merge_sort(arr[mid:])        # sort right half
    return merge(left, right)


def merge(left, right):
    result = []
    i = j  = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])              # add remaining
    result.extend(right[j:])
    return result

print(merge_sort([38, 27, 43, 3, 9, 82, 10]))  # [3, 9, 10, 27, 38, 43, 82]

Sorting Full Comparison:
AlgorithmBestAverageWorstSpaceUse WhenBubble SortO(n)O(n²)O(n²)O(1)Learning onlySelection SortO(n²)O(n²)O(n²)O(1)Memory limitedInsertion SortO(n)O(n²)O(n²)O(1)Nearly sorted / smallMerge SortO(n log n)O(n log n)O(n log n)O(n)Large data ⭐

💼 Interview Problems Solved Today
Easy
1. Find Maximum Element — O(n)
pythondef find_max(arr):
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val

print(find_max([3, 7, 1, 9, 4]))   # 9
2. Check if Array is Sorted — O(n)
pythondef is_sorted(arr):
    for i in range(len(arr) - 1):    # ← subtract INSIDE range!
        if arr[i] > arr[i + 1]:
            return False
    return True                       # ← OUTSIDE loop!

print(is_sorted([1, 2, 3, 4, 5]))   # True
print(is_sorted([1, 3, 2, 4, 5]))   # False
3. Reverse Array (In-Place) — O(n)
pythondef reverse_array(arr):
    left  = 0
    right = len(arr) - 1

    while left < right:
        arr[left], arr[right] = arr[right], arr[left]  # swap!
        left  += 1
        right -= 1

    return arr

print(reverse_array([1, 2, 3, 4, 5]))   # [5, 4, 3, 2, 1]

💡 Interview note: Don't use arr[::-1] — it uses extra memory! In-place = O(1) space!


Medium
4. Second Largest Element — O(n)
pythondef second_largest(arr):
    first  = arr[0]
    second = float('-inf')    # negative infinity — smallest possible!

    for num in arr:
        if num > first:
            second = first    # old first becomes second
            first  = num      # new biggest
        elif num > second and num != first:
            second = num      # update second

    return second

print(second_largest([3, 7, 1, 9, 4]))   # 7
print(second_largest([1, 1, 1, 1]))       # handles duplicates ✅
5. Find Pair with Given Sum — O(n) ⭐ Two-Pointer Technique!
pythondef find_pair(arr, target):
    left  = 0
    right = len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return True                   # pair found!
        elif current_sum < target:
            left  += 1                    # need bigger sum → move left right
        else:
            right -= 1                    # need smaller sum → move right left

    return False                          # no pair found

print(find_pair([2, 7, 11, 15], 9))    # True  (2 + 7 = 9)
print(find_pair([2, 7, 11, 15], 20))   # False

⚠️ Note: Array must be sorted for two-pointer to work!
Unsorted? Use a set → seen = set() → also O(n).


Hard
6. Maximum Subarray Sum — Kadane's Algorithm — O(n) 🔥 FAANG FAVOURITE!
pythondef max_subarray(arr):
    max_sum     = arr[0]    # best sum seen so far
    current_sum = arr[0]    # running sum of current subarray

    for num in arr[1:]:
        # should we EXTEND current subarray OR start fresh?
        current_sum = max(num, current_sum + num)
        max_sum     = max(max_sum, current_sum)

    return max_sum

print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))   # 6
# best subarray = [4, -1, 2, 1] = 6 ✅
How Kadane's works step by step:
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

num=-2  → current=-2,  max=-2
num=1   → current=1,   max=1    ← start fresh (1 > -2+1)
num=-3  → current=-2,  max=1
num=4   → current=4,   max=4    ← start fresh (4 > -2+4)
num=-1  → current=3,   max=4
num=2   → current=5,   max=5
num=1   → current=6,   max=6   ← new best!
num=-5  → current=1,   max=6
num=4   → current=5,   max=6

Answer = 6 ✅

💡 Key insight: At each step ask — is it better to extend the current subarray, or start fresh from this number?


🏆 Summary Table
ProblemTechniqueComplexityAsked atFind MaxLinear scanO(n)AllIs SortedCompare neighboursO(n)AllReverse ArrayTwo PointerO(n)/O(1)AllSecond LargestTrack two varsO(n)Mid-levelFind PairTwo PointerO(n)FAANGMax Subarray (Kadane)Dynamic ProgrammingO(n)Google/Amazon

🔑 Key Concepts to Remember
✅ float('-inf')    → smallest possible number (use for tracking max/second)
✅ Two Pointer      → left + right pointers moving inward → O(n) instead of O(n²)
✅ In-place swap    → arr[l], arr[r] = arr[r], arr[l] → O(1) space
✅ Kadane's trick   → max(num, current_sum + num) → extend or restart!
✅ Binary Search    → MUST be sorted! Cuts in half every step → O(log n)
✅ Merge Sort       → Best general-purpose sort → O(n log n) always

🔮 What's Next — Day 22
Day 22 → Strings & Hashing
├── String manipulation problems
├── Hash maps / dictionaries for O(1) lookup
├── Anagram check, frequency count
└── Sliding window technique

Day 21 of #100DaysOfAI | Phase 2: Math & DSA for AI
🔗 https://github.com/balaravi444/AI-ML-Learning-Journey
