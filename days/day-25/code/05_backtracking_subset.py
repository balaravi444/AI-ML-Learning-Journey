# Program 5 — Backtracking: Generate All Subsets
# Day 25 — Recursion & Backtracking

# ============================================
# Problem: Generate all subsets of a list
# Input:  [1, 2, 3]
# Output: [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
# ============================================

def generate_subsets(arr):
    result = []

    def backtrack(start, current):
        # Every state is a valid subset — add it!
        result.append(current[:])      # copy current subset

        for i in range(start, len(arr)):
            current.append(arr[i])     # CHOOSE
            backtrack(i + 1, current)  # EXPLORE
            current.pop()              # UN-CHOOSE (backtrack!)

    backtrack(0, [])
    return result


print("=== All Subsets ===")
subsets = generate_subsets([1, 2, 3])
for s in subsets:
    print(s)

print(f"\nTotal subsets: {len(subsets)}")   # 2^n = 8


# ============================================
# How backtracking works here:
# ============================================
# Start: current = []
#
# Choose 1:  current = [1]
#   Choose 2:  current = [1, 2]
#     Choose 3:  current = [1, 2, 3] ← save!
#     Undo 3:    current = [1, 2]
#   Undo 2:    current = [1]
#   Choose 3:  current = [1, 3] ← save!
#   Undo 3:    current = [1]
# Undo 1:    current = []
# Choose 2:  current = [2]
#   ... and so on!
