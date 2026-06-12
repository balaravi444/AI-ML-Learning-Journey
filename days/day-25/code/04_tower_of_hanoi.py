# Program 4 — Tower of Hanoi
# Day 25 — Recursion & Backtracking
# Classic recursion problem asked in FAANG interviews!

# ============================================
# Problem:
# Move n disks from source to destination
# Rules:
#   1. Only one disk at a time
#   2. Larger disk can never go on smaller disk
#   3. Can use auxiliary peg as helper
# ============================================

def tower_of_hanoi(n, source, destination, auxiliary):
    # Base case — only one disk, just move it!
    if n == 1:
        print(f"Move disk 1 from {source} → {destination}")
        return

    # Step 1 — Move n-1 disks from source to auxiliary
    tower_of_hanoi(n-1, source, auxiliary, destination)

    # Step 2 — Move largest disk to destination
    print(f"Move disk {n} from {source} → {destination}")

    # Step 3 — Move n-1 disks from auxiliary to destination
    tower_of_hanoi(n-1, auxiliary, destination, source)


print("=== Tower of Hanoi — 2 disks ===")
tower_of_hanoi(2, 'A', 'C', 'B')
# Move disk 1 from A → B
# Move disk 2 from A → C
# Move disk 1 from B → C

print("\n=== Tower of Hanoi — 3 disks ===")
tower_of_hanoi(3, 'A', 'C', 'B')

# Complexity:
# n disks → 2^n - 1 moves
# Time: O(2^n)  — exponential!
# Space: O(n)   — recursion depth

print("\n=== Move Count ===")
for n in range(1, 6):
    moves = (2**n) - 1
    print(f"{n} disks → {moves} moves")
# 1 disk  → 1 move
# 2 disks → 3 moves
# 3 disks → 7 moves
# 4 disks → 15 moves
# 5 disks → 31 moves
