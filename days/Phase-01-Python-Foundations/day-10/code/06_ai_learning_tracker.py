# Problem 6 - AI Learning Tracker Using Files
# Day 10 - File Handling

def add_entry(day, topic):
    with open("learning_tracker.txt", "a") as f:
        f.write(f"Day {day}: {topic}\n")
    print(f"Day {day} entry saved!")

def show_progress():
    print("\n--- AI Learning Progress ---")
    try:
        with open("learning_tracker.txt", "r") as f:
            for line in f:
                print(line.strip())
    except FileNotFoundError:
        print("No entries yet. Start tracking!")

add_entry(1, "Python Basics")
add_entry(2, "Operators and Conditions")
add_entry(3, "Control Flow")
add_entry(10, "File Handling")
show_progress()
