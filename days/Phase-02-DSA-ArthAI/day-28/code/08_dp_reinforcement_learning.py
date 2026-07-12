"""
Day 28 — DSA: Dynamic Programming
Topic: DP in Reinforcement Learning
Date: 15 June 2026
Author: Bala Ravi

This shows how DP concepts directly power
Reinforcement Learning — the AI behind:
- AlphaGo (beat world champion at Go!)
- ChatGPT training (RLHF!)
- Game playing AI
- Robotics control

The Bellman Equation IS a DP recurrence!
V(s) = max_a [R(s,a) + γ * V(s')]

This is EXACTLY like:
dp[i] = max(dp[i-1], dp[i-2] + nums[i])
"""


def value_iteration(
    states: list,
    actions: list,
    rewards: dict,
    transitions: dict,
    gamma: float = 0.9,
    iterations: int = 100
) -> dict:
    """
    Value Iteration — DP algorithm for RL!

    The Bellman Equation:
    V(s) = max_a [R(s,a) + γ * V(next_state)]

    This IS Dynamic Programming:
    - State = dp index
    - Value = dp value
    - Bellman equation = recurrence relation!

    Args:
        states: All possible states
        actions: All possible actions
        rewards: Reward for each (state, action)
        transitions: Next state for (state, action)
        gamma: Discount factor (0-1)
        iterations: Number of DP iterations

    Returns:
        Optimal value for each state
    """
    # Initialize all values to 0
    V = {state: 0 for state in states}

    for _ in range(iterations):
        new_V = {}

        for state in states:
            # Bellman equation — DP recurrence!
            action_values = []

            for action in actions:
                if (state, action) in transitions:
                    next_state = transitions[(state, action)]
                    reward = rewards.get((state, action), 0)

                    # V(s) = R + γ * V(next_s)
                    value = reward + gamma * V[next_state]
                    action_values.append(value)

            new_V[state] = max(action_values) if action_values else 0

        V = new_V

    return V


if __name__ == "__main__":
    print("=== DP in Reinforcement Learning ===\n")

    # Simple 4-state grid world
    # States: S0(start) → S1 → S2 → S3(goal)
    states = ["S0", "S1", "S2", "S3"]
    actions = ["move_right", "stay"]

    rewards = {
        ("S0", "move_right"): 0,
        ("S1", "move_right"): 0,
        ("S2", "move_right"): 10,  # reward at goal!
        ("S0", "stay"): -1,
        ("S1", "stay"): -1,
        ("S2", "stay"): -1,
        ("S3", "stay"): 0
    }

    transitions = {
        ("S0", "move_right"): "S1",
        ("S1", "move_right"): "S2",
        ("S2", "move_right"): "S3",
        ("S0", "stay"): "S0",
        ("S1", "stay"): "S1",
        ("S2", "stay"): "S2",
        ("S3", "stay"): "S3"
    }

    values = value_iteration(
        states, actions, rewards, transitions)

    print("Optimal Values (Bellman Equation = DP!):")
    for state, value in values.items():
        print(f"  {state}: {value:.2f}")

    print("\n=== Connection to ChatGPT Training ===")
    print("RLHF (Reinforcement Learning from Human Feedback)")
    print("Used to train ChatGPT uses the SAME Bellman equation!")
    print("V(response) = reward + γ * V(next_response)")
    print("This IS Dynamic Programming at massive scale! 🤯")
