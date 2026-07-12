# Problem 5 - Function-Based Eligibility Checker
# Day 04 - Functions & Variable Scope

def check_eligibility(marks, age):
  if marks >= 75 and age >= 18:
    return "Eligibile for AI program"
  else:
    return "Not Eligible"
  print(check_eligibility(80, 20))
