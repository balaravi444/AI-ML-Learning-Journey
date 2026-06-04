# Program 1 - GitHub API
# Day 17 - Python APIs

import requests

url = "https://api.github.com/users/balaravi444"

try:
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"Username  : {data['login']}")
        print(f"Repos     : {data['public_repos']}")
        print(f"Followers : {data['followers']}")
    else:
        print(f"API Error: {response.status_code}")

except requests.exceptions.ConnectionError:
    print("No internet connection!")
