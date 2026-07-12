# Program 3 - Save API Data to JSON
# Day 17 - Python APIs
# Real data collection pipeline!

import requests
import json

API_KEY = "your_api_key_here"
city = input("Enter city name: ")

url = "https://api.openweathermap.org/data/2.5/weather"

params = {
    "q": city,
    "appid": API_KEY,
    "units": "metric"
}

try:
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        # Save to JSON file
        with open("weather.json", "w") as f:
            json.dump(data, f, indent=4)

        print("✅ Weather data saved to weather.json!")

        # Load back and print
        with open("weather.json", "r") as f:
            loaded = json.load(f)

        print(f"City        : {loaded['name']}")
        print(f"Temperature : {loaded['main']['temp']}°C")
        print(f"Humidity    : {loaded['main']['humidity']}%")
    else:
        print("City not found!")

except requests.exceptions.ConnectionError:
    print("No internet connection!")
