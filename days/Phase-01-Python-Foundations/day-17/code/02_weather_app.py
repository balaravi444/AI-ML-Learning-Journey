# Program 2 - Live Weather App
# Day 17 - Python APIs

import requests

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

        print(f"City        : {data['name']}")
        print(f"Temperature : {data['main']['temp']}°C")
        print(f"Feels Like  : {data['main']['feels_like']}°C")
        print(f"Humidity    : {data['main']['humidity']}%")
        print(f"Weather     : {data['weather'][0]['description']}")
    else:
        print("City not found!")

except requests.exceptions.ConnectionError:
    print("No internet connection!")
