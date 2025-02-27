import os
import requests
from judgeval.common.tracer import Tracer

judgment = Tracer()

@judgment.observe(span_type="weather_tool")
def get_weather(city):
    """Get current weather data for a city."""
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if not API_KEY:
        print("Error: OPENWEATHER_API_KEY not found in environment variables")
        return None
        
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"  # for Celsius
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "weather": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"]
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing weather data: {e}")
        return None