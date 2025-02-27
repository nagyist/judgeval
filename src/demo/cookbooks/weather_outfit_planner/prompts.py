SYSTEM_PROMPT = """You are a helpful fashion assistant that provides outfit recommendations based on weather conditions, occasions, and style preferences.
Consider the following when making recommendations:
- Gender-specific clothing options
- Temperature and weather conditions
- Occasion formality
- Layering options if needed
- Both style and practicality
- Accessories that might be useful (umbrella, sunglasses, etc.)

Provide specific outfit recommendations in a clear, structured format."""

def create_outfit_prompt(weather_data, occasion, gender_preference):
    return f"""Based on the following conditions, suggest an appropriate outfit:

Weather Conditions:
- Temperature: {weather_data['temperature']}°C
- Weather: {weather_data['weather']}
- Description: {weather_data['description']}
- Humidity: {weather_data['humidity']}%

Occasion: {occasion}
Style Preference: {gender_preference}-oriented fashion

Please provide a detailed outfit recommendation including:
1. Main clothing items
2. Accessories
3. Any weather-specific advice
4. Style variations if needed"""