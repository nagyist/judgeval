import os
from openai import OpenAI
from dotenv import load_dotenv
from tools import get_weather
from prompts import SYSTEM_PROMPT, create_outfit_prompt
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer
from judgeval.common.tracer import Tracer

judgment = Tracer()

class WeatherOutfitPlanner:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.judgment_client = JudgmentClient()
        
    @judgment.observe(span_type="outfit_recommendation")
    def get_outfit_recommendation(self, city, occasion, gender_preference):
        """Get outfit recommendation based on weather, occasion, and gender preference."""
        # Get weather data
        weather_data = get_weather(city)
        if not weather_data:
            return "Sorry, couldn't fetch weather data for that location."
            
        # Create prompt
        prompt = create_outfit_prompt(weather_data, occasion, gender_preference)
        
        # Get GPT recommendation
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        recommendation = response.choices[0].message.content
        
        # Create example for evaluation
        example = Example(
            input=prompt,
            actual_output=recommendation,
            retrieval_context=[
                f"Weather: {weather_data}",
                f"Occasion: {occasion}",
                f"Style Preference: {gender_preference}"
            ]
        )
        
        # Run evaluation
        results = self.judgment_client.run_evaluation(
            examples=[example],
            scorers=[FaithfulnessScorer(threshold=0.7)],
            model="gpt-4",
            project_name="weather_outfit_planner",
            eval_run_name=f"outfit_recommendation_{city}_{occasion}"
        )
        
        return recommendation, results

def main():
    planner = WeatherOutfitPlanner()
    
    # Get user input
    city = input("Enter city (e.g., 'San Francisco'): ")
    occasion = input("Enter occasion (e.g., 'work meeting', 'casual dinner'): ")
    gender_preference = input("Enter style preference (masculine/feminine/gender-neutral): ").lower()
    
    # Validate gender_preference input
    valid_preferences = ['masculine', 'feminine', 'gender-neutral']
    if gender_preference not in valid_preferences:
        gender_preference = 'gender-neutral'
        print("Invalid preference, defaulting to gender-neutral style.")
    
    # Get recommendation and evaluation
    recommendation, evaluation_results = planner.get_outfit_recommendation(city, occasion, gender_preference)
    
    print("\nOutfit Recommendation:")
    print(recommendation)
    
    

if __name__ == "__main__":
    main()