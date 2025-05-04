import os, requests

NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID")
NUTRITIONIX_API_KEY = os.getenv("NUTRITIONIX_API_KEY")
NUTRITIONIX_API_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"

def get_nutrition_info(food_name, weight_grams):
    """
    Query NutritionIX for the given food name and weight (in grams).
    Returns a dict of nutrition info (calories, protein, fat, carbs, etc.).
    """
    if not NUTRITIONIX_APP_ID or not NUTRITIONIX_API_KEY:
        raise Exception("NutritionIX API credentials not set.")
    query_text = f"{weight_grams:.0f} grams {food_name}"
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY,
        "x-remote-user-id": "0",
        "Content-Type": "application/json"
    }
    data = {
        "query": query_text
    }
    response = requests.post(NUTRITIONIX_API_URL, json=data, headers=headers)
    response.raise_for_status()
    result = response.json()
    # The result has a "foods" list. We take the first item (best match).
    info = {}
    if result.get("foods"):
        food_data = result["foods"][0]
        # Extract key nutritional values
        info["food_name"] = food_data.get("food_name", food_name)
        info["serving_weight"] = food_data.get("serving_weight_grams", weight_grams)
        info["calories"] = food_data.get("nf_calories", 0)
        info["protein"] = food_data.get("nf_protein", 0)
        info["fat"] = food_data.get("nf_total_fat", 0)
        info["carbs"] = food_data.get("nf_total_carbohydrate", 0)
        # You could extract more fields (like fiber, sugars) if needed.
    return info
