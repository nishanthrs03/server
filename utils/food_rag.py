import os
import json
import numpy as np
import requests
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional, Any
import pickle

from dotenv import load_dotenv   # noqa: E402
import os
import tensorflow as tf
import tensorflow_hub as hub



import os
import time
import json
from typing import List, Union, Dict, Optional
import requests
from dotenv import load_dotenv


# Load the Universal Sentence Encoder model once
_USE_MODEL = None
_DETECTED_FOOD_EMBEDDINGS_CACHE = {}

_USE_MODEL = None
_DETECTED_FOOD_EMBEDDINGS_CACHE: Dict[str, List[float]] = {}




load_dotenv()                     

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cache directory for embeddings
CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
CACHE_DIR.mkdir(exist_ok=True)

# Path to the predefined food list
FOOD_LIST_PATH = Path(os.path.dirname(os.path.abspath(__file__)), "..", "data", "food_list.json")

# Embedding model to use
EMBEDDING_MODEL = "text-embedding-3-small"

# Cache for food embeddings
_food_embeddings_cache = {}
_detected_food_embeddings_cache = {}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDINGS_MODEL = "text-embedding-3-small"  # You can also use "text-embedding-3-large" for higher quality
EMBEDDINGS_ENDPOINT = "https://api.openai.com/v1/embeddings"



from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)  # Replace or load securely

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding




from typing import List
import numpy as np

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not a or not b:
        print("⚠️ Empty embedding(s) passed to cosine_similarity.")
        return 0.0
    
    # Convert to numpy arrays
    a_np = np.array(a)
    b_np = np.array(b)
    
    # Check for zero vectors
    if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
        print("⚠️ Zero-vector embedding(s) passed to cosine_similarity.")
        return 0.0
    
    # Calculate similarity and ensure we return a scalar
    sim = np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    
    # Convert to Python float to ensure it's a scalar
    # This handles cases where sim is a 0-dim array or has shape ()
    return float(sim)
def load_embeddings_from_file(file_path: str) -> Tuple[Dict[int, List[float]], Dict[int, str]]:
    """Load embeddings from existing JSON file"""
    try:
        with open(file_path, 'r') as f:
            embeddings_data = json.load(f)
            
        food_embeddings = {}
        food_id_to_name = {}
        
        for item in embeddings_data:
            food_id = item['id']
            food_embeddings[food_id] = item['embedding']
            food_id_to_name[food_id] = item['name']
            
        print(f"Loaded {len(food_embeddings)} embeddings from {file_path}")
        return food_embeddings, food_id_to_name
        
    except Exception as e:
        print(f"Error loading embeddings file: {e}")
        return {}, {}


def match_detected_foods(detected_foods: List[Dict[str, Any]], 
                         similarity_threshold: float = 0.70) -> List[Dict[str, Any]]:
    """
    Match detected foods against the predefined food list using semantic similarity
    """
    print(f"\n🔍 Matching {len(detected_foods)} detected food(s)...")
    

    # Load embeddings and mappings from file
    embeddings_path = CACHE_DIR / "food_embeddings.json"
    food_embeddings, food_id_to_name = load_embeddings_from_file(str(embeddings_path))
    
    if not food_embeddings:
        print("⛔ Could not load food embeddings")
        return []

    matched_foods = []

    for detected_food in detected_foods:
        food_name = detected_food.get("item", "")
        volume_ml=detected_food.get("volume_ml", "")
        if not food_name:
            print("⛔ Skipping empty item in detected_food.")
            continue

        print(f"\n➡️ Trying to match detected food: '{food_name}'")

        detected_embedding = get_embedding(food_name)
        if not detected_embedding or len(detected_embedding) == 0:
            print(f"⚠️ No embedding returned for '{food_name}'. Skipping.")
            continue

        best_match_id = None
        best_match_score = 0.0

        for food_id, food_embedding in food_embeddings.items():
            if not food_embedding:
                continue

            similarity = cosine_similarity(detected_embedding, food_embedding)

            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = food_id
        
        print(f"✅ Best match: {food_id_to_name.get(best_match_id, 'N/A')} (ID: {best_match_id}) with score {float(best_match_score):.4f}")

        if best_match_score >= similarity_threshold and best_match_id is not None:
            matched_food = detected_food.copy()
            matched_food["original_item"] = food_name
            matched_food["item"] = food_id_to_name[best_match_id]
            matched_food["volume_ml"]=volume_ml
            matched_food["confidence"] = float(best_match_score)  # Ensure float
            matched_food["food_id"] = best_match_id
            matched_foods.append(matched_food)
            print(f"🎯 Added to matched_foods ✅")
        else:
            print(f"❌ Match score below threshold ({similarity_threshold}). Skipping.")

    print(f"\n✅ Matching complete. Total matched foods: {len(matched_foods)}")
    return matched_foods

