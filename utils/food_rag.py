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

# Load the Universal Sentence Encoder model once
_USE_MODEL = None
_DETECTED_FOOD_EMBEDDINGS_CACHE = {}

def get_embedding(text: str | list) -> List[List[float]]:
    """Get embedding using Universal Sentence Encoder"""
    global _USE_MODEL
    
    try:
        # Load model if not already loaded
        if _USE_MODEL is None:
            _USE_MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            
        # Handle both single strings and lists
        if isinstance(text, list):
            # Process batch
            embeddings = _USE_MODEL(text).numpy().tolist()
            # Update cache
            for i, item in enumerate(text):
                _DETECTED_FOOD_EMBEDDINGS_CACHE[item] = embeddings[i]
            return embeddings
        else:
            # Process single item
            embedding = _USE_MODEL([text]).numpy()[0].tolist()
            _DETECTED_FOOD_EMBEDDINGS_CACHE[text] = embedding
            return [embedding]

    except Exception as e:
        print(f"Error generating embedding with USE model: {e}")
        return []


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

def _chat_with_retry(url, headers, json_data, retries=3, base_delay=2):
    """Helper function to retry API calls with exponential backoff"""
    for attempt in range(retries):
        resp = requests.post(url, headers=headers, json=json_data, timeout=30)
        if resp.status_code < 400:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(base_delay ** attempt)
            continue
        resp.raise_for_status()
# def get_embedding(text: str | list) -> List[List[float]]:
#     """Get embedding for a text using Gemini's embedding API"""
#     try:
#         from google import genai
#         if not os.getenv("GEMINI_API_KEY"):
#             raise ValueError("GEMINI_API_KEY not found in environment variables")
#         api_key=os.getenv("GEMINI_API_KEY")
#         client = genai.Client(api_key=api_key)

#         if isinstance(text, list):
#             results = client.models.embed_content(
#                 model="gemini-embedding-exp-03-07",
#                 contents=text)
#             embeddings = [result.embeddings[i].values for i in range(len(text))]
#             for i, item in enumerate(text):
#                 _detected_food_embeddings_cache[item] = embeddings[i]
#             return embeddings
#         else:
#             result = client.models.embed_content(
#                 model="gemini-embedding-exp-03-07",
#                 contents=text)
#             embedding = result.embeddings[0].values
#             _detected_food_embeddings_cache[text] = embedding
#             return [embedding]

#     except Exception as e:
#         print(f"Error generating embedding with Gemini: {e}")
#         return []  # Or handle the error as appropriate for your application





def load_food_embeddings() -> Dict[int, List[float]]:
    """Load or generate embeddings for the predefined food list"""
    global _food_embeddings_cache
    
    print("=== Starting load_food_embeddings ===")
    
    # If embeddings are already loaded, return them
    if _food_embeddings_cache:
        print(f"Using cached embeddings in memory. Keys: {len(_food_embeddings_cache)}")
        return _food_embeddings_cache
    
    # Path to the embeddings cache file
    embeddings_cache_path = CACHE_DIR / "1food_embeddings.pkl"
    print(f"Embeddings cache path: {embeddings_cache_path}")
    print(f"Cache file exists: {embeddings_cache_path.exists()}")
    
    # Load embeddings from cache if available
    if embeddings_cache_path.exists():
        try:
            print("Attempting to load embeddings from cache file...")
            with open(embeddings_cache_path, 'rb') as f:
                _food_embeddings_cache = pickle.load(f)
                print(f"Successfully loaded embeddings from cache. Count: {len(_food_embeddings_cache)}")
                
                # Debug the loaded embeddings
                print("Checking embedding shapes:")
                for food_id, embedding in list(_food_embeddings_cache.items())[:3]:  # Show first 3 for brevity
                    print(f"  Food ID {food_id}: Type={type(embedding)}, Shape={np.array(embedding).shape if hasattr(embedding, '__iter__') else 'scalar'}")
                
                # Check if any embeddings are empty
                empty_embeddings = [food_id for food_id, embedding in _food_embeddings_cache.items() 
                                  if hasattr(embedding, '__iter__') and len(embedding) == 0]
                if empty_embeddings:
                    print(f"WARNING: Found {len(empty_embeddings)} empty embeddings in cache: {empty_embeddings[:5]}")
                
                return _food_embeddings_cache
        except Exception as e:
            print(f"Error loading embeddings cache: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No cache file found. Will generate new embeddings.")
    
    # Generate embeddings for all foods in the list
    print("Loading food list...")
    food_list = load_food_list()
    print(f"Loaded {len(food_list)} foods. Generating embeddings...")

    food_names = [food["name"] for food in food_list]
    print(f"Generating embeddings for {len(food_names)} foods in batches...")

    embeddings = {}
    batch_size = 100
    for i in range(0, len(food_names), batch_size):
        batch_names = food_names[i:i + batch_size]
        print(f"Generating embeddings for batch {i//batch_size + 1} of {len(food_names) // batch_size + 1}...")
        try:
            batch_embeddings = get_embedding(batch_names)
            if batch_embeddings:
                for j, embedding in enumerate(batch_embeddings):
                    food_id = food_list[i + j]["id"]
                    embeddings[food_id] = embedding
                    emb_array = np.array(embedding) if hasattr(embedding, '__iter__') else embedding
                    print(f"  Generated embedding for {food_id}: Type={type(embedding)}, Shape={emb_array.shape if hasattr(emb_array, 'shape') else 'scalar'}")
                    if hasattr(embedding, '__iter__') and len(embedding) == 0:
                        print(f"  WARNING: Empty embedding generated for {food_id}: '{food_names[i+j]}'")
        except Exception as e:
            print(f"  ERROR generating embeddings for batch: {e}")
            import traceback
            traceback.print_exc()
    
    # Save embeddings to cache
    print(f"Generated {len(embeddings)} embeddings. Saving to cache...")
    try:
        # Ensure the cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory: {CACHE_DIR} (exists: {CACHE_DIR.exists()})")
        
        with open(embeddings_cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Successfully saved embeddings to {embeddings_cache_path}")
    except Exception as e:
        print(f"ERROR saving embeddings cache: {e}")
        import traceback
        traceback.print_exc()
    
    _food_embeddings_cache = embeddings
    print("=== Completed load_food_embeddings ===")

    return embeddings


def load_food_list() -> List[Dict[str, Any]]:
    """Load the predefined food list"""
    print("=== Starting load_food_list ===")
    
    # Create the data directory if it doesn't exist
    data_dir = FOOD_LIST_PATH.parent
    print(f"Food list directory: {data_dir}")
    data_dir.mkdir(exist_ok=True)
    print(f"Directory exists after mkdir: {data_dir.exists()}")
    
    # Create a default food list if it doesn't exist
    if not FOOD_LIST_PATH.exists():
        print(f"Food list file not found at {FOOD_LIST_PATH}. Creating default...")
        default_foods = [
            {"id": 1, "name": "apple"},
            {"id": 2, "name": "banana"},
            {"id": 3, "name": "orange"},
            {"id": 4, "name": "chicken breast"},
            {"id": 5, "name": "rice"},
            {"id": 6, "name": "broccoli"}
        ]
        try:
            with open(FOOD_LIST_PATH, 'w') as f:
                json.dump(default_foods, f, indent=2)
            print(f"Created default food list with {len(default_foods)} items")
        except Exception as e:
            print(f"ERROR creating default food list: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Found existing food list at {FOOD_LIST_PATH}")
    
    # Load the food list
    try:
        with open(FOOD_LIST_PATH, 'r') as f:
            food_list = json.load(f)
        print(f"Successfully loaded food list with {len(food_list)} items")
        # Print the first few foods for debugging
        for food in food_list[:3]:
            print(f"  Food: ID={food['id']}, Name='{food['name']}'")
        print("=== Completed load_food_list ===")
        return food_list
    except Exception as e:
        print(f"ERROR loading food list: {e}")
        import traceback
        traceback.print_exc()
        print("=== Failed load_food_list ===")
        return []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not a or not b:
        print("⚠️ Empty embedding(s) passed to cosine_similarity.")
        return 0.0
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        print("⚠️ Zero-vector embedding(s) passed to cosine_similarity.")
        return 0.0
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return sim
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
                         similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
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

            similarity = cosine_similarity(detected_embedding[0], food_embedding)  # [0] gets first embedding
            # print(f"    ↪ Compared with '{food_id_to_name[food_id]}': score={float(similarity):.4f}")

            if food_id != 41:  # Exclude "Unknown" (ID 41) unless it's the only match
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_id = food_id
            elif best_match_id is None: # if 41 is the only possible match
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

def add_food_to_list(food_name: str) -> Dict[str, Any]:
    """Add a new food to the predefined list"""
    food_list = load_food_list()
    
    # Check if food already exists
    for food in food_list:
        if food["name"].lower() == food_name.lower():
            return food
    
    # Generate a new ID
    new_id = max([food["id"] for food in food_list], default=0) + 1
    
    # Add the new food
    new_food = {"id": new_id, "name": food_name}
    food_list.append(new_food)
    
    # Save the updated food list
    with open(FOOD_LIST_PATH, 'w') as f:
        json.dump(food_list, f, indent=2)
    
    # Generate and cache embedding for the new food
    embedding = get_embedding(food_name)
    
    # Update the embeddings cache
    _food_embeddings_cache[new_id] = embedding
    
    # Save the updated embeddings cache
    embeddings_cache_path = CACHE_DIR / "food_embeddings.pkl"
    with open(embeddings_cache_path, 'wb') as f:
        pickle.dump(_food_embeddings_cache, f)
    
    return new_food




