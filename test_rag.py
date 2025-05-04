import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np

# Add the parent directory to the path so we can import the utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.food_rag import match_detected_foods, load_food_list
from utils.volume_estimation import estimate_volumes_via_gpt4o

def test_rag_pipeline(image_path, similarity_threshold=0.85):
    
    """
    Test the RAG pipeline with a sample image
    
    Args:
        image_path: Path to the image file
        similarity_threshold: Minimum similarity score to consider a match
    """
    print(f"Testing RAG pipeline with image: {image_path}")
    print(f"Similarity threshold: {similarity_threshold}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print("\n1. Detecting foods with GPT-4o...")
    detected_foods, raw_response = estimate_volumes_via_gpt4o(image, debug=True)
    
    if not detected_foods:
        print("No foods detected in the image.")
        print(f"GPT-4o response: {raw_response}")
        return
    
    print(f"\nDetected {len(detected_foods)} food items:")
    for i, food in enumerate(detected_foods, 1):
        print(f"  {i}. {food.get('item', 'unknown')} - {food.get('volume_ml', 0):.1f} ml")
    
    print("\n2. Loading predefined food list...")
    food_list = load_food_list()
    print(f"Loaded {len(food_list)} foods from the predefined list")
    
    print("\n3. Matching detected foods against predefined list...")
    matched_foods = match_detected_foods(detected_foods, similarity_threshold)
    
    if not matched_foods:
        print("No matching foods found in the predefined list.")
        print("Try adjusting the similarity threshold or adding the detected foods to your database.")
        return
    
    print(f"\nMatched {len(matched_foods)} food items:")
    for i, food in enumerate(matched_foods, 1):
        original = food.get('original_item', 'unknown')
        matched = food.get('item', 'unknown')
        confidence = food.get('confidence', 0)
        volume = food.get('volume_ml', 0)
        
        print(f"  {i}. Original: {original}")
        print(f"     Matched: {matched}")
        print(f"     Confidence: {confidence:.2f}")
        print(f"     Volume: {volume:.1f} ml")
        print()
    
    print("\nRAG pipeline test completed successfully!")

if __name__ == "__main__":
    # Check if an image path was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default test image if none was provided
        test_dir = Path(os.path.dirname(os.path.abspath(__file__)), "test_images")
        test_dir.mkdir(exist_ok=True)
        image_path = str(test_dir / "test_food.jpg")
        print(f"No image path provided. Please place a test image at: {image_path}")
        print("Usage: python test_rag.py [image_path] [similarity_threshold]")
        sys.exit(1)
    
    # Check if a similarity threshold was provided
    similarity_threshold = 0.85
    if len(sys.argv) > 2:
        try:
            similarity_threshold = float(sys.argv[2])
        except ValueError:
            print(f"Invalid similarity threshold: {sys.argv[2]}. Using default: {similarity_threshold}")
    
    # Run the test
    test_rag_pipeline(image_path, similarity_threshold)
