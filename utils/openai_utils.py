import os
import base64
import cv2
import requests
import json
import time

# Load OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------------------------------- #
#  Simple retry wrapper – handles 429 / 5xx with exponential back‑off
# --------------------------------------------------------------------------- #
def _chat_with_retry(payload, retries=3, base_delay=2):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code < 400:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(base_delay ** attempt)  # e.g., 1 s, 2 s, 4 s...
            continue
        resp.raise_for_status()

# --------------------------------------------------------------------------- #
#  Function: identify_food_item
#  Uses GPT‑4o Vision to identify the food item in a given image (cv2 array).
#  The image is resized and JPEG‑compressed to keep the request payload small.
# --------------------------------------------------------------------------- #
def identify_food_item(image_cv):
    """
    Identify the food item in the given image (cv2 image array) using GPT‑4o Vision.
    Returns the food name as a string.
    """
    # 1. Resize the image to cap the longest edge at 768 px (keeping aspect ratio)
    max_edge = 768
    h, w = image_cv.shape[:2]
    if max(h, w) > max_edge:
        scale = max_edge / float(max(h, w))
        image_cv = cv2.resize(image_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. JPEG‑compress at 50% quality (adjusted for smaller payload)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    ok, buffer = cv2.imencode(".jpg", image_cv, encode_params)
    if not ok:
        raise RuntimeError("Could not encode image")
    
    # 3. Encode to base‑64 and prepare a data‑URI
    b64_img = base64.b64encode(buffer).decode('utf-8')
    data_uri = f"data:image/jpeg;base64,{b64_img}"

    # 4. Build the payload for the GPT‑4o vision API call
    payload = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Identify the food item in this picture."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri,
                        "detail": "low"  # request fewer tokens in the reply
                    }
                }
            ]
        }],
        "max_tokens": 30
    }

    # 5. Call the OpenAI API
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions",
                         headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    
    # 6. Parse and return the food name (use first line if multiple)
    food_name = result["choices"][0]["message"]["content"].split('\n', 1)[0].strip()
    return food_name

# --------------------------------------------------------------------------- #
#  Function: get_food_density
#  Uses GPT‑4o‑mini to get the average density (in g/mL) for a given food name.
#  Expects GPT‑4o‑mini to return only the numeric value.
# --------------------------------------------------------------------------- #
def get_food_density(food_name):
    payload = {
        "model": "gpt-4o-mini",
        "max_tokens": 10,
        "messages": [{
            "role": "user",
            "content": f"Average density of {food_name} in g/mL? Reply with only the number."
        }]
    }

    data = _chat_with_retry(payload)
    answer = data["choices"][0]["message"]["content"].strip()

    try:
        # Extract numeric value from the answer string
        density_value = float(''.join(ch for ch in answer if ch.isdigit() or ch == '.'))
    except ValueError:
        density_value = None

    return density_value
