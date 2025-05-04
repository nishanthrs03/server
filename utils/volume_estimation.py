# E:\food_calorie_app\server\utils\volume_estimation.py

import os
import cv2
import base64
import json
import time
from google import genai
from dotenv import load_dotenv   # noqa: E402
import os
import os
import base64
import cv2
import requests
import json
import time
load_dotenv()                     

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

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


def estimate_volumes_via_gpt4o(image_cv, debug=False):
    """
    Uses GPT‑4o to identify food items and their volumes (in mL), using a 
    credit card in the image for scale. 
    Returns (parsed_list, raw_response_string).
      - parsed_list: a list of {"item": <str>, "volume_ml": <float>}
      - raw_response_string: the raw text from GPT-4o for debugging
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    # 1. Resize if too large
    max_dim = 768
    h, w = image_cv.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_cv = cv2.resize(image_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Encode as JPEG
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    ok, buffer = cv2.imencode(".jpg", image_cv, encode_params)
    if not ok:
        raise RuntimeError("Failed to encode image for GPT-4o volume estimation.")

    # 3. Base64 data URI
    b64_img = base64.b64encode(buffer).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_img}"

    # 4. System prompt with more detail
    system_prompt = (
        "You are an advanced vision AI with the ability to interpret images. "
        "You see an image.In the image are one or more "
        "food items. Please do the following:\n\n"
        "1. Identify each distinct food item.\n"
        "2. Estimate the volume of each item\n"
        "3. Return ONLY valid JSON (no extra text) in this format:\n"
        "[\n"
        "  {\n"
        '    "item": "food name",\n'
        '    "volume_ml": <number>\n'
        "  },\n"
        "  ...\n"
        "]\n\n"
        "If you cannot detect any food, return an empty JSON array []."
    )

    user_content = [
        {
            "type": "text",
            "text": "Here is the image. Identify each food item and its volume in mL"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": data_uri,
                "detail": "low"
            }
        }
    ]

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 250,
        "temperature": 0.0
    }

    data = _chat_with_retry(payload)
    raw_response = data["choices"][0]["message"]["content"].strip()

    # Attempt to parse JSON
    try:
        volumes_list = json.loads(raw_response)
        if not isinstance(volumes_list, list):
            volumes_list = []
    except json.JSONDecodeError:
        volumes_list = []

    if debug:
        # Return both parsed list and raw text for debugging
        return volumes_list, raw_response
    else:
        return volumes_list, None