# utils/volume_estimation.py

import cv2
import numpy as np

def estimate_volume(image_or_path):
    """
    Example function to estimate the volume (in mL) of the given segment.
    For now, returns a dummy value or uses simple logic:
    - Convert image to grayscale
    - Find area with OpenCV
    - Multiply by an assumed thickness
    - Return approximate volume
    """
    if isinstance(image_or_path, str):
        # If it's a file path
        image = cv2.imread(image_or_path)
    else:
        # If it's an image array
        image = image_or_path
    
    if image is None:
        raise FileNotFoundError("Unable to load the segment image for volume estimation.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Simple threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area_pixels = 0
    for cnt in contours:
        total_area_pixels += cv2.contourArea(cnt)
    
    # Convert pixels to real-world area (assume some calibration)
    px_to_cm = 1.0 / 50.0  # 1 px ~ 0.02 cm
    area_cm2 = total_area_pixels * (px_to_cm**2)
    
    # Assume thickness of 2 cm
    thickness_cm = 2.0
    volume_cm3 = area_cm2 * thickness_cm
    volume_ml = volume_cm3  # 1 cm^3 = 1 mL
    
    return volume_ml
