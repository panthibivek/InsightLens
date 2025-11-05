from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import cv2
import numpy as np

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

with open("beluww.png", "rb") as f:
    image_data = f.read()

image_part = types.Part.from_bytes(
    data=image_data, 
    mime_type="image/png"
)

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[ 
        image_part, 
        """Identify the main object in the image and provide a JSON response with:
        - box_2d: bounding box coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000
        - label: the specific name of the object (e.g., "cat", "laptop", "bottle" - NOT generic terms like "the object")
        
        Format: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "specific_object_name"}]
        Only return the JSON array, nothing else.""" 
    ],
)

print("API Response:")
print(response.text)


try:

    response_text = response.text.strip()
    if response_text.startswith("```json"):
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif response_text.startswith("```"):
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    detections = json.loads(response_text)
    

    img = cv2.imread("beluww.png")
    height, width = img.shape[:2]
    
    # Draw bounding boxes for each detection
    for detection in detections:
        box = detection["box_2d"]
        label = detection["label"]
        
        # Denormalize coordinates, turning it into pixel coordinates.
        ymin = int(box[0] * height / 1000)
        xmin = int(box[1] * width / 1000)
        ymax = int(box[2] * height / 1000)
        xmax = int(box[3] * width / 1000)
        

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        cv2.putText(
            img,
            label,
            (xmin + 5, ymin - 5),
            font,
            font_scale,
            (0, 0, 255),
            font_thickness
        )
    
    output_filename = "beluww_with_bbox.png"
    cv2.imwrite(output_filename, img)
    print(f"\nImage saved as: {output_filename}")
    
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print("Raw response:", response.text)
except Exception as e:
    print(f"Error processing image: {e}")
