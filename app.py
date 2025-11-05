import os
import json
from google import genai
from dotenv import load_dotenv
from google.genai import types

import cv2

load_dotenv()
API_KEY = os.getenv("API_KEY")

def myfunction(myprompt: str, image_url:str):
    client = genai.Client(api_key=API_KEY)
    with open(image_url, 'rb') as f:
        image_bytes = f.read()
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
            types.Part.from_text(
                text=myprompt
            ),
        ],
    )
    # print(response.text)
    return response.text

    
def build_rectangle(coords: list):
    image = cv2.imread(path_to_image)
    h, w = image.shape[:2]

    x_min = int(coords[0] / 1000 * w)
    y_min = int(coords[1] / 1000 * h)
    x_max = int(coords[2] / 1000 * w)
    y_max = int(coords[3] / 1000 * h)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.imshow("Image with Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



path_to_image = "E:/InsightLens/images/tiger.jpg"

response=myfunction("""
                    You are a vision API. Detect the main object and return only valid JSON.

RULES:
- Output only JSON (no text, no code fences, no explanations)
- JSON format must be exactly:
{"box_2d":[x_min, y_min, x_max, y_max]} normalized to integers 0–1000
- Coordinate values MUST be integers
- keys must match exactly: "box_2d"
- Do NOT include label, confidence, or any extra fields
- Do NOT wrap the JSON in ```json or backticks
""", path_to_image)

data=json.loads(response)
coordinates=data["box_2d"]
if not coordinates:
    print("No 'box_2d' found in the response.")
build_rectangle(coordinates)


