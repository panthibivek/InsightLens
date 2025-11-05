from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import cv2

# Load API key
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Load image
image_path = "cat.jpeg"
with open(image_path, "rb") as f:
    image_bytes = f.read()

image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

# Ask Gemini to detect object
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        image_part,
        """Return JSON only:
        [{"box_2d":[ymin,xmin,ymax,xmax],"label":"object_name"}]
        Values normalized to 0-1000."""
    ]
)

# Parse JSON safely
response_text = response.text
response_text = response_text.replace("```json", "").replace("```", "").strip()
detections = json.loads(response_text)

# Read image and draw boxes
img = cv2.imread(image_path)
h, w = img.shape[:2]

for det in detections:
    ymin, xmin, ymax, xmax = det["box_2d"]
    ymin, xmin, ymax, xmax = (
        int(ymin * h / 1000), int(xmin * w / 1000),
        int(ymax * h / 1000), int(xmax * w / 1000)
    )

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    cv2.putText(img, det["label"], (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Save result
output = "cat_with_bbox.png"
cv2.imwrite(output, img)
print(f"Saved: {output}")
