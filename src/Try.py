from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

with open("cat.jpeg", "rb") as f:
    image_data = f.read()

image_part = types.Part.from_bytes(
    data=image_data, 
    mime_type="image/png"
)


client = genai.Client(api_key=api_key)



response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[image_part, "Identify the object in the image and give json box_2d with values ymin, xmin, ymax, xmax coordinates of its bounding box and the object's name. normalize to 0-1000"],
)

print(response.text)