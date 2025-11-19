from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import multipart

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

class Question(BaseModel):
    question: str

@app.post("/ask_image")
async def ask_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                types.Part.from_text(
                    text=(
                        """Identify the main object in the image and provide a JSON response with:
                    - box_2d: bounding box coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000
                    - label: the specific name of the object (e.g., "cat", "laptop", "bottle" - NOT generic terms like "the object")

                    Format: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "specific_object_name"}]
                    Only return the JSON array, nothing else."""
                    )
                ),
            ],
        )
    ],
)

        return {"answer": response.text}
    except Exception as e:
        return {"error": str(e)}
