import os
import json
import logging
from google import genai
from dotenv import load_dotenv

from google.genai import types
from fastapi import FastAPI, Form, HTTPException
from starlette.responses import JSONResponse


logging.basicConfig(level=logging.INFO)

load_dotenv()
API_KEY = os.getenv("API_KEY")
print(f"API_KEY loaded: {API_KEY}")
if not API_KEY:
    logging.error("API_KEY environment variable is not set. Please set it before running.")

app = FastAPI(title="Simple Gemini Tiger Part Detector")

# --- Hardcoded Image Path (READ FROM SERVER'S LOCAL FILE SYSTEM) ---
# WARNING: This path must exist on the machine running the FastAPI server.
LOCAL_IMAGE_PATH = "E:/InsightLens/images/tiger.jpg"


# --- Synchronous Gemini API Call Function (Simple) ---
# Note: Removed async and complex retry logic for simplicity.
def get_part_detection_sync(prompt_text: str, image_bytes: bytes):
    """
    Calls the Gemini API to detect an object in an image and return JSON.
    This is a synchronous function call.
    """
    if not API_KEY:
        raise Exception("API Key is missing. Cannot proceed with API call.")

    client = genai.Client(api_key=API_KEY)
    
    # MIME type is based on the expected .jpg file
    mime_type = "image/jpeg"

    # Make the synchronous API call
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            types.Part.from_text(text=prompt_text)
        ],
    )
    
    return response.text

# --- Simple Welcome Endpoint (GET) ---

@app.get("/")
def welcome():
    """Simple check to see if the server is running."""
    return {"message": "Server is running! To use the main function, send a POST request to /detect_tiger_part"}

# --- Main Detection Endpoint (POST) ---

@app.post("/detect_tiger_part", summary="Detects a specific part of a tiger using a hardcoded local image path.")
def detect_tiger_part_api(
    part_of_tiger: str = Form(..., description="E.g., 'the tiger's nose' or 'the main tiger object'.")
):
    """
    Accepts the name of a part of the tiger, reads the image from the hardcoded 
    path, and returns a normalized 2D bounding box (0-1000) in JSON format.
    """
    
    # 1. Read the image file content from the hardcoded path
    try:
        if not os.path.exists(LOCAL_IMAGE_PATH):
            logging.error(f"File not found: {LOCAL_IMAGE_PATH}")
            raise HTTPException(
                status_code=500, # Using 500 because this is a server setup issue
                detail=f"Image file not found at hardcoded path: {LOCAL_IMAGE_PATH}. Please fix the path."
            )
            
        with open(LOCAL_IMAGE_PATH, 'rb') as f:
            image_bytes = f.read()

    except Exception as e:
        logging.error(f"Error reading local image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read local image file due to an unexpected error: {e}")

    # 2. Construct the strict, dynamic prompt
    detection_prompt = f"""
        You are a vision API. Detect the object described by the user: '{part_of_tiger}' in the image and return ONLY valid JSON.
        If the part '{part_of_tiger}' is not clearly visible or not present, return an empty box: {{"box_2d":[0, 0, 0, 0]}}.

        RULES:
        - Output only JSON (no text, no code fences, no explanations)
        - JSON format must be exactly: {{"box_2d":[x_min, y_min, x_max, y_max]}} normalized to integers 0–1000.
        - Coordinate values MUST be integers.
        - keys must match exactly: "box_2d"
        - Do NOT include label, confidence, or any extra fields.
        - Do NOT wrap the JSON in ```json or backticks.
    """

    # 3. Call the Gemini function
    try:
        gemini_response_text = get_part_detection_sync(detection_prompt, image_bytes)
        
        # 4. Attempt to parse the JSON response from the model
        return json.loads(gemini_response_text)
            
    except json.JSONDecodeError:
        # If the model breaks the rule and returns non-JSON text
        logging.error(f"Model returned non-JSON response: {gemini_response_text}")
        return JSONResponse(status_code=500, content={
            "error": "Model response was not valid JSON. Check raw response for model's output.",
            "raw_response_text": gemini_response_text
        })
    except Exception as e:
        # Catch critical errors like API key or connection issues
        logging.error(f"Critical error during API call: {e}")
        raise HTTPException(status_code=500, detail=f"A critical error occurred while calling the Gemini API: {e}")