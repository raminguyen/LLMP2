# LLMP/gemini.py
import google.generativeai as genai
import numpy as np
import io
import base64
from PIL import Image
from dotenv import load_dotenv
import os
import time
import requests

# Load environment variables from the .env file
load_dotenv()

# Load the API key for Gemini models from the .env file
gemini_api_key = os.getenv('gemini_api_key')

# Configure the Google Generative AI API
genai.configure(api_key=gemini_api_key)



class GeminiBaseModel:
    def __init__(self, model_name, generation_config):
        # Create the model with specific configuration
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )

    def preprocess_image(self, image):
        # Handle the case where the image is a file path, NumPy array, or tuple
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, tuple):
            image = self._handle_image_tuple(image)
        else:
            raise ValueError("Unsupported image format.")

        # Validate that the final image is a PIL.Image
        if not isinstance(image, Image.Image):
            raise ValueError("Expected a PIL.Image object.")

        # Convert the image to base64
        return self._image_to_base64(image)

    def _handle_image_tuple(self, image):
        if len(image) == 2 and isinstance(image[0], np.ndarray):
            return Image.fromarray(image[0])
        elif isinstance(image[0], Image.Image):
            return image[0]
        else:
            raise ValueError("Unexpected tuple format. Expected (np.ndarray, dict) or (PIL.Image, ...).")

    def _image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class GeminiModel(GeminiBaseModel):
    def __init__(self, model_name):
        # Define common configuration for the models
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 100,
            "response_mime_type": "text/plain",
        }
        super().__init__(model_name=model_name, generation_config=generation_config)

    def query(self, question, image):
        base64_image = self.preprocess_image(image)
        prompt = f"{question}\n![Image](data:image/png;base64,{base64_image})"
        time.sleep(5)
        try:
            chat_session = self.model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/mixed; boundary={boundary}",
        }

        response = requests.post(
            "https://api.google.com/gemini/v1/batch",
            headers=headers,
            data=batch_payload
        )

        if response.status_code == 200:
            return response.text
        else:
            print(f"Batch request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None


class GeminiProVision(GeminiModel):
    def __init__(self):
        super().__init__(model_name="gemini-1.5-pro")


class Gemini1_5Flash(GeminiModel):
    def __init__(self):
        super().__init__(model_name="gemini-1.5-flash")