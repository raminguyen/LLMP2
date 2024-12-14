import requests
import png
import io
import numpy as np
import base64
import os
from dotenv import load_dotenv
import time  # Import time for sleep functionality


# Load environment variables from the .env file
load_dotenv()

# Get the API key from the .env file
chatgpt_api_key = os.getenv('chatgpt_api_key')

if not chatgpt_api_key:
    raise ValueError("API key not found. Please ensure your .env file contains the API key.")


class GPTModel:
    
    def __init__(self, model_name):
        print(f"Initializing GPTModel with model_name: {model_name}")
        self.model_name = model_name
        self.api_key = chatgpt_api_key  # This should be a string

    def query(self, question, image=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"  # Should work if self.api_key is a string
        }

        # Prepare payload based on the model and whether an image is provided
        if image is not None:
            payload = self._prepare_payload_with_image(question, image)
        else:
            payload = self._prepare_payload_without_image(question)

        # Add a delay before making the API request
        time.sleep(5)  # Sleep for 5 seconds to avoid rate limits

        # Make the API request
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()

        # Handle any errors
        if 'error' in response_json:
            print('ERROR', response_json)
            return None

        # Return the content from the response
        if 'choices' in response_json:
            return response_json['choices'][0]['message']['content']

    def _prepare_payload_with_image(self, question, image):
        size = image.shape[0]
        grayscale = np.zeros((size, size), dtype=np.uint8)
        grayscale[image == 0] = 255
        grayscale[image == 1] = 0

        png_image = png.from_array(grayscale, 'L')

        outp = io.BytesIO()
        png_image.write(outp)
        png_bytes = outp.getvalue()
        outp.close()

        base64_image = base64.b64encode(png_bytes).decode('utf-8')

        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 400
        }

    def _prepare_payload_without_image(self, question):
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "max_tokens": 400
        }