import requests
import png
import io
import numpy as np
import base64
import os
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from transformers import BitsAndBytesConfig
from transformers import pipeline


class LLaVAModel:
    
    def __init__(self, model_name):
        print(f"Initializing LLaVAModel with model_name: {model_name}")
        self.model_name = model_name

    def query(self, question, image=None):

        size = image.shape[0]
        grayscale = np.zeros((size,size), dtype=np.uint8)
        grayscale[image==0] = 255
        grayscale[image==1] = 0
        
        png_image = png.from_array(grayscale, 'L')

        outp = io.BytesIO()
        png_image.write(outp)
        png_bytes = outp.getvalue()
        outp.close()

        base64_image = base64.b64encode(png_bytes).decode('utf-8')

        if self.model_name == "llava-hf/llava-1.5-7b-hf":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            pipe = pipeline("image-to-text", model="llava-hf/llava-1.5-7b-hf", model_kwargs={"quantization_config": quantization_config})

            prompt = f"""

            USER: <image>\n
            
            {question}
            
            \nASSISTANT:
            
            """

            outputs = pipe(base64_image, prompt=prompt, generate_kwargs={"max_new_tokens": 100})

            return outputs[0]["generated_text"]

        elif self.model_name == "raminguyen/fine-tuned-data":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            processor = LlavaNextProcessor.from_pretrained("raminguyen/fine-tuned-data")
    
            model = LlavaNextForConditionalGeneration.from_pretrained("raminguyen/fine-tuned-data", quantization_config=quantization_config, device_map="auto")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            pil_image = Image.fromarray(grayscale)
            inputs = processor(pil_image, prompt, return_tensors="pt").to("cuda:0")
            
            # autoregressively complete prompt
            output = model.generate(**inputs, max_new_tokens=100, pad_token_id = 2)

            return processor.decode(output[0])
        
        