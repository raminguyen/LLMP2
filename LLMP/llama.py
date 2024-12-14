import requests
import png
import io
import numpy as np
import base64
import os
import torch
from transformers import AutoModelForPreTraining, TrainingArguments, Trainer, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq, pipeline, AutoModel
from peft import LoraConfig, get_peft_model
from PIL import Image


class llama:
    
    def __init__(self, model_name):
        print(f"Initializing llamaModel with model_name: {model_name}")
        self.model_name = model_name

    def query(self, question, image=None):

        size = image.shape[0]
        grayscale = np.zeros((size,size), dtype=np.uint8)
        grayscale[image==0] = 255
        grayscale[image==1] = 0

        pil_image = Image.fromarray(grayscale)

        # Bits and Bytes configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Model and processor loading
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = AutoModelForPreTraining.from_pretrained(
            model_id,
            device_map="auto",   # Automatically map model to available devices (e.g., GPU)
            torch_dtype=torch.bfloat16,  # Using bfloat16 for reduced memory usage
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            cache_dir = "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/cache"
        )

        image = pil_image.convert("RGB")


        processor = AutoProcessor.from_pretrained(self.model_name)

        model.tie_weights()
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # For inference, you only need to process the inputs, without handling labels or setting -100
        batch = processor(text=text, images=image, return_tensors="pt", padding=True).to("cuda")
        
        # Run inference on the model with the preprocessed batch
        outputs = model.generate(**batch, max_length=1000)
        
        # Decode the generated output
        decoded_output = processor.decode(outputs[0], skip_special_tokens=True)
        return decoded_output