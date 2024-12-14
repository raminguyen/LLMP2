import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image


import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

class llamafinetuned:
    def __init__(self, adapter_name):
        print(f"Initializing llamaModel with adapter: {adapter_name}") 
        self.model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        
        # Bits and Bytes configuration for efficient loading
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load the base model from the cache or local directory
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            device_map="cuda:0",  # Use GPU 0
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir = "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/cache"
        )
        
        # Load the LoRA adapter for the base model
        self.model = PeftModel.from_pretrained(
            self.model, 
            adapter_name  # Load adapter from the specified path
        )
                
        # Load the processor from the same cache directory
        self.processor = AutoProcessor.from_pretrained(
            self.model_name
        )
    
    def query(self, question, image):
        # Convert NumPy image to grayscale
        size = image.shape[0]
        grayscale = np.zeros((size, size), dtype=np.uint8)
        grayscale[image == 0] = 255
        grayscale[image == 1] = 0
        
        # Convert NumPy array to PIL image and RGB formats
        pil_image = Image.fromarray(grayscale).convert("RGB")
        
        # Format the input text
        text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"<|image|>{question}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # Process the input text and image to create a batch
        batch = self.processor(text=text, images=pil_image, return_tensors="pt", padding=True).to("cuda")
        
        # Run inference using the model
        outputs = self.model.generate(**batch, max_length=500)
        
        # Decode the generated output
        decoded_output = self.processor.decode(outputs[0], skip_special_tokens=True)
        return decoded_output

