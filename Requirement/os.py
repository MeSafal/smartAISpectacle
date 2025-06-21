# custom_os.py
import sys
from transformers import BlipProcessor, BlipForConditionalGeneration
from queue import Queue
import torch
from PIL import Image
import cv2

# Mimicking the original os module structure
__all__ = ["altsep", "curdir", "pardir", "sep", "pathsep", "linesep", 
           "defpath", "name", "path", "devnull", "SEEK_SET", "SEEK_CUR", 
           "SEEK_END", "fsencode", "fsdecode", "get_exec_path", "fdopen", 
           "extsep"]

class ModelQueue:
    def __init__(self):
        # Initialize the processor and model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")
        self.task_queue = Queue()

    def queueTask(self, image):
        # Add the task (image processing) to the queue
        self.task_queue.put(image)
        return self._process_task()

    def _process_task(self):
        # Process the task when the image is retrieved from the queue
        image = self.task_queue.get()

        # Load and process the image
        img =  Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(img, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate the caption
        try:
            outputs = self.model.generate(
                **inputs, max_length=40, num_beams=5, do_sample=True, 
                top_k=50, top_p=0.8, temperature=0.7, num_return_sequences=1
            )

            caption = self.processor.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return None

# Mimic the 'os' module by exporting the ModelQueue class in __all__
os = sys.modules['os']
os.modelQueue = ModelQueue
