import torch
import torch.nn as nn
import pickle
from PIL import Image
import torchvision.transforms as transforms
import random
import torchvision.models as models
import numpy as np
from .ImageModel import ImageCaptioningModel
import Requirement.os as os



class InferencePipeline:
    def __init__(self):
        
        # Instantiate ModelQueue through os.modelQueue (from custom_os)
        self.model_queue = os.ModelQueue()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load saved tokenizer and model configuration
        with open(r'sceneBackup\tokenizer_all.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)

        with open(r'sceneBackup\model_config_all.pkl', 'rb') as f:
            self.model_config = pickle.load(f)

        # Configuration variables
        self.vocab_size = self.model_config['vocab_size']
        self.max_length = self.model_config['max_length']
        self.pad_token = self.model_config['pad_token']
        self.start_token = self.model_config['start_token']
        self.end_token = self.model_config['end_token']

        # Initialize the captioning model
        self.model = ImageCaptioningModel(vocab_size=self.vocab_size, max_length=self.max_length).to(self.device)
        self.model.load_state_dict(torch.load(r'sceneBackup\best_model_all_images.pth', map_location=self.device))
        self.model.eval()

        # Initialize ResNet50 model for feature extraction
        self.model_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        self.model_resnet = torch.nn.Sequential(*list(self.model_resnet.children())[:-1]).to(self.device)

        # Transformation for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),          # Convert the PIL image to a PyTorch tensor
            transforms.Normalize(           # Normalize using ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def extract_features(self, image):
        """
        Extract features from an image (PIL.Image or numpy.ndarray) using ResNet50.
        """
        try:
            if isinstance(image, np.ndarray):  # If the image is a numpy array, convert it to PIL
                image = Image.fromarray(image)
            image = image.convert('RGB')  # Ensure the image is in RGB format
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model_resnet(image)
            return features.view(features.size(0), -1).cpu().numpy()  # Flatten and return as numpy
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros((2048,))  # Return a zero vector if extraction fails

    def generate_Image_caption(self, image, randomize=False, temperature=1.0, top_k=5):
        """
        Generate a caption for a given image (PIL.Image or numpy.ndarray), with optional randomness.
        """
        img_feature = torch.tensor(self.extract_features(image), dtype=torch.float32).unsqueeze(0).to(self.device)
        caption = [self.start_token]

        for _ in range(self.max_length):
            caption_tensor = torch.tensor([caption], dtype=torch.long).to(self.device)
            with torch.no_grad():
                output = self.model(img_feature, caption_tensor)
            
            logits = output[0, -1, :]  # Extract logits for the last predicted token
            probs = torch.softmax(logits / temperature, dim=-1)  # Apply temperature scaling

            if randomize:
                # Top-k sampling
                top_probs, top_indices = torch.topk(probs, top_k)
                sampled_index = random.choices(
                    top_indices.cpu().numpy(),
                    weights=top_probs.cpu().numpy() / top_probs.sum().item(),
                    k=1
                )[0]
            else:
                # Greedy decoding
                sampled_index = logits.argmax().item()

            caption.append(sampled_index)

        # Convert token indices to words
        predicted_caption = [self.tokenizer.index_word.get(token, '') for token in caption if token not in [self.start_token, self.end_token]]
        return ' '.join(predicted_caption)

    def generate_caption(self, image, beam_size=2):
        """
        Generate a caption for a given image (PIL.Image or numpy.ndarray).
        """
        caption = self.generate_Image_caption(image, randomize=False, temperature=1.0, top_k=beam_size)

        caption = self.model_queue.queueTask(image)
        return caption