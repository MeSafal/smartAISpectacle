import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
from torchvision.models import resnet50, ResNet50_Weights

# Configuration (match your training setup)
class Config:
    # Paths
    checkpoint_path = r"scene\best_model.pth"
    tokenizer_file = r"scene\tokenizer.json"
    
    # Model parameters
    embedding_dim = 256
    hidden_dim = 1024
    
    # Data parameters
    image_size = 224
    max_seq_length = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture (must match training)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(2048, Config.embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, Config.embedding_dim)
        self.lstm = nn.LSTM(Config.embedding_dim, Config.hidden_dim, batch_first=True)
        self.fc = nn.Linear(Config.hidden_dim, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        outputs, _ = self.lstm(embeddings + features)
        return self.fc(outputs)

class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

class InferencePipeline:
    def __init__(self):
        self.device = Config.device
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((Config.image_size, Config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_tokenizer(self):
        """Load trained tokenizer from file"""
        if not os.path.exists(Config.tokenizer_file):
            raise FileNotFoundError(f"Tokenizer file {Config.tokenizer_file} not found")

        class Tokenizer:
            def __init__(self):
                self.word2idx = {}
                self.idx2word = {}
                self.special_tokens = ['<pad>', '<start>', '<end>', '<unk>']

            def load(self, file_path):
                with open(file_path, 'r') as f:
                    state = json.load(f)
                self.word2idx = state['word2idx']
                self.idx2word = {int(k): v for k, v in state['idx2word'].items()}

        tokenizer = Tokenizer()
        tokenizer.load(Config.tokenizer_file)
        return tokenizer

    def _load_model(self):
        """Load trained model weights"""
        if not os.path.exists(Config.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint {Config.checkpoint_path} not found")

        model = ImageCaptionModel(len(self.tokenizer.word2idx)).to(self.device)
        model.load_state_dict(torch.load(Config.checkpoint_path, map_location=self.device))
        model.eval()
        return model

    def _preprocess_image(self, image_path):
        """Load and transform image"""
        try:
            image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
        except IOError:
            raise ValueError(f"Invalid image file: {image_path}")

        return self.transform(image).unsqueeze(0).to(self.device)

    def generate_caption(self, image_path, temperature=1.0, beam_size=None):
        """
        Generate caption for an image
        :param image_path: Path to input image
        :param temperature: Softmax temperature (1.0 = normal sampling)
        :param beam_size: Use beam search if specified (None for greedy)
        :return: Generated caption string
        """
        # Validate inputs
        # if not os.path.exists(image_path):
        #     raise FileNotFoundError(f"Image file {image_path} not found")

        if temperature <= 0:
            raise ValueError("Temperature must be > 0")

        # Preprocess image
        image_tensor = self._preprocess_image(image_path)

        # Generate caption
        with torch.no_grad():
            features = self.model.encoder(image_tensor)
            
            if beam_size:
                return self._beam_search(features, beam_size)
            
            return self._greedy_decode(features, temperature)

    def _greedy_decode(self, features, temperature):
        """Greedy decoding implementation"""
        inputs = torch.tensor([[self.tokenizer.word2idx['<start>']]], device=self.device)
        hidden = None
        caption_words = []

        for _ in range(Config.max_seq_length):
            embeddings = self.model.decoder.embed(inputs)
            combined = embeddings + features.unsqueeze(1)
            lstm_out, hidden = self.model.decoder.lstm(combined, hidden)
            outputs = self.model.decoder.fc(lstm_out.squeeze(1))
            
            # Apply temperature
            scaled_outputs = outputs / temperature
            probabilities = torch.softmax(scaled_outputs, dim=-1)
            
            predicted = torch.multinomial(probabilities, 1)
            word_idx = predicted.item()

            if word_idx == self.tokenizer.word2idx['<end>']:
                break

            word = self.tokenizer.idx2word.get(word_idx, '<unk>')
            if word not in self.tokenizer.special_tokens:
                caption_words.append(word)

            inputs = predicted

        return ' '.join(caption_words)

    def _beam_search(self, features, beam_size=3):
        """Beam search implementation"""
        # Initial setup
        start_token = self.tokenizer.word2idx['<start>']
        end_token = self.tokenizer.word2idx['<end>']
        
        # Initialize beam
        sequences = [[[start_token], 0.0, None]]

        for _ in range(Config.max_seq_length):
            all_candidates = []
            
            for seq in sequences:
                tokens, score, hidden = seq
                if tokens[-1] == end_token:
                    all_candidates.append(seq)
                    continue

                # Prepare inputs
                inputs = torch.tensor([tokens[-1]], device=self.device).unsqueeze(0)
                current_hidden = hidden

                # Forward pass
                embeddings = self.model.decoder.embed(inputs)
                combined = embeddings + features
                lstm_out, new_hidden = self.model.decoder.lstm(combined, current_hidden)
                outputs = self.model.decoder.fc(lstm_out.squeeze(1))
                log_probs = torch.log_softmax(outputs, dim=-1)

                # Get top candidates
                top_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)
                top_probs = top_probs.squeeze().tolist()
                top_indices = top_indices.squeeze().tolist()

                # Create new candidates
                for i in range(beam_size):
                    token = top_indices[i]
                    new_score = score + top_probs[i]
                    new_seq = tokens + [token]
                    all_candidates.append([new_seq, new_score, new_hidden])

            # Select top beam_size candidates
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_size]

        # Select best sequence
        best_seq = sequences[0][0]
        
        # Convert to words
        caption_words = []
        for idx in best_seq[1:]:  # Skip start token
            if idx == end_token:
                break
            word = self.tokenizer.idx2word.get(idx, '<unk>')
            if word not in self.tokenizer.special_tokens:
                caption_words.append(word)
                
        return ' '.join(caption_words)
