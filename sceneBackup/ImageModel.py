import torch
import torch.nn as nn

# Model Definition
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim=256, hidden_dim=1024, num_layers=3):
        super(ImageCaptioningModel, self).__init__()
        self.image_encoder = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, img_features, captions):
        img_features = img_features.view(img_features.size(0), -1)
        img_features = self.image_encoder(img_features).unsqueeze(1)
        captions = self.embedding(captions)
        inputs = torch.cat((img_features, captions), 1)
        lstm_out, _ = self.lstm(inputs)
        attn_output = self.attention(lstm_out)
        outputs = self.fc(self.dropout(attn_output))        
        return outputs[:, :-1, :]

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim) 
        self.softmax = nn.Softmax(dim=1)  
    
    def forward(self, lstm_output):
        attn_weights = self.softmax(self.attention(lstm_output)) 
        return attn_weights * lstm_output 
