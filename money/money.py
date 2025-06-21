

def money_inference(image):
    import random
    import cv2
    import torch
    from torchvision import transforms
    from .classifier import MoneyClassifier


    # Define 10 different messages for each currency class (80 messages total)
    currency_messages = {
        'Rupees 50': [
            "It appears you have a Rupees 50 note in view!",
            "This looks like a crisp Rupees 50 bill.",
            "I detect a clear Rupees 50 note here.",
            "The image shows a bright Rupees 50 note.",
            "Surely, that's a Rupees 50 note!",
            "This is unmistakably a Rupees 50 note.",
            "I'm quite confident this is a Rupees 50 note.",
            "A Rupees 50 note is visible in the image.",
            "Your note is recognized as Rupees 50.",
            "I believe that's a Rupees 50 note."
        ],
        'Rupees 5': [
            "I see a small but significant Rupees 5 note.",
            "That looks like a Rupees 5 bill to me.",
            "The image reveals a Rupees 5 note clearly.",
            "It appears to be a Rupees 5 note.",
            "I'm noticing a Rupees 5 bill in the frame.",
            "Surely, that's a Rupees 5 note.",
            "This is clearly a Rupees 5 note.",
            "The currency here is a Rupees 5 note.",
            "I identify this as a Rupees 5 bill.",
            "It seems like a Rupees 5 note."
        ],
        'Rupees 500': [
            "It looks like you have a Rupees 500 note.",
            "This note appears to be a Rupees 500 bill.",
            "I recognize a Rupees 500 note in the image.",
            "The note in view is a Rupees 500 note.",
            "Surely, that's a Rupees 500 note.",
            "This is unmistakably a Rupees 500 note.",
            "I'm quite certain this is a Rupees 500 note.",
            "A Rupees 500 note seems to be present here.",
            "The currency is identified as Rupees 500.",
            "It appears to be a Rupees 500 bill."
        ],
        'Rupees 100': [
            "I detect a Rupees 100 note here.",
            "This seems to be a Rupees 100 bill.",
            "The image shows a Rupees 100 note clearly.",
            "It appears to be a Rupees 100 note.",
            "I'm pretty sure that's a Rupees 100 note.",
            "This is clearly a Rupees 100 note.",
            "Surely, that's a Rupees 100 bill.",
            "The note seems to be Rupees 100.",
            "I identify this as a Rupees 100 note.",
            "It looks like a Rupees 100 note."
        ],
        'Rupees 10': [
            "It appears you have a Rupees 10 note.",
            "This note seems to be a Rupees 10 bill.",
            "The image clearly shows a Rupees 10 note.",
            "I detect a Rupees 10 note here.",
            "Surely, that's a Rupees 10 note.",
            "This is unmistakably a Rupees 10 note.",
            "I'm confident this is a Rupees 10 note.",
            "The currency in view is a Rupees 10 note.",
            "I recognize this as a Rupees 10 bill.",
            "It seems to be a Rupees 10 note."
        ],
        'Rupees 1000': [
            "I see a Rupees 1000 note in the image.",
            "This appears to be a Rupees 1000 bill.",
            "The note is identified as Rupees 1000.",
            "Surely, that's a Rupees 1000 note.",
            "It looks like a Rupees 1000 note.",
            "This is clearly a Rupees 1000 note.",
            "I'm confident this is a Rupees 1000 bill.",
            "The image reveals a Rupees 1000 note.",
            "I detect a Rupees 1000 note here.",
            "It seems to be a Rupees 1000 note."
        ],
        'Rupees 20': [
            "I notice a Rupees 20 note in the image.",
            "This appears to be a Rupees 20 bill.",
            "The image clearly shows a Rupees 20 note.",
            "It seems to be a Rupees 20 note.",
            "Surely, that's a Rupees 20 note.",
            "This is unmistakably a Rupees 20 note.",
            "I'm quite sure this is a Rupees 20 note.",
            "The note is identified as Rupees 20.",
            "I recognize this as a Rupees 20 note.",
            "It looks like a Rupees 20 bill."
        ],
        'Unknown': [
            "I am not sure which currency this is.",
            "This currency note is unfamiliar to me.",
            "I cannot confidently identify this note.",
            "The note appears to be unrecognized.",
            "I'm uncertain about this currency.",
            "This currency remains a mystery to me.",
            "I don't have a clear match for this note.",
            "The note remains unidentified.",
            "I'm not sure about the currency here.",
            "This note is unknown to my system."
        ]
    }

    """
    Perform money inference on the provided image (a NumPy array in BGR format)
    and return a friendly message for the predicted Nepali currency denomination.
    
    This function lazy-loads heavy libraries and model resources only when called.
    """

    # Mapping numerical predictions to Nepali currency denominations
    class_mapping = {
        0: 'Rupees 50',
        1: 'Rupees 5',
        2: 'Rupees 500',
        3: 'Rupees 100',
        4: 'Rupees 10',
        5: 'Rupees 1000',
        6: 'Rupees 20',
        7: 'Unknown',
    }

    # Preprocess the image
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    frame_tensor = transform_pipeline(frame_rgb).unsqueeze(0)  # Add batch dimension

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load and prepare the model
    model = MoneyClassifier(8).to(device)
    model.load_state_dict(torch.load(r'money\best_money_classifier.pth', map_location=device))
    model.eval()

    frame_tensor = frame_tensor.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(frame_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        predicted_class = class_mapping[predicted.item()]

    # Select and return a random friendly message for the prediction
    message = random.choice(currency_messages.get(predicted_class, ["I can't identify this currency."]))
    return message
