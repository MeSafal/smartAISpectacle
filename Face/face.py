

def face_inference(frame):
    import cv2
    import torch
    import numpy as np
    import random
    from PIL import Image
    from torchvision import transforms
    # Load Haar cascade for face detection
    FACE_HAAR_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model architecture
    class FaceRecognitionModel(torch.nn.Module):
        def __init__(self, embedding_size=512):
            super().__init__()
            base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            self.features = torch.nn.Sequential(*list(base_model.children())[:-1])
            self.embedding = torch.nn.Linear(2048, embedding_size)
            self.bn = torch.nn.BatchNorm1d(embedding_size)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.embedding(x)
            return torch.nn.functional.normalize(self.bn(x), p=2, dim=1)

    # Load model and embeddings
    model = FaceRecognitionModel().to(device)
    model.load_state_dict(torch.load(r'Face\face_recognition_final.pth', map_location=device))
    model.eval()

    data = np.load(r'Face\saved_embeddings.npz')
    known_embeddings = data['embeddings']
    known_labels = data['labels']
    class_names = data['classes']

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define message templates for different scenarios
    no_face_messages = [
         "I’m sorry, I can’t see any face here.",
         "It looks like there’s no face in the frame.",
         "No face detected, please try again with a clear view.",
         "I don’t see a face; maybe step in front of the camera?",
         "Hmm, it appears there is no face to analyze.",
         "I couldn’t find any face in this frame.",
         "The camera doesn’t capture any face at the moment.",
         "No face is visible, please adjust your position.",
         "I’m not able to detect any face right now.",
         "It seems there's no face present in this scene."
    ]

    recognized_high_messages = [
         "I am absolutely sure this is {name}.",
         "There's no doubt—this is {name}.",
         "With high confidence, I recognize this as {name}.",
         "This face unmistakably belongs to {name}.",
         "I'm very certain that this is {name}.",
         "Clearly, this is {name} without any doubt.",
         "I can confidently say this is {name}.",
         "There's a strong match here; it's {name}.",
         "I'm almost 100% sure this is {name}.",
         "This is undeniably {name}."
    ]

    recognized_medium_messages = [
         "I think this might be {name}.",
         "It appears to be {name}, though I'm not completely sure.",
         "I believe this is {name}, but there’s a slight uncertainty.",
         "There's a good chance this is {name}.",
         "I lean towards this being {name}.",
         "It seems likely that this is {name}.",
         "I suspect this might be {name}.",
         "There's a fair match here, possibly {name}.",
         "I would say this is probably {name}.",
         "It looks like {name}, albeit with some uncertainty."
    ]

    recognized_low_messages = [
         "I'm not very sure, but this might be {name}.",
         "It could be {name}, though I'm not confident.",
         "I think this resembles {name}, but I'm not entirely sure.",
         "This might be {name}, but I have my doubts.",
         "There's a doubt of {name} here, but i am not sure tho.",
         "I see some similarities with {name}, yet I'm uncertain.",
         "It seems to have traits of {name}, but I'm not convinced.",
         "I'm leaning towards {name}, though the match isn't strong.",
         "This face vaguely reminds me of {name}.",
         "I suspect {name} but I'm not completely sure."
    ]

    unknown_messages = [
         "I don't recognize this face.",
         "I'm sorry, but this face doesn't match any known profiles.",
         "I am not sure who this might be.",
         "This face is unfamiliar to me.",
         "I can't seem to identify this face.",
         "I'm having trouble recognizing this face.",
         "I don't have enough confidence to say who this is.",
         "This face remains a mystery to me.",
         "I'm uncertain about this face's identity.",
         "I don't recognize this face with sufficient certainty."
    ]

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_HAAR_CASCADE.detectMultiScale(gray, 1.3, 5)

    # If no faces are detected, return a no-face message
    if len(faces) == 0:
        return random.choice(no_face_messages)

    # Process only the first detected face (for simplicity)
    (x, y, w, h) = faces[0]
    face_img = frame[y:y+h, x:x+w]
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model(tensor).cpu().numpy()[0]

    # Calculate cosine similarity between query embedding and all known embeddings
    similarities = np.dot(known_embeddings, query_embedding)

    # Find best match
    best_match_idx = np.argmax(similarities)
    best_match_label = class_names[best_match_idx]
    best_match_value = similarities[best_match_idx]

    print(f"\nBest match: {best_match_label} with similarity: {best_match_value:.2f}\n")

    # Determine which message to return based on similarity value
    if best_match_value > 0.4:
        message = random.choice(recognized_high_messages).format(name=best_match_label)
    elif best_match_value > 0.3:
        message = random.choice(recognized_medium_messages).format(name=best_match_label)
    elif best_match_value > 0.2:
        message = random.choice(recognized_low_messages).format(name=best_match_label)
    else:
        message = random.choice(unknown_messages)

    return message

# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)
#     print("Press SPACE to process the current frame for face recognition. Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Display the current frame continuously
#         cv2.imshow('Face Recognition', frame)

#         key = cv2.waitKey(1) & 0xFF

#         if key == ord(' '):
#             # Process the frame and get the message
#             message = face_inference(frame.copy())
#             # Overlay the message on the frame for visualization
#             output_frame = frame.copy()
#             cv2.putText(output_frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             cv2.imshow('Face Recognition', output_frame)
#             print(message)
#         elif key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
