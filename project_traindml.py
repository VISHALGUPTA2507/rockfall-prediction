import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from playsound import playsound
import time
import os

# ======== Parameters ========
model_path = "rockfall_resnet18.pt"  # Make sure this file is in the same folder
alert_sound = "alert.mp3"            # You can use any mp3/wav sound

# ======== Load Model ========
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v:k for k,v in class_to_idx.items()}

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(idx_to_class))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ======== Image Transform ========
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ======== Webcam ========
cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Starting webcam detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    
    # Transform and predict
    input_tensor = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output,1).item()
        label = idx_to_class[pred]

    # Show frame with label
    cv2.putText(frame, f"Prediction: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Rockfall Detection", frame)

    # Play alert if rockfall detected
    if label == "rockfall":
        if os.path.exists(alert_sound):
            playsound(alert_sound, block=False)
        else:
            print("Rockfall detected!")

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
