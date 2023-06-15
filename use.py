import torch

# Load the trained model
model = torch.load('path_to_saved_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # Set the model to evaluation mode

import cv2
import numpy as np

# Load and preprocess the input image
image = cv2.imread('path_to_input_image.jpg')
resized_image = cv2.resize(image, (224, 224))  # Resize to match the model's input size
input_image = resized_image.transpose(2, 0, 1)  # Convert to channel-first format (C, H, W)
input_tensor = torch.from_numpy(input_image).unsqueeze(0).float()  # Add batch dimension and convert to tensor
input_tensor = input_tensor / 255.0  # Normalize image
input_tensor = input_tensor.to(device)

# Perform inference
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)

# Get the predicted class
_, predicted_class = torch.max(probabilities, 1)
predicted_label = predicted_class.item()