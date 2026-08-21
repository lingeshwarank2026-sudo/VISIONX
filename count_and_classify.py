import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Import the model architecture from train_classifier
from train_classifier import CustomCNN

# Configuration
MODEL_PATH = "vehicle_classifier.pth"
VIDEO_PATH = "sample_traffic.mp4" # Replace with your video path
CLASSES = ['0', 'Bus', 'Motorcycle', 'car', 'truck'] # Exact CSV column order (after filename)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = CustomCNN(num_classes=len(CLASSES)).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded trained model successfully.")
    except Exception as e:
        print(f"Could not load model weights: {e}")
        print("Please run train_classifier.py first to train the model.")
    model.eval()
    return model

def predict_crop(model, crop_img):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert OpenCV BGR image to PIL RGB
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply sigmoid because we used BCEWithLogitsLoss
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
    # Get the class with highest probability (skip index 0 which is the noise '0' class)
    best_class_idx = np.argmax(probs)
    if probs[best_class_idx] > 0.5 and CLASSES[best_class_idx] != '0':
        return CLASSES[best_class_idx], probs[best_class_idx]
    # If the noise class wins, pick the best among real classes (index 1+)
    real_probs = probs[1:]
    real_best = np.argmax(real_probs)
    if real_probs[real_best] > 0.4:
        return CLASSES[real_best + 1], real_probs[real_best]
    return "Unknown", 0.0

def process_video():
    model = get_model()
    
    # Initialize OpenCV Background Subtractor for counting moving vehicles
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {VIDEO_PATH}")
        print("Please provide a valid video file.")
        return

    vehicle_count = 0
    # A simple crossing line to count vehicles
    counting_line_y = 400 
    offset = 10 # Allowable error in pixels

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize for faster processing
        frame = cv2.resize(frame, (800, 600))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fgMask = backSub.apply(gray)
        
        # Threshold to remove shadows and noise
        _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw counting line
        cv2.line(frame, (0, counting_line_y), (800, counting_line_y), (0, 255, 0), 2)
        
        for contour in contours:
            # Filter by area to remove noise
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                center_y = y + h // 2
                
                # Check if the vehicle center crosses the counting line
                if counting_line_y - offset < center_y < counting_line_y + offset:
                    vehicle_count += 1
                    cv2.line(frame, (0, counting_line_y), (800, counting_line_y), (0, 0, 255), 2) # Flash red
                    
                # Crop the detected region for classification
                crop = frame[y:y+h, x:x+w]
                if crop.size > 0:
                    vehicle_class, prob = predict_crop(model, crop)
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    label = f"{vehicle_class} ({prob:.2f})"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display vehicle count
        cv2.putText(frame, f"VISIONX Vehicle Count: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        cv2.imshow("VISIONX - Traffic Monitor", frame)
        
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
