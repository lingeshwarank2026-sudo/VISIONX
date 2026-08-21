import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import os

from train_classifier import CustomCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['0', 'Bus', 'Motorcycle', 'car', 'truck']
MODEL_PATH = "vehicle_classifier.pth"

# Load Model
model = CustomCNN(num_classes=5).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except Exception as e:
    print(f"Model load warning: {e}")
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_pil_image(image):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    class_probs = {
        'Bus': float(probs[1]),
        'Motorcycle': float(probs[2]),
        'Car': float(probs[3]),
        'Truck': float(probs[4])
    }
    best_class = max(class_probs, key=class_probs.get)
    return best_class, class_probs[best_class], class_probs

class VehicleClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VISIONX - AI Vehicle Classifier")
        self.root.geometry("620x720")
        self.root.configure(bg="#0f172a")
        
        # Title
        title_label = tk.Label(root, text="VISIONX AI Vehicle Identifier", font=("Arial", 18, "bold"), fg="#38bdf8", bg="#0f172a")
        title_label.pack(pady=(20, 5))
        
        subtitle = tk.Label(root, text="Upload any vehicle image to predict its class", font=("Arial", 10), fg="#94a3b8", bg="#0f172a")
        subtitle.pack(pady=(0, 15))
        
        # Upload Button
        self.btn = tk.Button(root, text="📁 Choose Photo", font=("Arial", 12, "bold"), bg="#6366f1", fg="white", activebackground="#4f46e5", padx=20, pady=8, relief="flat", cursor="hand2", command=self.upload_image)
        self.btn.pack(pady=10)
        
        # Image Display Area
        self.canvas = tk.Label(root, bg="#1e293b", width=360, height=260, text="No Image Selected\nClick 'Choose Photo' above", fg="#64748b", font=("Arial", 11))
        self.canvas.pack(pady=10)
        
        # Prediction Label
        self.result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), fg="#ffffff", bg="#0f172a")
        self.result_label.pack(pady=5)
        
        self.conf_label = tk.Label(root, text="", font=("Arial", 12), fg="#38bdf8", bg="#0f172a")
        self.conf_label.pack(pady=2)
        
        # Probability Bars Frame
        self.bar_frame = tk.Frame(root, bg="#0f172a")
        self.bar_frame.pack(fill="x", padx=60, pady=15)
        
        self.bars = {}
        for cat in ['Car', 'Bus', 'Motorcycle', 'Truck']:
            row = tk.Frame(self.bar_frame, bg="#0f172a")
            row.pack(fill="x", pady=4)
            lbl = tk.Label(row, text=f"{cat}:", width=12, anchor="w", fg="#cbd5e1", bg="#0f172a", font=("Arial", 10, "bold"))
            lbl.pack(side="left")
            pb = ttk.Progressbar(row, orient="horizontal", length=220, mode="determinate")
            pb.pack(side="left", padx=8)
            val_lbl = tk.Label(row, text="0%", width=8, anchor="e", fg="#94a3b8", bg="#0f172a", font=("Arial", 10))
            val_lbl.pack(side="right")
            self.bars[cat] = (pb, val_lbl)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if not file_path:
            return
        
        img = Image.open(file_path)
        
        # Resize for preview
        img_preview = img.copy()
        img_preview.thumbnail((360, 260))
        photo = ImageTk.PhotoImage(img_preview)
        self.canvas.config(image=photo, text="")
        self.canvas.image = photo
        
        # Run AI prediction
        best_class, conf, all_probs = predict_pil_image(img)
        
        icons = {'Car': '🚗', 'Bus': '🚌', 'Motorcycle': '🏍️', 'Truck': '🚚'}
        self.result_label.config(text=f"Result: {icons.get(best_class, '')} {best_class}")
        self.result_label.config(fg="#4ade80" if conf > 0.6 else "#fbbf24")
        self.conf_label.config(text=f"Confidence: {conf*100:.1f}%")
        
        for cat, (pb, val_lbl) in self.bars.items():
            percentage = all_probs.get(cat, 0.0) * 100
            pb['value'] = percentage
            val_lbl.config(text=f"{percentage:.1f}%")

if __name__ == '__main__':
    root = tk.Tk()
    app = VehicleClassifierGUI(root)
    root.mainloop()
