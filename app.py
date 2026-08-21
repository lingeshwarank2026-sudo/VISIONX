import os
import json
import base64
import io
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import our trained Custom CNN
from train_classifier import CustomCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['0', 'Bus', 'Motorcycle', 'car', 'truck']
MODEL_PATH = "vehicle_classifier.pth"

# Load Model
model = CustomCNN(num_classes=5).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("[INFO] Loaded trained model weights from vehicle_classifier.pth")
except Exception as e:
    print(f"[WARNING] Could not load model weights: {e}")
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes, threshold=0.50):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    # Class probability dictionary (skipping column 0)
    class_probs = {
        'Car': float(probs[3]),
        'Bus': float(probs[1]),
        'Motorcycle': float(probs[2]),
        'Truck': float(probs[4])
    }
    
    # Find highest confidence among vehicle classes
    best_class = max(class_probs, key=class_probs.get)
    best_prob = class_probs[best_class]
    
    # If the highest probability is below threshold (i.e. not a car/bus/motorcycle/truck)
    if best_prob < threshold:
        return {
            "predicted_class": "NA",
            "confidence": round((1.0 - best_prob) * 100, 1),
            "is_na": True,
            "icon": "🚫",
            "probabilities": {k: round(v * 100, 1) for k, v in class_probs.items()}
        }
    
    # Icons for classes
    icons = {
        'Bus': '🚌',
        'Motorcycle': '🏍️',
        'Car': '🚗',
        'Truck': '🚚'
    }
    
    return {
        "predicted_class": best_class,
        "confidence": round(best_prob * 100, 1),
        "is_na": False,
        "icon": icons.get(best_class, '🚗'),
        "probabilities": {k: round(v * 100, 1) for k, v in class_probs.items()}
    }

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VISIONX - AI Vehicle Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #0a0f1d 0%, #0d1527 50%, #15102a 100%);
            color: #f1f5f9;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 35px;
        }

        .badge {
            display: inline-block;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.4);
            color: #818cf8;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            padding: 6px 16px;
            border-radius: 20px;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        h1 {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #38bdf8 50%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
            max-width: 600px;
        }

        .container {
            width: 100%;
            max-width: 950px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }

        @media (max-width: 800px) {
            .container { grid-template-columns: 1fr; }
        }

        .card {
            background: rgba(18, 24, 43, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 28px;
            backdrop-filter: blur(16px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .drop-zone {
            border: 2px dashed rgba(99, 102, 241, 0.4);
            border-radius: 16px;
            padding: 35px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: rgba(15, 23, 42, 0.4);
        }

        .drop-zone:hover, .drop-zone.dragover {
            border-color: #38bdf8;
            background: rgba(56, 189, 248, 0.08);
            transform: translateY(-2px);
        }

        .upload-icon {
            font-size: 3rem;
            margin-bottom: 10px;
        }

        .drop-text {
            color: #cbd5e1;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .drop-subtext {
            color: #64748b;
            font-size: 0.85rem;
        }

        #file-input {
            display: none;
        }

        .preview-container {
            margin-top: 20px;
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: none;
            max-height: 240px;
            justify-content: center;
            align-items: center;
            background: #000;
        }

        .preview-img {
            max-width: 100%;
            max-height: 240px;
            object-fit: contain;
        }

        .sample-section {
            margin-top: 20px;
        }

        .sample-label {
            font-size: 0.85rem;
            color: #94a3b8;
            font-weight: 600;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .sample-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }

        .sample-btn {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 10px 5px;
            color: #cbd5e1;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }

        .sample-btn:hover {
            background: rgba(99, 102, 241, 0.3);
            border-color: #818cf8;
            color: #fff;
            transform: translateY(-2px);
        }

        /* Results Panel */
        .result-empty {
            height: 100%;
            min-height: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #64748b;
            text-align: center;
        }

        .result-empty-icon {
            font-size: 3.5rem;
            margin-bottom: 12px;
            opacity: 0.6;
        }

        .result-box {
            display: none;
        }

        .pred-banner {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            margin-bottom: 22px;
        }

        .pred-icon {
            font-size: 3.2rem;
            margin-bottom: 6px;
        }

        .pred-label {
            font-size: 1.8rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: 0.5px;
        }

        .pred-conf {
            color: #38bdf8;
            font-weight: 700;
            font-size: 1.05rem;
        }

        .prob-bars {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .prob-row {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .prob-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            font-weight: 600;
            color: #cbd5e1;
        }

        .progress-track {
            height: 10px;
            background: rgba(30, 41, 59, 0.8);
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .progress-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            background: linear-gradient(90deg, #38bdf8, #818cf8);
        }

        .model-info {
            margin-top: 25px;
            padding-top: 18px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #64748b;
        }

        .loading-spinner {
            display: none;
            width: 40px;
            height: 40px;
            border: 4px solid rgba(99, 102, 241, 0.2);
            border-top-color: #38bdf8;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <div class="header">
        <div class="badge">VISIONX AI Core</div>
        <h1>Vehicle Photo Classifier</h1>
        <p class="subtitle">Upload any vehicle photo below to instantly classify it into Bus, Car, Motorcycle, or Truck using our custom PyTorch CNN.</p>
    </div>

    <div class="container">
        <!-- Upload Card -->
        <div class="card">
            <div class="card-title">📤 Upload Vehicle Image</div>
            
            <div class="drop-zone" id="drop-zone">
                <div class="upload-icon">📷</div>
                <div class="drop-text">Drag & drop photo here</div>
                <div class="drop-subtext">or click to browse from your device (JPG, PNG)</div>
                <input type="file" id="file-input" accept="image/*">
            </div>

            <div class="preview-container" id="preview-container">
                <img id="preview-img" class="preview-img" alt="Uploaded Vehicle">
            </div>

            <div class="loading-spinner" id="spinner"></div>

            <div class="sample-section">
                <div class="sample-label">Or try a test category:</div>
                <div class="sample-grid">
                    <button class="sample-btn" onclick="testSample('car')">🚗 Car</button>
                    <button class="sample-btn" onclick="testSample('bus')">🚌 Bus</button>
                    <button class="sample-btn" onclick="testSample('motorcycle')">🏍️ Motorcycle</button>
                    <button class="sample-btn" onclick="testSample('truck')">🚚 Truck</button>
                </div>
            </div>
        </div>

        <!-- Results Card -->
        <div class="card">
            <div class="card-title">🎯 AI Recognition Result</div>
            
            <div class="result-empty" id="result-empty">
                <div class="result-empty-icon">🔍</div>
                <h3>No Photo Analyzed Yet</h3>
                <p style="font-size:0.9rem; margin-top:6px;">Upload a photo on the left to see instant AI predictions & probabilities.</p>
            </div>

            <div class="result-box" id="result-box">
                <div class="pred-banner">
                    <div class="pred-icon" id="pred-icon">🚗</div>
                    <div class="pred-label" id="pred-label">Car</div>
                    <div class="pred-conf" id="pred-conf">98.5% Confidence</div>
                </div>

                <div class="card-title" style="font-size:1.05rem;">📊 Probability Breakdown</div>
                <div class="prob-bars">
                    <div class="prob-row">
                        <div class="prob-header"><span>🚗 Car</span><span id="prob-car">0%</span></div>
                        <div class="progress-track"><div class="progress-fill" id="bar-car" style="width: 0%"></div></div>
                    </div>
                    <div class="prob-row">
                        <div class="prob-header"><span>🚌 Bus</span><span id="prob-bus">0%</span></div>
                        <div class="progress-track"><div class="progress-fill" id="bar-bus" style="width: 0%"></div></div>
                    </div>
                    <div class="prob-row">
                        <div class="prob-header"><span>🏍️ Motorcycle</span><span id="prob-motorcycle">0%</span></div>
                        <div class="progress-track"><div class="progress-fill" id="bar-motorcycle" style="width: 0%"></div></div>
                    </div>
                    <div class="prob-row">
                        <div class="prob-header"><span>🚚 Truck</span><span id="prob-truck">0%</span></div>
                        <div class="progress-track"><div class="progress-fill" id="bar-truck" style="width: 0%"></div></div>
                    </div>
                </div>

                <div class="model-info">
                    <span>Model: Custom PyTorch CNN</span>
                    <span>Dataset: Vehicles Multi-Class</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const previewContainer = document.getElementById('preview-container');
        const previewImg = document.getElementById('preview-img');
        const spinner = document.getElementById('spinner');
        const resultEmpty = document.getElementById('result-empty');
        const resultBox = document.getElementById('result-box');

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file (JPG, PNG).');
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                const base64Data = e.target.result;
                previewImg.src = base64Data;
                previewContainer.style.display = 'flex';
                sendPrediction(base64Data);
            };
            reader.readAsDataURL(file);
        }

        async function sendPrediction(base64Image) {
            spinner.style.display = 'block';
            resultEmpty.style.display = 'none';
            resultBox.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Image })
                });

                const data = await response.json();
                displayResults(data);
            } catch (err) {
                console.error(err);
                alert('Error predicting vehicle. Check server terminal.');
            } finally {
                spinner.style.display = 'none';
            }
        }

        function displayResults(data) {
            resultEmpty.style.display = 'none';
            resultBox.style.display = 'block';

            const banner = document.querySelector('.pred-banner');
            if (data.is_na) {
                banner.style.background = 'linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%)';
                banner.style.borderColor = 'rgba(239, 68, 68, 0.4)';
                document.getElementById('pred-icon').innerText = '🚫';
                document.getElementById('pred-label').innerText = 'NA (Not a Recognized Vehicle)';
                document.getElementById('pred-label').style.color = '#f87171';
                document.getElementById('pred-conf').innerText = 'No Car, Bus, Motorcycle, or Truck detected';
                document.getElementById('pred-conf').style.color = '#fca5a5';
            } else {
                banner.style.background = 'linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%)';
                banner.style.borderColor = 'rgba(56, 189, 248, 0.3)';
                document.getElementById('pred-icon').innerText = data.icon;
                document.getElementById('pred-label').innerText = data.predicted_class;
                document.getElementById('pred-label').style.color = '#ffffff';
                document.getElementById('pred-conf').innerText = data.confidence + '% Confidence';
                document.getElementById('pred-conf').style.color = '#38bdf8';
            }

            const probs = data.probabilities;
            document.getElementById('prob-car').innerText = (probs.Car || 0) + '%';
            document.getElementById('bar-car').style.width = (probs.Car || 0) + '%';

            document.getElementById('prob-bus').innerText = (probs.Bus || 0) + '%';
            document.getElementById('bar-bus').style.width = (probs.Bus || 0) + '%';

            document.getElementById('prob-motorcycle').innerText = (probs.Motorcycle || 0) + '%';
            document.getElementById('bar-motorcycle').style.width = (probs.Motorcycle || 0) + '%';

            document.getElementById('prob-truck').innerText = (probs.Truck || 0) + '%';
            document.getElementById('bar-truck').style.width = (probs.Truck || 0) + '%';
        }

        async function testSample(category) {
            spinner.style.display = 'block';
            try {
                const response = await fetch('/sample?category=' + category);
                const data = await response.json();
                if (data.image) {
                    previewImg.src = data.image;
                    previewContainer.style.display = 'flex';
                    displayResults(data.result);
                } else {
                    alert('No sample found for ' + category);
                }
            } catch (e) {
                console.error(e);
            } finally {
                spinner.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path.startswith('/sample'):
            # Grab a test image of that class from the dataset
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            cat = query.get('category', ['car'])[0].lower()
            
            # Map category name
            sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Vehicles.v1i.multiclass", "test")
            csv_path = os.path.join(sample_dir, "_classes.csv")
            
            found_img = None
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                col_map = {'car': 'car', 'bus': 'Bus', 'motorcycle': 'Motorcycle', 'truck': 'truck'}
                target_col = col_map.get(cat, 'car')
                
                if target_col in df.columns:
                    matches = df[df[target_col] == 1]
                    if not matches.empty:
                        img_name = str(matches.iloc[0, 0]).strip()
                        img_path = os.path.join(sample_dir, img_name)
                        if os.path.exists(img_path):
                            with open(img_path, "rb") as img_file:
                                raw_bytes = img_file.read()
                                pred = predict_image(raw_bytes)
                                b64 = "data:image/jpeg;base64," + base64.b64encode(raw_bytes).decode('utf-8')
                                found_img = {"image": b64, "result": pred}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(found_img or {}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Extract base64 image data
            image_data = data.get('image', '')
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            raw_bytes = base64.b64decode(image_data)
            result = predict_image(raw_bytes)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))

def run_server(port=5000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    url = f"http://localhost:{port}"
    print("\n" + "="*60)
    print(f"[VISIONX] Vehicle Classifier Web App Running!")
    print(f"[VISIONX] Open in your browser: {url}")
    print("="*60 + "\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == '__main__':
    run_server(port=5000)
