# VISIONX - Intelligent Computer Vision & Traffic Analysis System

## Project Overview
**VISIONX** is an AI-powered computer vision system designed to assist traffic authorities and smart cities in monitoring roads and managing traffic. It provides an end-to-end framework to classify vehicle types (Car, Bus, Motorcycle, Truck) and track & count vehicles moving across roadways in real-time.

## Problem Statement
Traffic congestion and road incidents pose significant challenges to modern urban infrastructure. Traffic authorities need automated systems capable of extracting meaningful insights from traffic camera feeds. The core problem **VISIONX** addresses is taking continuous video feeds or images and accurately distinguishing and counting different types of vehicles in real-time.

## Dataset Used
The dataset used for this system is `Vehicles.v1i.multiclass`.
*   **Format**: The dataset is organized in a Multi-Label Classification format (image-level labels provided via `_classes.csv` files), rather than Object Detection format (bounding boxes). 
*   **Classes**: Bus, Motorcycle, Car, Truck.
*   **Splits**: Train, Valid, Test splits containing preprocessed and augmented vehicle imagery.

## Methodology
To strictly adhere to guidelines ("No pretrained models unless mentioned otherwise") while providing a robust solution, **VISIONX** is split into two logical pipelines:

1.  **Vehicle Classification (Custom CNN)**:
    Since the provided dataset contains image-level labels (no bounding boxes), a **Custom Convolutional Neural Network (CNN)** was built and trained using PyTorch from scratch. 
    *   **Architecture**: It consists of 3 Convolutional blocks (Conv2d + ReLU + MaxPool) followed by a fully connected classifier with Dropout for regularization.
    *   **Loss Function**: `BCEWithLogitsLoss` was used because the problem is structured as a multi-label classification task.
    *   *Why?* A custom CNN represents a well-understood, simple solution built from the ground up, avoiding the "black-box" nature of large pretrained models.

2.  **Vehicle Detection & Counting (OpenCV)**:
    Without bounding box annotations in the dataset, training a custom object detector like YOLO or Faster R-CNN from scratch is not feasible. To solve the counting requirement:
    *   **VISIONX** uses **OpenCV's Background Subtraction (MOG2)** technique.
    *   As vehicles move across the traffic camera's frame, the background subtractor isolates the moving objects (foreground).
    *   Contour detection (`cv2.findContours`) groups these moving pixels into bounding boxes.
    *   A virtual "counting line" is drawn on the frame. When the center of a bounding box crosses this line, the `vehicle_count` increments.
    *   The localized bounding box crop is then passed to our **Custom CNN** to classify the vehicle type in real-time.

## Technologies Used
*   **Python 3.x**
*   **PyTorch & Torchvision**: For building, training, and running inference on the custom CNN.
*   **OpenCV (`cv2`)**: For video processing, background subtraction, and contour tracking.
*   **Pandas**: For parsing the dataset's `_classes.csv` files.
*   **NumPy & Pillow**: For image manipulation and tensor conversions.

## Installation Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lingeshwarank2026-sudo/VISIONX.git
   cd VISIONX
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   Extract the provided `Vehicles.v1i.multiclass.zip` dataset. Update the `DATASET_PATH` variable in `train_classifier.py` to point to the extracted folder.

4. **Train the Model**:
   ```bash
   python train_classifier.py
   ```
   This will train the CNN and output a `vehicle_classifier.pth` weight file.

5. **Run the Traffic Monitor**:
   Provide a sample traffic video named `sample_traffic.mp4` in the project directory, then run:
   ```bash
   python count_and_classify.py
   ```

## Results
*   The Custom CNN successfully learns feature representations of the four vehicle classes from the dataset, achieving reasonable loss convergence during training.
*   The OpenCV Background Subtractor effectively isolates moving vehicles in standard traffic camera angles, providing a lightweight and highly efficient counting mechanism that works purely on CPU.
*   The combined pipeline successfully counts and classifies vehicles simultaneously.

## Challenges Faced
*   **Dataset Format Constraint**: The primary challenge was that the assignment required "counting vehicles from images/videos", but the provided mandatory dataset only contained classification labels (no bounding boxes). Designing a pipeline that leverages traditional Computer Vision (Background Subtraction) for localization and the required dataset for classification was the breakthrough solution.
*   **Multi-Label Nuances**: Parsing the CSV required custom PyTorch Dataset handling since it utilized a multi-hot encoded vector structure rather than standard class-folder structures.

## Future Improvements
*   **Deep SORT Tracking**: Currently, counting relies on a basic coordinate-crossing logic. Implementing an object tracker like SORT/DeepSORT would prevent double-counting of slow-moving or occluded vehicles.
*   **YOLO Object Detection**: While this project avoided pretrained detection models to stick strictly to the rules, using Ultralytics YOLOv8 (which is permitted in the tools list) would significantly improve the bounding box stability over traditional background subtraction, especially in varied lighting conditions or static image counting.
*   **Data Augmentation**: Further augmenting the training dataset (rotation, zooming, color jitter) would improve the CNN's classification accuracy on blurry video crops.

## Screenshots
*(Include screenshots of your running `count_and_classify.py` window here, showing the counting line and classification bounding boxes)*
