import os
import cv2
import sqlite3
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet101
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, render_template, request

app = Flask(__name__)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the ResNet101 model for classification
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.resnet = resnet101(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Define the DeepLabV3 model for segmentation
def prepare_segmentation_model(num_classes=2):
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model

# Load models
def load_models():
    # Load ResNet101 for plant disease detection
    num_classes = 20
    disease_model = PlantDiseaseModel(num_classes)
    
    # Load the state dict
    state_dict = torch.load('model/ResNet_101_ImageNet_plant-model-84.pth', map_location=device)
    
    # Remove the 'module.' prefix from state dict keys if present
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load the modified state dict
    disease_model.load_state_dict(new_state_dict, strict=False)
    disease_model.to(device)
    disease_model.eval()

    # Load DeepLabV3 for image segmentation
    segmentation_model = prepare_segmentation_model()
    checkpoint = torch.load('model/model.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        segmentation_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        segmentation_model.load_state_dict(checkpoint)
    segmentation_model.to(device)
    segmentation_model.eval()

    return disease_model, segmentation_model

# Global variables for models
disease_model, segmentation_model = load_models()

# Define the class labels
class_labels = [
    "Apple Aphis spp", "Apple Erisosoma lanigerum",
    "Apple Monillia laxa", "Apple Venturia inaequalis",
    "Apricot Coryneum beijerinckii", "Apricot Monillia laxa",
    "Cancer symptom", "Cherry Aphis spp",
    "Downy mildew", "Drying symptom",
    "Gray mold", "Leaf scars",
    "Peach Monillia laxa", "Peach Parthenolecanium corni",
    "Pear Erwinia amylovora", "Plum Aphis spp",
    "RoughBark", "StripeCanker",
    "Walnut Eriophyies erineus", "Walnut Gnomonialeptostyla",
]

# Initialize database
def init_db():
    conn = sqlite3.connect('photos.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS photos
                 (id INTEGER PRIMARY KEY, original_filename TEXT, processed_filename TEXT, 
                  timestamp TEXT, disease_name TEXT)''')
    conn.commit()
    conn.close()

# Check if the database exists and initialize it if not
def check_and_initialize_db():
    if not os.path.exists('photos.db'):
        init_db()

# Preprocess image for disease detection
def preprocess_image_disease(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0).to(device)

# Preprocess image for segmentation
def preprocess_image_segmentation(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0).to(device)

# Perform disease detection
def detect_disease(model, image):
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Perform image segmentation
def segment_image(model, image):
    with torch.no_grad():
        output = model(image)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

# Draw bounding box and disease name
def draw_bounding_box(image, mask, disease_name):
    draw = ImageDraw.Draw(image)
    mask = np.array(mask)
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # Write disease name
    font = ImageFont.load_default()
    draw.text((x2+5, y1-5), disease_name, font=font, fill="red")
    
    return image

# Process image for disease detection
def process_image(image_path):
    # Preprocess image for disease detection
    input_image_disease = preprocess_image_disease(image_path)
    
    # Detect disease
    disease_id = detect_disease(disease_model, input_image_disease)
    disease_name = class_labels[disease_id]
    
    # Preprocess image for segmentation
    input_image_segmentation = preprocess_image_segmentation(image_path)
    
    # Segment image
    segmentation_mask = segment_image(segmentation_model, input_image_segmentation)
    
    # Load original image and apply mask
    original_image = Image.open(image_path)
    segmentation_mask_resized = Image.fromarray(segmentation_mask).resize(original_image.size)
    highlighted_image = Image.fromarray(np.array(segmentation_mask_resized) * 255).convert("RGBA")
    original_image = original_image.convert("RGBA")
    blended_image = Image.blend(original_image, highlighted_image, 0.5)
    
    # Draw bounding box and disease name
    result_image = draw_bounding_box(blended_image, segmentation_mask_resized, disease_name)
    
    # Save the result
    result_path = image_path.replace('.jpg', '_processed.png')
    result_image.save(result_path)
    
    return disease_name, result_path

# Capture a single photo
def capture_photo(photo_id, frame):
    image_folder = "captured_images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Generate unique filename based on photo_id and current timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    original_filename = f"photo_{photo_id}_{timestamp}.jpg"
    original_filepath = os.path.join(image_folder, original_filename)

    # Save the frame as an image
    cv2.imwrite(original_filepath, frame)

    # Process the image for disease detection
    disease_name, processed_filepath = process_image(original_filepath)

    # Ensure the database is initialized before saving
    check_and_initialize_db()

    # Save the photo details to the database
    conn = sqlite3.connect('photos.db', check_same_thread=False)
    c = conn.cursor()
    timestamp_db = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO photos (original_filename, processed_filename, timestamp, disease_name) VALUES (?, ?, ?, ?)", 
              (original_filename, os.path.basename(processed_filepath), timestamp_db, disease_name))
    conn.commit()
    conn.close()

    return os.path.basename(processed_filepath)

# API to capture a single photo from the video stream
@app.route('/capture', methods=['POST'])
def capture_from_stream():
    # Ensure the database is initialized before capturing
    check_and_initialize_db()

    # URL of the video stream
    video_url = "http://192.168.232.102:5000/video_feed"

    # Open video stream
    cap = cv2.VideoCapture(video_url)

    # Check if the video stream is open
    if not cap.isOpened():
        return jsonify({'error': 'Failed to open video stream'}), 500

    # Read a single frame from the stream
    ret, frame = cap.read()
    cap.release()  # Release the video capture after reading one frame

    # Check if frame was successfully captured
    if not ret:
        return jsonify({'error': 'Failed to capture frame'}), 500

    # Generate a photo ID (could be auto-increment or another logic)
    photo_id = int(datetime.now().timestamp())  # Simple unique ID generation based on timestamp

    # Capture and save the photo
    processed_filename = capture_photo(photo_id, frame)

    return jsonify({'message': 'Photo captured and processed successfully', 'filename': processed_filename}), 200

# API to view photo by ID
@app.route('/api/photos/<int:id>', methods=['GET'])
def get_photo(id):
    # Ensure the database is initialized before fetching photos
    check_and_initialize_db()

    conn = sqlite3.connect('photos.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT processed_filename FROM photos WHERE id=?", (id,))
    photo = c.fetchone()
    conn.close()

    if photo:
        image_folder = "captured_images"
        return send_from_directory(image_folder, photo[0])
    else:
        return jsonify({'error': 'Photo not found'}), 404

# API to get a list of all photos
@app.route('/api/photos', methods=['GET'])
def list_photos():
    # Ensure the database is initialized before listing photos
    check_and_initialize_db()

    conn = sqlite3.connect('photos.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, processed_filename, disease_name FROM photos")
    photos = c.fetchall()
    conn.close()

    return jsonify({'photos': [{'id': photo[0], 'filename': photo[1], 'disease_name': photo[2]} for photo in photos]})

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream video feed
def generate_frames():
    video_url = "http://192.168.232.102:5000/video_feed"
    cap = cv2.VideoCapture(video_url)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return app.response_class(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize the database on app startup if it doesn't exist
    check_and_initialize_db()
    app.run(debug=True)