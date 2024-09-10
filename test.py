import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet101
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the ResNet101 model for classification
class PlantDiseaseModel(nn.Module):
    def _init_(self, num_classes):
        super(PlantDiseaseModel, self)._init_()
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
# Draw bounding box and disease name
def draw_bounding_box(image, mask, disease_name):
    draw = ImageDraw.Draw(image)
    mask = np.array(mask)
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    
    # Check if there are any True values in rows and cols
    if np.any(rows) and np.any(cols):
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Write disease name
        font = ImageFont.load_default()
        draw.text((x2+5, y1-5), disease_name, font=font, fill="red")
    else:
        print("No bounding box could be drawn, no objects detected in the mask.")

    return image


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

# Main function
def process_image(image_path):
    disease_model, segmentation_model = load_models()
    
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
    
    # Save or display the result
    result_image.save("result.png")
    result_image.show()

# Usage
if __name__ == "__main__":
    process_image("")