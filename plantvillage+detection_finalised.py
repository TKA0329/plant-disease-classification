import streamlit as st
import zipfile
import torch
import io
import os 
from pathlib import Path
from torchvision.models import resnet18
import cv2
import numpy as np
from PIL import Image
import gdown

pth_path = Path("fc+3+4_RealWorld+PlantVillageModelV7.pth")
pt_path = Path("my_model_5.pt")

if not pth_path.exists():
    gdown.download("https://drive.google.com/file/d/1U1_wAo24jBxrFaQO1gC3DLHtXdoolh8i/view?usp=drive_link", str(pth_path), quiet=False)

if not pt_path.exists():
    gdown.download("https://drive.google.com/file/d/1emlbp3vo_ONumYaHO0o1JvvVuOO6kAj9/view?usp=drive_link", str(pt_path), quiet=False)
# Might be necessary if there's different class labels
# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# set up the folder for data.zip
extract_path = "unzipped_data_plant_disease"
os.makedirs(extract_path, exist_ok=True)

# But extractall() needs a directory path to unzip the files into. 
with zipfile.ZipFile("my_model_5.zip", "r") as zip_ref:
  zip_ref.extractall(extract_path)

#instantiate the model
model_loaded = resnet18(weights=None)
model_loaded.fc = torch.nn.Linear(in_features=512, out_features=38)

# Load the state dict
model_loaded.load_state_dict(torch.load("fc+3+4_RealWorld+PlantVillageModelV7.pth", map_location=torch.device('cpu')))

# Set to eval mode (important for testing/inference)
model_loaded.eval()

# load and instantiate YOLO-trained model
from ultralytics import YOLO
path = Path("unzipped_data_plant_disease/my_model_5.pt")
model = YOLO(path)

# upload image 
from torchvision import transforms
from PIL import Image
import torch

st.markdown("### Image Classification")

st.write("Avoid images with multiple leaves, hands, insects, or busy backgrounds! A plain background (or cropping tightly around the leaf) works best.")

image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

if image_file is not None:
    # Define the transform 
    image_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Load your image for image classification
    img_path = image_file 
    img = Image.open(img_path).convert("RGB") # PIL (RGB)
    img_transformed = image_transform(img).unsqueeze(0)  # add batch dimension
    
    # creating a path to save the YOLO-predicted image to
    
    results = model(img,  # must use img NOT img_path
                    save=True, # so original file used for drawing boxes ltr to prevent blue boxes
                    project="streamlit_outputs", 
                    name="results", 
                    exist_ok=True, 
                    conf=0.5, # → Only shows predictions with ≥ 50% confidence
                    iou=0.4)[0] # → Controls how strict NMS is when removing overlapping boxes. Lower = stricter.
    
    # Run the model
    with torch.inference_mode():
        output = model_loaded(img_transformed)

    # Get predicted class + index + probabilities
    probabilities = torch.softmax(output, dim=1)
    
    #finding the second largest probability 
    top2= torch.topk(probabilities, 2)
    top2_prob = ((top2.values[0]).tolist())[1]
    
    # Get 1st prob's predicted class
    pred_class = torch.argmax(output, dim=1).item()
    st.write(f"No.1 Predicted class index: {pred_class}")
    
    # Get 2nd prob's predicted class
    list_probs = probabilities.squeeze().tolist()
    pred_class_2 = list_probs.index(top2_prob)
    st.write(f"No.2 Predicted class index: {pred_class_2}")
    st.write("--------------------------------------------------------------")
    
    dict_of_classes = {'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 
                       'Apple___Cedar_apple_rust': 2, 
        'Apple___healthy': 3, 'Blueberry___healthy': 4, 
        'Cherry_(including_sour)___Powdery_mildew': 5, 
        'Cherry_(including_sour)___healthy': 6, 
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8, 
        'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11, 
        'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 
        'Grape___healthy': 14, 
        'Orange___Haunglongbing_(Citrus_greening)': 15, 
        'Peach___Bacterial_spot': 16, 
        'Peach___healthy': 17, 'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 
        'Potato___Early_blight': 20, 'Potato___Late_blight': 21, 'Potato___healthy': 22, 
        'Raspberry___healthy': 23, 'Soybean___healthy': 24, 'Squash___Powdery_mildew': 25, 
        'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27, 'Tomato___Bacterial_spot': 28,
            'Tomato___Early_blight': 29, 'Tomato___Late_blight': 30, 
        'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32, 
        'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34, 
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 
        'Tomato___Tomato_mosaic_virus': 36, 'Tomato___healthy': 37}

    st.write(f"Predicted label 1: {(next((k for k, v in dict_of_classes.items() if v == pred_class), None))}")
    st.write(f"No.1 Predicted Confidence: {(probabilities.squeeze().tolist())[pred_class]}")
    st.write("--------------------------------------------------------------")
    st.write(f"Predicted label 2: {(next((k for k, v in dict_of_classes.items() if v == pred_class_2), None))}")
    st.write(f"No.2 Predicted Confidence: {top2_prob}")
    st.write("--------------------------------------------------------------")
    
    st.markdown("#### Uploaded image: ")
    
    # displaying image on streamlit 
    buf = io.BytesIO()

    # original image 
    st.image(image_file, width=250)

    # transformed image for image classification (visualisation)
    #st.image((image_transform(img).permute(1,2,0)).numpy(), width=250)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
   
    # display image for disease spots identification
    if st.button("Display disease spots"):
        classes_names = ["Black/Brown spots", 
                         "Mildew",
                         "yellow-green discoloration",
                         "yellow/brown patches", 
                         "yellow/brown/red spots"]
        result_path = Path("streamlit_outputs/results/image0.jpg")
        image = cv2.imread(str(result_path)) # requires a str 
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(np.array(img), # NumPy (RGB)
                             cv2.COLOR_RGB2BGR #cv2.imread(...) loads the image in BGR # NumPy (BGR)
                             )

        for box in results.boxes:
            
            # box.xyxy[0] contains the bounding box coordinates: top-left (x1, y1) and bottom-right (x2, y2)
            [x1, y1, x2, y2] = box.xyxy[0]
            
            # convert to int so can be used in drawing functions 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # get the class
            cls = int(box.cls[0])

            # get the class name
            class_name = classes_names[cls]

            # get the respective colour
            colour = getColours(cls)

            # draw the rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), 
                          colour, 
                          2) # 1 is the thickness of box

            # put the class name and confidence on the image
            cv2.putText(image, f'{class_name} {box.conf[0]:.2f}', # display class name and confidence
                        (x1, y1 - 5), # shift slightly above box
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, # font size 
                        colour, # colour of text
                        1) # thickness of font must be int

        st.markdown("#### Image with disease spots highlighted: ")
        image = image = cv2.cvtColor(image,
                             cv2.COLOR_BGR2RGB # Convert back to RGB for display # NumPy (RGB again)
                             )
        image_pil = Image.fromarray(image) # PIL
        st.image(image_pil)

else:
   st.write("Please upload an image first!")
   