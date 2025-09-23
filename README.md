# Plant Disease Classification + Detection
This project explores plant disease classification and object detection using deep learning (PyTorch).

## Learning Journey
Took the FreeCodeCamp Introduction to PyTorch course, where I explored:
1. CNN architectures
2. Kernels, padding, stride, max pooling etc.
3. Training loops (backpropagation, gradient descent, etc.)
4. Project training done on Google Colab Pro

## Results
1. Achieved 90%+ accuracy on PlantVillage benchmark images using a custom CNN.
2. Identified dataset bias: accuracy dropped significantly on real-world images.
3. Addressed this by augmenting with noisy real-world samples (Google Images), applying transfer learning (ResNet18), experimenting with advanced augmentations etc. → improved generalization to ~70% accuracy.

## Methods & Experiments
1. Transfer Learning: ResNet18 (freezing/unfreezing layers, fine-tuning).
2. Training Strategies:
    - Mixup augmentation
    - Learning rate schedulers (StepLR, OneCycleLR, ReduceLROnPlateau)
    - BatchNorm & Dropout regularization
6. Optimizers: SGD w/ momentum, Adam w/ weight decay
7. Custom CNN: Deeper hidden units, regularization, data normalization.
8. Data Augmentation: TrivialAugmentWide, ColorJitter, and more.

## Model Evaluation & Visualization
1. Confusion Matrix
2. Grad-CAM & Grad-CAM++

## Object Detection (real-world datasets only)
1. Used YOLO for object detection
2. Pre-processed labels with LabelStudio

## Deployment
1. Saved/loaded model as .pth/.pt in VS Code
2. Deployed app on Streamlit 
    - Demo (just upload a picture of a leaf): [Streamlit Link](https://plant-disease-classification-vayyob3uqtbmtgjf5clzhg.streamlit.app/)
3. Implemented top-k softmax to display model’s top-2 predictions with confidence scores
    - Disclaimer: This model is trained on the 38 classes in the PlantVillage dataset, alongside a small real-world sample obtained from Google Images. Accuracy is moderate, so results are meant for demonstration purposes and may not be fully reliable.

## Next Goals
1. Push image classification accuracy beyond 75%+ on real-world data 
2. Improve object detection accuracy with further fine-tuning

## Datasets
1. **PlantVillage dataset:** Used for training and benchmarking. Available at [PlantVillage GitHub](https://github.com/spMohanty/PlantVillage-Dataset.git)  
2. **Real-world images:** Curated from Google Images to test generalization (not included in the repo due to licensing restrictions)

## References
1. Referenced [geeksforgeeks Object Detection with YOLO and OpenCV](https://www.geeksforgeeks.org/computer-vision/object-detection-with-yolo-and-opencv/) for customization

 ## Model Weights
1. Pre-trained model weights (`.pt` / `.pth`) are hosted on [Google Drive](https://drive.google.com/drive/folders/1SJD4w37yV43QSEeLKKjAJIl7sS5iMiMP?usp=drive_link)

## Notebooks:
1. experimental_notebook_for_fine_tuning_and_more.ipynb: An exploratory notebook that shows the process behind achieving the final model. May include messy code / experiments that did not make it to the final version.
2. Final_plantdisease_classification.ipynb: Final version. 