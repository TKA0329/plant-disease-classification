# Plant Disease Classification + Detection
This project explores plant disease classification and object detection using deep learning (PyTorch).

## Learning Journey
Took the FreeCodeCamp Introduction to PyTorch course, where I learned:
1. CNN architectures
2. Kernels, padding, stride, and max pooling
3. Training loops (backpropagation, gradient descent, etc.)

## Results
1. Achieved 90%+ accuracy on PlantVillage benchmark images.
2. Identified dataset bias: accuracy dropped significantly on real-world images.
3. Addressed this by augmenting with noisy real-world samples (Google Images), applying transfer learning (ResNet18), and experimenting with advanced augmentations → improved generalization to ~72% accuracy.

## Methods & Experiments
1. Transfer Learning: ResNet18 (freezing/unfreezing layers, fine-tuning).
2. Training Strategies:
3. Mixup augmentation
4. Learning rate schedulers (StepLR, OneCycleLR, ReduceLROnPlateau)
5. BatchNorm & Dropout regularization
6. Optimizers: SGD w/ momentum, Adam w/ weight decay
7. Custom CNN: Deeper hidden units, regularization, data normalization.
8. Data Augmentation: TrivialAugmentWide, ColorJitter, and more.

## Model Evaluation & Visualization
1. Confusion Matrix
2. Grad-CAM & Grad-CAM++

## Object Detection (real-world datasets only)
1. Used YOLO for detection
2. Pre-processed labels with LabelStudio

## Deployment
1. Saved/loaded model as .pt in VS Code
2. Deployed app on Streamlit
3. Implemented top-k softmax to display model’s top-2 predictions with confidence scores

## Next Goals
1. Push image classification accuracy beyond 75%+ on real-world data
2. Improve object detection accuracy with further fine-tuning