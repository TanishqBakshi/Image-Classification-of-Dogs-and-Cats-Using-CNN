# Image Classification of Dogs and Cats Using CNN

## Project Overview

This project classifies images of cats and dogs using Deep Learning and Convolutional Neural Networks (CNNs).

Initially, a custom CNN model was developed and trained. However, the model showed signs of overfitting and achieved lower test accuracy.

To improve performance, MobileNetV2 was used through Transfer Learning. The final model achieved significantly higher accuracy and better generalization on unseen images.

The project also integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of an image influenced the model's prediction.

---

## Objectives

- Classify images as either Cat or Dog
- Compare a Custom CNN with MobileNetV2
- Improve model performance using Transfer Learning
- Visualize model decision-making using Grad-CAM
- Deploy the model through a Streamlit web application

---

## Dataset

Dataset used:

- Cats and Dogs Dataset
- Two classes:
  - Cats
  - Dogs

Image Size:

- 224 × 224 pixels

---

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- OpenCV
- Streamlit
- Jupyter Notebook

---

## Model Architecture

### Custom CNN

Layers:

- Conv2D (32)
- MaxPooling2D
- Conv2D (64)
- MaxPooling2D
- Conv2D (128)
- MaxPooling2D
- Flatten
- Dense (128)
- Dropout (0.5)
- Dense (1)

### MobileNetV2

Transfer Learning model using MobileNetV2 as the feature extractor with a custom classification head.

---

## Model Performance

### Custom CNN

| Metric | Value |
|----------|----------|
| Training Accuracy | 96.7% |
| Validation Accuracy | 76.7% |
| Test Accuracy | 74.7% |

Observation:

The model showed significant overfitting.

---

### MobileNetV2

| Metric | Value |
|----------|----------|
| Training Accuracy | 98.6% |
| Validation Accuracy | 98.1% |
| Test Accuracy | 98.7% |

Observation:

MobileNetV2 achieved significantly better generalization and accuracy compared to the custom CNN.

---

## Grad-CAM Visualization

Grad-CAM was implemented to visualize which regions of an image contributed most to the model's prediction.

Benefits:

- Improves interpretability
- Helps verify model behavior
- Demonstrates Explainable AI (XAI)

---

## Streamlit Application

The project includes a Streamlit web application where users can:

1. Upload an image
2. Receive Cat/Dog prediction
3. View prediction confidence
4. View Grad-CAM visualization

Run the application:

```bash
streamlit run app/app.py
```

---

## Project Structure

```text
CNN_PROJECT
│
├── app
│   ├── app.py
│   └── gradcam_utils.py
│
├── models
│   ├── cnn_model.keras
│   └── mobilenet_model.keras
│
├── notebooks
│   ├── dataset_exploration.ipynb
│   ├── mobilenet_training.ipynb
│   └── gradcam.ipynb
│
├── results
│   └── gradcam_output.jpg
│
├── README.md
└── requirements.txt
```

---

## Future Improvements

- Multi-class image classification
- Better UI design using React
- Cloud deployment
- Real-time image capture using webcam
- Support for larger datasets

---

## Author

Tanishq Bakshi

B.Tech Computer Science Engineering

Amity University