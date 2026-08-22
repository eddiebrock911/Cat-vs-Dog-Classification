# 🐱🐶 Cat vs Dog Classification

<p align="center">
  <strong>Deep Learning • Computer Vision • TensorFlow • Neural Networks</strong>
</p>

<p align="center">
  A Deep Learning powered image classification application that identifies whether an uploaded image contains a <strong>Cat</strong> or a <strong>Dog</strong>.
</p>

<p align="center">
  <a href="https://imgclassifier-tilx.onrender.com">🚀 Live Demo</a> •
  <a href="https://github.com/eddiebrock911/Cat-vs-Dog-Classification">💻 Source Code</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Keras-Neural%20Network-D00000?style=for-the-badge&logo=keras&logoColor=white">
  <img src="https://img.shields.io/badge/Computer%20Vision-Image%20Classification-5C3EE8?style=for-the-badge">
  <img src="https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=black">
</p>

---

## 📌 Overview

**Cat vs Dog Classification** is a Deep Learning computer-vision project that uses a neural-network-based image classifier to distinguish between two classes:

* 🐱 **Cat**
* 🐶 **Dog**

The project demonstrates a complete workflow from image processing and model inference to deploying an image-classification application on the web.

Binary image classification is a common introductory Computer Vision problem and is useful for understanding how neural networks learn visual patterns from images.

---

## 🚀 Live Demo

### 🌐 Try the Application

**Live Demo:**
https://imgclassifier-tilx.onrender.com

Upload an image and let the trained model determine whether it is more likely to be a **Cat** or **Dog**.

---

## ✨ Features

* 🐱 Cat image detection
* 🐶 Dog image detection
* 🧠 Deep Learning-based classification
* 👁️ Computer Vision workflow
* 🖼️ Image upload and processing
* ⚡ Fast prediction
* 🌐 Web-based interface
* ☁️ Render deployment
* 📦 Python dependency management
* 🔧 Automated setup support
* 📱 Easy-to-use application workflow

---

## 🧠 How It Works

The application follows this pipeline:

```text
                 User
                  │
                  ▼
          ┌───────────────┐
          │ Upload Image  │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ Image Loading │
          └───────┬───────┘
                  │
                  ▼
          ┌────────────────┐
          │ Preprocessing  │
          └───────┬────────┘
                  │
                  ▼
          ┌────────────────┐
          │ Neural Network │
          └───────┬────────┘
                  │
                  ▼
          ┌────────────────┐
          │ Classification │
          └───────┬────────┘
                  │
            ┌─────┴─────┐
            ▼           ▼
         🐱 CAT       🐶 DOG
```

---

## 🔬 Computer Vision Pipeline

The general image-classification workflow is:

```text
Input Image
     ↓
Image Loading
     ↓
Resize / Preprocessing
     ↓
Normalization
     ↓
Tensor Conversion
     ↓
Neural Network
     ↓
Feature Extraction
     ↓
Binary Classification
     ↓
Cat / Dog Prediction
```

The model learns visual patterns such as:

* Shapes
* Edges
* Textures
* Fur patterns
* Facial structures
* Body features
* Spatial relationships

These learned representations are then used to distinguish the two classes.

---

## 🧩 Technology Stack

### 🐍 Programming

* Python

### 🧠 Deep Learning

* TensorFlow
* Keras
* Neural Networks

### 👁️ Computer Vision

* Image preprocessing
* Image classification
* Feature learning

### ☁️ Deployment

* Render

### 🛠️ Development

* Git
* GitHub
* Python virtual environments

The repository itself is categorized around Artificial Intelligence, Deep Learning, Neural Networks, and TensorFlow.

---

## 📂 Project Structure

```text
Cat-vs-Dog-Classification/
│
├── cat-vs-dog.py
│
├── requirements.txt
│
├── render.yaml
│
├── setup.sh
│
├── .gitattributes
│
└── README.md
```

### 📄 File Description

| File               | Description                     |
| ------------------ | ------------------------------- |
| `cat-vs-dog.py`    | Main Python application         |
| `requirements.txt` | Python dependencies             |
| `render.yaml`      | Render deployment configuration |
| `setup.sh`         | Setup / initialization script   |
| `.gitattributes`   | Git configuration               |
| `README.md`        | Project documentation           |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/eddiebrock911/Cat-vs-Dog-Classification.git
```

### 2. Navigate to the Project

```bash
cd Cat-vs-Dog-Classification
```

### 3. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## 📦 Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run the main Python application:

```bash
python cat-vs-dog.py
```

If the application starts a local web server, open the URL shown in your terminal.

---

## 🧠 Model Prediction

The classifier performs binary classification:

```text
Input
  │
  ▼
Neural Network
  │
  ├───────────────┐
  ▼               ▼
 Cat Probability  Dog Probability
  │               │
  └───────┬───────┘
          ▼
    Final Prediction
```

The class with the stronger model output becomes the predicted category.

---

## 📊 Classification Concept

This project uses **binary classification**.

The model learns a decision boundary between:

```text
Class 0 → Cat 🐱
Class 1 → Dog 🐶
```

A simplified prediction concept can be represented as:

```python
prediction = model.predict(image)
```

The resulting model output is then converted into the corresponding class label.

---

## 🖼️ Example Workflow

```text
┌─────────────────────┐
│    Select Image     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Preprocess Image   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Run Neural Network  │
└──────────┬──────────┘
           │
           ▼
     ┌─────┴─────┐
     │           │
     ▼           ▼
  🐱 CAT       🐶 DOG
```

---

## 🎯 Learning Objectives

This project was created to practice and understand:

* Deep Learning fundamentals
* Neural networks
* Image classification
* Computer Vision
* Image preprocessing
* TensorFlow/Keras
* Binary classification
* Model inference
* Python application development
* Cloud deployment

---

## 🔮 Future Improvements

The project can be extended with:

* [ ] Display prediction confidence
* [ ] Add image preview
* [ ] Improve UI/UX
* [ ] Add drag-and-drop upload
* [ ] Add batch image prediction
* [ ] Add confusion matrix
* [ ] Add accuracy/loss graphs
* [ ] Add data augmentation
* [ ] Add transfer learning
* [ ] Experiment with MobileNetV2
* [ ] Experiment with ResNet
* [ ] Add model explainability
* [ ] Add prediction history
* [ ] Add REST API
* [ ] Add automated testing
* [ ] Add CI/CD using GitHub Actions

---

## 📈 Possible Advanced Architecture

A future version could use Transfer Learning:

```text
                 Input Image
                      │
                      ▼
             Pretrained CNN
                      │
              Feature Extraction
                      │
                      ▼
                Dense Layer
                      │
                      ▼
               Binary Output
                /          \
               /            \
             🐱              🐶
            Cat             Dog
```

Possible backbones include:

* MobileNetV2
* EfficientNet
* ResNet
* VGG16

Transfer learning can provide a stronger starting point by reusing visual features learned from large image datasets.

---

## ☁️ Deployment

The project includes:

```text
render.yaml
```

for deployment configuration.

The deployed application is available at:

**https://imgclassifier-tilx.onrender.com**

Deployment architecture:

```text
GitHub
   │
   ▼
Render
   │
   ▼
Python Environment
   │
   ▼
Application
   │
   ▼
Public Web App
```

---

## ⚠️ Limitations

A Cat-vs-Dog classifier may not perform perfectly on every image.

Performance can be affected by:

* Poor image quality
* Unusual camera angles
* Multiple animals
* Occluded animals
* Very dark images
* Unusual backgrounds
* Images outside the training distribution
* Animals other than cats or dogs

Therefore, predictions should be treated as model outputs rather than guaranteed classifications.

---

## 🧪 Testing Ideas

For evaluating the model, useful test cases include:

```text
✓ Clear cat image
✓ Clear dog image
✓ Different cat breeds
✓ Different dog breeds
✓ Indoor images
✓ Outdoor images
✓ Low-light images
✓ Different backgrounds
✓ Different image resolutions
```

---

## 📚 What I Learned

Through this project, I practiced the complete journey of a Computer Vision application:

```text
Machine Learning
      ↓
Deep Learning
      ↓
Neural Networks
      ↓
Computer Vision
      ↓
Image Classification
      ↓
Python Application
      ↓
Deployment
```

This project helped bridge the gap between **training an AI model** and **turning that model into an actual usable application**.

---

## 👨‍💻 Author

### Ankit

**AI / Machine Learning Developer**

GitHub:
https://github.com/eddiebrock911

Repository:
https://github.com/eddiebrock911/Cat-vs-Dog-Classification

Live Demo:
https://imgclassifier-tilx.onrender.com

---

## ⭐ Support the Project

If you found this project useful:

⭐ **Star the repository**

🍴 **Fork the repository**

🐛 **Report bugs**

💡 **Suggest improvements**

---

## 📜 License

This project is provided for educational and experimental purposes.

Please verify the licensing terms of any external dataset, pretrained model, or third-party resource used with the project before redistribution.

---

<p align="center">

<strong>🐱 + 🐶 + 🧠 = Computer Vision</strong>

<br><br>

Built with <strong>Python • TensorFlow • Deep Learning</strong>

<br><br>

Made by <strong>Ankit</strong> 🚀

</p>
