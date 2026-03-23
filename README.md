# 🚗 Car Damage Detection using Deep Learning

An end-to-end Deep Learning project that detects and classifies **car damage types** from images using a trained ResNet50 model.

This project demonstrates the complete pipeline from model training to deployment using Streamlit.

---

## 🚀 Live Demo
👉 https://car-damage-detection-ai-mohamed-aslam.streamlit.app/

---

## 🚀 Project Overview

Manual vehicle damage inspection is time-consuming and error-prone, especially in insurance and automobile industries.

This application uses Deep Learning to automatically:

- Detect car damage  
- Classify damage type  
- Provide confidence scores  
- Visualize prediction probabilities  

---

## 🎯 Objective

To build a deep learning-based image classification system that:

- Classifies car damage into predefined categories  
- Provides prediction confidence  
- Handles real-world image inputs  
- Deploys as an interactive web application  

---

## 📊 Dataset Details

- Total Images: ~1700  
- Type: Labeled car damage images  
- Views: Third-quarter front & rear  

### 🧾 Classes (6 Categories)

- Front Normal  
- Front Crushed  
- Front Breakage  
- Rear Normal  
- Rear Crushed  
- Rear Breakage  

---

## 🧠 Model Architecture

- Model: **ResNet50 (Transfer Learning)**  
- Framework: **PyTorch**  
- Pretrained on ImageNet  
- Fine-tuned last layers for classification  

---

## 🔄 Deep Learning Workflow

### 1. Data Preprocessing
- Image resizing (224x224)  
- Normalization using ImageNet stats  

### 2. Model Training
- Transfer Learning using ResNet50  
- Frozen base layers  
- Fine-tuned last layers  

### 3. Evaluation
- Validation Accuracy: **~80%**  

### 4. Inference Pipeline
- Image upload  
- Transformation  
- Model prediction  
- Softmax probability output  

### 5. Deployment
- Streamlit-based interactive dashboard  

---

## ⚙️ Features

- 📤 Upload car images  
- 🤖 AI-based damage classification  
- 📊 Confidence score display  
- 📈 Class probability visualization  
- 🌙 Modern dark UI with animations  
- ⚡ Optimized model loading using caching  

---

## 📸 Sample Output

![App Screenshot](app_screenshot.jpg)

---

## 🌐 Web Application

The application allows users to:

- Upload a car image  
- Get predicted damage type  
- View confidence score  
- Analyze probability distribution across all classes  

---

## 🛠 Tech Stack

- Python  
- PyTorch  
- Torchvision  
- Streamlit  
- Pandas  
- Pillow  

---
## 📁 Project Structure

```bash
Car-damage-detection-AI/
│
├── app.py                 # Streamlit UI application
├── model_helper.py        # Model loading & prediction logic
│
├── model/
│   └── saved_model.pth    # Trained deep learning model
│
├── .streamlit/
│   └── config.toml        # Streamlit configuration (theme)
│
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
```

---

## ⚙️ Installation & Run

```bash
git clone https://github.com/aslam347/Car-damage-detection-AI.git
cd Car-damage-detection-AI
pip install -r requirements.txt
streamlit run app.py
```

---

## 📜 Requirements

```txt
streamlit==1.55.0
torch==2.2.2
torchvision==0.17.2
numpy==1.26.4
pillow
pandas
```

---

## 💡 Key Learnings

- Deep Learning using Transfer Learning (ResNet50)  
- Image classification using CNN  
- Data preprocessing for images  
- Model optimization and inference  
- Handling real-world deployment issues  
- Streamlit dashboard development  
- Debugging dependency conflicts (Torch, NumPy, Python)  
- End-to-end AI project deployment  

---

## 👨‍💻 Author

**Mohamed Aslam M**  
AI Engineer | Machine Learning Enthusiast  



