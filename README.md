# Skin-Cancer-Detection-DL
🩺 Skin Cancer Detection using Deep Learning (HAM10000)

This project implements a deep learning–based skin cancer classification system using the HAM10000 dataset.
It compares multiple Convolutional Neural Network (CNN) architectures, including EfficientNet and ResNet, to identify the best-performing model.

This work is designed as an MS Data Science thesis / final-year project with a research-grade pipeline.

📌 Project Objectives

Classify skin lesion images into multiple disease categories

Compare multiple CNN models (EfficientNet & ResNet)

Apply preprocessing, augmentation, and transfer learning

Evaluate models using medical metrics (AUC, Sensitivity, Specificity)

Select the best model for skin cancer detection

🗂 Dataset

HAM10000: Human Against Machine with 10000 training images

10,015 dermatoscopic images

7 diagnostic classes:

akiec – Actinic keratoses

bcc – Basal cell carcinoma

bkl – Benign keratosis-like lesions

df – Dermatofibroma

mel – Melanoma

nv – Melanocytic nevi

vasc – Vascular lesions

Source: Kaggle
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

🧠 Models Used
Model	Role
ResNet-50	Baseline
EfficientNet-B0 / B3	Proposed main model
📂 Project Structure
skin-cancer-detection/
│
├── data/
│   └── HAM10000/
│       ├── images/
│       ├── HAM10000_metadata.csv
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── efficientnet.py
│   └── resnet.py
│
├── dataset.py
├── train.py
├── evaluate.py
├── prepare_data.py
├── main.py
│
├── requirements.txt
├── .gitignore
└── README.md

🧩 File Descriptions
prepare_data.py

Loads original HAM10000 metadata

Creates stratified train.csv and test.csv

Ensures class balance

dataset.py

Custom PyTorch Dataset class

Links CSV labels with image files

Applies image transforms

Returns (image, label) tensors

models/efficientnet.py

Loads pretrained EfficientNet

Replaces final classification layer

models/resnet.py

Loads pretrained ResNet

Replaces final classification layer

train.py

Training loop

Validation loop

Loss computation

Accuracy tracking

Model saving

evaluate.py

Confusion matrix

Classification report

ROC–AUC

Sensitivity & Specificity

main.py

Full training pipeline

Dataset loading

Model selection

Training execution

Evaluation

🔧 Installation
Step 1: Clone Repository
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection

Step 2: Create Virtual Environment
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

Step 3: Install Dependencies
pip install -r requirements.txt

📥 Dataset Setup
Option A — Manual Download

Download HAM10000 from Kaggle

Extract into:

data/HAM10000/images/


Place HAM10000_metadata.csv inside:

data/HAM10000/

Option B — Automatic Split
python prepare_data.py


This creates:

data/HAM10000/train.csv
data/HAM10000/test.csv

🚀 Training Models
Train EfficientNet-B0
python main.py --model efficientnet --version b0

Train ResNet-50
python main.py --model resnet

📊 Evaluation

After training:

python evaluate.py --model efficientnet


Outputs:

Confusion matrix

Classification report

ROC-AUC

Sensitivity

Specificity

🧪 Data Augmentation

Applied during training:

Random horizontal flip

Random rotation

Brightness/contrast adjustment

Resize to 224×224

Normalization

🏥 Medical Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC–AUC

Sensitivity

Specificity

🔍 Transfer Learning

Pretrained ImageNet weights

Final layers fine-tuned

Early layers optionally frozen

🎯 Best Model Selection

The final model is selected using:

ROC–AUC

Sensitivity (Melanoma class)

Specificity

Generalization performance

🧾 Requirements

Key dependencies:

torch
torchvision
pandas
numpy
scikit-learn
matplotlib
opencv-python
tqdm
Pillow

📎 Reproducibility

Fixed random seeds

Stratified splitting

Locked dependency versions

⚠ Notes

Dataset files are excluded via .gitignore

Model weights are not committed

GPU recommended for training

📚 Future Work

Vision Transformers (ViT)

EfficientNetV2

Self-supervised learning

Mobile deployment