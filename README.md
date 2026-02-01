# Deepfake Detection Suite - EfficientNetB0 (v2.0)

This project is a refined iteration of the initial deepfake detection Capstone project. It implements a production-ready Flask service using **EfficientNetB0** optimized for **224x224** image inputs. This version addresses previous limitations by utilizing a balanced dataset and incorporating forensic analysis into the development pipeline.

**Link to Previous Attempt (v1.0):** [Capstone2 v1.0](https://github.com/gconsulting78-debug/machine-learning-zoomcamp-Capstone2.git)

---

## 📊 Version Improvements
* **Balanced Dataset:** Trained on 140,000 images (70k Real / 70k Fake) to eliminate class bias.
* **Optimized Architecture:** Utilizes EfficientNetB0 at its native resolution (224x224) for peak weight efficiency and performance.
* **Forensic EDA:** Analysis includes Error Level Analysis (ELA) and Fast Fourier Transform (FFT) to identify digital fingerprints of GAN-generated content.

---

## 📂 Project Structure
```text
deepfake_project/
├── models/
│   └── best_model_finetuned.keras  # Fine-tuned EfficientNetB0 Model
├── data/
│   └── samples/                   # Test images for API validation
├── app2.py                        # Flask API Service
├── requirements.txt               # Project Dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project Documentation

---

## 🚀 Local Execution Guide

Follow these steps to set up the environment and run the inference service on your local machine.

### 1. Repository Setup
```bash
# Clone the repository
git clone [https://github.com/gconsulting78-debug/machine-learning-zoomcamp-Capstone2b.git](https://github.com/gconsulting78-debug/machine-learning-zoomcamp-Capstone2b.git)
cd deepfake_project
2. Environment Configuration
Virtual environments ensure that dependency versions remain consistent with the training environment.

Bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
🖥️ Running the Inference Server
1. Start the Flask API
Bash

# Start the service
python app2.py
The service will initialize the EfficientNetB0 model and listen on http://127.0.0.1:5000.

2. Testing the API
Open a separate terminal window and use curl to send a sample image for prediction:

Bash

curl -X POST -F "image=@data/samples/fake_1.jpg" [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)
🛠 Technical Specifications
Architecture: EfficientNetB0 (ImageNet weights)

Input Dimension: 224 x 224 x 3 (RGB)

Preprocessing: Standardized pixel scaling [0, 255]

Forensic EDA: Error Level Analysis (ELA) and FFT Frequency checks.

API Service: Flask with JSON response handling.
