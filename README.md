# 🛡️ Deepfake Image Detection: EfficientNetB0 (v2.0)

## 📖 Project Overview
This project is an advanced iteration of the initial Deepfake detection pipeline. It focuses on the binary classification of images to detect AI-generated content using a fine-tuned **EfficientNetB0** architecture. By analyzing digital textures and frequency artifacts, the model identifies subtle signatures left by GANs and diffusion models that are often invisible to the human eye.

This version (v2.0) improves upon the previous submission by utilizing a perfectly balanced dataset and a production-ready Flask API for real-time inference.

### 🔗 Project Links
- **Previous Attempt (v1.0):** [Capstone2 v1.0 Repository](https://github.com/gconsulting78-debug/machine-learning-zoomcamp-Capstone2.git)
- **Model Architecture:** EfficientNetB0 (Native 224x224 input)
- **Dataset:** 140,000 High-Resolution Images (70k Real / 70k Fake)

---

## 📂 Project Structure
The repository is organized for portability and clear evaluation:

```text
deepfake_project/
├── models/
│   └── best_model_finetuned.keras  # Fine-tuned EfficientNetB0 weights
├── data/
│   └── samples/                   # Sample images for API testing
├── app2.py                        # Flask API Service
├── requirements.txt               # Version-locked dependencies
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```
---

## 🚀 Local Execution Guide

Follow these steps to set up the environment and run the inference service on your local machine.

### 1. Repository Setup

```bash
# Clone the repository
git clone [https://github.com/gconsulting78-debug/machine-learning-zoomcamp-Capstone2b.git](https://github.com/gconsulting78-debug/machine-learning-zoomcamp-Capstone2b.git)
cd deepfake_project
```
### 2. Environment Configuration

Virtual environments ensure that dependency versions remain consistent with the training environment and prevent conflicts with system-wide packages.

```bash
# Create a virtual environment
python3 -m venv venv
```
### 3. Activate the virtual environment
#### On macOS/Linux:
source venv/bin/activate  
#### On Windows: 
venv\Scripts\activate

### 4. Install required packages from the requirements file
pip install -r requirements.txt
### ⚠️ Compatibility Note: Keras 3 & TensorFlow
This project is specifically configured to use **Keras 3.10.0** with **TensorFlow 2.15.0**. 

* **Why?** This ensures the specific fine-tuning layers and model weights from version 2.0 load correctly without serialization errors.
* **Environment Tip:** If you encounter a `dependency conflict` warning during installation, it is due to the overlap between TensorFlow's internal Keras and the standalone Keras 3 package. The provided `app2.py` is designed to handle this specifically in a Python 3.10+ environment.
---

## 🖥️ Running the Inference Server

### 1. Start the Flask API

```bash
# Start the service
python app2.py
```
### 2. Testing the API

Once the server is active, open a **separate terminal window** to test the prediction endpoint. Use the following `curl` command to send one of the sample images provided in the repository:

```bash
# Send a sample fake image for prediction
curl -X POST -F "image=@data/samples/fake_1.jpg" [http://127.0.0.1:9696/predict](http://127.0.0.1:9696/predict)
```
**JSON Response:**

```json
{
  "confidence": "40.54%",
  "prediction": "Fake",
  "status": "success"
}
```
