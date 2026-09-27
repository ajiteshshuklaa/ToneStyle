ToneStyle
.AI-Based Skin Tone & Color Recommendation System
.ToneStyle is a real-time AI-powered computer vision application that detects a user’s skin tone and undertone using a webcam and provides personalized color recommendations based on color theory.

Project Overview
-Choosing the right colors based on skin tone is often challenging. ToneStyle solves this problem by combining Computer Vision, Deep Learning, and Machine Learning to automatically analyze facial skin and suggest suitable colors in real time.
.The system captures live video, detects the face, analyzes skin characteristics, and recommends colors that best complement the user’s complexion.

 Key Features
- Real-time webcam-based skin tone detection
- CNN-based classification using MobileNetV2
- Accurate face detection using SSD (OpenCV DNN)
- Undertone detection (Warm, Cool, Neutral) using LAB color space
- Multi-frame averaging for stable predictions
- Ensemble Voting Classifier for experimental ML-based classification
- Personalized color recommendations based on tone + undertone
-  Optimized for real-time performance (low latency, smooth output)
  
 Tech Stack
1.Languages:
-Python
2.Libraries & Frameworks:
-OpenCV, TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Joblib
3.Core Domains:
-Computer Vision, Deep Learning, Machine Learning

 How It Works (Pipeline)
Webcam Input → Captures real-time video using OpenCV
Face Detection → Uses SSD model to detect face region
Preprocessing → Resize, normalize, and prepare input image
Skin Tone Prediction → CNN (MobileNetV2) predicts tone (Light/Medium/Dark)
Color Analysis → Convert RGB → LAB color space
Undertone Detection → Based on LAB values (a, b) and ITA calculation
Multi-frame Smoothing → Averages predictions over frames for stability
Color Recommendation → Suggests suitable colors based on tone + undertone

 Project Structure
ToneStyle/
│
├── webcam_skin_detect.py
├── predict_skin.py
├── skin_tone_model.keras
├── tonestyle_color_dataset.csv
├── test1.jpg
└── README.md

 ow to Run
1. Clone Repository
git clone https://github.com/yourusername/ToneStyle.git
cd ToneStyle2. Install Dependencies
pip install opencv-python tensorflow numpy pandas scikit-learn joblib
3. Run the Application
python webcam_skin_detect.py

- Press Q to exit the webcam

 Dataset
Custom experimental dataset (CSV) created for:
Skin tone
Undertone
Color recommendation mapping

Used for:
Validation
Feature-based classification
Testing ML models

 Models Used
CNN (MobileNetV2) → Skin tone classification
SSD (Deep Learning) → Face detection
Voting Classifier (Scikit-learn) → Ensemble-based classification

Key Concepts Implemented
Computer Vision
Deep Learning (CNN)
Machine Learning (Ensemble Learning)
LAB Color Space
Feature Engineering (L, a, b, ITA)
Real-Time Processing 

Author
Aashi Singh
AI & Computer Vision Enthusiast
