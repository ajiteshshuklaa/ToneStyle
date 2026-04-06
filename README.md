ToneStyle  
AI-Based Skin Tone & Color Recommendation System  
ToneStyle is a real-time AI-powered computer vision application that detects a user’s skin tone and undertone using a webcam and provides personalized color recommendations based on color theory.



 Project Overview:
 Choosing the right colors based on skin tone is often challenging. ToneStyle solves this problem by combining Computer Vision, Deep Learning, and Machine Learning to automatically analyze facial skin and suggest suitable colors in real time.

The system captures live video, detects the face, analyzes skin characteristics, and recommends colors that best complement the user’s complexion.

Key Features
- Real-time webcam-based skin tone detection
- CNN-based classification using MobileNetV2
- Accurate face detection using SSD (OpenCV DNN)
- Undertone detection (Warm, Cool, Neutral) using LAB color space
- Multi-frame averaging for stable predictions
- Ensemble Voting Classifier for experimental ML-based classification
- Personalized color recommendations based on tone + undertone
- Optimized for real-time performance (low latency, smooth output)
- The system analyzes facial skin region and suggests colors that best complement the detected skin tone.

  Technologies Used
  Languages:Python

Libraries & Frameworks:OpenCV, TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Joblib

Core Domains:Computer Vision, Deep Learning, Machine Learning

How It Works (Pipeline):
1.Webcam Input → Captures real-time video using OpenCV
2.Face Detection → Uses SSD model to detect face region
3.Preprocessing → Resize, normalize, and prepare input image
4.Skin Tone Prediction → CNN (MobileNetV2) predicts tone (Light/Medium/Dark)
5.Color Analysis → Convert RGB → LAB color space
6.Undertone Detection → Based on LAB values (a, b) and ITA calculation
7.Multi-frame Smoothing → Averages predictions over frames for stability
8.Color Recommendation → Suggests suitable colors based on tone + undertone


000Project Structure
ToneStyle/
│
├── webcam_skin_detect.py
├── predict_skin.py
├── skin_tone_model.keras
├── tonestyle_color_dataset.csv
├── test1.jpg
└── README.md
How to Run:
1.Clone Repository:
git clone https://github.com/yourusername/ToneStyle.git
cd ToneStyle.
2.Install Dependencies:
pip install opencv-python tensorflow numpy pandas scikit-learn joblib
3.Run the Application:
python webcam_skin_detect.py

Dataset
1.Custom experimental dataset (CSV) created for:
-Skin tone
-Undertone
-Color recommendation mapping
2.Used for:
-Validation
-Feature-based classification
-Testing ML models

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
Ajitesh Shukla
AI & Computer Vision Enthusiast





👨‍💻 Author
Ajitesh Shukla
AI & Computer Vision Enthusiast
