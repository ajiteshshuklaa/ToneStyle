🎨 ToneStyle  
AI-Based Skin Tone & Color Recommendation System  

ToneStyle is an AI-powered computer vision application that detects a user's skin tone using a webcam and recommends suitable colors based on undertone analysis.


📌 Project Overview

ToneStyle uses:
- 📷 Real-time webcam capture (OpenCV)
- 🧠 Deep Learning model (.keras)
- 📊 CSV-based color dataset
- 🎯 Skin tone & undertone classification
- 🎨 Personalized color recommendations

The system analyzes facial skin region and suggests colors that best complement the detected skin tone.

 🛠️ Technologies Used

- Python 3.x
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas
- CSV Dataset


 📂 Project Structure
ToneStyle/
│
├── webcam_skin_detect.py
├── skin_tone_model.keras
├── tonestyle_color_dataset.csv
├── test1.jpg
├── predict_skin.py
└── README.md

⚙️ How It Works

1. Captures real-time video from webcam
2. Detects face region
3. Extracts skin area
4. Predicts:
   - Skin Tone
   - Undertone
5. Displays recommended colors on screen


▶️ How To Run

1️⃣ Clone the Repository

```bash
git clone https://github.com/ajiteshshuklaa/ToneStyle.git
cd ToneStyle

2️⃣ Install Dependencies
pip install opencv-python tensorflow numpy pandas

3️⃣ Run the Application
python webcam_skin_detect.py

🎯 Features
✅ Real-time skin tone detection
✅ Undertone classification
✅ Personalized color recommendations
✅ FPS display
✅ Clean OpenCV interface

💡 Future Improvements
Add GUI interface
Deploy as web application
Improve model accuracy
Add fashion recommendations
Mobile app version

👨‍💻 Author
Ajitesh Shukla
AI & Computer Vision Enthusiast
