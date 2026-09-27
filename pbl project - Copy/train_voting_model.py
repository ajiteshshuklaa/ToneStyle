import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib

print("🚀 Script Started...")

# ---------------- LOAD DATASET ----------------
try:
    df = pd.read_csv("lab_dataset.csv")   # ✅ UPDATED FILE NAME
    print("✅ Dataset loaded successfully")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# ---------------- DEBUG CHECK ----------------
print("\n📊 Dataset Preview:")
print(df.head())

print("\n📌 Columns in dataset:")
print(df.columns)

# ---------------- FEATURE SELECTION ----------------
required_columns = ["L", "a", "b", "ITA", "skin_tone"]

for col in required_columns:
    if col not in df.columns:
        print(f"❌ Missing column: {col}")
        exit()

X = df[["L", "a", "b", "ITA"]]
y = df["skin_tone"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n✅ Data split completed")

# ---------------- MODELS ----------------
lr = LogisticRegression(max_iter=200)
svm = SVC(probability=True)
rf = RandomForestClassifier(n_estimators=100)

# ---------------- VOTING CLASSIFIER ----------------
voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svm', svm),
        ('rf', rf)
    ],
    voting='soft'
)

# ---------------- TRAIN ----------------
print("\n🧠 Training model...")
voting.fit(X_train, y_train)

print("✅ Training completed")

# ---------------- EVALUATION ----------------
accuracy = voting.score(X_test, y_test)
print(f"\n🎯 Voting Classifier Accuracy: {accuracy:.2f}")

# ---------------- SAVE MODEL ----------------
joblib.dump(voting, "voting_model.pkl")

print("💾 Model saved as voting_model.pkl")