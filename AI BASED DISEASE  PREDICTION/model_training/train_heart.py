import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# ---------------- BASE PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")


# ---------------- LOAD DATA ----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)


# ---------------- FIND TARGET COLUMN ----------------
possible_targets = ["target", "output", "num", "HeartDisease"]

target_col = None
for col in possible_targets:
    if col in data.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError(
        f"No valid target column found. Available columns: {list(data.columns)}"
    )


# ---------------- HANDLE CATEGORICAL DATA ----------------
label_encoder = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = label_encoder.fit_transform(data[col])


# ---------------- SPLIT FEATURES & LABEL ----------------
X = data.drop(columns=[target_col])
y = data[target_col]


# ---------------- TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------- TRAIN MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ---------------- SAVE MODEL ----------------
os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)


print("✅ Heart Disease Model Trained Successfully")
print(f"📁 Model saved at: {MODEL_PATH}")
