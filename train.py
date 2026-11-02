
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "crop_model.pkl"

# Create model directory
MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
data = pd.read_csv(DATA_PATH)

# Check for missing values
print("Missing values:")
print(data.isnull().sum())

# Remove duplicate rows
data = data.drop_duplicates()

# Separate features and target
X = data[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]
]

y = data["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and expected feature order
joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
    },
    MODEL_PATH,
)

print(f"\nModel saved to: {MODEL_PATH}")
