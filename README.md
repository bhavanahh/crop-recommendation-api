# Crop Recommendation API 🌱

A Machine Learning project that predicts a suitable crop based on soil nutrients and environmental conditions using a Random Forest Classifier. The trained model is deployed as a REST API using FastAPI and Render.

## Live Demo

- **Live API:** https://crop-recommendation-api-97x8.onrender.com/
- **API Documentation:** https://crop-recommendation-api-97x8.onrender.com/docs
- **Health Check:** https://crop-recommendation-api-97x8.onrender.com/health

## Features

- Predicts suitable crops from soil and weather data.
- Uses a Random Forest Classification model.
- Accepts input through a REST API.
- Validates input using Pydantic.
- Provides interactive API documentation through Swagger UI.
- Deployed online using Render.

## Dataset

The dataset contains the following input features:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

**Target:** Crop label

## Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Test Accuracy:** 99.55%
- **Train-Test Split:** 80:20
- **Random State:** 42
- **Number of Estimators:** 100

The accuracy represents the model's performance on the test split of the dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pydantic
- Render

## Project Structure

```text
crop-recommendation-api/
├── data/
│   └── Crop_recommendation.csv
├── model/
│   └── crop_model.pkl
├── train.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/bhavanahh/crop-recommendation-api.git
cd crop-recommendation-api
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train the model (optional)

```bash
python train.py
```

This creates the trained model in the `model/` directory.

### 4. Start the API

```bash
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to access the interactive API documentation.

## API Usage

### Endpoint

`POST /predict`

### Example Request

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

### Example Response

```json
{
  "predicted_crop": "rice",
  "input": {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 202.9
  }
}
```

The predicted crop depends on the input values.

## Deployment

The FastAPI application is deployed on Render and uses the saved machine learning model to generate predictions.

**Note:** The free hosting instance may sleep after inactivity, so the first request can take longer to respond.
