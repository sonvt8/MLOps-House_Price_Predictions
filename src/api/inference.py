import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from .schemas import HousePredictionRequest, PredictionResponse


# Load model pipeline and feature names
MODEL_PATH = Path(__file__).parent.parent / "models" / "trained" / "model_pipeline.joblib"
FEATURE_NAMES_PATH = Path(__file__).parent.parent / "models" / "trained" / "feature_names.json"

try:
    model_pipeline = joblib.load(MODEL_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model pipeline or feature names: {str(e)}")


def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """Predict house price based on input features."""
    try:
        # Prepare input data as DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Calculate derived features that are expected by the model
        current_year = datetime.now().year
        input_data["house_age"] = current_year - input_data["year_built"]
        input_data["bed_bath_ratio"] = input_data["bedrooms"] / input_data["bathrooms"]
        input_data["price_per_sqft"] = 0  # This will be calculated after prediction

        # Calculate total_rooms if not provided
        if input_data["total_rooms"].isna().any():
            input_data["total_rooms"] = input_data["bedrooms"] + input_data["bathrooms"]

        # Make prediction using the pipeline
        predicted_price = model_pipeline.predict(input_data)[0]

        # Calculate price_per_sqft after prediction
        input_data["price_per_sqft"] = predicted_price / input_data["sqft"]

        # Convert numpy.float32 to Python float and round to 2 decimal places
        predicted_price = round(float(predicted_price), 2)

        # Confidence interval (10% range)
        confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]
        confidence_interval = [round(float(value), 2) for value in confidence_interval]

        # Get feature importance if available
        features_importance = {}
        try:
            if hasattr(model_pipeline.named_steps["model"], "feature_importances_"):
                importances = model_pipeline.named_steps["model"].feature_importances_
                features_importance = dict(zip(feature_names, importances.tolist()))
        except Exception:
            pass  # Feature importance not available

        return PredictionResponse(
            predicted_price=predicted_price,
            confidence_interval=confidence_interval,
            features_importance=features_importance,
            prediction_time=datetime.now().isoformat(),
        )

    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")


def batch_predict(requests: list[HousePredictionRequest]) -> list[float]:
    """Perform batch predictions."""
    try:
        # Prepare input data as DataFrame
        input_data = pd.DataFrame([req.dict() for req in requests])

        # Calculate derived features that are expected by the model
        current_year = datetime.now().year
        input_data["house_age"] = current_year - input_data["year_built"]
        input_data["bed_bath_ratio"] = input_data["bedrooms"] / input_data["bathrooms"]
        input_data["price_per_sqft"] = 0  # Dummy value for compatibility

        # Calculate total_rooms if not provided
        if input_data["total_rooms"].isna().any():
            input_data["total_rooms"] = input_data["bedrooms"] + input_data["bathrooms"]

        # Make predictions using the pipeline
        predictions = model_pipeline.predict(input_data)

        # Convert to list of floats and round to 2 decimal places
        return [round(float(pred), 2) for pred in predictions]

    except Exception as e:
        raise ValueError(f"Error during batch prediction: {str(e)}")
