"""
Tomato Model Loader - Heroku-Ready
"""

from pathlib import Path
import pickle
import numpy as np
import warnings
import pandas as pd
import traceback

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class TomatoModelLoader:
    """Loads Random Forest model and encoders from pickle files."""

    def __init__(self):
        self.model = None
        self.categorical_encoders = None
        self.target_encoder = None
        self.metadata = None
        self.is_trained = False

        # Absolute Heroku path for models
        self.models_dir = Path('/app/models')
        self.model_path = self.models_dir / 'rf_model.pkl'
        self.cat_encoders_path = self.models_dir / 'categorical_encoders.pkl'
        self.target_encoder_path = self.models_dir / 'target_encoder.pkl'
        self.metadata_path = self.models_dir / 'metadata.pkl'

        self.load_model()

    def load_model(self):
        """Load trained model and encoders from pickle files."""
        try:
            required_files = [
                self.model_path,
                self.cat_encoders_path,
                self.target_encoder_path,
                self.metadata_path,
            ]

            print("Checking model files:")
            for f in required_files:
                print(f"{f} exists? {f.exists()}")

            if not all(f.exists() for f in required_files):
                print("Model files not found. Make sure they are in /app/models")
                self.is_trained = False
                return False

            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            # Load categorical encoders
            with open(self.cat_encoders_path, 'rb') as f:
                self.categorical_encoders = pickle.load(f)
            # Load target encoder
            with open(self.target_encoder_path, 'rb') as f:
                self.target_encoder = pickle.load(f)
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

            self.is_trained = True
            print("Model loaded successfully!")
            print(f"Accuracy: {self.metadata.get('accuracy', 0):.3f}")
            print(f"Training date: {self.metadata.get('training_date', 'Unknown')}")
            return True

        except Exception as e:
            print("Error loading model:", traceback.format_exc())
            self.is_trained = False
            return False

    def reload_model(self):
        """Reload the model from files."""
        self.is_trained = False
        self.model = None
        self.categorical_encoders = None
        self.target_encoder = None
        self.metadata = None
        return self.load_model()

    def encode_features(
        self,
        rainfall_mm,
        temperature_c,
        market_day,
        school_open,
        disease_alert,
        last_week_demand,
        week,
        month,
    ):
        """Encode input features for prediction."""
        if not self.is_trained:
            raise ValueError("Model not loaded.")

        try:
            last_week_encoded = self.categorical_encoders['Last_Week_Demand'].transform([last_week_demand])[0]
            month_encoded = self.categorical_encoders['Month'].transform([month])[0]

            features = pd.DataFrame([[
                rainfall_mm,
                temperature_c,
                1 if market_day else 0,
                1 if school_open else 0,
                1 if disease_alert == 'Presence' else 0,
                last_week_encoded,
                month_encoded,
                2024  # default year
            ]], columns=[
                'Rainfall_mm', 'Temperature_C', 'Market_Day', 'School_Open',
                'Disease_Alert', 'Last_Week_Demand', 'Month', 'Year'
            ])

            return features

        except Exception as e:
            raise ValueError(f"Feature encoding failed: {traceback.format_exc()}")

    def predict(
        self,
        rainfall_mm=75.0,
        temperature_c=22.0,
        market_day=True,
        school_open=True,
        disease_alert='Absence',
        last_week_demand='Medium',
        week=1,
        month='January',
    ):
        """Make demand prediction using the trained model."""
        if not self.is_trained:
            raise ValueError("Model not loaded.")

        try:
            features = self.encode_features(
                rainfall_mm, temperature_c, market_day, school_open,
                disease_alert, last_week_demand, week, month
            )

            pred_encoded = self.model.predict(features)[0]
            pred_proba = self.model.predict_proba(features)[0]
            confidence = np.max(pred_proba)
            prediction = self.target_encoder.inverse_transform([pred_encoded])[0]

            return prediction, confidence

        except Exception as e:
            raise ValueError(f"Prediction failed: {traceback.format_exc()}")

    def get_model_info(self):
        """Get information about the loaded model."""
        if not self.is_trained:
            return {'is_trained': False, 'message': 'Model not loaded.'}

        return {
            'is_trained': True,
            'model_type': 'Random Forest Classifier',
            'accuracy': self.metadata.get('accuracy', 0),
            'training_date': self.metadata.get('training_date', 'Unknown'),
            'features': self.metadata.get('features', []),
            'target': self.metadata.get('target', 'Market_Demand'),
        }


# Global model instance
predictor = TomatoModelLoader()

