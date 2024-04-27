import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.utils import save_object
from src.utils import evaluate_models
import os
import logging

class PredictPipeline:
    def __init__(self):
        pass
    
    def fit_preprocessor(self, X_train):
        try:
            preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')
            preprocessor = load_object(file_path=preprocessor_path)
            preprocessor.fit(X_train)  # Fit the preprocessor to training data
            save_object(preprocessor_path, preprocessor)  # Save the fitted preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
   # This function will predict the target variable
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # print the recived features
            print("Recived Features : {features}")

            # Transform the input features
            data_scaled = preprocessor.transform(features)

            print("data_scaled/pred_df : ",data_scaled)
            logging.info("data_scaled/pred_df : {data_scaled}")

            # Predict using the model
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        abtest: str,
        vehicleType: str,
        gearbox: str,
        powerPS: int,
        brand: str,
        model: str,
        kilometer: int,
        fuelType: str,
        notRepairedDamage: str,
        yearOfRegistration: int):
        
        self.abtest = abtest
        self.vehicleType = vehicleType
        self.gearbox = gearbox
        self.powerPS = powerPS
        self.brand = brand
        self.model = model
        self.kilometer = kilometer
        self.fuelType = fuelType
        self.notRepairedDamage = notRepairedDamage
        self.yearOfRegistration = yearOfRegistration

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "abtest": [self.abtest],
                "vehicleType": [self.vehicleType],
                "gearbox": [self.gearbox],
                "powerPS": [self.powerPS],
                "brand": [self.brand],
                "model": [self.model],
                "kilometer": [self.kilometer],
                "fuelType": [self.fuelType],
                "notRepairedDamage": [self.notRepairedDamage],
                "yearOfRegistration": [self.yearOfRegistration],
            }

            # Create a DataFrame with the same columns as used during training
            columns = ['abtest', 'vehicleType', 'gearbox', 'powerPS', 'brand', 'model', 'kilometer', 'fuelType', 'notRepairedDamage', 'yearOfRegistration']
            df = pd.DataFrame(custom_data_input_dict, columns=columns)

            return df

        except Exception as e:
            raise CustomException(e, sys)