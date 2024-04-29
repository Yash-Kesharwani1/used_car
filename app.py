import os
from flask import Flask, request, render_template
import math
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData
# from src.components.data_ingestion import DataIngestions
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import load_object
import logging
import numpy as np
from src.exception import CustomException

application = Flask(__name__)
app = application

PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

# Route for a home page
@app.route("/")
def index():
    return render_template('index.html')

# if __name__ == "__main__":
#     # Load the trained model and data transformation objects
#     data_transformation = DataTransformation()
#     model_trainer = ModelTrainer()

#     # Initiate the Flask app
#     app.run(debug=True)




@app.route('/predict_datapoint', methods=['GET','POST'])
def predict_datapoint():
    # try:
    #     # Receive features from request
    #     features = request.get_json()

    #     # Load preprocessor object
    #     preprocessor_path = data_transformation.load_preprocessor()
    #     preprocessor = load_object(file_path=preprocessor_path)

    #     # Transform the input features
    #     features = np.array(features).reshape(1, -1)
    #     data_scaled = preprocessor.transform(features)

    #     logging.info("Columns of pred_df : ", pred_df.columns)
    #     logging.info("Recived Features : {features}")
    #     logging.info("data_scaled/pred_df : ", data_scaled)

    #     # Load the trained model
    #     model_path = model_trainer.load_model()
    #     model = load_object(file_path=model_path)

    #     # Predict using the model
    #     preds = model.predict(data_scaled)
    #     return jsonify({'predicted_price': preds[0]})

    # except CustomException as e:
    #     return jsonify({'error': str(e)}), 500
    
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            abtest = request.form.get('abtest'),
            vehicleType= request.form.get('vehicleType'),
            gearbox= request.form.get('gearbox'),
            powerPS= int(request.form.get('powerPS')),
            brand= request.form.get('brand'),
            model= request.form.get('model'),
            kilometer= int(request.form.get('kilometer')),
            fuelType= request.form.get('fuelType'),
            notRepairedDamage= request.form.get('notRepairedDamage'),
            yearOfRegistration= int(request.form.get('yearOfRegistration'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results = math.floor(results[0]))


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)