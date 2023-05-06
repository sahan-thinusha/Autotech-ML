from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('dataset.csv')

# Encode categorical variables
le_make = LabelEncoder()
data['car_make'] = le_make.fit_transform(data['car_make'])

le_model = LabelEncoder()
data['car_model'] = le_model.fit_transform(data['car_model'])

le_part = LabelEncoder()
data['part_name'] = le_part.fit_transform(data['part_name'])

# Train a random forest regressor
X = data.drop(['repair_time_in_hours'], axis=1)
y = data['repair_time_in_hours']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

@app.route('/', methods=['POST'])
def predict_repair_time():
    car_model = request.json['car_model']
    part = request.json['part']
    make = request.json['make']
    encoded_car_model = le_model.transform([car_model])[0]
    encoded_part = le_part.transform([part])[0]
    features = [le_make.transform([make])[0], encoded_car_model, encoded_part]
    estimated_repair_time = rf.predict([features])[0]
    return jsonify({'estimated_repair_time': estimated_repair_time})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
