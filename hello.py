from logging import debug
from flask import Flask, request
# from src.utils import load_model
import numpy as np
import pickle

app = Flask(__name__)

best_model_path = '/home/root@DESKTOP/codes/ML-Ops_Scikit/models/best_clf_0.6715.pkl'

def load_model(path):
    print("\nloading the model...")
    load_file = open(path, "rb")
    loaded_model = pickle.load(load_file)
    return loaded_model


@app.route("/")
def hello_world():
    print("Server started!!")
    return "<p>Hello, World!</p>"


# curl http://localhost:5000/predict -X POST  -H 'Content-Type: application/json' -d '{"image": ["1.0", "2.0", "3.0"]}'

clf = load_model(best_model_path)
print("***model loaded***\n")

@app.route("/predict", methods=['POST'])
def predict():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)

    prediction = clf.predict(image)
    print("image:\n", image)
    print("prediction:", prediction[0], end="\n\n")

    return str(prediction[0])

    

if __name__ == '__main__':
    app.run(debug=True)

