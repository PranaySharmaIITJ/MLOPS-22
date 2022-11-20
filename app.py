from flask import Flask,request
from joblib import load

app = Flask(__name__)

@app.route("/")
def hello_world():
	return "<p>Hello, World!</p>"

model_path = 'best_svm_model.joblib'

@app.route("/predict",methods=['POST'])
def predict_digit():
	image_1 = request.json['image_1']
	image_2 = request.json['image_2']
	model = load(model_path)
	print('done......loading')
	predicted_1 = model.predict([image_1])
	predicted_2 = model.predict([image_2])
	if int(predicted_1[0]) == int(predicted_2[0]):
		return 'The Images belong to same digit'
	else:
		return 'The Images belong to different digits'

if __name__=="__main__":
	app.run(debug=True)