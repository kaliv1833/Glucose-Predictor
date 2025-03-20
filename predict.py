#this is an example of importing and using model
from joblib import load

model = load("Glucose_Predictor.joblib")

#data should be by order: pregnancies, BloodPressure, insullin receive, bmi, age, outcome(0 if has not diabet, 1 if has diabet)
new_data = [[0,100,0,25,34,0]]

predict = model.predict(new_data)

print(predict)
