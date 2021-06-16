from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, request, render_template
import joblib as jb

app = Flask(__name__)
model = jb.load('stroke.joblib')

dataset_X = [[1.0, 0, 2.0, 1.0, 0, 25.0, 0, 0, 79.20, 38.5]]
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('stroke.html')


@app.route('/predictstroke', methods=['POST'])
def predictstroke():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have High posibilities to get a stroke!"
    elif prediction == 0:
        pred = "You don't have High posibilities to get a stroke."
    output = pred
    return render_template('stroke.html', predicted='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
