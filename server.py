from flask import Flask, request, render_template
from prediction import predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/result', methods=['POST'])
def submit():
    #prepare input
    requests = request.form.to_dict()

    Q1 = requests['Q1']

    Q2 = requests['Q2']

    prediction = predict(Q1, Q2)

    if prediction:
        pred = " Two Questions match each other "
    else:
        pred = " Two Questions are different "

    return render_template("result.html", Question1 = Q1, Question2 = Q2, output = pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0' ,debug=1 ,port=4000)
