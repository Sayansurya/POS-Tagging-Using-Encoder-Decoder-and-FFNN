from flask import Flask, request, jsonify, redirect, url_for, render_template, session
from flask_session import Session
import json
from torch_utils import ffnn_prediction, enc_ffnn_predictions, enc_dec_predictions

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.form['comment']:
            session['comment'] = request.form['comment']
            # print(session.get('comment'))
            return redirect(url_for('predict'))
    else:
        return render_template('home.html')


@app.route('/predict')
def predict():
    sentence = session.get('comment')
    results_ffnn = ffnn_prediction(sentence)
    results_enc = enc_ffnn_predictions(sentence)
    results_enc_dec = enc_dec_predictions(sentence)
    return render_template('output.html', comment = sentence, results_ffnn = results_ffnn, results_enc = results_enc, results_enc_dec = results_enc_dec)


if(__name__ == '__main__'):
    app.run(debug=True)