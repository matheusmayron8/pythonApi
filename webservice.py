from flask import Flask, jsonify, request
from sklearn.externals import joblib
import peewee
import pandas as pd

#instancia do Flask
app = Flask(__name__)

#POST /api/face
@app.route('/api/face', methods=['POST'])
def check_face():
    dados = request.json
    base = pd.DataFrame.from_dict(dados, orient='index')
    clf = joblib.load('./model.pkl')
    return jsonify({'status':200, 'mensagem':'1'})

if __name__ == '__main__':
    app.run(debug=True)