import pickle
import json
from models import Model
from flask import Blueprint,render_template
import numpy as np

predict = Blueprint('predict', __name__)


@predict.route('/predict/<modelid>')
def prediction(modelid):
    modelfile = Model.query.filter_by(modelid=modelid).first()
    classnames1 = modelfile.classnames
    feature_names = modelfile.feature_names
    modelfile = open(modelfile.location, 'rb')
    feature_names = json.loads(feature_names)
    json.loads(classnames1)
    return render_template('predict.html',columnnames=feature_names,classnames=classnames1,modelid=modelid)

@predict.route('/predict/<modelid>/values/<predict_values>', methods=['GET'])
def predict_values(modelid,predict_values):
     modelfile = Model.query.filter_by(modelid=modelid).first()
     classnames1 = modelfile.classnames
     classnames1 = json.loads(classnames1)
     classnames1 = list(classnames1)
     modelfile = open(modelfile.location, 'rb')
     predictv =predict_values.split(",")
     predictv = np.array(predictv)
     data = pickle.load(modelfile)
     dtest = data.predict([predictv])
     
     return json.dumps(classnames1[int(dtest)])
