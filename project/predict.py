import pickle
import json
from models import Model
from flask import Blueprint,render_template
import numpy as np

predict = Blueprint('predict', __name__)


@predict.route('/predict/<modelid>')
def prediction(modelid):
# open a file, where you stored the pickled data
    modelfile = Model.query.filter_by(modelid=modelid).first()
    print("Modelfile location",modelfile.location)
    classnames1 = modelfile.classnames
    
    modelfile = open(modelfile.location, 'rb')
    
    model = pickle.load(modelfile)
    print(model)
    columnnames = model.feature_names_in_
    print(columnnames[0])
  
    

    return render_template('predict.html',columnnames=columnnames,classnames=classnames1,modelid=modelid)

@predict.route('/predict/<modelid>/values/<predict_values>', methods=['GET','POST'])
def predict_values(modelid,predict_values):
     modelfile = Model.query.filter_by(modelid=modelid).first()
     #print("Modelfile location",modelfile.location)
     classnames1 = modelfile.classnames
     modelfile = open(modelfile.location, 'rb')
     test =predict_values.split(",")
     test = np.array(test)

     #print("Predict values",test)
     data = pickle.load(modelfile)

     dtest = data.predict([test])
     print(json.loads(classnames1))
     print("prediction",dtest)
     
     return json.dumps(dtest.tolist())
