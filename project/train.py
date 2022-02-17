import os
from flask import Blueprint,render_template, redirect, url_for, request, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from . import db
from .models import File,Model
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from flask import current_app
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle
from io import BytesIO
import base64
train = Blueprint('train', __name__)


@train.route('/train', methods=['POST','GET'])
def train_data():
    print('args:', request.args)
    print('form:', request.form)
    target = request.args.get("target")
    print(target)
    fileid = session['fileid']
    print(request.args.getlist("nullable"))
    nullable = request.args.getlist("nullable")
    mean = request.args.getlist("mean")
    mode = request.args.getlist("mode")
  #  print(nullable)
  #  print(mean)
   # print(mode)
    args = request.args
    filtered_dict = dict(filter(lambda item: "Col" in item[0], args.items()))
    if filtered_dict :
        print("dict",filtered_dict)
        filelocation = session['filelocation']
        df = pd.read_csv(filelocation)
        df.rename(columns=filtered_dict,inplace=True)
        print(df.columns)
    else:
        filtered_dict = dict(filter(lambda item: "Col" in item[0], args.items()))
        print("dict",filtered_dict)
        filelocation = session['filelocation']
        df = pd.read_csv(filelocation)
        df.rename(columns=filtered_dict,inplace=True)
        print(df.columns)

    print("dict",filtered_dict)
    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    df.rename(columns=filtered_dict,inplace=True)
    print(df.columns)
    df.to_csv (filelocation, index = None, header=True)
    file = File.query.filter_by(fileid=fileid).first()
    file.location = filelocation
    db.session.commit()
    train_data = pd.read_csv(filelocation)
    #data = session['df']
    #train_data = pd.DataFrame(data)
    #train_data.rename(columns=filtered_dict,inplace=True)
   # dict_obj = train_data.to_dict('list')
    #session['df'] = dict_obj
    print(train_data)
   # print(train_data["Age"])

    #train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)

    for x in mean:
        print(x)
        train_data[x].fillna(train_data[x].mean(), inplace=True)
   # print(train_data["Age"])
   
    for z in mode:
        print(z)
        train_data[z].fillna(train_data[z].mode()[0], inplace=True)
    #print(train_data["Age"])
    #print(train_data.columns)
    #print(train_data)
    
    train_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

   # train_data.replace({'Col4':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

    #print(train_data)
    features = train_data.columns
    nullable.append(target)
    X = train_data.drop(columns = (nullable),axis=1)
    print(X)
    y=train_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
    #train_data["Fare"].fillna(train_data["Fare"].mean(), inplace=True)

   # rfc=RandomForestClassifier()
   # rfc.fit(X_train, y_train)
   # rfc.score(X_test,y_test)
   # rfc=RandomForestClassifier(random_state=35)
  #  rfc.fit(X_train, y_train)
    model = SVC()
    model.fit(X_train, y_train)
    X_train_prediction = model.predict(X_train)
   # training_data_accuracy = accuracy_score(y_train, X_train_prediction)
   # print('Accuracy score of training data : ', training_data_accuracy)

    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(y_test, X_test_prediction)
    print('Accuracy score of test data : ', test_data_accuracy)
    #print(f'Test Accuracy: {accuracy_score(y_test, X_test_prediction)}')
    print("test accuracy: ",model.score(X_test,y_test))
   # print('F1 Score: %.3f' % f1_score(y_test, X_test_prediction))
    #print(df.info())
#https://www.analyticsvidhya.com/blog/2021/07/titanic-survival-prediction-using-machine-learning/
  #  user = current_user.id
    filename = os.path.join(current_app.config['UPLOAD_FOLDER'], 'models/test.sav')
    print(os.path.join(current_app.config['UPLOAD_FOLDER'], '\models'))
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    
    #print(f1_score(y_test, X_test_prediction, average="macro"))
   # print(precision_score(y_test, X_test_prediction, average="macro"))
   # print(recall_score(y_test, X_test_prediction, average="macro"))
    accuracy = str(test_data_accuracy)
    cv = classification_report(y_test, X_test_prediction)
    print(cv)
    classreport = report_to_df(cv)
    matrix = plot_confusion_matrix(model, X_test, y_test,
                                 cmap=plt.cm.Blues)
    img = BytesIO()
    plt.title('Confusion matrix for our classifier')
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    projectid = session['projectid']
    classreportsql =str(classreport.to_json())
    
    tables=[classreport.to_html(index=False,classes='data')]
    print(tables)
    class_report = str(tables)
    model = Model(projectid=projectid, accuracy=accuracy,confusion_matrix=plot_url,class_report=class_report)

    db.session.add(model)
    db.session.commit()


    print(classreport.to_json())
    return render_template('train.html', plot_url=plot_url,accuracy=accuracy,tables=tables)

def report_to_df(report):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+[x for x in report[0] if x!='']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value!='']
        if row!=[]:
            values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    return df