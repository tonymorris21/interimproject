import os
from flask import Blueprint,render_template, redirect, url_for, request, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from . import db
from .models import File
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from flask import current_app
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
train = Blueprint('train', __name__)


@train.route('/train', methods=['POST','GET'])
def train_data():
    print('args:', request.args)
    print('form:', request.form)
    target = request.args.get("target")
    print(target)
    
    print(request.args.getlist("nullable"))
    nullable = request.args.getlist("nullable")
    mean = request.args.getlist("mean")
    mode = request.args.getlist("mode")
    print(nullable)
    print(mean)
    print(mode)

    filelocation = session['filelocation']
    train_data = pd.read_csv(filelocation)
    print(train_data["Age"])
    #train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
    for x in mean:
        print(x)
        train_data[x].fillna(train_data[x].mean(), inplace=True)
    print(train_data["Age"])
    
    for z in mode:
        print(z)
        train_data[z].fillna(train_data[z].mode()[0], inplace=True)
    #print(train_data["Age"])
    print(train_data.columns)
    #print(train_data)
    train_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
    print(train_data)
    features = train_data.columns
    nullable.append(target)
    X = train_data.drop(columns = (nullable),axis=1)
    print(X)
    y=train_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
    #train_data["Fare"].fillna(train_data["Fare"].mean(), inplace=True)

   # rfc=RandomForestClassifier()
   # rfc.fit(X_train, y_train)
   # rfc.score(X_test,y_test)
   # rfc=RandomForestClassifier(random_state=35)
  #  rfc.fit(X_train, y_train)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(y_train, X_train_prediction)
    print('Accuracy score of training data : ', training_data_accuracy)
    #print("test accuracy: ",rfc.score(X_test,y_test))
    #print(df.info())
#https://www.analyticsvidhya.com/blog/2021/07/titanic-survival-prediction-using-machine-learning/
    return render_template('train.html')