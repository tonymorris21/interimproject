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
train = Blueprint('train', __name__)


@train.route('/train', methods=['POST','GET'])
def train_data():
    print('args:', request.args)
    print('form:', request.form)
    target = request.args.get("target")
    print(target)
    filelocation = session['filelocation']
    train_data = pd.read_csv(filelocation)
    train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
    X=train_data[["Pclass", "Age", "SibSp", "Parch", "Fare"]]
    y=train_data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
    train_data["Fare"].fillna(train_data["Fare"].mean(), inplace=True)

    rfc=RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc.score(X_test,y_test)
    rfc=RandomForestClassifier(random_state=35)
    rfc.fit(X_train, y_train)

    print("test accuracy: ",rfc.score(X_test,y_test))
    #print(df.info())

    return render_template('train.html')