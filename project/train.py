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


@train.route('/train')
def train_data():

    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    X= df[["Pclass","Age","SibSp","Parch","Fare"]]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)
    #test_data["Fare"].fillna(test_data["Fare"].mean(), inplace=True)
    rfc=RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc.score(X_test,y_test)
    rfc=RandomForestClassifier(random_state=35)
    rfc.fit(X_train, y_train)

    print("test accuracy: ",rfc.score(X_test,y_test))
    print(df.info())

    return render_template('train.html')