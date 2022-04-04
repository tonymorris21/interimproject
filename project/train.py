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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import uuid
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import json 
from datetime import date
from urllib import parse
import json
from flask import send_file
from bs4 import BeautifulSoup
import seaborn as sn
train = Blueprint('train', __name__)


@train.route('/train/<fileid>/target/<target>', methods=['POST','GET'])
def trainpage(fileid,target):
    return render_template('algorithmselect.html')

@train.route('/train/<fileid>/algorithm/<algorithm>/target/<target>/param/<param>', methods=['POST','GET'])
def train_data(fileid,algorithm,target,param):
    print("test train")
    
    print(request.get_data())
    
    parsed = parse.unquote(param)
    print(parsed)
    parsed= parsed.replace("{","")
    parsed= parsed.replace("}","")
    parsed= parsed.replace('"','')
    parsed = parsed.split(",")
    elem = {}
    file = File.query.filter_by(fileid=fileid).first()
    dfd = pd.read_csv(file.location)
    df = pd.read_feather(file.featherlocation)
    print("test",)

    X = df.drop(target, 1)
    Y = df.filter([target])
    classnames = dfd[target].unique()
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize
    print("test",classnames)
    #if key 'Normalisation' is present in elem, then use it as Normalisation
    if 'Normalisation' in elem:
        if elem['Normalisation'] == 'yes':
            X = preprocessing.normalize(X)


    if 'randomstate' in elem:
        randomstate = elem['randomstate']
    else:
        randomstate = None
    #if key 'split' is present in elem, then use it as split
    if 'split' in elem:
        split = elem['split']
    else:
        split = 0.25

    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=split,random_state=randomstate)
    if bool(elem):
        for i in parsed:
            first = i.split(":")
            elem[first[0]] = first[1]
        print(elem['Kernel'])
    # if key 'Kernel' is present in elem, then use it as kernel
    if 'Kernel' in elem:
        Kernel = elem['Kernel']
    else:
        Kernel = 'rbf'
    # if key 'C' is present in elem, then use it as C
    if 'k' in elem:
        k = elem['k']
    else:
        k = 5

    # if key 'var_smoothing' is present in elem, then use it as var_smoothing
    if 'var_smoothing' in elem:
        var_smoothing = elem['var_smoothing']
    else:
        var_smoothing = 1e-09


    #if key 'ntree' is present in elem, then use it as ntree
    if 'ntree' in elem:
        ntree = elem['ntree']
    else:
        ntree = 100
    #if key 'max_features' is present in elem, then use it as max_features
    if 'max_features' in elem:
        max_features = elem['max_features']
    else:
        max_features = 'auto'


    if algorithm == "RF" : modelid, plot_url, accuracy, tables = RFC(max_features,ntree,dfd[target],X_train,X_test,y_train,y_test,fileid,target,list(X.columns.values))
    elif algorithm == "NB" : modelid, plot_url, accuracy, tables = naivebayes(var_smoothing,dfd[target],X_train,X_test,y_train,y_test,fileid,target,list(X.columns.values))
    elif algorithm == "KNN" : modelid, plot_url, accuracy, tables = KNN(k,dfd[target],X_train,X_test,y_train,y_test,fileid,target,list(X.columns.values))
    elif algorithm == "SVC" : modelid, plot_url, accuracy, tables = svc(Kernel,dfd[target],X_train,X_test,y_train,y_test,fileid,target,list(X.columns.values))
#https://www.analyticsvidhya.com/blog/2021/07/titanic-survival-prediction-using-machine-learning/
    print("test")
    return render_template('modelinfo.html', modelid=modelid,confusion_matrix=plot_url,accuracy=accuracy,tables=tables,algorithm=algorithm,target=target)



def naivebayes(var_smoothing,targetdf,X_train,X_test,y_train,y_test,fileid,target,feature_names):
 

    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    gaussian.feature_names = feature_names
    Y_pred = gaussian.predict(X_test) 
    accuracy = accuracy_score(y_test,Y_pred)
    algorithm = "NB"
    modelid, plot_url, accuracy, tables = toDatabase(target,algorithm,targetdf.unique(),gaussian,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables

def KNN(k,targetdf,X_train,X_test,y_train,y_test,fileid,target,feature_names):
    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_feather(file.featherlocation)

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    knn.feature_names = feature_names
    Y_pred = knn.predict(X_test) 

    accuracy = accuracy_score(y_test,Y_pred)


    algorithm = "KNN"
    modelid, plot_url, accuracy, tables = toDatabase(target,algorithm,targetdf.unique(),knn,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables
def RFC (max_features,ntree,targetdf,X_train,X_test,y_train,y_test,fileid,target,feature_names):


    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf.feature_names = feature_names
    Y_pred = rf.predict(X_test) 

    accuracy = accuracy_score(y_test,Y_pred)

    algorithm = "RF"
    modelid, plot_url, accuracy, tables = toDatabase(target,algorithm,targetdf.unique(),rf,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables

def svc(Kernel,targetdf,X_train,X_test,y_train,y_test,fileid,target,feature_names):


    svc = SVC(kernel=Kernel)
    svc.fit(X_train, y_train)
    svc.feature_names = feature_names
    Y_pred = svc.predict(X_test) 

    accuracy = accuracy_score(y_test,Y_pred)
    algorithm = "SVC"
    modelid, plot_url, accuracy, tables = toDatabase(target,algorithm,targetdf.unique(),svc,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables

def toDatabase(target,algorithm,classnames,model,accuracy,y_test,X_test,Y_pred):

    cv = classification_report(y_test, Y_pred,target_names=classnames)

    classreport = report_to_df(cv)

    cm = confusion_matrix(y_test, Y_pred)

    ax= plt.subplot()
    sn.heatmap(cm, annot=True, fmt='g', ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classnames)
    ax.yaxis.set_ticklabels(np.flip(classnames))
    img = BytesIO()


    plt.title('Confusion matrix for our classifier')
    plt.savefig(img, format='png')
    plt.close()


    img.seek(0)


    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    projectid = session['projectid']



    createddate = date.today()


    tables=[classreport.to_html(index=False,classes='table table-sm')]



    class_report = str(tables)
    projectid = session['projectid']
    filename = os.path.join(current_app.config['UPLOAD_FOLDER'], 'models\\', str(uuid.uuid4())+'_model.sav')

    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    model = Model(projectid=projectid, location = filename,target=target,algorithm=algorithm,classnames=classnames,createddate=createddate,accuracy=accuracy,confusion_matrix=plot_url,class_report=class_report)

    db.session.add(model)
    db.session.commit()

    modelid = model.modelid


    return modelid,plot_url, accuracy, tables

def classnamemap(le,inverse):
    classnames = dict(zip(le, inverse))
    return classnames

@train.route('/modelEvaluation/<modelid>', methods=['POST','GET'])
def modelinfo(modelid):
    model = Model.query.filter_by(modelid=modelid).first()
    confusion_matrix = model.confusion_matrix
    accuracy = model.accuracy
    classreport = model.class_report
    classreport = str(classreport).replace('[', '')
    classreport = str(classreport).replace(']', '')
    classreport = str(classreport).replace("'", '')
    classreport = str(classreport).replace("\\n", '')
    classreport.strip('/n')
    soup = BeautifulSoup(classreport)
    soup.get_text(strip=True)
    return render_template('modelinfo.html',modelid=model.modelid,target = model.target,algorithm = model.algorithm,accuracy=accuracy,confusion_matrix=confusion_matrix,tables=soup)



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

@train.route('/train/<modelid>/downloadModel', methods=['POST','GET'])
def downloadModel(modelid):
    model = Model.query.filter_by(modelid=modelid).first()
    
    return send_file(model.location,
                     mimetype='application/octet-stream',
                     attachment_filename='data.pickle',
                     as_attachment=True)