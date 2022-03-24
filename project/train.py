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
train = Blueprint('train', __name__)


@train.route('/train/<fileid>/target/<target>', methods=['POST','GET'])
def trainpage(fileid,target):
    print("test train page")
    return render_template('algorithmselect.html')

@train.route('/train/<fileid>/algorithm/<algorithm>/target/<target>', methods=['POST','GET'])
def train_data(fileid,algorithm,target):
    print("test train")
    modelid, plot_url, accuracy, tables = svc(fileid,target)
    

    #target = request.args.get("target")
    #print(target)
    #fileid = session['fileid']
    #print(request.args.getlist("nullable"))
    #nullable = request.args.getlist("nullable")
    #mean = request.args.getlist("mean")
    #mode = request.args.getlist("mode")
  #  print(nullable)
  #  print(mean)
   # print(mode)
    #args = request.args
    #filtered_dict = dict(filter(lambda item: "Col" in item[0], args.items()))
    #if filtered_dict :
       # print("dict",filtered_dict)
       # filelocation = session['filelocation']
       # df = pd.read_csv(filelocation)
      #  df.rename(columns=filtered_dict,inplace=True)
     #   print(df.columns)
    #else:
      #  filtered_dict = dict(filter(lambda item: "Col" in item[0], args.items()))
      #  print("dict",filtered_dict)
      #  filelocation = session['filelocation']
     #   df = pd.read_csv(filelocation)
    #    df.rename(columns=filtered_dict,inplace=True)
   #     print(df.columns)

   # print("dict",filtered_dict)
   # filelocation = session['filelocation']
    #df = pd.read_csv(filelocation)
   # df.rename(columns=filtered_dict,inplace=True)
   # print(df.columns)
   # df.to_csv (filelocation, index = None, header=True)
   # file = File.query.filter_by(fileid=fileid).first()
   # file.location = filelocation
   # db.session.commit()
    #train_data = pd.read_csv(filelocation)
    #data = session['df']
    #train_data = pd.DataFrame(data)
    #train_data.rename(columns=filtered_dict,inplace=True)
   # dict_obj = train_data.to_dict('list')
    #session['df'] = dict_obj
   # print(train_data)
   # print(train_data["Age"])

    #train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)

   # for x in mean:
    #    print(x)
    #    train_data[x].fillna(train_data[x].mean(), inplace=True)
   # print(train_data["Age"])
   
   # for z in mode:
    #    print(z)
    #    train_data[z].fillna(train_data[z].mode()[0], inplace=True)
    #print(train_data["Age"])
    #print(train_data.columns)
    #print(train_data)
    
    #train_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

   # train_data.replace({'Col4':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

    #print(train_data)


    #train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
    #train_data["Fare"].fillna(train_data["Fare"].mean(), inplace=True)

#https://www.analyticsvidhya.com/blog/2021/07/titanic-survival-prediction-using-machine-learning/
    print("test")
    return render_template('train.html', modelid=modelid,plot_url=plot_url,accuracy=accuracy,tables=tables)

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

def naivebayes(fileid,target):
    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_csv(file.location)
    X = df.drop(target, 1)
    
    Y = df.filter([target])
    le = LabelEncoder()
    y = le.fit_transform(Y)
    
    inverse = le.inverse_transform(y)
    columnnames = X.columns.values
    d  = classnamemap(y,inverse)
    classnames1 = {str(k):str(v) for k,v in d.items()}
    classnames = json.dumps(classnames1, indent = 4) 
    print(X.columns.values)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    gaussian.feature_names = list(X.columns.values)
    Y_pred = gaussian.predict(X_test) 
    accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for Naive Bayes\n',cm)
    print('accuracy_Naive Bayes: %.3f' %accuracy)
    print('precision_Naive Bayes: %.3f' %precision)
    print('recall_Naive Bayes: %.3f' %recall)
    print('f1-score_Naive Bayes : %.3f' %f1)
    modelid, plot_url, accuracy, tables = toDatabase(classnames,gaussian,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables

def KNN(fileid,target):
    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_csv(file.location)
    X = df.drop(target, 1)
    Y = df.filter([target])
    le = LabelEncoder()
    y = le.fit_transform(Y)
    print(Y)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test) 
    accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_knn = round(knn.score(X_train, y_train) * 100, 2)
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for Naive Bayes\n',cm)
    print('accuracy_Naive KNNs: %.3f' %accuracy)
    print('precision_Naive Bayes: %.3f' %precision)
    print('recall_Naive Bayes: %.3f' %recall)
    print('f1-score_Naive Bayes : %.3f' %f1)
    modelid, plot_url, accuracy, tables = toDatabase(knn,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables

def svc(fileid,target):

    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_csv(file.location)
    X = df.drop(target, 1)
    Y = df.filter([target])
    le = LabelEncoder()
    y = le.fit_transform(Y)
    inverse = le.inverse_transform(y)
    d  = classnamemap(y,inverse)
    classnames1 = {str(k):str(v) for k,v in d.items()}
    classnames = json.dumps(classnames1, indent = 4) 
    
    print(Y)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    svc = SVC()
    svc.fit(X_train, y_train)
    svc.feature_names = list(X.columns.values)
    Y_pred = svc.predict(X_test) 
    accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_svc = round(svc.score(X_train, y_train) * 100, 2)
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for Naive Bayes\n',cm)
    print('accuracy_Naive SVC: %.3f' %accuracy)
    print('precision_Naive Bayes: %.3f' %precision)
    print('recall_Naive Bayes: %.3f' %recall)
    print('f1-score_Naive Bayes : %.3f' %f1)
    modelid, plot_url, accuracy, tables = toDatabase(classnames,svc,accuracy,y_test,X_test,Y_pred)
    return modelid, plot_url, accuracy, tables

def toDatabase(classnames,model,accuracy,y_test,X_test,Y_pred):


    cv = classification_report(y_test, Y_pred)
    print(cv)
    classreport = report_to_df(cv)
    matrix = plot_confusion_matrix(model, X_test, Y_pred,
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
    projectid = session['projectid']
    filename = os.path.join(current_app.config['UPLOAD_FOLDER'], 'models\\', str(uuid.uuid4())+'_model.sav')
    print(os.path.join(current_app.config['UPLOAD_FOLDER'], '\models'))
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    model = Model(projectid=projectid, location = filename,classnames=classnames,accuracy=accuracy,confusion_matrix=plot_url,class_report=class_report)

    db.session.add(model)
    db.session.commit()

    modelid = model.modelid

    return modelid,plot_url, accuracy, tables

def classnamemap(le,inverse):
    classnames = dict(zip(le, inverse))
    return classnames