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
from pandas.plotting import scatter_matrix
import csv
from flask import jsonify
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import collections
from collections import Counter

import seaborn as sn
filedata = Blueprint('filedata', __name__)


@filedata.route('/filedata/<fileid>')
def file_data(fileid):
    file = File.query.filter_by(fileid=fileid).first()
    session['filename'] = file.name
    session['filelocation'] = file.location
    session['fileid'] = file.fileid
    filename = session['filename']
    filelocation = session['filelocation']
    print(filelocation)
    print(filename)
   
    sniffer = csv.Sniffer()
    sample_bytes = 2096

    hasheader = sniffer.has_header(
        open(filelocation).read(sample_bytes))
    print(hasheader)
    df = pd.read_csv(file.location)
    print("Generate report",df)
    continousdf,categoricaldf = generate_report(df)
    #print(list(df.columns))
    #print(df.isnull().sum())
    numcolumns = len(df.columns)
    columnTypesDict = []
    for name, column in df.iteritems():
        unique_count = column.unique().shape[0]
        total_count = column.shape[0]
        if unique_count / total_count < 0.05:
            columnTypesDict.append(name)
  #  print("Categorigcal columsn",columnTypesDict)
    if not hasheader:
        #print("test")
        addheader =[]
        i=0
        for x in df.columns:

            addheader.append("Col"+str(i))
            i+=1
        df.columns = addheader
       # print(addheader)
    numrows = df.shape[0]
    dataTypeDict = dict(df.dtypes)
    tbl = zip(list(df.columns),list(df.dtypes))
#https://flutterq.com/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table/
    test = df.isna().sum()
    #print(test)
    dataframe = pd.DataFrame(
        {
            "Feature" : list(df.columns),
            "Data type" : list(df.dtypes),
            "Count": list(df.nunique()),
            "Missing(Count)" : list(df.isna().sum()),
            "Cardinality": list(df.nunique())
            
        }
    )
    print("Continous column values",continousdf.columns.values)
    dataframe.insert(2, "Nullable", 'Checkbox')
    tables=[dataframe.to_html(index=False,classes='data')]
    body = ''
    target = list(df.columns)
    dict_obj = df.to_dict('list')
    filelocationandname = filelocation
    df.to_csv (filelocationandname, index = None, header=True)
    df.loc[df.isnull().sum(1)>1].index
    rowsmissingonefeature = [0,0,0,0,0,0,0]
    numberoffeatures = df.shape[1]
    count = 0
    missingfeatures = df.isnull().sum(axis=1).tolist()
    print("Missing features",df.isnull().sum(axis=1).tolist())
    print("Missing features target",df["Col4"].isnull().sum().tolist())
    missingtarget = 0
    occurrences = collections.Counter(missingfeatures)
    print(occurrences)
    dict(occurrences)
    columns = " " + df.columns
    index_list = list(df.index.values)
    columnlist = list(df.columns)
    columnlist.insert(0," ")
    test = len(continousdf["Feature"].values)
    catcount =  len(categoricaldf["Feature"].values)
    
    contcount =  len(continousdf["Feature"].values)
    catcountpt = str(catcount/(contcount+catcount)*100)
    contcountpt = str(contcount/(contcount+catcount)*100)
    print(catcountpt,contcountpt)
    return render_template('filedata.html',catcount=catcount,contcount = contcount,catcountpt = catcountpt, contcountpt = contcountpt,missingtarget=missingtarget, occurrences = occurrences,indexlist = index_list,rowdata = list(df[df.isnull().any(axis=1)].values.tolist()),tables=[df.to_html()], titles=[''],totalrows = numrows, featurecount = numcolumns ,fileid = fileid,hasheader=hasheader,df =df,column_names=dataframe.columns.values,row_data=list(dataframe.values.tolist()),target = target,columns = columnlist,dataTypeDict = dataTypeDict,check_box = "Nullable", numcolumns = numcolumns, zip=zip,datasetname = filename, numrows =numrows,continouscolumnnames = continousdf.columns.values,continousrow_data=list(continousdf.values.tolist()),continouscolumns=list(continousdf.columns),categoricalcolumnnames = categoricaldf.columns.values,categoricalrow_data=list(categoricaldf.values.tolist()),categorical=list(categoricaldf.columns))
def generate_report(df):
    #print("count",df.describe(include='all'))
    #print(df)
    columnTypesDict = []
    for name, column in df.iteritems():
        unique_count = column.unique().shape[0]
        total_count = column.shape[0]
        if unique_count / total_count < 0.05:
            columnTypesDict.append(name)

    
    con = df
    cat = df
    #for categorical 
    non_floats= []
    for col in cat:
        if df[col].dtypes != "object":
            if col not in columnTypesDict:
                non_floats.append(col)
    cat = cat.drop(columns=non_floats)
    allcolumns = df.columns
    categoricalcols = cat.columns
    #print("Categorical Count", categorical.count())
    continous = con.drop(columns = categoricalcols ,axis=1)
    #print("continous Count", continous.describe(include = 'all'))
   # print(continous)

    modelist = modeList(cat)
    modeListcount = modeListCount(cat)
    secondmodeListcount = secondmodeListCount(cat)
    secondmodelist = secondmodeList(cat)
    print("1st mode",modelist)
    print("second mode",secondmodelist)
    modePercentagelist = modePercentageList(cat)
    secondmodePercentagelist = secondmodePercentageList(cat)
    pd.set_option("display.precision", 2)
    categoricaldf = pd.DataFrame(
        {
            "Feature": list(cat.columns),
            "Count": list(cat.count()),
            "% Missing": list(cat.isna().sum()/len(cat)*100),
            "Cardinality":list(cat.nunique()),
            "Mode":modelist,
            "Mode Freq": modeListcount,
            "Mode %":modePercentagelist,
            "2nd Mode": secondmodelist,
            "2nd Mode Freq":secondmodeListcount,
            "2nd Mode %": secondmodePercentagelist
        }
    )
    continousmin = continous.min()
   # print(continous.columns)

    
   # print(continous.describe(include = 'all'))
    continousdf = pd.DataFrame(
        {
            "Feature": list(continous.columns),
            "Count": list(continous.count()),
            "% Missing": list(continous.isna().sum()/len(continous)*100),
            "Cardinality":list(continous.nunique()),
            "Min":list(continous.min()),
            "1st Quartile":list(continous.quantile(0.25)),
            "Mean":list(continous.mean()),
            "Median":list(continous.median()),
            "3rd Quartile":list(continous.quantile(0.75)),
            "Max":list(continous.max()),
            "Std. Deviation": list(continous.std())

            
        }
    )
    print(continousdf)
    #print(cat)
    continousdf = np.round(continousdf, decimals = 2)
    categoricaldf = np.round(categoricaldf, decimals = 2)
    return continousdf,categoricaldf

def modeList(df):
    modeList = []
    for columns in df:
        modeList.append(get_most_frequent(df[columns]))
    return modeList
def secondmodeList(df):
    modeList = []
    for columns in df:
        modeList.append(second_get_most_frequent(df[columns]))
    return modeList

def modeListCount(df):
    modeListCount = []
    for columns in df:
        modeListCount.append(get_most_frequent_count(df[columns]))
    return modeListCount
def secondmodeListCount(df):
    secondmodeListCount = []
    for columns in df:
        secondmodeListCount.append(second_get_most_frequent_count(df[columns]))
    return secondmodeListCount
def modePercentageList(df):
    modePercentageList = []
    for columns in df: 
        modePercentageList.append(getmodepercentage(df[columns]))
    return modePercentageList
def secondmodePercentageList(df):
    secondmodePercentageList = []
    for columns in df: 
        secondmodePercentageList.append(getsecondmodepercentage(df[columns]))
    return secondmodePercentageList

def get_most_frequent(x):
    a = x.value_counts()
    
    first = sorted(dict(a).items(), key=lambda x: -x[1])[0]
    return first[0]

def second_get_most_frequent(x):
    a = x.value_counts()

    first = sorted(dict(a).items(), key=lambda x: -x[1])[1]
    return first[0]

def get_most_frequent_count(x):
    a = x.value_counts()

    first = sorted(dict(a).items(), key=lambda x: -x[1])[0]
    return first[1]
def second_get_most_frequent_count(x):
    a = x.value_counts()

    first = sorted(dict(a).items(), key=lambda x: -x[1])[1]
    return first[1]

def getmodepercentage(x):
    a = get_most_frequent_count(x)
    allmissing = x.isna().sum()
    count = len(x)
    total = (a/(count-allmissing)*100)
    return total
def getsecondmodepercentage(x):
    a = second_get_most_frequent_count(x)
    allmissing = x.isna().sum()
    count = len(x)

    total = (a/(count-allmissing)*100)
    return total

@filedata.route('/filedata/<fileid>/<featurename>', methods=['GET', 'POST'])
def testfunc(fileid,featurename):
    print("Test feature")
    datatype = request.form.get("datatype")
    
    featurename = featurename
    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_csv(file.location)

    if datatype == 'continuous':
        plot_url = generateContinuousGraphs(df[featurename],featurename)
        return plot_url
    if datatype == 'categorical':
        print("Categorical feature")
        plot_url = generateCategoricalGraphs(df[featurename],featurename)
        return plot_url





def generateContinuousGraphs(feature,featurename):
    fig = plt.figure()
    sn.distplot(a=feature, hist=True)
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url

def generateCategoricalGraphs(feature,featurename):
    
    fig = plt.figure()
    feature.value_counts().plot(kind='bar', xlabel=featurename, ylabel='Frequency')
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    #comapare 
    return plot_url
@filedata.route('/filedata/<fileid>/correlations', methods=['GET', 'POST'])
def correlationGraphs(fileid):
    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    fig = plt.figure()
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
    plot_url = generatescattermatrix(df)
    images = plot_url + "," + plot_url2
    return images
def generatescattermatrix(df):
    fig = plt.figure()
    scatter_matrix(df)
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url
@filedata.route('/filedata/<fileid>/dropfeature/<featurename>', methods=['GET', 'POST'])
def dropFeature(fileid,featurename):
    featurename = featurename
    print(featurename)
    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_csv(file.location)
    print(df)
    print(file.location)
    df = df.drop(featurename, 1)
    df.to_csv(file.location ,mode='w+',index=False )
    print("feature dropped")
    print(df)
    return redirect(url_for('filedata.file_data',fileid = fileid))

@filedata.route('/filedata/<fileid>/newgraph/<xaxis>/<yaxis>', methods=['GET', 'POST'])
def newgraph(xaxis,yaxis,fileid):
    print('args:', request.args)
    print('form:', request.form)
    yaxis = yaxis
    xaxis = xaxis
    print("Xaxis",xaxis)
    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    fig = plt.figure()
    sn.catplot(x=xaxis, y=yaxis, data=df)
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url2
@filedata.route('/filedata/<fileid>/bivariate/<target>', methods=['GET', 'POST'])
def generatebivariate(target,fileid):
    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    columns = len(df.columns)
    images = ""
    for columns in df:
        if(columns==target):
            print("target")
        else:
            sn.catplot(x=columns, y=target, data=df)
            img = BytesIO()
            plt.tight_layout()
            plt.savefig(img,format='png')
            plt.close()
            img.seek(0)
            plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
            images = images + "," + plot_url2
         
               
        
    return str(images)

@filedata.route('/filedata/<fileid>/missingvalues/<target>', methods=['GET', 'POST'])
def getMissingFeatures(fileid,target):
    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    missingfeatures = df.isnull().sum(axis=1).tolist()
    print(type(target)==str)
    print("Missing features",df.isnull().sum(axis=1).tolist())
    occurrences = collections.Counter(missingfeatures)
    b = dict(occurrences)
    missingtarget = df[target].isnull().sum().tolist()
    print(b)
    missingtargets = str(missingtarget)
    d = dict(occurrences)
    d[99]=missingtargets
    
    return jsonify(d)
@filedata.route('/filedata/<fileid>/dropMissingValue/<target>', methods=['GET', 'POST'])
def dropMissingFeature(fileid,target):
    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    if(type(target)==str):
        df = df.dropna(subset=[target])
    else:
        df.dropna(axis=0, thresh=target)

    return "success"