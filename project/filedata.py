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
import matplotlib
import matplotlib.pyplot as plt
import base64
import collections
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sn
from flask import send_file
from sklearn.preprocessing import OneHotEncoder
from numpy import array
import missingno as msno
filedata = Blueprint('filedata', __name__)

matplotlib.use('Agg')
@filedata.route('/filedata/<fileid>')
def file_data(fileid):
    print("request args",request.args)
    file = File.query.filter_by(fileid=fileid).first()
    session['filename'] = file.name
    session['filelocation'] = file.location
    session['fileid'] = file.fileid
    filename = session['filename']
    filelocation = session['filelocation']
    #print(filelocation)
  #  print(filename)
   
    sniffer = csv.Sniffer()
    sample_bytes = 2096

    hasheader = sniffer.has_header(
        open(filelocation).read(sample_bytes))
   # print(hasheader)
    df = pd.read_csv(file.location)
    msno.matrix(df, sparkline=False, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0))
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
   # print("Generate report",df)
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
   # if not hasheader:
        #print("test")
    #    addheader =[]
    #    i=0
    #    for x in df.columns:

      #      addheader.append("Col"+str(i))
     #       i+=1
     #   df.columns = addheader
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
   # print("Continous column values",continousdf.columns.values)
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
  #  print("Missing features",df.isnull().sum(axis=1).tolist())
    #print("Missing features target",df["Col4"].isnull().sum().tolist())
    missingtarget = 0
    occurrences = collections.Counter(missingfeatures)
  #  print(occurrences)
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
    dfrows = df[df.isnull().any(axis=1)]
    
    
    
   # print(dfrows.head())
    rowdatacolumns = dfrows.columns.values
   # print(rowdatacolumns)
   # print(catcountpt,contcountpt)
    test = list(dfrows.index.values)
    dfd = df
    dfd.insert(0,'Row',list(df.index.values))
    dfrows.insert(0, 'Row', test)
    dfrows["Drop"] = "dropbutton"
        
    tablerowdata = dfrows.to_html(classes = 'table rowdatatable')
    return render_template('filedata.html',fulltable=list(dfd.values.tolist()),fulltablecolumns = dfd.columns.values, plot_url=plot_url,indexrows = list(dfrows.index.values.tolist()),titlesgr=dfrows.columns.values,rowdatacolumns = rowdatacolumns,catcount=catcount,contcount = contcount,catcountpt = catcountpt, contcountpt = contcountpt,missingtarget=missingtarget, occurrences = occurrences,indexlist = index_list,rowdata = list(dfrows.values.tolist()),tables=[df.to_html()], titles=[''],totalrows = numrows, featurecount = numcolumns ,fileid = fileid,hasheader=hasheader,df =df,column_names=dataframe.columns.values,row_data=list(dataframe.values.tolist()),target = target,columns = columnlist,dataTypeDict = dataTypeDict,check_box = "Nullable", numcolumns = numcolumns, zip=zip,datasetname = filename, numrows =numrows,continouscolumnnames = continousdf.columns.values,continousrow_data=list(continousdf.values.tolist()),continouscolumns=list(continousdf.columns),categoricalcolumnnames = categoricaldf.columns.values,categoricalrow_data=list(categoricaldf.values.tolist()),categorical=list(categoricaldf.columns))

@filedata.route('/filedata/<fileid>/contvalue/<contvalue>', methods=['GET', 'POST']) 
def setContFeatures(fileid,contvalue):
    contvars =[]
    if session.get("contfeature") is not None:
        contvars = session["contfeature"] 
    contvars.append(contvalue)
    session["contfeature"] = contvars
    file_data(fileid)
    return "200"
@filedata.route('/filedata/<fileid>/catvalue/<catvalue>', methods=['GET', 'POST']) 
def setCatFeatures(fileid,catvalue):
    contvars =[]
    if session.get("catfeature") is not None:
        contvars = session["catfeature"] 

    if session.get("contfeature") is not None:
     cont = session.get("contfeature")
     if(catvalue in cont):
        cont.remove(catvalue)
        print("cont values ",cont)
        session["contfeature"] =cont
    contvars.append(catvalue)
    session["catfeature"] = contvars
    file_data(fileid)
    return "200"
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
    if session.get("contfeature") is not None:
        cont = session.get("contfeature")
        print(cont)
        at = session.get("catfeature")
        print(at)
        for x in cont:

            non_floats.append(x)

    for col in cat:
        if df[col].dtypes != "object":
            if col not in columnTypesDict:
                non_floats.append(col)
   # print(non_floats)
    if session.get("catfeature") is not None:
        cont = session.get("catfeature")
        print("cont value",cont)
        for x in cont:

            non_floats.remove(x)
    cat = cat.drop(columns=non_floats)
    allcolumns = df.columns
    categoricalcols = cat.columns
    print(categoricalcols)

    
    #print("Categorical Count", categorical.count())
    continous = con.drop(columns = categoricalcols ,axis=1)
    #print("continous Count", continous.describe(include = 'all'))
   # print(continous)

    modelist = modeList(cat)
    modeListcount = modeListCount(cat)
    secondmodeListcount = secondmodeListCount(cat)
    secondmodelist = secondmodeList(cat)
   # print("1st mode",modelist)
   # print("second mode",secondmodelist)
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
            "2nd Mode %": secondmodePercentagelist,
            "":"Checkbox"
        }
    )
    continousmin = continous.min()
   # print(continous.columns)

    
   # print(continous.describe(include = 'all'))
  #  print(list(continous.std()))

    quartile1 = []
    for x in continous.columns:
        try:
            quant = continous[x].quantile(0.25)
            if quant:
             #   print(continous[x].quantile(0.25))
                quartile1.append(quant)
        except (RuntimeError, TypeError, NameError):
            quartile1.append("N/A")
    quartile3 = []
    for x in continous.columns:
        try:
            quant1 = continous[x].quantile(0.75)
            if quant1:
             #   print(continous[x].quantile(0.75))
                quartile3.append(quant1)
        except (RuntimeError, TypeError, NameError):
            quartile3.append("N/A")
    mean1 = []
    for x in continous.columns:
        try:
            mean = continous[x].mean()
            if mean:
                
                mean1.append(mean)
              #  print(str(mean))
        except Exception as e:
            mean1.append("N/A")
    median1 = []
    for x in continous.columns:
        try:
            median = continous[x].median()
            if median:
                
                median1.append(median)
             #   print(str(median))
        except Exception as e:
            median1.append("N/A")
    std1 = []
    for x in continous.columns:
        try:
            std = continous[x].std()
            if std:
                
                std1.append(median)
               # print(str(std))
        except Exception as e:
            std1.append("N/A")
 #   print("mean",median1)
    continousdf = pd.DataFrame(
        {
            "Feature": array(continous.columns),
            "Count": array(continous.count()),
            "% Missing": array(continous.isna().sum()/len(continous)*100),
            "Cardinality":array(continous.nunique()),
            "Min":array(continous.min()),
            "1st Quartile":quartile1,
            "Mean":mean1,
            "Median":median1,
            "3rd Quartile":quartile3,
            "Max":array(continous.max()),
            "Std. Deviation": std1,
            "":"Checkbox"
            
        }
    )
   # print(continousdf)
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
        binsize = int(np.ceil(np.log2(len(df[featurename]))) + 1)
        plot_url = generateContinuousGraphs(df[featurename],featurename,binsize)
        return plot_url
    if datatype == 'categorical':
        print("Categorical feature")
        plot_url = generateCategoricalGraphs(df[featurename],featurename)
        return plot_url




def generateContinuousGraphs(feature,featurename,binsize):
    fig = plt.figure()
    q75,q25 = np.percentile(feature, [75 ,25])
    iqr = q75 - q25
    print(iqr)
    print(len(feature))
    a= sn.histplot(data=feature, stat="frequency" , bins=int(binsize),kde=True)
    a.set_title("Histogram for "+ featurename + " values")
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    p= sn.boxplot(y=feature)
    p.set_title("Box Plot for "+ featurename + " values")
    img = BytesIO()
    
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
    images = plot_url + "," + plot_url2
    return images
@filedata.route('/filedata/<fileid>/generateContinuousGraphs/feature/<feature>/binsize/<binsize>/charttype/<charttype>', methods=['GET', 'POST'])
def generateContinuousGraphsd(fileid,feature,binsize,charttype):
    fig = plt.figure()

    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_csv(file.location)
    if(charttype=="distplot"):
        sn.distplot(a=df[feature], hist=True, bins=int(binsize))
    
        img = BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    if(charttype=="boxplot"):
        sn.boxplot(y=df[feature])
        set_title("Box Plot for "+ df[feature].name + " values")
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
    return str(images)
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
    print(occurrences)
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
@filedata.route('/filedata/<fileid>/dropRowbyIndex/<index1>', methods=['GET', 'POST'])
def dropRowByIndex(fileid,index1):
    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    index = index1
    df.drop(df.index[int(index1)], inplace = True)
    file = File.query.filter_by(fileid=fileid).first()
    df.to_csv(file.location ,mode='w+',index=False )
    print(df.head)
    return "200"
@filedata.route('/filedata/<fileid>/missingvalues/<featurename>/change/<change>', methods=['GET', 'POST'])
def missingValues(fileid,featurename,change):
    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    print("Value changing is equal to",change)
    if change =="mean":
        df[featurename].fillna(df[featurename].mean(), inplace=True)
    if change == "mode":
        print("Changing to mode",df[featurename].mode()[0])
        df[featurename].fillna(df[featurename].mode()[0], inplace=True)
    if change == "median":
        df[featurename].fillna(df[featurename].median(), inplace=True)
    file = File.query.filter_by(fileid=fileid).first()
    df.to_csv(file.location ,mode='w+',index=False )

    fig = plt.figure()
    binsize = int(np.ceil(np.log2(len(df[featurename]))) + 1)
    a= sn.distplot(a=df[featurename], hist=True, bins=int(binsize),hist_kws=dict(edgecolor="black", linewidth=2))
    a.set_title("Histogram for "+ df[featurename].name + " after changes")
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url
@filedata.route('/filedata/<fileid>/encoding/<featurename>/change/<change>', methods=['GET', 'POST'])
def encodingValues(fileid,featurename,change):
    filelocation = session['filelocation']
    df = pd.read_csv(filelocation)
    print("Value changing is equal to",change)
    if change =="OrdinalEncoder":
        ord_enc = OrdinalEncoder()
        obj_df[featurename] = ord_enc.fit_transform(obj_df[[featurename]])
    if change == "mode":
        enc = OneHotEncoder(handle_unknown='ignore')
        name = featurename + '_new'
        enc_df = pd.DataFrame(enc.fit_transform(df[[name]]).toarray())
        df = df.join(enc_df)
    file = File.query.filter_by(fileid=fileid).first()
    df.to_csv(file.location ,mode='w+',index=False )
    print(df.head)
    return "200"
@filedata.route('/filedata/<fileid>/downloadcleandata', methods=['GET', 'POST'])
def downloaddata(fileid):
    file = File.query.filter_by(fileid=fileid).first()
    csv = file.location
    return send_file(file.location,
                     mimetype='text/csv',
                     attachment_filename='data.csv',
                     as_attachment=True)

@filedata.route('/filedata/<fileid>/clamp/<featurename>/upper/<uppervalue>/lower/<lowervalue>', methods=['GET', 'POST'])
def clampTransformation(fileid, featurename,uppervalue,lowervalue):
    file = File.query.filter_by(fileid=fileid).first()
    df = pd.read_csv(file.location)
    df[featurename] = df[featurename].clip(upper=float(uppervalue),lower=float(lowervalue))
    print(df[featurename].max())
    print(file.location)
    df.to_csv(file.location ,mode='w+',index=False )
    return "200"

@filedata.route('/filedata/whiskerValues/Q1/<Q1>/Q3/<Q3>', methods=['GET', 'POST'])
def whiskerValues(Q1,Q3):
    print(Q1)
    print(Q3)
    Q1 = Q1.strip()
    print("test",Q1)
    Q3 = Q3.strip()

    uppervalue = float(Q3)+(1.5*(float(Q3)-float(Q1)))
    lowervalue=float(Q1)-1.5*(float(Q3)-float(Q1))

    values = str(uppervalue) + "," + str(lowervalue)
    return values
