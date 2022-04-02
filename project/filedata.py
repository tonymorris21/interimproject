from locale import normalize
import os
from flask import Blueprint,render_template, redirect, url_for, request, flash, session
from pydantic import Json
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
from sklearn.preprocessing import LabelEncoder
from numpy import array
import missingno as msno
import vaex
import time
import pyarrow.feather as feather
import random
import json
from random import sample
filedata = Blueprint('filedata', __name__)

matplotlib.use('Agg')
@filedata.route('/filedata/<fileid>')
def file_data(fileid):
    
    start = time.time()
    print("request args",request.args)
    file = File.query.filter_by(fileid=fileid).first()
    session['filename'] = file.name
    session['filelocation'] = file.featherlocation
    session['fileid'] = file.fileid
    filename = session['filename']
    filelocation = session['filelocation']
   
    sniffer = csv.Sniffer()
    sample_bytes = 2096

   # print(hasheader)
    start_time = time.time()
    print(filelocation)
    df = feather.read_feather(filelocation,use_threads=True)
    
    print("feather",df.head)
    #https://matthewrocklin.com/blog/work/2017/01/12/dask-dataframes
    #https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html
    #kdnuggets.com/2020/06/machine-learning-dask.html
    print("--- %s seconds ---" % (time.time() - start_time))
    msno.matrix(df, sparkline=False, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0))
    img = BytesIO()
    plt.title("Nullity Matrix",fontsize=20)
    plt.tight_layout()
   
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
   # print("Generate report",df)
    contcolumns,continousdf,categoricaldf = generate_report(df)
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
    #df.to_csv (filelocationandname, index = None, header=True)
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
    end = time.time()
    print("The time of execution of above program is :", end-start)
    return render_template('filedata.html',contcolumns = list(contcolumns.tolist()),fulltablecolumns = dfd.columns.values, plot_url=plot_url,indexrows = list(dfrows.index.values.tolist()),titlesgr=dfrows.columns.values,rowdatacolumns = rowdatacolumns,catcount=catcount,contcount = contcount,catcountpt = catcountpt, contcountpt = contcountpt,missingtarget=missingtarget, occurrences = occurrences,indexlist = index_list,rowdata = list(dfrows.values.tolist()),tables=[df.to_html()], titles=[''],totalrows = numrows, featurecount = numcolumns ,fileid = fileid,df =df,column_names=dataframe.columns.values,row_data=list(dataframe.values.tolist()),target = target,columns = columnlist,dataTypeDict = dataTypeDict,check_box = "Nullable", numcolumns = numcolumns, zip=zip,datasetname = filename, numrows =numrows,continouscolumnnames = continousdf.columns.values,continousrow_data=list(continousdf.values.tolist()),continouscolumns=list(continousdf.columns),categoricalcolumnnames = categoricaldf.columns.values,categoricalrow_data=list(categoricaldf.values.tolist()),categorical=list(categoricaldf.columns))

@filedata.route('/filedata/<fileid>/contvalue/<contvalue>', methods=['GET', 'POST']) 
def setContFeatures(fileid,contvalue):
 
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    data =getFeatureData(df,df[contvalue],contvalue,"Continuous")
    return data
@filedata.route('/filedata/<fileid>/catvalue/<catvalue>', methods=['GET', 'POST']) 
def setCatFeatures(fileid,catvalue):

    print("catfeatures",session.get("catfeature"))
    file_data(fileid)
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    data =getFeatureData(df,df[catvalue],catvalue,"Categorical")
    print(data)
    return data
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
    print("contfeatures",session.get("contfeature"))
    print("catfeatures",session.get("catfeature"))
    if session.get("contfeature") is not None:
        cont = session.get("contfeature")
        print(cont)
        at = session.get("catfeature")
        t = list(dict.fromkeys(cont))
        print("cont",cont)
        non_floats = [i for i in t if i not in non_floats]

    for col in cat:
        if df[col].dtypes != "object":
            if col not in columnTypesDict:
                non_floats.append(col)
   # print(non_floats)
    print(non_floats)
    try:
        if session.get("catfeature") is not None:
            cont = session.get("catfeature")
            
            t = list(dict.fromkeys(cont))
            non_floats = [i for i in non_floats if i not in t]

           
    except:
        print("test")
    print("non floats",non_floats)
    cat = cat.drop(columns=non_floats)
    allcolumns = df.columns
    categoricalcols = cat.columns
    print(categoricalcols)

    
    #print("Categorical Count", categorical.count())
    continous = con.drop(columns = categoricalcols ,axis=1)
    contcolumns = continous.columns
    
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
    pd.options.display.float_format = '${:,.2f}'.format
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
                
                std1.append(std)
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
    
    return array(continous.columns),continousdf,categoricaldf

def modeList(df):
    modeList = []
    for columns in df:
        modeList.append(get_most_frequent(df[columns]))
    return modeList
def secondmodeList(df):
    modeList = []
    for columns in df:
        try:
            modeList.append(second_get_most_frequent(df[columns]))
        except:
            modeList.append(0)
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
    print("Second get most frequent",a)
    first = sorted(dict(a).items(), key=lambda x: -x[1])[1]
    print(first)
    return first[0]

def get_most_frequent_count(x):
    a = x.value_counts()

    first = sorted(dict(a).items(), key=lambda x: -x[1])[0]
    return first[1]
def second_get_most_frequent_count(x):
    a = x.value_counts()
    try:
        first = sorted(dict(a).items(), key=lambda x: -x[1])[1]
    except:
        return 0
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
def generateGraphs(fileid,featurename):
    print("Test feature")
    datatype = request.form.get("datatype2")
    
    featurename = featurename
    filelocation = session['filelocation']
    print("filelocation",filelocation)
    df = pd.read_feather(filelocation)
    print("datatype",datatype)
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
    a= sn.histplot(data=feature, stat="frequency" , bins=int(binsize),kde=True)
    a.set_title("Histogram for "+ featurename + " values")
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    p= sn.boxplot(y=feature)
    p.set_title("Box Plot for "+ featurename + " values")
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
    
    images = plot_url + "," + plot_url2
    return images
@filedata.route('/filedata/<fileid>/generateContinuousGraphs/feature/<feature>/binsize/<binsize>/charttype/<charttype>', methods=['GET', 'POST'])
def generateContinuousGraphsd(fileid,feature,binsize,charttype):
    fig = plt.figure()
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)

    if(charttype=="distplot"):
        a= sn.histplot(data=df[feature], stat="frequency" , bins=int(binsize),kde=True)
        a.set_title("Histogram for "+ df[feature].name)
    
        img = BytesIO()
        
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
    if(charttype=="after changes"):
        a= sn.histplot(data=df[feature], stat="frequency" , bins=int(binsize),kde=True)
        a.set_title("Histogram for "+ df[feature].name + "after changes")
        plt.tight_layout()
        img = BytesIO()
        
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

def generateCategoricalGraphs(feature,featurename):
    
    fig = plt.figure()
    
    feature.value_counts().plot(kind='bar', xlabel=featurename, ylabel='Density')
    img = BytesIO()
    plt.title("Bar plot for " + featurename)
    plt.tight_layout()
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url

@filedata.route('/filedata/<fileid>/correlations/<cont>', methods=['GET', 'POST'])
def correlationGraphs(fileid,cont):

    filelocation = session['filelocation']

    df = pd.read_feather(filelocation)
    fig = plt.figure()
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)

    img = BytesIO()

    plt.suptitle("Correlation Matrix")
    
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
    plot_url = generatescattermatrix(df,cont)
    images = plot_url + "," + plot_url2

    return str(images)

def generatescattermatrix(df,cont):

    fig = plt.figure()
    cont = cont.split(',')
    cont.pop()

    df = df.sample(frac = 0.3)
    scatter_matrix(df[cont])
    img = BytesIO()

    plt.suptitle("Scatter Plot Matrix")
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url
@filedata.route('/filedata/<fileid>/dropfeature/<featurename>', methods=['GET', 'POST'])
def dropFeature(fileid,featurename):
    featurename = featurename
    print(featurename)
    print(fileid)
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)

    df = df.drop(featurename, 1)
    df.to_feather(filelocation)
    print("feature dropped")
    print(df)
    return redirect(url_for('filedata.file_data',fileid = fileid))

@filedata.route('/filedata/<fileid>/newgraph/<xaxis>/<yaxis>', methods=['GET', 'POST'])
def newgraph(xaxis,yaxis,fileid):

    yaxis = yaxis
    xaxis = xaxis
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
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
    #get total execution time of this function and store in time
    start_time = time.time()


    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    columns = len(df.columns)
    images = ""
    for columns in df:
        if(columns==target):
            print("target")
        else:
            sn.catplot(y=columns, x=target, data=df)
            img = BytesIO()
            plt.tight_layout()
            plt.suptitle("Scatter plot matrix for "+columns)

            plt.savefig(img,format='png')
            plt.close()
            img.seek(0)
            plot_url2 = base64.b64encode(img.getvalue()).decode('utf8')
            images = images + "," + plot_url2
         
               
    end_time = time.time() - start_time
    print("Total time to generate bivariate",end_time)
    return str(images)

@filedata.route('/filedata/<fileid>/missingvalues/<target>', methods=['GET', 'POST'])
def getMissingFeatures(fileid,target):
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    missingfeatures = df.isnull().sum(axis=1).tolist()
    print(type(target)==str)
    print("Missing features",df.isnull().sum(axis=1).tolist())
    occurrences = collections.Counter(missingfeatures)

    b = dict(occurrences)
    missingtarget = df[target].isnull().sum().tolist()
    #store rows in a df[column] missing values for each df[column] with column name in dictionary
    missingfeatures = df.isnull().sum(axis=0).tolist()


    missingfeatures = dict(zip(df.columns, missingfeatures))
    missingfeatures["totalrows"] = df.shape[0]
    print("Missing values",missingfeatures)
    missingtargets = str(missingtarget)
    d = dict(occurrences)
    d[99]=missingtargets
    
    return jsonify(missingfeatures)
@filedata.route('/filedata/<fileid>/dropMissingValue/<target>', methods=['GET', 'POST'])
def dropMissingFeature(fileid,target):
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    df = df.dropna(how='any' ,subset=[target])
    df.reset_index(drop=True).to_feather(filelocation)
    return "success"

@filedata.route('/filedata/<fileid>/missingvalues/<featurename>/change/<change>', methods=['GET', 'POST'])
def missingValues(fileid,featurename,change):
    datatype = request.form.get("datatype")
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    print("Value changing is equal to",change)
    if change =="mean":
        df[featurename].fillna(df[featurename].mean(), inplace=True)
    if change == "mode":
        print("Changing to mode",df[featurename].mode()[0])
        df[featurename].fillna(df[featurename].mode()[0], inplace=True)
    if change == "median":
        df[featurename].fillna(df[featurename].median(), inplace=True)
    if change == "remove":
        print("Remove")
        print(featurename)
        df.dropna(subset = [featurename], inplace=True)
        print(df.head)
    file = File.query.filter_by(fileid=fileid).first()
    df.reset_index(drop=True).to_feather(file.featherlocation)

    fig = plt.figure()
    binsize = int(np.ceil(np.log2(len(df[featurename]))) + 1)
    a= sn.histplot(data=df[featurename], stat="frequency" , bins=int(binsize),kde=True)
    a.set_title("Histogram for "+ df[featurename].name + " after changes")
    img = BytesIO()
    
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    #if datatype is eqaul to  "continuous"  then set datatype = "Continuous"
    if datatype == "continuous":
        datatype = "Continuous"

    print("datatype is",datatype)
    
    data =getFeatureData(df,df[featurename],featurename,datatype)
    dataarray = []
    dataarray.append(data)
    dataarray.append(plot_url)
    #return dataarray in json format
    return json.dumps(dataarray)
    
@filedata.route('/filedata/<fileid>/encoding/<featurename>/change/<change>', methods=['GET', 'POST'])
def encodingValues(fileid,featurename,change):
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    print("Value changing is equal to",change)
    if change =="OrdinalEncoder":
        ord_enc = OrdinalEncoder()
        df[featurename] = ord_enc.fit_transform(df[[featurename]])
    if change =="LabelEncoding":
        label_enc = LabelEncoder()
        df[featurename] = label_enc.fit_transform(df[[featurename]])
       
    if change == "OneHotEncoder":
        enc = OneHotEncoder(handle_unknown='ignore')
        name = featurename + '_new'
        enc_df = pd.DataFrame(enc.fit_transform(df[[featurename]]).toarray())
        
        df = df.join(enc_df)
    file = File.query.filter_by(fileid=fileid).first()
    df.to_feather(file.featherlocation)
    print(df.head)
    data =getFeatureData(df,df[featurename],featurename,"Categorical")
    return data
@filedata.route('/filedata/<fileid>/downloadcleandata', methods=['GET', 'POST'])
def downloaddata(fileid):
    file = File.query.filter_by(fileid=fileid).first()
    csv = file.location
    return send_file(file.location,
                     mimetype='text/csv',
                     attachment_filename='data.csv',
                     as_attachment=True)

@filedata.route('/filedata/<fileid>/clamp/<featurename>/datatype/<datatype>/upper/<uppervalue>/lower/<lowervalue>', methods=['GET', 'POST'])
def clampTransformation(fileid, featurename,datatype,uppervalue,lowervalue):
    file = File.query.filter_by(fileid=fileid).first()
    filelocation = session['filelocation']
    df = pd.read_feather(filelocation)
    df[featurename] = df[featurename].clip(upper=float(uppervalue),lower=float(lowervalue))
    df.to_feather(file.featherlocation)
    data =getFeatureData(df,df[featurename],featurename,"Continuous")
    return data
def getFeatureData(df,feature,featurename,datatype):
    if datatype=="Continuous":
        data = []
        float_formatter = "{:.2f}".format
        data.append(featurename)
        data.append(int(feature.count()))
        data.append(feature.isna().sum()/len(feature)*100)#missing
        data.append(int(feature.nunique()))#cardinality
        data.append(float_formatter(float(feature.min())))#min
        data.append(float_formatter(feature.quantile(0.25)))#1st
        data.append(float_formatter(feature.mean()))#mean
        data.append(float_formatter(feature.median()))#median
        data.append(float_formatter(feature.quantile(0.75)))#3rd
        data.append(float_formatter(feature.max()))#max
        print("max",feature.max())
        data.append(float_formatter(feature.std()))#std
        print(data)
        return json.dumps(data)
    else:
        data = []
        float_formatter = "{:.2f}".format
        
        
        d =df[featurename].value_counts().index.tolist()
        f = df[featurename].value_counts().tolist()
        allmissing = df[featurename].isna().sum()
        data.append(featurename)
        data.append(int(feature.count()))
        data.append(feature.isna().sum()/len(feature)*100)
        data.append(int(feature.nunique()))
        data.append(float_formatter(d[0]))
        data.append(float_formatter(f[0]))
        data.append(float_formatter(f[0]/len(feature)*100))
        data.append(float_formatter(d[1]))
        data.append(float_formatter(f[1]))
        data.append(float_formatter(f[1]/len(feature)*100))
        print(data)
        return json.dumps(data)

@filedata.route('/filedata/whiskerValues/Q1/<Q1>/Q3/<Q3>', methods=['GET', 'POST'])
def whiskerValues(Q1,Q3):
    Q1 = Q1.strip()
    Q3 = Q3.strip()
    uppervalue = float(Q3)+(1.5*(float(Q3)-float(Q1)))
    lowervalue=float(Q1)-1.5*(float(Q3)-float(Q1))
    values = str(uppervalue) + "," + str(lowervalue)
    return values


https://library-cc.tudublin.ie/articles/3928395.3124/1.PDF