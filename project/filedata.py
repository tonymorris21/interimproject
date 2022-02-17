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
import csv
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

    hasheader = (sniffer.has_header(
        open(filename).read(sample_bytes)))
    print(hasheader)
    df = pd.read_csv(filelocation)
    #print(list(df.columns))
    #print(df.isnull().sum())
    numcolumns = len(df.columns)

    if not hasheader:
        print("test")
        addheader =[]
        i=0
        for x in df.columns:

            addheader.append("Col"+str(i))
            i+=1
        df.columns = addheader
        print(addheader)
    numrows = df.shape[0]
    dataTypeDict = dict(df.dtypes)
    tbl = zip(list(df.columns),list(df.dtypes))
#https://flutterq.com/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table/
    test = df.isna().sum()
    print(test)
    dataframe = pd.DataFrame(
        {
            "Column Name" : list(df.columns),
            "Data type" : list(df.dtypes),
            "Missing(Count)" : list(df.isna().sum()),
            "Distinct Values": list(df.nunique())
        }
    )
    dataframe.insert(2, "Nullable", 'Checkbox')
    tables=[dataframe.to_html(index=False,classes='data')]
    body = ''
    target = list(df.columns)
    dict_obj = df.to_dict('list')
    filelocationandname = filelocation
    df.to_csv (filelocationandname, index = None, header=True)
    return render_template('filedata.html',hasheader=hasheader,df =df,column_names=dataframe.columns.values,row_data=list(dataframe.values.tolist()),target = target,columns = list(df.columns),dataTypeDict = dataTypeDict,check_box = "Nullable", numcolumns = numcolumns, zip=zip,datasetname = filename, numrows =numrows)