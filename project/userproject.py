import os
from flask import Blueprint,render_template, redirect, url_for, request, flash,session
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from . import db
from .models import File,Model
from .models import Project
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename

from sqlalchemy.dialects.postgresql import UUID
import uuid
from flask import current_app
import string
import json
import json2html
import pandas as pd
from json2html import *
from pandas.io.json import json_normalize
from markupsafe import Markup
from flask import escape
from bs4 import BeautifulSoup
import urllib.request
userproject = Blueprint('userproject', __name__)

@userproject.route('/userproject')
def project():

    print("hello")
   
    data = Project.query.filter_by(userid=current_user.id).all()
    return render_template('project.html',data=data)

@userproject.route('/userproject', methods=['POST'])
def createproject():

    name = request.form.get('name')
    projectid = str(uuid.uuid4())
    new_project = Project(projectid=projectid,userid=current_user.id, projectName=name)

    db.session.add(new_project)
    db.session.commit()
    session['projectid'] = projectid
    return projectinfo(projectid)

@userproject.route('/createproject')
def projectcreate():

    return render_template('createproject.html')

@userproject.route('/project/<projectid>')
def projectinfo(projectid):

    print("Projectid",projectid)
    model = Model.query.filter_by(projectid=projectid).all()
    file = File.query.filter_by(projectid=projectid).all()
    project = Project.query.filter_by(projectid=projectid).all()
    print("Project 0",project[0].projectName)
    session['projectname'] = project[0].projectName
    for row in model:
        print ("Name: ",row.accuracy)
    #confusion_matrix = model.confusion_matrix
   # accuracy = model.accuracy
    if len(model) >0 :

        return render_template('projectinfo.html',file =file,model =model , projectname = project[0].projectName)
    if len(file)<0:
        return render_template('projectinfo.html', file = file,model=model)
    return render_template('projectinfo.html',file = file,model=model)
@userproject.route('/model/<modelid>')
def modelinfo(modelid):
    model = Model.query.filter_by(modelid=modelid).all()
    confusion_matrix = model[0].confusion_matrix
    accuracy = model[0].accuracy
    classreport = model[0].class_report
    classreport = str(classreport).replace('[', '')
    classreport = str(classreport).replace(']', '')
    classreport = str(classreport).replace("'", '')
    classreport = str(classreport).replace("\\n", '')
    classreport.strip('/n')
    print(classreport)
    print(classreport)
    soup = BeautifulSoup(classreport)
    soup.get_text(strip=True)

    return render_template('modelinfo.html',accuracy=accuracy,confusion_matrix=confusion_matrix,tables=soup)
def object_as_dict(obj):
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}



