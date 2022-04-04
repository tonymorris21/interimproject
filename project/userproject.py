
from flask import Blueprint,render_template, request,session
from . import db
from .models import File,Model
from .models import Project
from flask_login import current_user
import uuid
from json2html import *
from datetime import date


userproject = Blueprint('userproject', __name__)

@userproject.route('/userproject')
def project():


    data = Project.query.filter_by(userid=current_user.id).all()
    return render_template('project.html',data=data)

@userproject.route('/userproject', methods=['POST'])
def createproject():

    name = request.form.get('name')
    projectid = str(uuid.uuid4())
    projectcreation = date.today()
    new_project = Project(projectid=projectid,userid=current_user.id, projectName=name,projectcreation=projectcreation)

    db.session.add(new_project)
    db.session.commit()
    session['projectid'] = projectid
    return projectinfo(projectid)

@userproject.route('/project/<projectid>')
def projectinfo(projectid):

    
    model = Model.query.filter_by(projectid=projectid).all()
    file = File.query.filter_by(projectid=projectid).all()
    project = Project.query.filter_by(projectid=projectid).first()
    session['projectid'] = projectid
    return render_template('projectinfo.html',file = file,model=model, projectname = project.projectName)
    
@userproject.route('/deleteproject/<projectid>')
def deleteproject(projectid):

    project = Project.query.get(projectid)
    
    db.session.delete(project)
    db.session.commit()
    data = Project.query.filter_by(userid=current_user.id).all()
    return render_template('project.html',data=data)

@userproject.route('/deletemodel/<modelid>', methods=['POST'])
def deletemodel(modelid):

    project = Model.query.get(modelid)
    db.session.delete(project)
    db.session.commit()
    return "success"



