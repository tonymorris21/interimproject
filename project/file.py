import os
from flask import Blueprint,render_template, redirect, url_for, request, flash, session
from __init__ import db
from models import File
from models import Model
from models import Project
from flask_login import current_user
from werkzeug.utils import secure_filename
from flask import current_app
from datetime import date
import pandas as pd

file = Blueprint('file', __name__)

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@file.route('/upload')
def upload():
    return render_template('upload.html', name=current_user.name)

@file.route('/upload', methods=['POST'])
def upload_file():

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file was uploaded')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):

 

            filename = secure_filename(file.filename)

            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))

            featherfilename = filename+'.feather'
            featherlocation = os.path.join(current_app.config['UPLOAD_FOLDER'] ,featherfilename)
            df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            df.to_feather(featherlocation)

            projectid = session['projectid']
            fileuploaddate = date.today()
            new_file = File(projectid=projectid, name=filename, featherlocation = featherlocation,
            location=os.path.join(current_app.config['UPLOAD_FOLDER'], filename),fileuploaddate = fileuploaddate)

            db.session.add(new_file)
            db.session.commit()

            session['filename'] = filename
            session['filelocation'] = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            return redirect(url_for('userproject.projectinfo',projectid = projectid))
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(url_for('main.index'))


@file.route('/deletefile/<fileid>')
def deletefile(fileid):
    file = File.query.get(fileid)
    
    db.session.delete(file)
    db.session.commit()
    projectid = session['projectid']

    model = Model.query.filter_by(projectid=projectid).all()
    file = File.query.filter_by(projectid=projectid).all()
    project = Project.query.filter_by(projectid=projectid).first()

    session['projectname'] = project.projectName

    return render_template('projectinfo.html',file = file,model=model,projectname = project.projectName)
