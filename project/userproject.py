import os
from flask import Blueprint,render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from . import db
from .models import Project
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from flask import current_app
userproject = Blueprint('userproject', __name__)

@userproject.route('/userproject')
def project():
    return render_template('project.html')
    
@userproject.route('/userproject', methods=['POST'])
def createproject():

    name = request.form.get('name')

    new_project = Project(userid=current_user.id, projectName=name)

    db.session.add(new_project)
    db.session.commit()

    return render_template('upload.html', projectName = name)