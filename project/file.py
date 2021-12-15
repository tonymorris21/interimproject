import os
from flask import Blueprint,render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from . import db
from .models import File
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from flask import current_app
file = Blueprint('file', __name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@file.route('/upload')
def upload():
    return render_template('upload.html', name=current_user.name)

@file.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':   
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            print( "",current_user.email,filename)
            flash('File successfully uploaded')
            new_file = File(userid=current_user.id, name=filename, location=os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            db.session.add(new_file)
            db.session.commit()
            return redirect(url_for('main.index'))
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(url_for('main.index'))
