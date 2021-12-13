import os
from flask import Blueprint,render_template, Flask, flash, request, redirect
from . import db
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user
main = Blueprint('main', __name__)
from flask import current_app


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# https://github.com/shuhaowu/projecto/tree/master/projecto/apiv1/files
# https://www.digitalocean.com/community/tutorials/how-to-structure-large-flask-applications

@main.route('/')
def index():
	return render_template('index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)


@main.route('/profile', methods=['POST'])
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
            return redirect('/profile')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)
