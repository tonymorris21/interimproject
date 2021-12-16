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

filedata = Blueprint('filedata', __name__)


@filedata.route('/filedata')
def file_data():

    filename = session['filename']
    filelocation = session['filelocation']
    print(filename)
    df = pd.read_csv(filelocation)
    print(list(df.columns))
    

    return render_template('filedata.html',columns = list(df.columns) )
