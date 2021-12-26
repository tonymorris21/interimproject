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

train = Blueprint('train', __name__)


@train.route('/train')
def train_data():

#https://medium.com/swlh/building-a-machine-learning-model-step-by-step-with-the-titanic-dataset-e3462d849387
    return render_template('train.html')