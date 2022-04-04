from flask_login import UserMixin
from . import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

class File(db.Model):
    fileid = db.Column(db.Integer,primary_key=True)
    projectid = db.Column(db.Integer)
    name = db.Column(db.String(300))
    featherlocation = db.Column(db.String(300))
    location = db.Column(db.String(300))
    fileuploaddate = db.Column(db.Date)

class Project(db.Model):
    projectid = db.Column(db.String(400),primary_key=True)
    userid = db.Column(db.Integer)
    projectName = db.Column(db.String(300))
    projectcreation = db.Column(db.Date)

class Model(db.Model):
    modelid = db.Column(db.Integer,primary_key=True)
    projectid = db.Column(db.Integer)
    algorithm = db.Column(db.String(100))
    target = db.Column(db.String(100))
    location = db.Column(db.String(10000))
    classnames = db.Column(db.String(10000))
    createddate = db.Column(db.Date)
    accuracy = db.Column(db.Integer)
    confusion_matrix = db.Column(db.String(10000))
    class_report = db.Column(db.String(10000))