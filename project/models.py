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
    location = db.Column(db.String(300))

class Project(db.Model):
    projectid = db.Column(db.Integer,primary_key=True)
    userid = db.Column(db.Integer)
    projectName = db.Column(db.String(300))