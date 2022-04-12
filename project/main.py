
from flask import Blueprint,render_template, redirect,url_for

from flask_login import login_required, current_user
main = Blueprint('main', __name__)
from flask_login import current_user

@main.route('/')
def index():
    if current_user.is_authenticated:
	    return redirect(url_for('userproject.project'))
    else:
        return render_template('login.html')




