
from flask import Blueprint,render_template, redirect,url_for

from flask_login import login_required, current_user
main = Blueprint('main', __name__)
from flask_login import current_user

# https://github.com/shuhaowu/projecto/tree/master/projecto/apiv1/files
# https://www.digitalocean.com/community/tutorials/how-to-structure-large-flask-applications
# https://github.com/AmolMavuduru/AutoML-HackUTD19
#https://stackoverflow.com/questions/62682674/how-to-get-dynamic-html-table-entries-in-a-form-to-flask
@main.route('/')
def index():
    if current_user.is_authenticated:
	    return redirect(url_for('userproject.project'))
    else:
        return render_template('login.html')




