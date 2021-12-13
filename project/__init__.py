import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager


db = SQLAlchemy()

def create_app():
    app = Flask(__name__,template_folder='./templates',static_folder='./static')

    app.config['SECRET_KEY'] = 'secret-key-goes-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    db.init_app(app)
    from .models import User
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)
    path = os.getcwd()
    UPLOAD_FOLDER = os.path.join(path, 'uploads')

    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    from .models import User
    @login_manager.user_loader
    def load_user(user_id):

        return User.query.get(int(user_id))
    return app
if __name__ == '__main__':
    app.run(threaded=True)