from flask import render_template, Blueprint

bp = Blueprint('common', __name__)


@bp.route('/about')
def about():
    return render_template('about.html')


@bp.route('/contact')
def contact():
    return render_template('contact.html')
