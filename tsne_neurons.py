import requests
from flask import (
    Blueprint, render_template, current_app
)

bp = Blueprint('tsne_neurons', __name__)


@bp.route('/tsne_neurons/<perplexity>', strict_slashes=False, methods=['GET', 'POST'])
def tsne_neurons(perplexity):
    url = current_app.config['REST_KERAS_URL'] + 'tsne_neurons/' + str(perplexity)
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(url, headers=headers)
    result = response.json()

    return render_template('tsne_neurons.html',
                           plotly_all_questions=result,
                           perplexity=perplexity
                           )
