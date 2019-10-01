import json

import requests
from flask import (
    Blueprint, render_template, request, current_app
)
from wtforms import Form, SelectField, IntegerField, BooleanField, SubmitField, validators

bp = Blueprint('pos', __name__)

# defaults values for the visualization pages
DEFAULT_PERPLEXITY = 5
TOTAL_PAIRS = 1790


@bp.route('/pos/ptb', strict_slashes=False, methods=['GET', 'POST'])
def pos_ptb():
    url = current_app.config['REST_KERAS_URL'] + 'pos/ptb'
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(url, headers=headers)
    result = response.json()
    intro = 'Proof of concept created using Penn Treebank tags. For each question-answer pair, the part-of-speech tag with the highest amount of attention was recorded. Only the tags with the highest attention score were recorded and the others discarded. This was repeated for each neuron.'
    return render_template('pos.html', plotly_pos=result, intro=intro)


@bp.route('/pos/upos', strict_slashes=False, methods=['GET', 'POST'])
def pos_upos():
    url = current_app.config['REST_KERAS_URL'] + 'pos/upos'
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(url, headers=headers)
    result = response.json()
    intro = 'For each question-answer pair, each properly aligned universal dependencies universal part-of-speech tag\'s (UPOS) attention score was recorded. The result shows an overall percentage out of 100 of which tags a neuron paid attention to.'
    return render_template('pos.html', plotly_pos=result, intro=intro)


@bp.route('/pos/xpos', strict_slashes=False, methods=['GET', 'POST'])
def pos_xpos():
    url = current_app.config['REST_KERAS_URL'] + 'pos/xpos'
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(url, headers=headers)
    result = response.json()
    intro = 'For each question-answer pair, each properly aligned universal dependencies language-specific part-of-speech tag\'s (XPOS) attention score was recorded. The result shows an overall percentage out of 100 of which tags a neuron paid attention to.'
    return render_template('pos.html', plotly_pos=result, intro=intro)
