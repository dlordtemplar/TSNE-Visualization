import random

import requests
from flask import (
    Blueprint, render_template, request, session
)

bp = Blueprint('neuron', __name__)

# defaults values for the visualization pages
DEFAULT_NUM_TEXTS = 5


@bp.route('/neuron', defaults={'neuron': 0}, strict_slashes=False, methods=['GET', 'POST'])
@bp.route('/neuron/<int:neuron>', strict_slashes=False, methods=['GET', 'POST'])
def display_neuron(neuron):
    # Parameters
    if 'random' in session.keys():
        old_session_random = session['random']
    if 'num_texts' in session.keys():
        old_session_num_texts = session['num_texts']

    parameter_changed = False
    if 'random' in session.keys():
        old_session_random = session['random']
    else:
        old_session_random = False

    if request.method == 'POST':
        if 'scale' in request.values.keys():
            session['scale'] = True
        else:
            if 'scale' in session.keys():
                if session['scale']:
                    session['scale'] = False
            else:
                session['scale'] = False

        if 'random' in request.values.keys():
            session['random'] = True
            session['manual_indices'] = False
            parameter_changed = True
        else:
            if 'random' in session.keys():
                if session['random']:
                    session['random'] = False
                    parameter_changed = True
            else:
                session['random'] = False

        if 'texts_number' in request.values.keys():
            if request.values['texts_number'] != '':
                session['num_texts'] = int(request.values['texts_number'])
                parameter_changed = True
            elif 'num_texts' not in session.keys():
                session['num_texts'] = DEFAULT_NUM_TEXTS
        elif 'num_texts' not in session.keys():
            session['num_texts'] = DEFAULT_NUM_TEXTS

        if 'texts_indices' in request.values.keys():
            if request.values['texts_indices'] != '':
                parameter_changed = True
                if request.values['texts_indices'] == 'all':
                    session['indices'] = list(range(len(qa_pairs)))
                else:
                    session['indices'] = [int(x) for x in
                                          request.values['texts_indices'].replace(' ', '').split(',') if
                                          x != '']
                session['num_texts'] = len(session['indices'])
                session['manual_indices'] = True
        else:
            if 'manual_indices' not in session.keys():
                session['manual_indices'] = False
    else:
        if 'manual_indices' not in session.keys():
            session['manual_indices'] = False
        if 'scale' not in session.keys():
            session['scale'] = False
        if 'random' not in session.keys():
            session['random'] = False
        if 'num_texts' not in session.keys():
            session['num_texts'] = DEFAULT_NUM_TEXTS

    if not session['manual_indices']:
        if session['random']:
            if old_session_random != session['random'] or old_session_num_texts != session['num_texts']:
                session['indices'] = random.sample(range(0, len(qa_pairs)), session['num_texts'])
        else:
            session['indices'] = list(range(0, session['num_texts']))

    url = ' http://127.0.0.1:5000/neuron/' + str(neuron)
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    response = requests.post(url, headers=headers)
    result = response.json()

    return render_template('neuron.html',
                           neuron=result['neuron'],
                           neuron_num=result['neuron_num'], random=session['random'], indices=session['indices'],
                           scale=session['scale'], activated_words=result['activated_words'],
                           antiactivated_words=result['antiactivated_words'],
                           asked_questions=result['asked_questions'],
                           # plotly
                           pl_ca_heatmap_points=result['pl_ca_heatmap_points'],
                           pl_wa_heatmap_points=result['pl_wa_heatmap_points'],
                           indexed_correct_answers=result['indexed_correct_answers'],
                           indexed_highlighted_correct_answers=result['indexed_highlighted_correct_answers'],
                           indexed_wrong_answers=result['indexed_wrong_answers'],
                           indexed_highlighted_wrong_answers=result['indexed_highlighted_wrong_answers']
                           )
