import requests
from flask import (
    Blueprint, render_template, request, session
)

bp = Blueprint('pair', __name__)
MODEL_DIR = 'out/data/semeval/models'
DATASET_PATH = 'resources/datasets/semeval/train/'

# deep learning settings
MAX_LENGTH = 200

# defaults values for the visualization pages
DEFAULT_NUM_TEXTS = 5
DEFAULT_PERPLEXITY = 5


@bp.route('/pair', defaults={'pair_num': 0}, strict_slashes=False, methods=['GET', 'POST'])
@bp.route('/pair/<int:pair_num>', strict_slashes=False, methods=['GET', 'POST'])
def pair(pair_num):
    if request.method == 'POST':
        if 'perplexity' in request.values.keys():
            if request.values['perplexity'] != '':
                session['perplexity'] = int(request.values['perplexity'])
            else:
                session['perplexity'] = DEFAULT_PERPLEXITY
        elif 'perplexity' not in session.keys():
            session['perplexity'] = DEFAULT_PERPLEXITY

        if 'scale' in request.values.keys():
            session['scale'] = True
        else:
            if 'scale' in session.keys():
                if session['scale']:
                    session['scale'] = False
            else:
                session['scale'] = False

        if 'neuron_num_ca' in request.values.keys():
            if request.values.get("neuron_num_ca") != 'None':
                session['neuron_display_ca'] = int(request.values.get("neuron_num_ca"))
            else:
                session['neuron_display_ca'] = 'None'
        if 'neuron_num_wa' in request.values.keys():
            if request.values.get("neuron_num_wa") != 'None':
                session['neuron_display_wa'] = int(request.values.get("neuron_num_wa"))
            else:
                session['neuron_display_wa'] = 'None'
    else:
        if 'perplexity' not in session.keys():
            session['perplexity'] = DEFAULT_PERPLEXITY
        if 'neuron_display_ca' not in session.keys():
            session['neuron_display_ca'] = 'None'
        if 'neuron_display_wa' not in session.keys():
            session['neuron_display_wa'] = 'None'
        if 'scale' not in session.keys():
            session['scale'] = False

    url = ' http://127.0.0.1:5000/pair/' + str(pair_num)
    # headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    headers = {'Accept-Charset': 'UTF-8'}
    params = {
        'perplexity': session['perplexity'],
        'neuron_display_ca': session['neuron_display_ca'],
        'neuron_display_wa': session['neuron_display_wa'],
        'scale': session['scale']
    }
    response = requests.post(url, headers=headers, params=params)
    result = response.json()

    return render_template('pair.html', question=result['question'],
                           highlighted_wrong_answers=result['highlighted_wrong_answers'],
                           highlighted_correct_answers=result['highlighted_correct_answers'],
                           wrong_answers=result['wrong_answers'],
                           correct_answers=result['correct_answers'],
                           pair_num=result['pair_num'],
                           neuron_num=result['neuron_num'],
                           neuron_display_ca=session['neuron_display_ca'],
                           neuron_display_wa=session['neuron_display_wa'],
                           scale=session['scale'],
                           texts_len=result['texts_len'],
                           scores_ca=result['scores_ca'],
                           scores_wa=result['scores_wa'],
                           # plotly
                           plotly_tsne=result['plotly_tsne'],
                           pl_ca_heatmaps=result['pl_ca_heatmaps'],
                           pl_wa_heatmaps=result['pl_wa_heatmaps']
                           )
