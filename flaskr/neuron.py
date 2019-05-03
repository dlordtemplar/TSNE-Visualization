import random
import re
import json

import requests
from flask import (
    Blueprint, render_template, request
)
from wtforms import Form, SelectField, IntegerField, TextField, BooleanField, SubmitField, validators, ValidationError

bp = Blueprint('neuron', __name__)

# defaults values for the visualization pages
DEFAULT_NUM_TEXTS = 5
NEURON_TOTAL = 128
NUM_TEXT_MAX = 20
TOTAL_PAIRS = 1790


class NeuronForm(Form):
    def validate_text_indices(form, field):
        if field.data:
            value = field.data.strip()
            if input != '':
                result = re.fullmatch(r'\s*\d*(\s*,\s*\d*)*\s*', value)
                if result is None:
                    raise ValidationError('Invalid input syntax. Example: 0, 1, 2, 3, 4')

    choices = [(i, i) for i in range(NEURON_TOTAL)]
    neuron_num = SelectField('Neuron', coerce=int, choices=choices, default=0)
    num_texts = IntegerField('Number of QA pairs', validators=[validators.NumberRange(min=1, max=NUM_TEXT_MAX)])
    text_indices = TextField('or input indices of QA pairs')
    random = BooleanField('Random texts')
    submit = SubmitField('Submit')


@bp.route('/neuron', strict_slashes=False, methods=['GET', 'POST'])
def display_neuron():
    if request.method == 'POST' and request.form:
        form = NeuronForm(request.form)

        if not form.validate():
            return render_template('neuron.html', form=form)
    else:
        form = NeuronForm()
        form.neuron_num.default = 0
        form.num_texts.default = DEFAULT_NUM_TEXTS
        form.process()

    if form.random.data:
        indices = random.sample(range(TOTAL_PAIRS), form.num_texts.data)
    else:
        if form.text_indices.data:
            to_int_array = list(map(int, form.text_indices.data.split(',')))
            indices = to_int_array
        else:
            indices = list(range(form.num_texts.data))

    url = ' http://127.0.0.1:5000/neuron/'
    headers = {'content-type': 'application/json; charset=utf-8'}
    data = {
        'neuron': form.neuron_num.data,
        'indices': indices,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()

    return render_template('neuron.html',
                           form=form,
                           indices=indices,
                           activated_words=result['activated_words'],
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
