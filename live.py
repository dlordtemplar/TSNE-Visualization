import json

import requests
from flask import (
    Blueprint, render_template, request, current_app
)
from wtforms import Form, SubmitField, IntegerField, TextAreaField, validators, SelectField, BooleanField, ValidationError

bp = Blueprint('live', __name__)

# defaults values for the visualization pages
DEFAULT_PERPLEXITY = 5
NEURON_TOTAL = 128
TOTAL_PAIRS = 1790


def pair_num_check(form, field):
    if isinstance(field.data, int) and (0 > field.data or field.data >= TOTAL_PAIRS):
        raise ValidationError('Field must be >= 0 and < ' + str(TOTAL_PAIRS) + '.')


class LiveForm(Form):
    pair_num = IntegerField('Load pair #', validators=[pair_num_check])
    choices_with_none = [(i, i) for i in range(NEURON_TOTAL)]
    choices_with_none.insert(0, (-1, 'None'))
    perplexity = IntegerField('Perplexity', validators=[validators.NumberRange(min=0)],
                              default=DEFAULT_PERPLEXITY)
    neuron = SelectField('Neuron #', coerce=int, choices=choices_with_none, default=-1)
    scale = BooleanField('Scale neurons', default=False)
    question = TextAreaField('Question')
    correct_answers = TextAreaField('Correct answers')
    wrong_answers = TextAreaField('Wrong answers')
    load = SubmitField('Load...')
    submit = SubmitField('Submit')


@bp.route('/live/random', strict_slashes=False, methods=['GET'])
def live_random():
    form = LiveForm()
    form.process()

    url = current_app.config['REST_KERAS_URL'] + 'live/random'
    headers = {'content-type': 'application/json; charset=utf-8'}
    data = {
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()
    form.question.data = result['question']
    correct_answers = ''
    for ca in result['correct_answers']:
        correct_answers += ca + '\n'
    form.correct_answers.data = correct_answers
    wrong_answers = ''
    for wa in result['wrong_answers']:
        wrong_answers += wa + '\n'
    form.wrong_answers.data = wrong_answers

    return render_template('live_input.html',
                           form=form,
                           question=result['question'],
                           wrong_answers=result['wrong_answers'],
                           correct_answers=result['correct_answers'],
                           pair_num=result['pair_num']
                           )


@bp.route('/live', strict_slashes=False, methods=['GET', 'POST'])
def live():
    if request.method == 'POST' and request.form:
        form = LiveForm(request.form)
        if not form.validate():
            return render_template('live_input.html', form=form)

        if form.load.data:
            url = current_app.config['REST_KERAS_URL'] + 'live/load'
            headers = {'content-type': 'application/json; charset=utf-8'}
            data = {
                'pair_num': form.pair_num.data
            }
            response = requests.post(url, headers=headers, data=json.dumps(data))
            result = response.json()
            form.question.data = result['question']
            correct_answers = ''
            for ca in result['correct_answers']:
                correct_answers += ca + '\n'
            form.correct_answers.data = correct_answers
            wrong_answers = ''
            for wa in result['wrong_answers']:
                wrong_answers += wa + '\n'
            form.wrong_answers.data = wrong_answers

            return render_template('live_input.html',
                                   form=form,
                                   question=result['question'],
                                   wrong_answers=result['wrong_answers'],
                                   correct_answers=result['correct_answers'],
                                   pair_num=result['pair_num']
                                   )
    else:
        form = LiveForm()
        form.process()
        return render_template('live_input.html', form=form)

    url = current_app.config['REST_KERAS_URL'] + 'live'
    headers = {'content-type': 'application/json; charset=utf-8'}
    data = {
        'question': form.question.data,
        'correct_answers': form.correct_answers.data,
        'wrong_answers': form.wrong_answers.data,
        'perplexity': form.perplexity.data,
        'neuron': form.neuron.data,
        'scale': form.scale.data
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()

    return render_template('live.html',
                           form=form,
                           question=result['question'],
                           highlighted_wrong_answers=result['highlighted_wrong_answers'],
                           highlighted_correct_answers=result['highlighted_correct_answers'],
                           wrong_answers=result['wrong_answers'],
                           correct_answers=result['correct_answers'],
                           scores_ca=result['scores_ca'],
                           scores_wa=result['scores_wa'],
                           # plotly
                           plotly_tsne=result['plotly_tsne'],
                           pl_ca_heatmaps=result['pl_ca_heatmaps'],
                           pl_wa_heatmaps=result['pl_wa_heatmaps']
                           )
