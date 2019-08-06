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


# class PairForm(Form):
#     choices_pair = [(i, i) for i in range(TOTAL_PAIRS)]
#     pair_num = SelectField('Pair', coerce=int, choices=choices_pair, default=0)
#     choices_with_none = [(i, i) for i in range(NEURON_TOTAL)]
#     choices_with_none.insert(0, (-1, 'None'))
#     ca_neuron = SelectField('Neuron #', coerce=int, choices=choices_with_none, default=-1)
#     wa_neuron = SelectField('Neuron #', coerce=int, choices=choices_with_none, default=-1)
#     perplexity = IntegerField('Perplexity', validators=[validators.NumberRange(min=0)],
#                               default=DEFAULT_PERPLEXITY)
#     scale = BooleanField('Scale neurons')
#     submit = SubmitField('Submit')


@bp.route('/pos', strict_slashes=False, methods=['GET', 'POST'])
def pair():
    # if request.method == 'POST' and request.form:
    #     form = PairForm(request.form)
    #     if not form.validate():
    #         return render_template('pair.html', form=form)
    # else:
    #     form = PairForm()
    #     form.process()

    url = current_app.config['REST_KERAS_URL'] + 'pos/'
    headers = {'content-type': 'application/json; charset=utf-8'}
    data = {
        # 'pair_num': form.pair_num.data,
        # 'perplexity': form.perplexity.data,
        # 'ca_neuron': form.ca_neuron.data,
        # 'wa_neuron': form.wa_neuron.data,
        # 'scale': form.scale.data
    }
    response = requests.post(url, headers=headers) #, data=json.dumps(data))
    result = response.json()
    print(result)

    return render_template('pos.html',
                           plotly_pos=result
                           )
