import requests
from flask import (
    Blueprint, render_template, request, redirect, url_for, current_app
)

bp = Blueprint('passage', __name__)


attention_threshold: float = 0  # for displaying highlights in the text


@bp.route('/passage', methods=['GET', 'POST'], strict_slashes=False)
def passage():
    global attention_threshold
    if request.method == 'POST':
        if 'attention_threshold' in request.form:
            attention_threshold = int(float(request.form['attention_threshold']))

        if 'question_input' in request.form:
            received_question = request.form['question_input']
            return redirect(
                url_for('passage.process_question', received_question=received_question))

    url = current_app.config['REST_TORCH_URL'] + 'passage/'
    headers = {'Accept-Charset': 'UTF-8'}
    params = {
        'attention_threshold': attention_threshold
    }
    response = requests.post(url, headers=headers, params=params)
    result = response.json()

    return render_template('passage.html',
                           texts=result['texts'],
                           val=result['val'],
                           question=result['question'],
                           error=result['error'],
                           checked=result['checked'],
                           cos_scores=result['cos_scores'],
                           run_time=result['run_time'],
                           correct_answer_id=result['correct_answer_id'])


@bp.route('/passage/<received_question>', methods=['GET', 'POST'])
def process_question(received_question):
    global attention_threshold
    if 'attention_threshold' in request.form:
        attention_threshold = int(float(request.form['attention_threshold']))

    if 'question_input' in request.form:
        received_question = request.form['question_input']

    url = current_app.config['REST_TORCH_URL'] + 'passage/' + received_question
    headers = {'Accept-Charset': 'UTF-8'}
    params = {
        'attention_threshold': attention_threshold,
        'received_question': received_question
    }
    response = requests.post(url, headers=headers, params=params)
    result = response.json()

    return render_template('passage.html',
                           texts=result['texts'],
                           val=result['val'],
                           question=result['question'],
                           error=result['error'],
                           checked=result['checked'],
                           cos_scores=result['cos_scores'],
                           run_time=result['run_time'],
                           correct_answer_id=result['correct_answer_id'])
