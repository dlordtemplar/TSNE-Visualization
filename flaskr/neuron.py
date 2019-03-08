import pickle
import random
from threading import Lock

import tensorflow as tf
from flask import (
    Blueprint, render_template, request, session
)
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

from loading_preprocessing_TC import *

bp = Blueprint('neuron', __name__)
MODEL_DIR = 'out/data/semeval/models'
DATASET_PATH = 'resources/datasets/semeval/train/'

# data
qa_pairs = None

# model
model = None
graph = None

# deep learning settings
MAX_LENGTH = 200

# defaults values for the visualization pages
DEFAULT_NUM_TEXTS = 5

bp._before_request_lock = Lock()
bp._got_first_request = False


# @bp.before_request
# def init():
#     if bp._got_first_request:
#         return
#     with bp._before_request_lock:
#         if bp._got_first_request:
#             return
#         bp._got_first_request = True
#
#         # first request, execute what you need.
#         setup()


@bp.before_app_first_request
def setup():
    global model
    if not model:
        model = load_environment()


def load_environment():
    """Load documents index for search engine, pre-trained embeddings, vocabulary, parameters and the model."""
    global embeddings, vocabulary_encoded, params
    global tokenizer, vocabulary_encoded, vocabulary_inv
    global model, cos_qa, model_filename
    global current_loaded
    global qa_pairs, answer_texts

    with open('flaskr/tokenizer.p', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('flaskr/embedding_matrix.p', 'rb') as handle:
        embeddings = pickle.load(handle)
    vocabulary_encoded = tokenizer.word_index
    vocabulary_inv = {v: k for k, v in vocabulary_encoded.items()}
    model = load_model('model_visualization_siamesedeeplstm')
    qa_pairs, answer_texts = load_data()
    return model


def load_model(new_model_filename):
    """Load a pretrained model from PyTorch / Keras checkpoint.
    Args:
        new_model_filename (string): the name of the model used when saving its weights and architecture to
        either a binary (PyTorch) or a .h5 and a .json (Keras)

    Returns:
        error (string): The error message displayed to a user. If empty, counts as no error.
    """
    global model, model_filename
    print("Loading model:", new_model_filename)
    try:
        json_file = open("out/data/semeval/models/" + new_model_filename + '.json',
                         'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        global graph
        graph = tf.get_default_graph()
        # load weights into new model
        model.load_weights("out/data/semeval/models/" + new_model_filename + ".h5")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_filename = new_model_filename
        return model
    except Exception as e:
        print(e)
        error = "<div class=\"alert alert-warning\"> Sorry, there is something wrong with the model: <br> " + str(
            e) + "</div>"
        return error


def load_data():
    """Load SemEval 2017 files from .xml and convert them into pandas dataframes.
    Args:

    Returns:
        train (pandas dataframe): QA-pairs in a format question - correct answers (ids) - pool (ids; incorrect answers).
        If there are multiple correct answers to a single question, they are split into multiple QA - pairs.
        answer_texts_train (pandas dataframe): answer texts and their ids.
    """
    files = [DATASET_PATH + 'SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
             DATASET_PATH + 'SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml']
    train_xml = read_xml(files)
    train, answer_texts_train = xml2dataframe_Labels(train_xml, 'train')
    answer_texts_train.set_index('answer_id', drop=False, inplace=True)
    return train, answer_texts_train


def prepare_data(texts):
    """Tokenize texts and pad resulting sequences of words using Keras functions."""
    global tokenizer, embeddings
    tokens = tokenizer.texts_to_sequences(texts)
    padded_tokens = pad_sequences(tokens, maxlen=MAX_LENGTH, value=embeddings.shape[0] - 1)
    return tokens, padded_tokens


def visualize_model_deep(model, question_lstm=True):
    """Retrieve weights of the second shared LSTM to visualize neuron activations."""
    recurrent_layer = model.get_layer('SharedLSTM2')
    output_layer = model.layers[-1]

    inputs = []
    inputs.extend(model.inputs)

    outputs = []
    outputs.extend(model.outputs)
    if question_lstm:
        outputs.append(recurrent_layer.get_output_at(1))
    else:
        outputs.append(recurrent_layer.get_output_at(0))

    global graph
    with graph.as_default():
        all_function = K.function(inputs, outputs)
        output_function = K.function([output_layer.input], model.outputs)
    return all_function, output_function


def highlight_neuron(rnn_values, texts, tokens, scale, neuron):
    """Generate HTML code where each word is highlighted according to a given neuron activity on it."""
    tag_string = "<span data-toggle=\"tooltip\" title=\"SCORE\"><span style = \"background-color: rgba(COLOR, OPACITY);\">WORD</span></span>"
    old_texts = texts
    texts = []
    for idx in range(0, len(old_texts)):
        current_neuron_values = rnn_values[idx, :, neuron]
        current_neuron_values = current_neuron_values[-len(tokens[idx]):]
        words = [vocabulary_inv[x] for x in tokens[idx]]
        current_strings = []
        if scale:
            scaled = [
                ((x - min(current_neuron_values)) * (2) / (
                        max(current_neuron_values) - min(current_neuron_values))) + (
                    -1)
                for x in current_neuron_values]
        else:
            scaled = current_neuron_values
        for score, word, scaled_score in zip(current_neuron_values, words, scaled):
            if score > 0:
                color = '195, 85, 58'
            else:
                color = '63, 127, 147'
            current_string = tag_string.replace('SCORE', str(score)).replace('WORD', word).replace('OPACITY', str(
                abs(scaled_score))).replace('COLOR', color)
            current_strings.append(current_string)
        texts.append(' '.join(current_strings))
    return texts


@bp.route('/neuron', defaults={'neuron': 0}, strict_slashes=False, methods=['GET', 'POST'])
@bp.route('/neuron/<int:neuron>', strict_slashes=False, methods=['GET', 'POST'])
def display_neuron(neuron):
    global answer_texts, qa_pairs, vocabulary_inv, model

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
        print(request.values)

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

    # Start actual visualization
    all_highlighted_wrong_answers = []
    all_wrong_answers = []

    min_ca = 1
    min_wa = 1
    max_ca = -1
    max_wa = -1

    activated_words = []
    activated_words_values = []
    antiactivated_words = []
    antiactivated_words_values = []

    activation_per_word_data = {}

    # plotly
    pl_ca_heatmaps_indexed = {}
    pl_wa_heatmaps_indexed = {}
    indexed_correct_answers = {}
    indexed_highlighted_correct_answers = {}
    indexed_wrong_answers = {}
    indexed_highlighted_wrong_answers = {}

    for i in session['indices']:
        print('Generating activations for QA pair', i)
        row = qa_pairs.iloc[i]
        correct_answers = answer_texts.loc[row['answer_ids']]['answer'].values
        wrong_answers = answer_texts.loc[row['pool']]['answer'].values
        question = row['question']
        q_tokens, q_padded_tokens = prepare_data([question])
        ca_tokens, ca_padded_tokens = prepare_data(correct_answers)
        wa_tokens, wa_padded_tokens = prepare_data(wrong_answers)
        all_function_deep, output_function_deep = visualize_model_deep(model, False)
        if len(correct_answers) > 0:
            scores_ca, rnn_values_ca = all_function_deep([q_padded_tokens * len(correct_answers), ca_padded_tokens])
            neuron_num = rnn_values_ca.shape[-1]
            all_values_ca = rnn_values_ca[:, :, neuron:neuron + 1]
            if np.min(all_values_ca) < min_ca:
                min_ca = np.min(all_values_ca)
            if np.max(all_values_ca) > max_ca:
                max_ca = np.max(all_values_ca)
            highlighted_correct_answers = highlight_neuron(rnn_values_ca, correct_answers, ca_tokens,
                                                           session['scale'],
                                                           neuron)

            if i not in indexed_highlighted_correct_answers:
                indexed_highlighted_correct_answers[i] = [highlighted_correct_answers]
            else:
                indexed_highlighted_correct_answers[i].append(highlighted_correct_answers)

            current_ca = [[vocabulary_inv[x] for x in ca_tokens[idx]] for idx in range(len(ca_tokens))]
            if i not in indexed_correct_answers:
                indexed_correct_answers[i] = current_ca
            else:
                indexed_correct_answers[i].append(current_ca)

            activation_per_word_data['ca_firings' + str(i)] = rnn_values_ca[:, :, neuron].flatten()
            activation_per_word_data['ca_text' + str(i)] = [
                vocabulary_inv[token] if token in vocabulary_inv.keys() else '<pad>' for x in ca_padded_tokens for
                token
                in x]
        else:
            if i not in indexed_highlighted_correct_answers:
                indexed_highlighted_correct_answers[i] = []
            else:
                indexed_highlighted_correct_answers[i].append([])

            if i not in indexed_correct_answers:
                indexed_correct_answers[i] = []
            else:
                indexed_correct_answers[i].append([])

            activation_per_word_data['ca_text' + str(i)] = []
            activation_per_word_data['ca_firings' + str(i)] = []

        if len(wrong_answers) > 0:
            scores_wa, rnn_values_wa = all_function_deep([q_padded_tokens * len(wrong_answers), wa_padded_tokens])
            neuron_num = rnn_values_wa.shape[-1]
            all_values_wa = rnn_values_wa[:, :, neuron:neuron + 1]
            if np.min(all_values_wa) < min_wa:
                min_wa = np.min(all_values_wa)
            if np.max(all_values_wa) > max_wa:
                max_wa = np.max(all_values_wa)
            highlighted_wrong_answers = highlight_neuron(rnn_values_wa, wrong_answers, wa_tokens, session['scale'],
                                                         neuron)
            all_highlighted_wrong_answers.append(highlighted_wrong_answers)

            if i not in indexed_highlighted_wrong_answers:
                indexed_highlighted_wrong_answers[i] = [highlighted_wrong_answers]
            else:
                indexed_highlighted_wrong_answers[i].append(highlighted_wrong_answers)

            current_wa = [np.array([vocabulary_inv[x] for x in wa_tokens[idx]]) for idx in range(len(wa_tokens))]
            if i not in indexed_wrong_answers:
                indexed_wrong_answers[i] = current_wa
            else:
                indexed_wrong_answers[i].append(current_wa)

            activation_per_word_data['wa_firings' + str(i)] = rnn_values_wa[:, :, neuron].flatten()
            activation_per_word_data['wa_text' + str(i)] = [
                vocabulary_inv[token] if token in vocabulary_inv.keys() else '<pad>' for x in wa_padded_tokens for
                token
                in x]
        else:
            all_highlighted_wrong_answers.append([])

            if i not in indexed_highlighted_wrong_answers:
                indexed_highlighted_wrong_answers[i] = []
            else:
                indexed_highlighted_wrong_answers[i].append([])

            all_wrong_answers.append([])

            if i not in indexed_wrong_answers:
                indexed_wrong_answers[i] = []
            else:
                indexed_wrong_answers[i].append([])

            activation_per_word_data['wa_text' + str(i)] = []
            activation_per_word_data['wa_firings' + str(i)] = []

        # Point generation for correct answers
        if parameter_changed or request.method == 'GET':
            if len(correct_answers) > 0:
                for idx in range(0, len(ca_tokens)):
                    words = [vocabulary_inv[x] for x in ca_tokens[idx]]
                    heatmap_points = {'z': rnn_values_ca[idx, -len(ca_tokens[idx]):, neuron:neuron + 1].tolist(),
                                      'y': words,
                                      'type': 'heatmap'}
                    if i in pl_ca_heatmaps_indexed:
                        pl_ca_heatmaps_indexed[i].append(heatmap_points)
                    else:
                        pl_ca_heatmaps_indexed[i] = [heatmap_points]

            # Same as above, but for wrong answers
            if len(wrong_answers) > 0:
                for idx in range(0, len(wa_tokens)):
                    words = [vocabulary_inv[x] for x in wa_tokens[idx]]
                    heatmap_points = {'z': rnn_values_wa[idx, -len(wa_tokens[idx]):, neuron:neuron + 1].tolist(),
                                      'y': words,
                                      'type': 'heatmap'}
                    if i in pl_wa_heatmaps_indexed:
                        pl_wa_heatmaps_indexed[i].append(heatmap_points)
                    else:
                        pl_wa_heatmaps_indexed[i] = [heatmap_points]

    all_firings = [x for i in session['indices'] for x in activation_per_word_data['wa_firings' + str(i)]] + [x for
                                                                                                              i in
                                                                                                              session[
                                                                                                                  'indices']
                                                                                                              for x
                                                                                                              in
                                                                                                              activation_per_word_data[
                                                                                                                  'ca_firings' + str(
                                                                                                                      i)]]
    all_tokens = [x for i in session['indices'] for x in activation_per_word_data['wa_text' + str(i)]] + [x for i in
                                                                                                          session[
                                                                                                              'indices']
                                                                                                          for x in
                                                                                                          activation_per_word_data[
                                                                                                              'ca_text' + str(
                                                                                                                  i)]]
    all_firings = np.array(all_firings)
    all_tokens = np.array(all_tokens)
    p_high = np.percentile([x for i, x in enumerate(all_firings) if all_tokens[i] != '<pad>'], 90)
    p_low = np.percentile([x for i, x in enumerate(all_firings) if all_tokens[i] != '<pad>'], 10)

    for ind, x in enumerate(all_firings):
        if x >= p_high:
            activated_words.append(all_tokens[ind])
            activated_words_values.append(x)
        elif x <= p_low:
            antiactivated_words.append(all_tokens[ind])
            antiactivated_words_values.append(x)

    seen = set()
    activated_words = [x for x in activated_words if not (x in seen or seen.add(x))]
    seen = set()
    antiactivated_words = [x for x in antiactivated_words if not (x in seen or seen.add(x))]
    asked_questions = qa_pairs['question']

    return render_template('neuron.html',
                           neuron=neuron,
                           neuron_num=neuron_num, random=session['random'], indices=session['indices'],
                           scale=session['scale'], activated_words=activated_words,
                           antiactivated_words=antiactivated_words,
                           asked_questions=asked_questions,
                           # plotly
                           pl_ca_heatmap_points=pl_ca_heatmaps_indexed,
                           pl_wa_heatmap_points=pl_wa_heatmaps_indexed,
                           indexed_correct_answers=indexed_correct_answers,
                           indexed_highlighted_correct_answers=indexed_highlighted_correct_answers,
                           indexed_wrong_answers=indexed_wrong_answers,
                           indexed_highlighted_wrong_answers=indexed_highlighted_wrong_answers
                           )
