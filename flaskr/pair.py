import math
import os
import pickle
from threading import Lock

# plotting settings, TODO: delete later
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
matplotlib.style.use('ggplot')

import tensorflow as tf
from flask import (
    Blueprint, render_template, request, session
)
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE

from loading_preprocessing_TC import *

sns.set()

bp = Blueprint('qa_pair', __name__)
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
DEFAULT_PERPLEXITY = 5

bp._before_request_lock = Lock()
bp._got_first_request = False


# @bp.before_app_first_request
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
        print('loaded environment')
        print('model:', model)


def load_environment():
    """Load documents index for search engine, pre-trained embeddings, vocabulary, parameters and the model."""
    global embeddings, vocabulary_encoded, params
    global tokenizer, vocabulary_encoded, vocabulary_inv
    global model, cos_qa, model_filename
    global current_loaded
    global qa_pairs, answer_texts
    print("Loading keras visualization setup")

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
        print("\tLoaded model structure from disk")
        # load weights into new model
        model.load_weights("out/data/semeval/models/" + new_model_filename + ".h5")
        print("\tLoaded model weights from disk")
        print("Loaded model from disk")
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


# TODO: just use the points, but remove the matplotlib drawing+saving
def tsne_plot(pair_num, model, labels, perplexity=40):
    """Creates a TSNE model and plots it"""
    cmap = matplotlib.colors.ListedColormap(['red', 'green', 'blue'])
    # TODO: Move out to global var
    num_labels_dict = {'ca': 1, 'wa': 0, 'q': 3}
    num_labels = np.array([num_labels_dict[x] for x in labels])
    tokens = []
    for word in model.keys():
        tokens.append(model[word])

    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=2500, random_state=23,
                      metric="cosine")
    new_values = tsne_model.fit_transform(tokens)

    # TODO: delete
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    trace_question_x = []
    trace_ca_x = []
    trace_wa_x = []
    trace_question_y = []
    trace_ca_y = []
    trace_wa_y = []
    trace_question_text = []
    trace_ca_text = []
    trace_wa_text = []

    for label_index in range(len(labels)):
        if labels[label_index] == 'q':
            trace_question_x.append(new_values[label_index][0])
            trace_question_y.append(new_values[label_index][1])
            trace_question_text.append('Q')
        elif labels[label_index] == 'ca':
            trace_ca_x.append(new_values[label_index][0])
            trace_ca_y.append(new_values[label_index][1])
            trace_ca_text.append('CA' + str(len(trace_ca_x)))
        elif labels[label_index] == 'wa':
            trace_wa_x.append(new_values[label_index][0])
            trace_wa_y.append(new_values[label_index][1])
            trace_wa_text.append('WA' + str(len(trace_wa_x)))

    marker_blue = {
        'size': 20,
        'color': 'rgb(0, 0, 255)'
    }
    marker_green = {
        'size': 20,
        'color': 'rgb(0, 204, 0)'
    }
    marker_red = {
        'size': 20,
        'color': 'rgb(255, 0, 0)'
    }
    trace_question = {
        'name': 'Question',
        'x': trace_question_x,
        'y': trace_question_y,
        'type': 'scatter',
        'mode': 'markers+text',
        'hoverinfo': 'name',
        'text': trace_question_text,
        'textposition': 'top right',
        'marker': marker_blue
    }
    trace_ca = {
        'name': 'Correct answer',
        'x': trace_ca_x,
        'y': trace_ca_y,
        'type': 'scatter',
        'mode': 'markers+text',
        'hoverinfo': 'name',
        'text': trace_ca_text,
        'textposition': 'top right',
        'marker': marker_green
    }
    trace_wa = {
        'name': 'Wrong answer',
        'x': trace_wa_x,
        'y': trace_wa_y,
        'type': 'scatter',
        'mode': 'markers+text',
        'hoverinfo': 'name',
        'text': trace_wa_text,
        'textposition': 'top right',
        'marker': marker_red
    }
    plotly_tsne = [trace_question, trace_ca, trace_wa]

    # TODO: delete
    x = np.array(x)
    y = np.array(y)
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=num_labels, cmap=cmap)
    fig.savefig('flaskr/static/tsne_semeval_siamese_current_qapair' + str(pair_num) + '_' + str(perplexity) + '.png',
                bbox_inches='tight')
    plt.close()

    return plotly_tsne


@bp.route('/pair', defaults={'pair_num': 0}, strict_slashes=False, methods=['GET', 'POST'])
@bp.route('/pair/<int:pair_num>', strict_slashes=False, methods=['GET', 'POST'])
def pair(pair_num):
    global answer_texts, qa_pairs, vocabulary_inv, model

    row = qa_pairs.iloc[pair_num]
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
    else:
        scores_ca = []
        rnn_values_ca = []
    if len(wrong_answers) > 0:
        scores_wa, rnn_values_wa = all_function_deep([q_padded_tokens * len(wrong_answers), wa_padded_tokens])
        neuron_num = rnn_values_wa.shape[-1]
    else:
        scores_wa = []
        rnn_values_wa = []

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

    plotly_tsne = []
    # generate TSNE
    if request.method == 'GET':
        # TODO: delete
        already_exists = False
        path = "flaskr/static/"
        labels = ['q'] + ['ca'] * len(correct_answers) + ['wa'] * len(wrong_answers)
        model_dict_wa = {}
        model_dict_ca = {}
        if len(correct_answers) > 0:
            model_dict_ca = {i + 1: np.max(rnn_values_ca[i, :, :], axis=1) for i in range(len(correct_answers))}
        if len(wrong_answers) > 0:
            model_dict_wa = {i + 1: np.max(rnn_values_wa[i - len(correct_answers), :, :], axis=1) for i in
                             range(len(correct_answers), len(wrong_answers) + len(correct_answers))}
        model_dict = {**model_dict_ca, **model_dict_wa}
        all_function_deep_q, output_function_deep_q = visualize_model_deep(model, True)
        _, rnn_values = all_function_deep_q([q_padded_tokens, [ca_padded_tokens[0]]])
        question_vector = rnn_values[0]
        model_dict[0] = np.max(question_vector, axis=1)
        plotly_tsne = tsne_plot(pair_num, model_dict, labels, session['perplexity'])

    # plotly
    pl_ca_heatmaps = []
    pl_wa_heatmaps = []
    # generate heatmaps
    if request.method == 'GET':
        # plotly
        if len(correct_answers) > 0:
            for idx in range(0, len(ca_tokens)):
                words = [vocabulary_inv[x] for x in ca_tokens[idx]]
                heatmap_points = {'z': rnn_values_ca[idx, -len(ca_tokens[idx]):, :].tolist(),
                                  'y': words,
                                  'type': 'heatmap'}
                pl_ca_heatmaps.append(heatmap_points)
        # Same as above, but for wrong answers
        if len(wrong_answers) > 0:
            for idx in range(0, len(wa_tokens)):
                words = [vocabulary_inv[x] for x in wa_tokens[idx]]
                heatmap_points = {'z': rnn_values_wa[idx, -len(wa_tokens[idx]):, :].tolist(),
                                  'y': words,
                                  'type': 'heatmap'}
                pl_wa_heatmaps.append(heatmap_points)

        if len(correct_answers) > 0:
            for idx in range(0, len(ca_tokens)):
                already_exists = [False, False]
                path = "flaskr/static/"
                for filename in os.listdir(path):
                    if 'current_correct_qapair' + str(pair_num) + '_' + str(idx) + '.png' in filename:
                        already_exists[0] = True
                        print(filename, 'already exists')
                    if 'thumbnail_current_correct_qapair' + str(pair_num) + '_' + str(idx) + '.png' in filename:
                        already_exists[1] = True
                        print(filename, 'already exists')
                    if already_exists[1] and already_exists[0]:
                        break
                words = [vocabulary_inv[x] for x in ca_tokens[idx]]
                if not already_exists[0]:
                    a4_dims = (math.ceil(0.1 * 1 * neuron_num), math.ceil(1 * 0.1 * len(words)))
                    chunk_size = 32
                    for j in range(0, neuron_num, chunk_size):
                        fig, ax = plt.subplots(figsize=a4_dims)
                        sns_plot = sns.heatmap(ax=ax,
                                               data=rnn_values_ca[idx, -len(ca_tokens[idx]):, j:j + chunk_size],
                                               yticklabels=words,
                                               cmap=sns.diverging_palette(220, 20, n=7),
                                               xticklabels=range(j, j + chunk_size))
                        sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=0, fontsize=8)
                        fig.savefig(
                            'flaskr/static/current_correct_qapair' + str(pair_num) + '_' + str(idx) + '_chunk_' + str(
                                j) + '.png',
                            bbox_inches='tight')
                        plt.close()
                if not already_exists[1]:
                    a4_dims = (5, 5)
                    fig, ax = plt.subplots(figsize=a4_dims)
                    sns_plot = sns.heatmap(ax=ax, data=rnn_values_ca[idx, -len(ca_tokens[idx]):, :],
                                           yticklabels=words,
                                           cmap=sns.diverging_palette(220, 20, n=7))
                    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=0, fontsize=8)
                    fig.savefig(
                        'flaskr/static/thumbnail_current_correct_qapair' + str(pair_num) + '_' + str(idx) + '.png',
                        bbox_inches='tight')
                    plt.close()
        if len(wrong_answers) > 0:
            for idx in range(0, len(wa_tokens)):
                already_exists = [False, False]
                path = "flaskr/static/"
                for filename in os.listdir(path):
                    if 'current_wrong_qapair' + str(pair_num) + '_' + str(idx) + '.png' in filename:
                        already_exists[0] = True
                        print(filename, 'already exists')
                    if 'thumbnail_current_wrong_qapair' + str(pair_num) + '_' + str(idx) + '.png' in filename:
                        already_exists[1] = True
                        print(filename, 'already exists')
                    if already_exists[1] and already_exists[0]:
                        break
                words = [vocabulary_inv[x] for x in wa_tokens[idx]]

                if not already_exists[0]:
                    a4_dims = (math.ceil(0.1 * 1 * neuron_num), math.ceil(1 * 0.1 * len(words)))
                    chunk_size = 32
                    for j in range(0, neuron_num, chunk_size):
                        fig, ax = plt.subplots(figsize=a4_dims)
                        sns_plot = sns.heatmap(ax=ax,
                                               data=rnn_values_wa[idx, -len(wa_tokens[idx]):, j:j + chunk_size],
                                               yticklabels=words, xticklabels=range(j, j + chunk_size),
                                               cmap=sns.diverging_palette(220, 20, n=7))
                        sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=0, fontsize=8)
                        fig.savefig(
                            'flaskr/static/current_wrong_qapair' + str(pair_num) + '_' + str(idx) + '_chunk_' + str(
                                j) + '.png',
                            bbox_inches='tight')
                        plt.close()
                if not already_exists[1]:
                    a4_dims = (5, 5)
                    fig, ax = plt.subplots(figsize=a4_dims)
                    sns_plot = sns.heatmap(ax=ax, data=rnn_values_wa[idx, -len(wa_tokens[idx]):, :],
                                           yticklabels=words,
                                           cmap=sns.diverging_palette(220, 20, n=7))
                    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=0, fontsize=8)
                    fig.savefig(
                        'flaskr/static/thumbnail_current_wrong_qapair' + str(pair_num) + '_' + str(idx) + '.png',
                        bbox_inches='tight')
                    plt.close()

    # generate text highlighting based on neuron activity

    highlighted_correct_answers = correct_answers
    highlighted_wrong_answers = wrong_answers
    if session['neuron_display_ca'] != 'None':
        highlighted_correct_answers = highlight_neuron(rnn_values_ca, correct_answers, ca_tokens, session['scale'],
                                                       session['neuron_display_ca'])
    if session['neuron_display_wa'] != 'None':
        highlighted_wrong_answers = highlight_neuron(rnn_values_wa, wrong_answers, wa_tokens, session['scale'],
                                                     session['neuron_display_wa'])

    return render_template('visualization_qapair.html', question=question,
                           highlighted_wrong_answers=highlighted_wrong_answers,
                           highlighted_correct_answers=highlighted_correct_answers,
                           wrong_answers=wrong_answers, correct_answers=correct_answers, i=pair_num,
                           neuron_num=neuron_num,
                           neuron_display_ca=session['neuron_display_ca'],
                           neuron_display_wa=session['neuron_display_wa'], scale=session['scale'],
                           texts_len=len(qa_pairs),
                           scores_ca=scores_ca, scores_wa=scores_wa,
                           # plotly
                           plotly_tsne=plotly_tsne,
                           pl_ca_heatmaps=pl_ca_heatmaps,
                           pl_wa_heatmaps=pl_wa_heatmaps
                           )
