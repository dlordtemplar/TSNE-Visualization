import matplotlib.pyplot as plt
import matplotlib

# %matplotlib inline

matplotlib.style.use('ggplot')
# from __future__ import print_function
from keras import backend as K
from keras.engine import Input, Model, InputSpec
from keras.layers import Dense, Activation, Dropout, Lambda
from keras.layers import Embedding, LSTM
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils.data_utils import get_file
from keras.datasets import imdb
import tensorflow as tf

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.1
from keras import backend as K

tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
import numpy as np
import random
import sys
import pdb
import scipy
import pickle
import keras
from sklearn.utils import class_weight
import gensim

# sys.path.append("../../codes")
from loading_preprocessing_TC import *
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set()
from sklearn.manifold import TSNE
from keras.models import model_from_json

DATA_FOLDER = "../out/data/semeval/"
max_length = 200

# # Model reconstruction from JSON file
with open(DATA_FOLDER + 'models/model_visualization_siamesedeeplstm.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(DATA_FOLDER + 'models/model_visualization_siamesedeeplstm.h5')

print(model.layers)


def visualize_model(model, question_lstm=True):
    recurrent_layer = model.get_layer('SharedLSTM1')
    output_layer = model.layers[-1]

    inputs = []
    inputs.extend(model.inputs)

    outputs = []
    outputs.extend(model.outputs)
    if question_lstm:
        outputs.append(recurrent_layer.get_output_at(1))
    else:
        outputs.append(recurrent_layer.get_output_at(0))

    all_function = K.function(inputs, outputs)
    output_function = K.function([output_layer.input], model.outputs)
    return all_function, output_function


def get_compare_embeddings(original_embeddings, tuned_embeddings, vocab, dimreduce_type="tsne", random_state=0):
    """ Compare embeddings drift. """
    if dimreduce_type == "pca":
        from sklearn.decomposition import PCA
        dimreducer = PCA(n_components=2, random_state=random_state)
    elif dimreduce_type == "tsne":
        from sklearn.manifold import TSNE
        dimreducer = TSNE(n_components=2, random_state=random_state)
    else:
        raise Exception("Wrong dimreduce_type.")

    reduced_original = dimreducer.fit_transform(original_embeddings)
    reduced_tuned = dimreducer.fit_transform(tuned_embeddings)

    def compare_embeddings(word):
        if word not in vocab:
            return None
        word_id = vocab[word]
        original_x, original_y = reduced_original[word_id, :]
        tuned_x, tuned_y = reduced_tuned[word_id, :]
        return original_x, original_y, tuned_x, tuned_y

    return compare_embeddings


EMBEDDING_FOLDER = "../resources/embeddings/"
EMBEDDING_DIM = 300
GLOVE_PATH = EMBEDDING_FOLDER + "glove.6B." + str(EMBEDDING_DIM) + "d.txt"
WORD2VEC_PATH = EMBEDDING_FOLDER + "GoogleNews-vectors-negative300.bin"
DATA_FOLDER = "../out/data/semeval/"
max_length = 200

answer_texts_train = pickle.load(open(DATA_FOLDER + "answer_texts_train_Ling.p", "rb"))
train = pickle.load(open(DATA_FOLDER + "train-expanded_Ling.p", "rb"))
answer_texts_dev = pickle.load(open(DATA_FOLDER + "answer_texts_dev_Ling.p", "rb"))
dev = pickle.load(open(DATA_FOLDER + "dev-Labels_Ling.p", "rb"))

answer_texts_train = answer_texts_train.set_index('answer_id')

correct_answer_ids = set(train['answer_id'].values)
incorrect_answer_ids = [x for x in answer_texts_train.index.values if x not in correct_answer_ids]
incorrect_answer_texts = answer_texts_train.loc[incorrect_answer_ids]

texts = list([x.text for x in train['question'].values]) + list([x.text for x in train['answer'].values]) + list(
    [x.text for x in incorrect_answer_texts['answer'].values])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

vocabulary = tokenizer.word_index
vocabulary_inv = {v: k for k, v in vocabulary.items()}
embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True, unicode_errors='ignore')

embedding_matrix = np.zeros((len(vocabulary) + 2, EMBEDDING_DIM))  # oov + pad vector
oov_vector = np.zeros(EMBEDDING_DIM)
for word, i in vocabulary.items():
    if word in embeddings_index.wv.vocab:
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector
    elif word.lower() in embeddings_index.wv.vocab:
        embedding_vector = embeddings_index[word.lower()]
        embedding_matrix[i] = embedding_vector
    elif word.capitalize() in embeddings_index.wv.vocab:
        embedding_vector = embeddings_index[word.capitalize()]
        embedding_matrix[i] = embedding_vector
    elif word.lower().capitalize() in embeddings_index.wv.vocab:
        embedding_vector = embeddings_index[word.lower().capitalize()]
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = oov_vector

questions = []
wrong_answers = []
for idx, row in train.iterrows():
    for y in row['pool']:
        wrong_answers.append(answer_texts_train.loc[y]['answer'].text)
        questions.append(row['question'].text)
correct_answers = []
for idx, row in train.iterrows():
    correct_answers.append(row['answer'].text)
    questions.append(row['question'].text)

data = [(x, 0) for x in wrong_answers] + [(x, 1) for x in correct_answers]

data_answers = [x[0] for x in data]
data_questions = questions
data_targets = [x[1] for x in data]

X_train_a_text, X_validation_a_text, X_train_q_text, X_validation_q_text, Y_train, Y_validation = train_test_split(
    data_answers, data_questions, data_targets, test_size=0.2)

X_train_a = tokenizer.texts_to_sequences(X_train_a_text)
X_validation_a = tokenizer.texts_to_sequences(X_validation_a_text)
X_train_q = tokenizer.texts_to_sequences(X_train_q_text)
X_validation_q = tokenizer.texts_to_sequences(X_validation_q_text)

X_train_a = pad_sequences(X_train_a, maxlen=max_length, value=embedding_matrix.shape[0] - 1)
X_validation_a = pad_sequences(X_validation_a, maxlen=max_length, value=embedding_matrix.shape[0] - 1)
X_train_q = pad_sequences(X_train_q, maxlen=max_length, value=embedding_matrix.shape[0] - 1)
X_validation_q = pad_sequences(X_validation_q, maxlen=max_length, value=embedding_matrix.shape[0] - 1)

Y_train = np.array(Y_train)
Y_validation = np.array(Y_validation)

# First LSTM

a4_dims = (8, 8)

all_function, output_function = visualize_model(model, False)
scores, rnn_values = all_function([X_train_a[:1], X_train_q[:1]])
print(scores.shape, rnn_values.shape)
words = [vocabulary_inv[i] for i in X_train_a[0] if i != len(vocabulary_inv) + 1]
fig, ax = plt.subplots(figsize=a4_dims)
sns_plot = sns.heatmap(ax=ax, data=rnn_values[0, -len(words):, :].T, xticklabels=words,
                       cmap=sns.diverging_palette(220, 20, n=7))
fig.savefig('../out/static/rnn_weights_semeval_siamese_a_LSTM1.png', bbox_inches='tight')

all_function, output_function = visualize_model(model, True)
scores, rnn_values = all_function([X_train_a[:1], X_train_q[:1]])
print(scores.shape, rnn_values.shape)
words = [vocabulary_inv[i] for i in X_train_q[0] if i != len(vocabulary_inv) + 1]
fig, ax = plt.subplots(figsize=a4_dims)
sns_plot = sns.heatmap(ax=ax, data=rnn_values[0, -len(words):, :].T, xticklabels=words,
                       cmap=sns.diverging_palette(220, 20, n=7))
fig.savefig('../out/static/rnn_weights_semeval_siamese_q_LSTM1.png', bbox_inches='tight')


# Second LSTM

def visualize_model_deep(model, question_lstm=True):
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

    all_function = K.function(inputs, outputs)
    output_function = K.function([output_layer.input], model.outputs)
    return all_function, output_function


all_function_deep, output_function_deep = visualize_model_deep(model, False)
scores, rnn_values = all_function_deep([X_train_a[:1], X_train_q[:1]])
print(scores.shape, rnn_values.shape)
words = [vocabulary_inv[i] for i in X_train_a[0] if i != len(vocabulary_inv) + 1]
fig, ax = plt.subplots(figsize=a4_dims)
sns_plot = sns.heatmap(ax=ax, data=rnn_values[0, -len(words):, :].T, xticklabels=words,
                       cmap=sns.diverging_palette(220, 20, n=7))
fig.savefig('../out/static/rnn_weights_semeval_siamese_deep_a_LSTM2.png', bbox_inches='tight')

all_function_deep, output_function_deep = visualize_model_deep(model, True)
scores, rnn_values = all_function_deep([X_train_a[:1], X_train_q[:1]])
print(scores.shape, rnn_values.shape)
words = [vocabulary_inv[i] for i in X_train_q[0] if i != len(vocabulary_inv) + 1]
fig, ax = plt.subplots(figsize=a4_dims)
sns_plot = sns.heatmap(ax=ax, data=rnn_values[0, -len(words):, :].T, xticklabels=words,
                       cmap=sns.diverging_palette(220, 20, n=7))
fig.savefig('../out/static/rnn_weights_semeval_siamese_deep_q_LSTM2.png', bbox_inches='tight')

# Weights histogram

weights = model.get_layer('SharedLSTM1').get_weights()
fig, axs = plt.subplots()
axs.hist(weights, bins='auto')
fig.savefig('../out/static/histogram_weights_semeval_siamese_LSTM1.png', bbox_inches='tight')

weights = model.get_layer('SharedLSTM2').get_weights()
fig, axs = plt.subplots()
axs.hist(weights, bins='auto')
fig.savefig('../out/static/histogram_weights_semeval_siamese_deep_LSTM2.png', bbox_inches='tight')

# Attention

example_question = [train['question'][0].text]
example_correct_answers = [x.text for x in answer_texts_train.loc[train['answer_ids'][0]]['answer']]
example_wrong_answers = [x.text for x in answer_texts_train.loc[train['pool'][0]]['answer']]
example_answers = example_correct_answers + example_wrong_answers
plot_data_q = tokenizer.texts_to_sequences(example_question * len(example_answers))
plot_data_q = pad_sequences(plot_data_q, maxlen=max_length, value=embedding_matrix.shape[0] - 1)
plot_data_a = tokenizer.texts_to_sequences(example_answers)
plot_data_a = pad_sequences(plot_data_a, maxlen=max_length, value=embedding_matrix.shape[0] - 1)


def visualize_model_attention(model):
    attention_layer = model.get_layer('attention_scores')
    output_layer = model.layers[-1]

    inputs = []
    inputs.extend(model.inputs)

    outputs = []
    outputs.extend(model.outputs)
    outputs.append(attention_layer.output)

    all_function = K.function(inputs, outputs)
    output_function = K.function([output_layer.input], model.outputs)
    return all_function, output_function


all_function, output_function = visualize_model_attention(model)

# attention_scores has a shape #texts x #words (including padding) x attention dimensionality
scores, attention_scores = all_function(([plot_data_a, plot_data_q]))
print(scores.shape, attention_scores.shape)

# By default, padding in Keras is done before the sequence, so the attention scores on words are:

tokens_num = tokenizer.texts_to_sequences([example_answers[0]])[0]
example_text_length = len(tokens_num)
tokens_words = [vocabulary_inv[x] for x in tokens_num]

atts = attention_scores[0, -example_text_length:, :].flatten()

fig, ax = plt.subplots()
bar_width = 0.35
rects1 = ax.bar(tokens_words, atts, bar_width, color='b',
                label='Attention')
fig.savefig('../out/static/histogram_attention_semeval_siamese.png', bbox_inches='tight')


# TSNE on Sentence Embeddings

def tsne_plot(model, labels, filename, perplexity=40):
    "Creates and TSNE model and plots it"
    tokens = []
    cmap = matplotlib.colors.ListedColormap(['red', 'green', 'blue'])
    num_labels_dict = {'ca': 1, 'wa': 0, 'q': 3}
    num_labels = [num_labels_dict[x] for x in labels]
    for word in model.keys():
        tokens.append(model[word])

    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=num_labels, cmap=cmap)
    plt.savefig(''.join(['../out/static/', filename, '.png']), bbox_inches='tight')


all_function_deep, output_function_deep = visualize_model_deep(model, True)
scores, rnn_values = all_function_deep([plot_data_a, plot_data_q])
question_vector = rnn_values[0]
# rnn_values[0]
all_function_deep, output_function_deep = visualize_model_deep(model, False)
scores, rnn_values = all_function_deep([plot_data_a, plot_data_q])
a4_dims = (15, 5)
print(scores.shape, rnn_values.shape)
words = [vocabulary_inv[i] for i in plot_data_a[0] if i != len(vocabulary_inv) + 1]
fig, axs = plt.subplots(1, 2, figsize=a4_dims)
sns_plot = sns.heatmap(ax=axs[0], data=rnn_values[0, -len(words):, :].T, xticklabels=words,
                       cmap=sns.diverging_palette(220, 20, n=7), vmin=-1, vmax=1)
words = [vocabulary_inv[i] for i in plot_data_a[1] if i != len(vocabulary_inv) + 1]
sns_plot = sns.heatmap(ax=axs[1], data=rnn_values[1, -len(words):, :].T, xticklabels=words,
                       cmap=sns.diverging_palette(220, 20, n=7), vmin=-1, vmax=1)
fig.savefig('../out/static/sentence embeddings.png', bbox_inches='tight')

model_dict = {i + 1: np.max(rnn_values[i, :, :], axis=1) for i, x in enumerate(plot_data_a)}
for i, x in model_dict.items():
    print(example_answers[i - 1])
    print(i - 1 <= len(example_correct_answers), scipy.spatial.distance.cosine(np.max(question_vector, axis=1), x))

model_dict[0] = np.max(question_vector, axis=1)
labels = ['q'] + ['ca'] * len(example_correct_answers) + ['wa'] * len(example_wrong_answers)
tsne_plot(model_dict, labels, 'tsne_semeval_siamese_40', 40)

tsne_plot(model_dict, labels, 'tsne_semeval_siamese_5', 5)

# Most activated words per neuron

num_texts = 100
_, rnn_values = all_function_deep([X_train_a[:num_texts], X_train_q[:num_texts]])
texts = [[vocabulary_inv[x] for x in y if x != 30747] for y in X_train_a[:num_texts]]
padded_texts = [
    [token for token in x] + [0] * (200 - len([token for token in x])) if len([token for token in x]) < 200 else x[:200]
    for x in texts]
print(rnn_values.shape)

neuron_total_num = 50
activated_words = []
activated_words_values = []
antiactivated_words = []
antiactivated_words_values = []
activation_per_word_data = pd.DataFrame()
activation_per_word_data['text'] = [token for x in padded_texts for token in x]
for neuron_num in range(0, neuron_total_num):
    activation_per_word_data[neuron_num] = rnn_values[:, :, neuron_num].flatten()[::-1]
    p_high = np.percentile(activation_per_word_data[neuron_num], 90)
    p_low = np.percentile(activation_per_word_data[neuron_num], 10)
    activated_words.append(activation_per_word_data[activation_per_word_data[neuron_num] >= p_high]['text'].values)
    activated_words_values.append(
        activation_per_word_data[activation_per_word_data[neuron_num] >= p_high][neuron_num].values)
    antiactivated_words.append(activation_per_word_data[activation_per_word_data[neuron_num] <= 0]['text'].values)
    antiactivated_words_values.append(
        activation_per_word_data[activation_per_word_data[neuron_num] <= 0][neuron_num].values)
    num_print = 10

for neuron_num in range(0, neuron_total_num):
    print('Neuron ' + str(neuron_num) + ' max: ' + str(max(rnn_values[:, :, neuron_num].flatten())) + ' min: ' +
          str(min(rnn_values[:, :, neuron_num].flatten())) + ' median: ' +
          str(np.median(rnn_values[:, :, neuron_num].flatten())))
    print(activated_words[neuron_num][:num_print])
    print(activated_words_values[neuron_num][:num_print])
    print(sum([isinstance(x, int) for x in activated_words[neuron_num]]) / len(activated_words[neuron_num]))
    print(antiactivated_words[neuron_num][:num_print])
    print(antiactivated_words_values[neuron_num][:num_print])
    print(sum([isinstance(x, int) for x in antiactivated_words[neuron_num]]) / len(antiactivated_words[neuron_num]))
    print('\n')
