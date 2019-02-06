# SiameseLSTM_visualization_semeval.ipynb

import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')
# %matplotlib inline
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

with open('tokenizer.p', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('embedding_matrix.p', 'wb') as handle:
    pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
class_weights = {i: x for i, x in enumerate(list(class_weights))}
print(class_weights)

adam = keras.optimizers.Adam(clipnorm=1.)

SEED = 42
MAX_LENGTH = 200
M = 96
N = 64
gaussian_noise = 0
unidirectional = False
trainable_embeddings = False
mean_pooling = False
initializer = keras.initializers.he_normal(seed=SEED)
dropout = False
embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=MAX_LENGTH, trainable=trainable_embeddings)

a_input = Input(shape=(MAX_LENGTH,), dtype='int32')
q_input = Input(shape=(MAX_LENGTH,), dtype='int32')

embedded_a = embedding_layer(a_input)
embedded_q = embedding_layer(q_input)

if gaussian_noise != 0:
    embedded_a = keras.layers.GaussianNoise(gaussian_noise)(embedded_a)
    embedded_q = keras.layers.GaussianNoise(gaussian_noise)(embedded_q)

if unidirectional:
    if dropout:
        shared_lstm = keras.layers.LSTM(M, return_sequences=True, recurrent_dropout=0.2, dropout=0.5,
                                        kernel_initializer=initializer, name='SharedLSTM1')
        shared_lstm2 = keras.layers.LSTM(N, return_sequences=True, recurrent_dropout=0.2, dropout=0.5,
                                         kernel_initializer=initializer, name='SharedLSTM2')
    else:
        shared_lstm = keras.layers.CuDNNLSTM(M, return_sequences=True, kernel_initializer=initializer,
                                             name='SharedLSTM1')
        shared_lstm2 = keras.layers.CuDNNLSTM(N, return_sequences=True, kernel_initializer=initializer,
                                              name='SharedLSTM2')
    N_output = N
else:
    if dropout:
        shared_lstm = Bidirectional(keras.layers.LSTM(M, return_sequences=True, recurrent_dropout=0.2, dropout=0.5,
                                                      kernel_initializer=initializer), name='SharedLSTM')
        shared_lstm2 = Bidirectional(keras.layers.LSTM(N, return_sequences=True, recurrent_dropout=0.2, dropout=0.5,
                                                       kernel_initializer=initializer), name='SharedLSTM')
    else:
        shared_lstm = Bidirectional(keras.layers.CuDNNLSTM(M, return_sequences=True, kernel_initializer=initializer),
                                    name='SharedLSTM1')
        shared_lstm2 = Bidirectional(keras.layers.CuDNNLSTM(N, return_sequences=True, kernel_initializer=initializer),
                                     name='SharedLSTM2')
    N_output = 2 * N

a_lstm_intermediate = shared_lstm(embedded_a)
a_lstm_intermediate = keras.layers.BatchNormalization()(a_lstm_intermediate)
a_lstm_output = shared_lstm2(a_lstm_intermediate)
a_lstm_output_viz = Lambda(lambda x: x[:, -1, :], output_shape=(N_output,), name='a_lstm_output_viz')(
    a_lstm_output)  # only needed for visualization

q_lstm_intermediate = shared_lstm(embedded_q)
q_lstm_intermediate = keras.layers.BatchNormalization()(q_lstm_intermediate)
q_lstm_output = shared_lstm2(q_lstm_intermediate)
q_lstm_output_viz = Lambda(lambda x: x[:, -1, :], output_shape=(N_output,), name='q_lstm_output_viz')(
    q_lstm_output)  # only needed for visualization

O_q = GlobalMaxPooling1D(name='max_pool_q')(q_lstm_output)
q_vec = Dense(N_output, name='W_qm')(O_q)
q_vec = RepeatVector(200)(q_vec)

a_vec = TimeDistributed(Dense(N_output, name='W_am'))(a_lstm_output)

m = Add()([q_vec, a_vec])
m = Activation(activation='tanh')(m)

s = TimeDistributed(Dense(1, name='w_ms'))(m)
s = keras.layers.Softmax(axis=1, name='attention_scores')(s)
# name='attended_a'
h_hat_a = Multiply()([a_lstm_output, s])

O_a = GlobalMaxPooling1D(name='max_pool_attended_a')(h_hat_a)
# print("O_q[0:5]", O_q[0:5])
# print("O_a[0:5]", O_a[0:5])
x = Dot(axes=-1, normalize=True)([O_q, O_a])
# x = Flatten()(x)

model = Model([a_input, q_input], x)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['acc'])

model.summary()

model.fit([X_train_a, X_train_q], Y_train, validation_data=([X_validation_a, X_validation_q], Y_validation),
          epochs=4, batch_size=20, class_weight=class_weights)

model_json = model.to_json()
with open(DATA_FOLDER + "models/model_visualization_siamesedeeplstm.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(DATA_FOLDER + "models/model_visualization_siamesedeeplstm.h5")
print("Saved model to disk")
