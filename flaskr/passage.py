import os
import pickle
from operator import itemgetter
from threading import Lock

import tensorflow as tf
import whoosh
from flask import (
    Blueprint, render_template, request, redirect, url_for, Markup, escape
)
from nltk import WordNetLemmatizer
from whoosh import qparser
from whoosh.fields import *
from whoosh.filedb.filestore import FileStorage
from whoosh.index import create_in

from loading_preprocessing_TC import *

bp = Blueprint('passage', __name__)
DATASET_PATH = 'resources/datasets/insuranceQA/'
WHOOSH_DATASET_PATH = 'resources/datasets/'
WHOOSH_PATH = 'whoosh/'
WHOOSH_PATH_QA = 'whoosh/indexdir/'
storage = None
ix = None

cuda_option = True  # whether to use cuda or not in PyTorch

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

# TODO: Do all these need to be global?
cos_scores = ['']
# TODO: Handle this more elegantly
checked = ["", "", "checked"]  # for displaying tick marks on which subset of the dataset (test1, test2, ...) is chosen
run_time = 0  # time of search engine query processing
correct_answer_id = -1  # for displaying a star icon in case the correct answer is known
lemmatizer = WordNetLemmatizer()
texts = []  # texts of found answers to the question
visible_scores = []  # attention scores > threshold for display
attention_scores = []  # original attention scores by the model
attention_threshold = 0  # for displaying highlights in the text
tag_string = "<data-toggle=\"tooltip\" title=\"SCORE\"><span style = \"background-color: rgba(255, 0, 0, OPACITY);\">WORD</span>"
tag_results = []  # complete HTML code for answer texts highlighted according to the attention scores


@bp.before_app_first_request
def setup():
    global model
    if not os.path.isdir(WHOOSH_PATH):
        os.makedirs(WHOOSH_PATH_QA)
        whoosh_setup()
    if not model:
        model = load_environment()


def whoosh_setup():
    import preprocess_corpus_classes as pr
    dl = pr.DataLexical(data_dir=WHOOSH_DATASET_PATH, corpus_type="insqa-v1",
                        vocab_path=DATASET_PATH + "V1/vocabulary", datareader=None, dataframe=None,
                        lemmatize_option=False, encoded=True)

    decoded = dl.decode_texts(dl.corpus_dataframe)

    decoded.head()

    stem_ana = whoosh.analysis.StemmingAnalyzer()
    ngt = whoosh.analysis.StandardAnalyzer() | whoosh.analysis.NgramFilter(minsize=2, maxsize=4)

    schema = Schema(question=TEXT(stored=True, analyzer=stem_ana), doc_id=NUMERIC(stored=True),
                    answer=TEXT(analyzer=stem_ana, stored=True), ngrams=NGRAMWORDS)
    ix = create_in(WHOOSH_PATH_QA, schema)

    writer = ix.writer()

    begin_time = datetime.datetime.now()
    for ind, row in decoded.iterrows():
        writer.add_document(question=row['question'], doc_id=ind, answer=row['answer'],
                            ngrams=[token.text for token in ngt(row['question'])])
    writer.commit()
    finish_time = datetime.datetime.now() - begin_time
    print('Time taken to create whoosh files:', finish_time)


def load_environment():
    """Load documents index for search engine, pre-trained embeddings, vocabulary, parameters and the model."""
    global embeddings, vocabulary_encoded, params, model, cos_qa, model_filename, ix, storage

    storage = FileStorage(WHOOSH_PATH_QA)
    ix = storage.open_index()

    embeddings_path = DATASET_PATH + 'INSQAV1_embeddings'
    vocabulary_path = DATASET_PATH + 'INSQAV1_vocabulary_encoded'
    params_path = DATASET_PATH + 'INSQAV1_params'
    model_filename = '2017-12-18 11_09_56.842347insqav1_attention_pool-500_we-adapt'

    embeddings = pickle.load(open(embeddings_path, 'rb'))
    vocabulary_encoded = pickle.load(open(vocabulary_path, 'rb'))
    params = pickle.load(open(params_path, 'rb'))

    model = BiLSTM(params['rnn_size'], params['embedding_size'], params['vocab_size'], cuda_option, embeddings,
                   params['batch_size'])
    if cuda_option:
        model.cuda()
    else:
        model.cpu()
    print("Loading model:", model_filename)
    checkpoint = torch.load(DATASET_PATH + model_filename)
    model.load_state_dict(checkpoint['state_dict'])
    model.batch_size = params['batch_size']
    cos_qa = nn.CosineSimilarity(dim=1, eps=1e-8)

    return model


def clear_interface_variables():
    """Return to default (before processing any request) values for interface output variables.
    Needed for switching between InsuranceQA and SemEval pages."""
    global attention_threshold, attention_scores, visible_scores
    global texts, ix
    global checked, run_time
    global cos_scores, correct_answer_id, error, tag_results, valid
    print('Setting interface variables to default values (~ clearing a webpage).')
    texts = []
    attention_scores = []
    visible_scores = []
    attention_threshold = 0
    error = ""
    highlight_attention(texts, visible_scores, 200)


def highlight_attention(tokens, scores, max_length):
    """Generate HTML for the answer texts so that the words are highlighted according to the attention score"""
    global attention_threshold, visible_scores, tag_results, tag_string
    tag_results = []
    for i_t, text in enumerate(tokens):
        tags = []
        for i_w, word in enumerate(text[:max_length]):
            tags.append(
                tag_string.replace("OPACITY", str(10 * scores[i_t][i_w])).replace("WORD", word).replace("SCORE",
                                                                                                        str(
                                                                                                            scores[
                                                                                                                i_t][
                                                                                                                i_w])))
        tag_results.append(' '.join(tags))


def handle_requests(request_form, received_question, processed_question):
    """Receive input from a user and act accordingly:
    # 1. Load the specified model
    # (form "model_file"; file upload)
    2. Change threshold for attention scores (used for displaying higlights in the text)
    (form "attention_threshold"; slider)
    3. Change the dataset which is used by the search engine (either test 1 or test 2, or both, or all the texts)
    (forms use-test1, use-test2, use-all; check boxes)
    4. Preprocess new question and run model on it.
    (form 'question_input'; text field)"""

    global attention_threshold, attention_scores, visible_scores
    global embeddings, vocabulary_encoded
    global model, params, cuda_option, cos_qa
    global texts, lemmatizer, ix
    global checked, run_time
    global cos_scores, correct_answer_id, error, tag_results, valid

    error = ""

    # if 'model_file' in request_form.keys():
    #     correct_answer_id = -1
    #     error = load_model(request_form['model_file'])

    if 'attention_threshold' in request_form.keys():
        print("Changed attention threshold value.")
        attention_threshold = int(request_form['attention_threshold']) / 1000

    if 'use-test1' in request_form.keys() or 'use-test2' in request_form.keys() or 'use-all' in request_form.keys():
        correct_answer_id = -1
        if 'use-all' in request_form.keys():
            if request_form['use-all'] == 'on':
                print("Switched to using all index.")
                storage = FileStorage(WHOOSH_PATH_QA)
                checked = ["", "", "checked"]
        elif 'use-test1' in request_form.keys() and 'use-test2' in request_form.keys():
            if request_form['use-test1'] == 'on' and request_form['use-test2'] == 'on':
                print("Switched to using all tests index.")
                storage = FileStorage(WHOOSH_PATH + "indexdir_tests")
                checked = ["checked", "checked", ""]
        elif 'use-test1' in request_form.keys():
            if request_form['use-test1'] == 'on':
                print("Switched to using test1 index.")
                storage = FileStorage(WHOOSH_PATH + "indexdir_test1")
                checked = ["checked", "", ""]
        elif 'use-test2' in request_form.keys():
            if request_form['use-test2'] == 'on':
                print("Switched to using test2 index.")
                storage = FileStorage(WHOOSH_PATH + "indexdir_test2")
                checked = ["", "checked", ""]
        ix = storage.open_index()

    if 'question_input' in request_form.keys():
        correct_answer_id = -1
        received_question = escape(request_form['question_input'])
        received_question = received_question.replace("/", "")
        print('Received question: ', received_question)
        processed_question = preprocess_text(received_question)
        print('Processed question: ', processed_question)
        valid, error = validate_question(received_question)
        if valid:
            print("Validation success.")
        else:
            print("Validation failure.")
            texts = []
            attention_scores = []
            visible_scores = []
            cos_scores = [""]
            tag_results = []
            correct_answer_id = -1

    if received_question != "..." and valid:
        model_predict(received_question, processed_question)

    return received_question, processed_question


def preprocess_text(text):
    """Lowercase, remove question mark, lemmatize and omit OOV.
    Args:
        text (string): a text (usually, a user-defined question) to preprocess.
    Returns:
        result (string): a preprocessed text (no question marks, no double spaces, lowercased, lemmatized, no OOV).
    """
    global vocabulary_encoded
    result = ' '.join(text.replace("?", "").strip().split()).split()
    result[0] = result[0].lower()
    result = [lemmatizer.lemmatize(word, pos='v') for word in result]
    result = [w if w in vocabulary_encoded.keys() else '' for w in result]
    result = ' '.join(result)
    return result


# - - - demo - - - #

import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np


## Preparing texts for deep learning model

def prepare_batch(texts, params, vocabulary_encoded, embeddings, max_len, volatile=False):
    vectorized_seqs = [[vocabulary_encoded[w] for w in text] for text in texts]
    seq_lengths = torch.LongTensor([len(x) for x in vectorized_seqs])
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())) + params['vocab_size']
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        if seqlen < max_len:
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq[:seqlen])
        else:
            seq_tensor[idx, :max_len] = torch.LongTensor(seq[:max_len])
    seq_lengths[seq_lengths > max_len] = max_len
    seq_lengths_sorted, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor_sorted = seq_tensor[perm_idx]
    return seq_tensor_sorted, perm_idx, seq_lengths_sorted.numpy()


## Deep Learning model

class BiLSTM(nn.Module):
    def __init__(self, hidden_dim, embedding_size, vocab_size, cuda_option, embeddings, batch_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_size)
        self.embeddings.weight.data.copy_(embeddings.weight.data)
        np.random.seed(1)
        torch.manual_seed(2)
        torch.cuda.manual_seed(2)
        torch.cuda.manual_seed_all(2)
        self.lstm = nn.LSTM(embedding_size, hidden_dim, bidirectional=True)  # (seq_len, batch, input_size) ->
        # (seq_len, batch, hidden_size * num_directions)
        self.cuda_option = cuda_option
        self.hidden = self.init_hidden(batch_size)
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim * 2, False)
        self.lin2 = nn.Linear(hidden_dim * 2, hidden_dim * 2, False)
        self.lin3 = nn.Linear(hidden_dim * 2, 1, False)
        self.tahn = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (
            autograd.Variable(transfer_data(torch.zeros(2, batch_size, self.hidden_dim), self.cuda_option),
                              requires_grad=True),
            autograd.Variable(transfer_data(torch.zeros(2, batch_size, self.hidden_dim), self.cuda_option),
                              requires_grad=True))

    def forward(self, sentence, lens, perm_idx, is_eval=False, volatile=False, attention=(False, None),
                wrong_answer_mode=False):
        # unpack output, transfer it to GPU and pack again (not done in batchify() to save memory)
        #         attention = (False, None)
        verbose = False
        sentence = self.embeddings(
            autograd.Variable(transfer_data(sentence, self.cuda_option), volatile=volatile).long())
        sentence = sentence.transpose(0, 1)
        packed_input = nn.utils.rnn.pack_padded_sequence(sentence, lens)

        # apply BiLSTM
        if not is_eval:
            packed_output, self.hidden = self.lstm(packed_input, self.hidden)
        else:
            packed_output, _ = self.lstm(packed_input)
        del packed_input

        # unpack output and transpose it to be batch_size x rnn_size*2 x max_len
        unpacked_output, lengths = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=-10.0)
        unpacked_output = torch.transpose(torch.transpose(unpacked_output, 0, 2), 0, 1)
        if verbose:
            print('unpacked_output: ', unpacked_output)
        del packed_output
        if attention[0]:

            # mask padding for LSTM output
            mask = autograd.Variable(torch.ones(unpacked_output.size())).cuda()
            for i, l in enumerate(lengths):
                if l < unpacked_output.size(2):
                    mask[i, :, l:] = 0
            if verbose:
                print('mask: ', mask)
            # apply W_{am} and W_{qm}
            a_lin = self.lin1(torch.transpose(unpacked_output * mask, 1, 2))
            if verbose:
                print('a_lin: ', a_lin)
            q_lin = self.lin2(torch.transpose(attention[1].repeat(1, 1, a_lin.size(1)), 1, 2))
            if verbose:
                print('q_lin: ', q_lin)
            # obtain m_{a, q}(t)
            if wrong_answer_mode:
                m = q_lin.repeat(a_lin.size(0), 1, 1) + a_lin
            else:
                m = q_lin + a_lin
            # if wrong_answer_mode:
            m = m * torch.transpose(mask, 1, 2)
            if verbose:
                print('m: ', m)
            attentions = self.tahn(self.lin3(m))
            if verbose:
                print('attentions: ', attentions)

            mask_ = (mask[:, 0:1, :] == 0)
            mask_ = torch.transpose(mask_, -1, -2)

            padded_attention = attentions.clone()

            padded_attention.masked_fill_(mask_, -float('inf'))

            softmax_attentions = self.softmax(padded_attention)

            # change padding value from 0 to -10 again to correctly maxpool
            unpacked_output = unpacked_output * torch.transpose(softmax_attentions.repeat(1, 1, 2 * self.hidden_dim),
                                                                1, 2)
            unpacked_output.masked_fill_((mask == 0), -10.0)

        # restore original order
        if self.cuda_option:
            perm_idx = perm_idx.cuda()
            unpacked_output = unpacked_output[perm_idx, :, :]
            perm_idx = perm_idx.cpu()
        else:
            unpacked_output = unpacked_output[perm_idx, :, :]

        # maxpool
        result, _ = unpacked_output.max(2, keepdim=True)
        del unpacked_output

        if verbose:
            print('result: ', result)
        if attention[0]:
            return result, softmax_attentions
        else:
            return result


class QALSTM(nn.Module):
    def __init__(self, hidden_dim, embedding_size, vocab_size, cuda_option, embeddings):
        dropout_on = False
        super(QALSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_size)
        self.embeddings.weight.data.copy_(embeddings.weight.data)
        np.random.seed(1)
        torch.manual_seed(2)
        torch.cuda.manual_seed(2)
        torch.cuda.manual_seed_all(2)
        self.lstm = nn.LSTM(embedding_size, hidden_dim, bidirectional=True)
        self.cuda_option = cuda_option
        self.hidden = self.init_hidden()
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim * 2, False)
        self.lin2 = nn.Linear(hidden_dim * 2, hidden_dim * 2, False)
        self.lin3 = nn.Linear(hidden_dim * 2, 1, False)
        self.tahn = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        if dropout_on:
            self.dropout = nn.Dropout(p=0.5)

    def init_hidden(self, minibatch_size=1):
        w = nn.init.xavier_normal(torch.Tensor(2, minibatch_size, self.hidden_dim))
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(transfer_data(w, self.cuda_option), requires_grad=True),
                autograd.Variable(transfer_data(w, self.cuda_option), requires_grad=True))

    def forward(self, sentence, lens, perm_idx, is_eval=False, volatile=False, attention=(False, None),
                wrong_answer_mode=False):
        attention_on = True
        dropout_on = False
        pool_type = 'max'
        verbose = False
        if volatile:
            with torch.no_grad():
                sentence = self.embeddings(autograd.Variable(transfer_data(sentence, self.cuda_option)).long())
        else:
            sentence = self.embeddings(autograd.Variable(transfer_data(sentence, self.cuda_option)).long())
        sentence = sentence.transpose(0, 1)
        packed_input = nn.utils.rnn.pack_padded_sequence(sentence, lens)

        # apply BiLSTM
        packed_output, self.hidden = self.lstm(packed_input, self.hidden)
        del packed_input

        # unpack output and transpose it to be batch_size x rnn_size*2 x max_len
        unpacked_output, lengths = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=-10.0)
        unpacked_output = torch.transpose(torch.transpose(unpacked_output, 0, 2), 0, 1)
        if verbose:
            print('unpacked_output: ', unpacked_output)
        del packed_output
        if attention_on:
            if attention[0]:

                # mask padding for BiLSTM output
                mask = autograd.Variable(torch.ones(unpacked_output.size())).cuda()
                for i, l in enumerate(lengths):
                    if l < unpacked_output.size(2):
                        mask[i, :, l:] = 0
                if verbose:
                    print('mask: ', mask)

                # apply W_{am} and W_{qm}
                a_lin = self.lin1(torch.transpose(unpacked_output * mask, 1, 2))
                if verbose:
                    print('a_lin: ', a_lin)
                q_lin = self.lin2(torch.transpose(attention[1].repeat(1, 1, a_lin.size(1)), 1, 2))
                if verbose:
                    print('q_lin: ', q_lin)

                # obtain m_{a, q}(t)
                if wrong_answer_mode:
                    m = q_lin.repeat(a_lin.size(0), 1, 1) + a_lin
                else:
                    m = q_lin + a_lin

                m = m * torch.transpose(mask, 1, 2)
                if verbose:
                    print('m: ', m)
                attentions = self.lin3(self.tahn(m))
                if verbose:
                    print('attentions: ', attentions)

                mask_ = (mask[:, 0:1, :] == 0)
                mask_ = torch.transpose(mask_, -1, -2)

                padded_attention = attentions.clone()

                padded_attention.masked_fill_(mask_, -float('inf'))

                softmax_attentions = self.softmax(padded_attention)

                # change padding value from 0 to -10 again to correctly maxpool
                unpacked_output = unpacked_output * torch.transpose(
                    softmax_attentions.repeat(1, 1, 2 * self.hidden_dim),
                    1, 2)
                unpacked_output.masked_fill_((mask == 0), -10.0)

        # restore original order after sorting for pack_padded_sequence operation (see PyTorch docs for details)
        if self.cuda_option:
            perm_idx = perm_idx.cuda()
            unpacked_output = unpacked_output[perm_idx, :, :]
            perm_idx = perm_idx.cpu()
        else:
            unpacked_output = unpacked_output[perm_idx, :, :]

        # pool output vectors of BiLSTM's hidden layers to fix the dimension
        if pool_type == 'max':
            result, _ = unpacked_output.max(2, keepdim=True)
        elif pool_type == 'mean':
            result = unpacked_output.mean(2, keepdim=True)

        if dropout_on:
            result = self.dropout(result)
        del unpacked_output

        if verbose:
            print('result: ', result)

        if attention[0]:
            return result, softmax_attentions
        else:
            return result


def transfer_data(x, cuda_option):
    if cuda_option:
        return x.cuda()
    else:
        return x.cpu()


# - - - demo - - - #

# - - - demo_preprocessing - - - #

from itertools import groupby
from string import punctuation
import re

punc = set(punctuation) - set('.')


## Preprocessing query text

def clean_punctuation(text):
    clean_text = []
    for k, g in groupby(text):
        if k in punc:
            clean_text.append(k)
        else:
            clean_text.extend(g)
    clean_text = ''.join(clean_text)
    clean_text = re.sub('([.,!?()*\\\\"\'-:;0-9=\$%\&_])', r' \1 ', clean_text)
    clean_text = re.sub('\s{2,}', ' ', clean_text)
    return clean_text


emoticons = {":-)": "happy", ":)": "happy", ":-]"":]": "happy", ":-3": "happy", ":3": "happy", ":->": "happy",
             ":>": "happy", \
             "8-)": "happy", "8)": "happy", ":-}": "happy", ":}": "happy", ":o)": "happy", ":c)": "happy",
             ":^)": "happy", \
             "=]": "happy", "=)": "happy", ":-D": "happy", ":D": "laugh", "8-D": "laugh", "8D": "laugh", "x-D": "laugh", \
             "xD": "laugh", "X-D": "laugh", "XD": "laugh", "=D": "laugh", "=3": "happy", "B^D": "laugh", ":-(": "sad", \
             ":(": "sad", ":-c": "sad", ":c": "sad", ":-<": "sad", ":<": "sad", ":-[": "sad", ":[": "sad",
             ":-||": "sad", \
             ">:[": "angry", ":{": "sad", ":@": "sad", ">:(": "angry", ";-)": "wink", ";)": "wink", "*-)": "wink", \
             "*)": "wink", ";-]": "wink", ";]": "wink", ";^)": "wink", ":-,": "wink", ";D": "laugh", \
             ":-/": "scepticism", ":/": "scepticism", ":-.": "scepticism", ">:\\": "angry", ">:/": "angry", \
             ":\\": "scepticism", "=/": "scepticism", "=\\": "scepticism", ":L": "scepticism", "=L": "scepticism", \
             ":S": "scepticism"}
emoticons_re = {}
for key, val in emoticons.items():
    new_key = key
    for c in new_key:
        if c in ['[', '\\', '^', '$', '.', '|', '?', '*', '+', '(', ')']:
            new_key = new_key.replace(c, "\\" + c)
        new_key = new_key.replace("\\\|", "\\|")
    regex = re.compile(new_key + "+")
    emoticons_re[regex] = val


def extract_emoticons(text, tag=0):
    global emoticons_re
    extracted_emoticons = []
    transformed_text = text
    for emoticon in emoticons_re.keys():
        if emoticon.search(text):
            for m in emoticon.finditer(text):
                extracted_emoticons.append((m.group(), emoticons_re[emoticon]))
                if tag:
                    placeholder = " [EMOTICON:" + emoticons_re[emoticon] + "] "
                else:
                    placeholder = " " + emoticons_re[emoticon] + " "
                transformed_text = transformed_text.replace(m.group(), placeholder)
    return transformed_text


# - - - demo_preprocessing - - - #

@bp.route('/passage', methods=['GET', 'POST'], strict_slashes=False)
def setup_framework():
    """Set interface variables to their default values.
    Load required models and data"""
    global attention_threshold, attention_scores, visible_scores
    global embeddings, vocabulary_encoded
    global model, params, cuda_option, cos_qa, model_filename
    global texts, lemmatizer, ix
    global checked, run_time
    global cos_scores, correct_answer_id, error, tag_results, valid

    if not model:
        load_environment()

    clear_interface_variables()
    received_question = "..."

    if request.method == 'POST':
        print(request.form)
        if 'question_input' in request.form.keys():
            received_question, processed_question = handle_requests(request.form, received_question, None)
            return redirect(
                url_for('passage.process_question', received_question=received_question, processed_question=None))
        else:
            handle_requests(request.form, received_question, None)
    return render_template('passage.html', texts=tag_results, val=1000 * attention_threshold,
                           question=received_question.replace("?", "") + "?", error=Markup(error), checked=checked,
                           cos_scores=sorted(cos_scores, reverse=True), run_time=run_time,
                           correct_answer_id=correct_answer_id, model_filename=model_filename)


def validate_question(text):
    """Check validity of the question:
    1. filter by length 2. auto-correction 3. OOV"""
    global valid, ix

    # empty or too long questions are not allowed
    if len(text) == 0:
        error = "<div class=\"alert alert-warning\"> Sorry, the question appears to be empty. Try again? </div>"
        return False, error
    elif len(text) > 150:
        error = "<div class=\"alert alert-warning\"> Sorry, the question is too long. Try to use only 150 characters." \
                " </div>"
        return False, error

    mparser = qparser.MultifieldParser(["answer"], schema=ix.schema)

    # auto-correction built in Whoosh
    with ix.searcher() as s:
        q = mparser.parse(text.replace("?", ""))
        corrected = s.correct_query(q, text)
        if corrected.query != q:
            error = "<div class=\"alert alert-warning\"> Did you mean: <a href=\"" + url_for('passage.process_question',
                                                                                             received_question=corrected.string) + "\">" + corrected.string + "</a>?</div>"
            return False, error

    # the question is valid, but contains lemmata which are not in vocabulary, so a warning is displayed
    oov_num = 0
    for word in text.replace("?", "").split():
        if not (lemmatizer.lemmatize(word, pos='v') in vocabulary_encoded.keys()):
            if not (lemmatizer.lemmatize(word.lower(), pos='v')) in vocabulary_encoded.keys():
                oov_num += 1
    if oov_num != 0:
        if oov_num < len(text.split()):
            warning = "<div class=\"alert alert-warning\"> The question has words that are not in vocabulary." \
                      " if you rephrase it, you might get better results. </div>"
            return True, warning
        else:
            error = "<div class=\"alert alert-warning\"> Sorry, could not understand your input." \
                    " </div>"
            return False, error

    return True, ""


def model_predict(received_question, processed_question):
    """Run a pre-trained loaded model on a received question."""
    pytorch_predict(received_question, processed_question)


def pytorch_predict(received_question, processed_question):
    """Retrieve candidate answers and run a PyTorch model on them in test mode, obtaining cosine similarity
    and attention scores."""
    global attention_threshold, attention_scores, visible_scores
    global embeddings, vocabulary_encoded
    global model, params, cuda_option, cos_qa, model_filename
    global texts, lemmatizer, ix
    global checked, run_time
    global cos_scores, correct_answer_id, error, tag_results, valid

    correct_answer_id = -1

    print("Received question:", received_question)
    print("Processed question:", processed_question)
    print("Attention threshold:", attention_threshold)

    answers, questions = retrieve_candidates(processed_question, received_question)

    # if the question input by the user is the same as one of the retrieved, we know the exact correct answer
    print(processed_question.split())
    for i, q in enumerate(questions):
        print('Question:', q)
        tmp_proc_q = [x.lower() for x in processed_question.split()]
        tmp_q = [x.lower() for x in q]
        if tmp_q == tmp_proc_q:
            correct_answer_id = i
    print("Found correct answer on rank" + str(correct_answer_id))

    # tokenize texts of the question and the answers, map them to numerical indices, and pad
    question_repeated = [processed_question.split()] * len(answers)
    volatile = True
    questions_input_packed, perm_idx_q, lens_q = prepare_batch(question_repeated, params,
                                                               vocabulary_encoded, model.embeddings,
                                                               params['max_len'], volatile)
    _, perm_idx_q = perm_idx_q.sort(0)
    answers_input_packed, perm_idx_a, lens_a = prepare_batch(answers, params, vocabulary_encoded,
                                                             model.embeddings,
                                                             params['max_len'], volatile)

    # resort answers depending on the length
    sorted_answers = [answers[i] for i in perm_idx_a]

    _, perm_idx_a = perm_idx_a.sort(0)

    # run the model and get cosine and attention scores
    questions_output = transfer_data(model(questions_input_packed, lens_q, perm_idx_q, True, True), cuda_option)
    out, softmax_attentions = model(answers_input_packed, lens_a, perm_idx_a, True, True,
                                    (True, questions_output))
    answers_output = transfer_data(out, cuda_option)
    cos_scores = cos_qa(questions_output, answers_output)

    # sort the answers based on their cosine similarity to the question
    cos_scores = list(cos_scores.data.cpu().numpy().flatten())
    print(cos_scores)
    texts = [text for _, text in sorted(zip(cos_scores, sorted_answers), key=itemgetter(0))]

    # scale attention scores to make them visible
    attention_scores = [list(a_s.data.cpu().numpy().flatten()) for _, a_s in
                        sorted(zip(cos_scores, softmax_attentions), key=itemgetter(0))]
    visible_scores = [[x if x > attention_threshold else 0 for x in y] for y in attention_scores]

    highlight_attention(texts, visible_scores, params['max_len'])


# text preprocessing


def filter_term(term, vocabulary):
    """If a word is not present in a vocabulary, check whether its lowercased or capitalized versions are present.
    If not and word is not a punctuation sign, then it is OOV. Otherwise, it is replaced by different case.
    Punctuation signs are omitted completely.

    Args:
        term (string): a word to check/
        vocabulary (dictionary): a vocabulary in a format {token: index}.
    Returns:
         (string): a version of the word that is in the vocabulary OR an empty string (punctuation) OR OOV tag.
    """
    if term in vocabulary.keys():
        return term
    else:
        if term.lower() in vocabulary.keys():
            return term.lower()
        elif term.capitalize() in vocabulary.keys():
            return term.capitalize()
        elif term.lower().capitalize() in vocabulary.keys():
            return term.lower().capitalize()
        else:
            if term in string.punctuation:
                return ''
            else:
                return 'OOV'


def retrieve_candidates(processed_question, received_question):
    """Find candidate answers among corpus documents and preprocess them.
    By default searches only based on question text; if no candidates are found, searches again based on
    question + answer; if no candidates are found, outputs an error message.
    Args:
        processed_question (string): pre-processed text of a user defined question.
        received_question (string): raw text of a user defined question.
    Returns:
        answers (list of strings): list of texts of candidate answers (preprocessed; without OOV words).
        questions (list of strings): list of texts of questions corresponding to the candidate answers.
        answers_with_oov (list of strings): list of texts of candidate answers. (with OOV words; for displaying
        to user).
        """
    global vocabulary_encoded, run_time
    mparser = qparser.MultifieldParser(["question"], schema=ix.schema)
    begin_time = datetime.datetime.now()
    with ix.searcher() as searcher:
        query = mparser.parse(processed_question)
        candidates_number = 20
        results = searcher.search(query, limit=candidates_number)
        finish_time = datetime.datetime.now() - begin_time
        print("Found texts in ", finish_time)
        run_time = finish_time
        if len(results) == 0:
            mparser_tmp = qparser.MultifieldParser(["answer", "question"], schema=ix.schema)
            query = mparser_tmp.parse(processed_question)
            results = searcher.search(query, limit=candidates_number)
            if len(results) == 0:
                cos_scores = [""]
                error = "<div class=\"alert alert-warning\"> Sorry, no answers were found. Try again? </div>"
                print('Stat', 5)
                # return False, error
                return render_template('passage.html', texts=[], val=1000 * attention_threshold,
                                       question=received_question.replace("?", "") + "?",
                                       error=Markup(error), checked=checked,
                                       cos_scores=sorted(cos_scores, reverse=True), run_time=run_time,
                                       correct_answer_id=correct_answer_id, model_filename=model_filename)
            else:  # preprocessing of retrieved candidates
                questions = [res['question'].split() for res in results]
                answers = [res['answer'].split() for res in results]
                print('Stat', 2)
                return answers, questions
        else:  # preprocessing of retrieved candidates
            questions = [res['question'].split() for res in results]
            answers = [res['answer'].split() for res in results]
            print('Stat', 4)
            return answers, questions


@bp.route('/passage/<received_question>', methods=['GET', 'POST'])
def process_question(received_question, processed_question=None):
    global attention_threshold, attention_scores, visible_scores
    global embeddings, vocabulary_encoded
    global model, params, cuda_option, cos_qa, model_filename
    global texts, lemmatizer, ix
    global checked, run_time
    global cos_scores, correct_answer_id, error, tag_results, valid
    print("Processing text", received_question)
    if not model:
        load_environment()

    if processed_question is None:
        processed_question = preprocess_text(received_question)

    highlight_attention(texts, visible_scores, 200)
    error = ""

    valid, error = validate_question(received_question)
    if not valid:
        print("Validation failure.")
        texts = []
        attention_scores = []
        visible_scores = []
        cos_scores = [""]
        tag_results = []
        correct_answer_id = -1
        return render_template('passage.html', texts=tag_results, val=1000 * attention_threshold,
                               question="...", error=Markup(error), checked=checked,
                               cos_scores=sorted(cos_scores, reverse=True), run_time=run_time,
                               correct_answer_id=correct_answer_id, model_filename=model_filename)

    if request.method == 'POST':
        print(request.form)
        if 'question_input' in request.form.keys():
            received_question, processed_question = handle_requests(request.form, received_question, None)
            return redirect(
                url_for('passage.process_question', received_question=received_question, processed_question=None))
        else:
            handle_requests(request.form, received_question, processed_question)
            return render_template('passage.html', texts=tag_results, val=1000 * attention_threshold,
                                   question=received_question.replace("?", "") + "?", error=Markup(error),
                                   checked=checked,
                                   cos_scores=sorted(cos_scores, reverse=True), run_time=run_time,
                                   correct_answer_id=correct_answer_id, model_filename=model_filename)
    else:
        model_predict(received_question, processed_question)

    if not valid:
        return render_template('passage.html', texts=tag_results, val=1000 * attention_threshold,
                               question="...", error=Markup(error),
                               checked=checked,
                               cos_scores=sorted(cos_scores, reverse=True), run_time=run_time,
                               correct_answer_id=correct_answer_id, model_filename=model_filename)
    else:
        return render_template('passage.html', texts=tag_results, val=1000 * attention_threshold,
                               question=received_question.replace("?", "") + "?", error=Markup(error),
                               checked=checked,
                               cos_scores=sorted(cos_scores, reverse=True), run_time=run_time,
                               correct_answer_id=correct_answer_id, model_filename=model_filename)
