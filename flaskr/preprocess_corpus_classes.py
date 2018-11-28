# coding: utf-8

# <h1 id="tocheading">Preprocessing Texts</h1>
# <div id="toc"></div>

# In[1]:


# get_ipython().run_cell_magic('javascript', '', "$.getScript('ipython_notebook_toc.js')")

# In[1]:


import codecs
import os
import re

import gensim
import numpy as np
import pandas as pd
import sklearn as sk
import spacy
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
pd.options.mode.chained_assignment = None  # default='warn'
data_dir_global = "datasets/"
spacy_model = spacy.load('en_core_web_sm')

# # Reader

# Module for reading ark, insqa v2, wikiqa corpora into a pandas dataframe. This dataframe will be intermediate representation for internal processing.
# 
# Also, the module includes a function to output the corpus information.

# In[6]:


class DataReader():
    data_dir = ""  # directory with the datasets
    corpus_dir = ""  # directory with the particular dataset
    custom_corpus_path = ""  # relative path to the particular dataset
    corpus_type = ""  # InsuranceQA, ark, etc.
    corpus_dataframe = pd.DataFrame()
    num_q = 0
    num_a = 0

    def __init__(self, data_dir="", corpus_type="", custom_corpus_path=""):
        self.data_dir = data_dir
        self.corpus_type = corpus_type
        self.custom_corpus_path = custom_corpus_path
        if data_dir == "":
            print("Please, specify the directory with the corpus via data_dir parameter.")
            return -1
        if corpus_type == "":
            print("Please, specify corpus type via corpus_type parameter. Allowed types: ark, insqa-v1, insqa-v2.")
            return -1

    def info(self):
        print("Corpus type: ", self.corpus_type)
        print("in a directory: ", self.corpus_dir)
        print("Number of questions: ", self.num_q)
        print("Number of answers: ", self.num_a)
        print("Is there a pool?", 'pool' in self.corpus_dataframe.columns.values)
        print("Is there a split into train/test/dev?", 'split_type' in self.corpus_dataframe.columns.values)
        if 'pool' in self.corpus_dataframe.columns.values:
            if True in pd.isnull(self.corpus_dataframe['pool']):
                print(
                    "WARNING: pool values are not defined for parts of the dataset. Print .corpus_dataframe['pool'] for more information.")

    def read_corpus(self):
        if self.corpus_type == "ark":
            if self.custom_corpus_path != "":
                self.corpus_dir = [self.data_dir + self.custom_corpus_path + "ark/S08/",
                                   self.data_dir + self.custom_corpus_path + "ark/S09/",
                                   self.data_dir + self.custom_corpus_path + "ark/S10/"]
            else:
                self.corpus_dir = [self.data_dir + "ark/S08/", self.data_dir + "ark/S09/", self.data_dir + "ark/S10/"]
            dataframe = self.read_corpus_ark()
        elif self.corpus_type == "insqa-v1":
            if self.custom_corpus_path != "":
                self.corpus_dir = self.data_dir + self.custom_corpus_path
            else:
                self.corpus_dir = self.data_dir + "insuranceQA/V1/"
            dataframe = self.read_corpus_insqa('v1')
        elif self.corpus_type == "insqa-v2":
            if self.custom_corpus_path != "":
                self.corpus_dir = self.data_dir + self.custom_corpus_path
            else:
                self.corpus_dir = self.data_dir + "insuranceQA/V2/"
            dataframe = self.read_corpus_insqa('v2')
        else:
            print(
                "Failed to read corpus: the type of the corpus is not recognized. Allowed types: ark, insqa-v1, insqa-v2.")
            return -1
        self.corpus_dataframe = dataframe
        return dataframe

    def read_corpus_insqa(self, version="v2"):
        if version == "v1":
            return self.read_corpus_insqa_v1()
        else:
            return self.read_corpus_insqa_v2()

    def read_corpus_insqa_v1(self):
        # read question and answers
        qa_pairs = pd.read_table(self.corpus_dir + "question.train.token_idx.label", header=None,
                                 names=['question', 'answer_ids'],
                                 dtype={'question': str, 'answer_ids': object})

        qa_pairs['answer_ids'] = qa_pairs['answer_ids'].apply(str.split)
        qa_pairs['answer_id'] = qa_pairs['answer_ids']
        lst_col = 'answer_id'
        qa_pairs_expanded = pd.DataFrame({col: np.repeat(qa_pairs[col].values, qa_pairs[lst_col].str.len())
                                          for col in qa_pairs.columns.difference([lst_col])
                                          }).assign(**{lst_col: np.concatenate(qa_pairs[lst_col].values)})[
            qa_pairs.columns.tolist()]

        answers = pd.read_table(self.corpus_dir + "answers.label.token_idx", header=None, names=['answer_id', 'answer'],
                                dtype={'answer_id': str, 'answer': str})

        full_qa_pairs = qa_pairs_expanded.merge(answers, on='answer_id', how='left')
        full_qa_pairs['split_type'] = 'train'

        # read train/test/dev split + pool information and concatenate them into one corpus_dataframes
        dfs_tmp = []
        for option in ['test1', 'test2', 'dev']:
            file_tmp = self.corpus_dir + "question." + option + ".label.token_idx.pool"
            df_tmp = pd.read_table(file_tmp, header=None, names=['answer_ids', 'question', 'pool'])
            df_tmp['answer_ids'] = df_tmp['answer_ids'].apply(str.split)
            df_tmp['split_type'] = option
            dfs_tmp.append(df_tmp)
        split_tmp = pd.concat(dfs_tmp)

        split_tmp['answer_id'] = split_tmp['answer_ids']
        lst_col = 'answer_id'

        split_tmp_expanded = pd.DataFrame({col: np.repeat(split_tmp[col].values, split_tmp[lst_col].str.len())
                                           for col in split_tmp.columns.difference([lst_col])
                                           }).assign(**{lst_col: np.concatenate(split_tmp[lst_col].values)})[
            split_tmp.columns.tolist()]

        split_tmp_expanded = split_tmp_expanded.merge(answers, on='answer_id', how='left')

        full_df = pd.concat([split_tmp_expanded, full_qa_pairs])
        full_df.reset_index(drop=True, inplace=True)
        self.num_q = len(qa_pairs) + len(split_tmp)
        self.num_a = len(qa_pairs_expanded) + len(split_tmp_expanded)

        return full_df

    def read_corpus_insqa_v2(self):
        # read question and answers
        data_q_file = "InsuranceQA.question.anslabel.token.encoded"
        qa_pairs = pd.read_table(self.corpus_dir + data_q_file, header=None, names=['domain', 'question', 'answer_ids'],
                                 dtype={'domain': str, 'question': str, 'answer_ids': object})
        qa_pairs['answer_ids'] = qa_pairs['answer_ids'].apply(lambda x: [i for i in x.split()])
        qa_pairs['answer_id'] = qa_pairs['answer_ids']
        lst_col = 'answer_id'
        qa_pairs_expanded = pd.DataFrame({col: np.repeat(qa_pairs[col].values, qa_pairs[lst_col].str.len())
                                          for col in qa_pairs.columns.difference([lst_col])
                                          }).assign(**{lst_col: np.concatenate(qa_pairs[lst_col].values)})[
            qa_pairs.columns.tolist()]

        data_a_file = "InsuranceQA.label2answer.token.encoded"
        answers = pd.read_table(self.corpus_dir + data_a_file, header=None, names=['answer_id', 'answer'],
                                dtype={'answer_id': str, 'answer': str})
        full_qa_pairs = qa_pairs_expanded.merge(answers, on='answer_id', how='left')

        # read train/test/dev split + pool information and concatenate them into one corpus_dataframes
        pool_size = 500
        dfs_tmp = []
        for option in ['train', 'test', 'valid']:
            file_tmp = self.corpus_dir + "InsuranceQA.question.anslabel.token." + str(
                pool_size) + ".pool.solr." + option + ".encoded"
            df_tmp = pd.read_table(file_tmp, header=None, names=['domain', 'question', 'answer_ids', 'pool'])
            if option != 'valid':
                df_tmp['split_type'] = option
            else:
                df_tmp['split_type'] = 'dev'
            dfs_tmp.append(df_tmp)
        split_tmp = pd.concat(dfs_tmp)

        # handle multiple correct answers
        split_tmp['answer_ids'] = split_tmp['answer_ids'].apply(lambda x: [i for i in x.split()])
        lst_col = 'answer_ids'
        split_tmp_expanded = pd.DataFrame({col: np.repeat(split_tmp[col].values, split_tmp[lst_col].str.len())
                                           for col in split_tmp.columns.difference([lst_col])
                                           }).assign(**{lst_col: np.concatenate(split_tmp[lst_col].values)})[
            split_tmp.columns.tolist()]
        split_tmp_expanded.rename(columns={'answer_ids': 'answer_id'}, inplace='True')

        # combine all information
        full_df = full_qa_pairs.merge(split_tmp_expanded.drop(axis=1, labels=['domain']), on=['question', 'answer_id'],
                                      how='inner')

        self.num_q = len(qa_pairs)
        self.num_a = len(qa_pairs_expanded)

        return full_df

    def read_corpus_ark(self):
        corpus_dataframes = []
        for data_folder in self.corpus_dir:
            qa_pairs = pd.read_table(data_folder + "question_answer_pairs.txt", encoding='ISO-8859-1', header=0,
                                     dtype={'Answer': str, 'Question': str})
            qa_pairs['Answer'] = qa_pairs['Answer'].astype('str')
            qa_pairs['Question'] = qa_pairs['Question'].astype('str')
            # normalize format for answers and questions (remove full stops and convert to lower case)
            qa_pairs['Answer'] = qa_pairs['Answer'].apply(lambda x: x.strip('.').lower())
            qa_pairs['Question'] = qa_pairs['Question'].apply(lambda x: x.lower())
            # drop missing answers
            qa_pairs = qa_pairs[qa_pairs.Answer != 'nan']
            qa_pairs = qa_pairs[qa_pairs.Answer != '']
            qa_pairs = qa_pairs[qa_pairs.Answer != ' ']
            # drop full duplicates
            qa_pairs.drop_duplicates(inplace=True, subset=['Question', 'Answer'])
            qa_pairs['ArticleFile'] = qa_pairs['ArticleFile'].astype('str')
            texts = []
            # collect paragraphs
            for idx, row in qa_pairs.iterrows():
                if row['ArticleFile'] != 'nan':
                    article_path = data_folder + row['ArticleFile'] + '.txt.clean'
                    with open(article_path, 'rt', encoding="ISO-8859-1") as article_file:
                        texts.append(article_file.read().replace('\n', ' '))
                else:
                    texts.append(' ')
            qa_pairs['ArticleText'] = texts
            corpus_dataframe = qa_pairs[['Question', 'Answer', 'ArticleText']]
            corpus_dataframe.columns = ['question', 'answer', 'paragraph']
            corpus_dataframe['question'] = corpus_dataframe['question'].astype(str)
            corpus_dataframe['answer'] = corpus_dataframe['answer'].astype(str)
            corpus_dataframe['paragraph'] = corpus_dataframe['paragraph'].astype(str)
            corpus_dataframes.append(corpus_dataframe)
        full_corpus_dataframe = pd.concat(corpus_dataframes)
        full_corpus_dataframe.reset_index(inplace=True, drop=True)
        self.num_q = len(full_corpus_dataframe)
        self.num_a = len(full_corpus_dataframe)
        return full_corpus_dataframe


class DataAugmenter():
    data_dir = ""
    corpus_type = ""
    corpus_dir = ""
    corpus_dataframe = pd.DataFrame()

    def __init__(self, data_dir="", corpus_type="", dataframe=None):
        self.data_dir = data_dir
        self.corpus_type = corpus_type
        if self.data_dir != "":
            if self.corpus_type == "":
                print("Please, specify corpus type via corpus_type parameter. Allowed types: ark, insqa-v1, insqa-v2.")
                return -1
            else:
                self.corpus_dataframe = DataReader(data_dir=self.data_dir, corpus_type=self.corpus_type).read_corpus()
        else:
            if data_dir == "":
                print("Please, specify the directory with the corpus via data_dir parameter.")
                return -1
            else:
                try:
                    self.corpus_dataframe = dataframe
                except Exception as e:
                    print("ERROR: The corpus path appears to be incorrect, failed to load.")
                    return -1

    def generate_split_corpus_dataframe(self, train_ratio=0.9):
        if 'split_type' in self.corpus_dataframe.columns.values:
            print("\t WARNING: There is already a split. If you want to re-generate it, delete the existing one first.")
            train = self.corpus_dataframe[self.corpus_dataframe['split_type'] == 'train']
            test = self.corpus_dataframe[self.corpus_dataframe['split_type'] == 'test']
            dev = self.corpus_dataframe[self.corpus_dataframe['split_type'] == 'dev']
        else:
            test_ratio = 1 - train_ratio
            train_dev, test = sk.model_selection.train_test_split(self.corpus_dataframe, test_size=test_ratio)
            train, dev = sk.model_selection.train_test_split(self.corpus_dataframe,
                                                             test_size=(test_ratio / train_ratio))
            train['split_type'] = 'train'
            test['split_type'] = 'test'
            dev['split_type'] = 'dev'
            self.corpus_dataframe = pd.concat([train, test, dev])
            print("\t Generated split.")
        return [train, dev, test]

    def generate_wrong_answers_pool(self, dataframe_original, pool_size=50):
        """
        Generate the pool of wrong answers indices for the questions (see Insurance QA corpus format).
        """
        if 'pool' not in dataframe_original.columns.values:
            print("\t There are no pools. Generating pool values.")
            dataframe = dataframe_original.copy()
            if 'answer_id' not in dataframe.columns.values:
                dataframe.loc[:, 'answer_id'] = dataframe.index.values
            pool = [np.random.choice(list(set(dataframe['answer_id']) - set([idx])), pool_size, replace=False) for idx
                    in dataframe['answer_id']]
            for idx, row in dataframe.iterrows():
                dataframe.set_value(idx, 'pool', ' '.join([str(x) for x in pool[idx]]))
            dataframe['answer_id'] = dataframe['answer_id'].astype(str)
            print("Generated pools of wrong answers.")
            return dataframe
        else:
            if True in pd.isnull(dataframe_original['pool']):
                print("\t There are already some pools, but the rest is incomplete. Generating missing pool values.")
                dataframe = dataframe_original.copy()
                missing_pool_ans_ids = dataframe[pd.isnull(dataframe['pool']) == True]['answer_id']
                pool = [np.random.choice(list(set(dataframe['answer_id']) - set([idx])), pool_size, replace=False) for
                        idx in missing_pool_ans_ids]
                for idx in missing_pool_ans_ids:
                    dataframe.set_value(idx, 'pool', ' '.join([str(x) for x in pool[dataframe.index.get_loc(idx)]]))
                print("Generated pools of wrong answers.")
                return dataframe
            else:
                print("\t There are already pools. If you want to re-generate them, delete the existing ones first.")
            return dataframe_original

    def convert_dataframe2insqav1(self, pool_size=50, split_ratio=0.9):
        print("1. Generating pool of wrong answers...")
        self.corpus_dataframe = self.generate_wrong_answers_pool(self.corpus_dataframe, pool_size)
        if 'answer_ids' not in self.corpus_dataframe.columns.values:
            print('Generating answer_ids column (contains ids of all correct answers per question)')
            self.corpus_dataframe.loc[:, 'answer_ids'] = [x.split() for x in
                                                          self.corpus_dataframe.groupby('answer_id', as_index=False,
                                                                                        sort=False)['answer_id'].apply(
                                                              ' '.join)]
        print("2. Splitting into train/test/dev in proportion %0.2f/%0.2f ..." % split_ratio, (1 - split_ratio))
        [train, dev, test] = self.generate_split_corpus_dataframe(split_ratio)

    def save_list2file(self, lst, f):
        for elem in lst:
            f.write(elem)
            f.write(" ")
        f.write("\n")

    def save_corpus_dataframe2insqav1_answers(self, output_path):
        """
        Save the corpus dataframe (in InsQA v1 format) answers file.
        """
        with codecs.open(output_path, "w", "utf-8") as output_file:
            print("Writing output to " + output_path)
            cntr = 0
            for idx, row in self.corpus_dataframe.iterrows():
                print("%0.0f %%" % (100 * cntr / len(self.corpus_dataframe)), end='\r')
                output_file.write(str(cntr) + "\t")
                cntr += 1
                self.save_list2file(row['answer'].split(), output_file)
            output_file.write("\n")

    def save_dataframe2insqav1_split(self, dataframe, option, output_path):
        """
        Save the corpus dataframe (in InsQA v1 format) train/test/dev files.
        """
        with codecs.open(output_path, "w", "utf-8") as output_file:
            print("Writing output to " + output_path)
            cntr = 0
            for idx, row in dataframe.iterrows():
                print("%0.0f %%" % (100 * cntr / len(dataframe)), end='\r')
                cntr += 1
                if option != "train":
                    self.save_list2file(row['answer_ids'], output_file)
                self.save_list2file(row['question'].split(), output_file)
                output_file.write("\t")
                self.save_list2file(row['answer_ids'], output_file)
                if option != "train":
                    self.save_list2file(row['pool'].split(), output_file)
                output_file.write("\n")

    def save_dataframe2makecorpus(dataframe, corpus_filename):
        """
        Save the dataframe in the  following format (compatible with QALSTM Lua version):
        #  #question
        # "question text"
        # #answer
        # "answer text"
        """
        with codecs.open(corpus_filename, "w", "utf-8") as output_file:
            number_processed_questions = 0
            for index, row in dataframe.iterrows():
                number_processed_questions += 1
                output_file.write("\n#question\n\t " + row['question'] + "\n#answer\n\t " + row['answer'] + "\n")
        print("Successfully saved a corpus file %s" % (number_processed_questions, corpus_filename))

    def save_dataframe2insqav1(self, pool_size=50, split_ratio=0.9):
        """
        Save the corpus dataframe (in InsQA v1 format) files (compatible with QALSTM Lua version).
        """
        print("Generating pool of wrong answers...")
        self.corpus_dataframe = self.generate_wrong_answers_pool(self.corpus_dataframe, pool_size)
        if 'answer_ids' not in self.corpus_dataframe.columns.values:
            print('Generating answer_ids column (contains ids of all correct answers per question)')
            self.corpus_dataframe.loc[:, 'answer_ids'] = [x.split() for x in
                                                          self.corpus_dataframe.groupby('answer_id', as_index=False,
                                                                                        sort=False)['answer_id'].apply(
                                                              ' '.join)]
        print("Splitting into train/test/dev...")
        [train, dev, test] = self.generate_split_corpus_dataframe(split_ratio)
        self.save_corpus_dataframe2insqav1_answers(self.corpus_type + "_answers.label.token_idx.pool")
        self.save_dataframe2insqav1_split(train, "train", self.corpus_type + "_question.train.token_idx.pool")
        self.save_dataframe2insqav1_split(test, "test", self.corpus_type + "_question.test.label.token_idx.pool")
        self.save_dataframe2insqav1_split(dev, "dev", self.corpus_type + "_question.dev.label.token_idx.pool")


# # Lexical

# Module for generating and training word2vec vectors, and creating vocabularies.

# In[8]:


## custom class for more efficient looping over training corpus for word2vec in gensim

class Sentences(object):
    df = pd.DataFrame()
    lemmatize_option = False

    def __init__(self, df, lemmatize_option):
        self.df = df
        self.lemmatize_option = lemmatize_option

    def __iter__(self):
        res = []
        for index, row in self.df.iterrows():
            q_column_name = 'question'
            a_column_name = 'answer'
            if self.lemmatize_option:
                q_column_name += "_lemmata"
                a_column_name += "_lemmata"
            for token in row[q_column_name].split():
                res.append(token)
            for token in row[a_column_name].split():
                res.append(token)
        yield res


# In[9]:


class DataLexical():
    data_dir = ""
    corpus_type = ""
    vocab_type = ""
    vocab_path = ""
    corpus_dataframe = pd.DataFrame()
    encoded = False
    lemmatize_option = False
    wv_model = None
    vocab_c2w = {}
    vocab_w2c = {}
    vocab_w2n = {}
    vocab_w = []

    def __init__(self, data_dir="", corpus_type="", vocab_path="", datareader=None, dataframe=None,
                 lemmatize_option=False, encoded=False):
        self.data_dir = data_dir
        self.corpus_type = corpus_type
        self.vocab_path = vocab_path
        if self.data_dir != "":
            self.corpus_dataframe = DataReader(data_dir=self.data_dir, corpus_type=self.corpus_type).read_corpus()
        else:
            if datareader is None:
                try:
                    self.corpus_dataframe = dataframe
                except Exception as e:
                    print(
                        "Please, specify either a directory to load the corpus from or the dataframe with the corpus.")
            else:
                self.corpus_dataframe = datareader.corpus_dataframe
                self.corpus_type = datareader.corpus_type
                self.vocab_path = datareader.vocab_path
        self.encoded = encoded
        self.lemmatize_option = lemmatize_option
        if self.lemmatize_option:
            self.corpus_dataframe = self.lemmatize_texts(None, True)
        self.corpus_dataframe['question'] = [' '.join(self.preprocess_text(x)) for x in
                                             self.corpus_dataframe['question']]
        self.corpus_dataframe['answer'] = [' '.join(self.preprocess_text(x)) for x in self.corpus_dataframe['answer']]

    #### VOCABULARIES

    def generate_vocabs(self):
        if len(self.vocab_path) > 0:
            if self.vocab_type == 'c2w':
                print("Loading c2w and generating w2c ...")
                self.vocab_c2w = self.load_vocab(self.vocab_path)
                self.vocab_w2c = self.generate_vocab_w2c()
            elif self.vocab_type == 'w2c':
                print("Loading w2c and generating c2w ...")
                self.vocab_w2c = self.load_vocab(self.vocab_path)
                self.vocab_c2w = self.generate_vocab_c2w()
            else:
                print("Please, specify vocab type as either 'c2w' or 'w2c'. ")
        else:
            print("Generating c2w and w2c ...")
            self.vocab_w2c = self.generate_vocab_w2c()
            self.vocab_c2w = self.generate_vocab_c2w()
        return 0

    def generate_vocab_w(self, save=True):
        print("Generating word vocabulary.")
        vocab = set()
        if self.encoded:
            print("\t WARNING: the texts are encoded. The resulting vocabulary will contain codes, not words.")
        for idx, row in self.corpus_dataframe.iterrows():
            print("Progress: %0.0f %%" % (100 * idx / len(self.corpus_dataframe)), end='\r')
            if self.corpus_type == "ark":
                vocab.update((self.preprocess_text(row['question'])) + self.preprocess_text(
                    (row['answer'])) + self.preprocess_text((row['paragraph'])))
            else:
                vocab.update((self.preprocess_text(row['question'])) + self.preprocess_text((row['answer'])))
        vocab = list(vocab)
        vocab.sort()
        print("\t w vocabulary is generated")
        if save:
            self.save_vocab(vocab, "w")
        return vocab

    def generate_vocab_c2w(self, save=True):
        print("Generating code to word (c2w) vocabulary.")
        if len(self.vocab_w2c) == 0:
            if len(self.vocab_w) == 0:
                if self.encoded:
                    vocab_c2w = self.load_vocab(vocab_path=self.vocab_path, vocab_type='c2w', save=False)
                else:
                    self.generate_vocab_w()
                    vocab_c2w = {('idx_' + str(idx)): word for idx, word in enumerate(self.vocab_w)}
        else:
            vocab_c2w = {v: k for k, v in self.vocab_w2c.items()}
        print("\t c2w vocabulary is generated.")
        if save:
            self.save_vocab(vocab_c2w, "c2w")
        return vocab_c2w

    def generate_vocab_w2c(self, save=True):
        print("Generating word to code (w2c) vocabulary.")
        if len(self.vocab_c2w) == 0:
            if len(self.vocab_w) == 0:
                self.vocab_w = self.generate_vocab_w()
            vocab_w2c = {word: ('idx_' + str(idx)) for idx, word in enumerate(self.vocab_w)}
        else:
            vocab_w2c = {v: k for k, v in self.vocab_c2w.items()}
        print("\t w2c vocabulary is generated.")
        if save:
            print("\t w2c vocabulary is updated. ")
            self.save_vocab(vocab_w2c, "w2c")
        return vocab_w2c

    def generate_vocab_w2n(self, save=True):
        print("Generating Word-to-Number (w2n) vocabulary.")
        if len(self.vocab_w) == 0:
            self.vocab_w = self.generate_vocab_w()
        vocab_w2n = {word: idx for idx, word in enumerate(self.vocab_w)}
        print("\t w2n vocabulary is generated.")
        if save:
            self.save_vocab(vocab_w2n, "w2n")
        return vocab_w2n

    def save_vocab(self, vocab, vocab_type='c2w'):
        if vocab_type == 'c2w':
            self.vocab_c2w = vocab
        elif vocab_type == 'w2c':
            self.vocab_w2c = vocab
        elif vocab_type == 'w':
            self.vocab_w = vocab
        elif vocab_type == 'w2n':
            self.vocab_w2n = vocab
        else:
            print('\t Unknown type of vocabulary. Failed to save.')
            return -1
        print("\t Saved " + vocab_type + " vocabulary to the class variable.")

    def load_vocab(self, vocab_path="", vocab_type='c2w', save=True):
        if vocab_path == "":
            vocab_path = self.vocab_path
        print("Loading " + self.vocab_type + " vocabulary from " + vocab_path)
        if self.vocab_path != "" and self.vocab_path != vocab_path:
            print("\t Warning: overwriting vocabulary path.")
        self.vocab_path = vocab_path
        if vocab_type == 'c2w':
            with codecs.open(self.vocab_path, encoding='utf8') as input_file:
                content = input_file.readlines()
            vocab = {x.split()[0]: x.split()[1] for x in content}
        else:
            print("\t Sorry, this type of vocabulary load is not supported yet.")
        if save:
            self.save_vocab(vocab, vocab_type)
        print("\t Vocabulary loaded.")
        return vocab

    #### ENCODING-DECODING

    def word2code(self, word):
        return self.vocab_w2c[word]

    def code2word(self, code):
        return self.vocab_c2w[code]

    def text2code(self, text):
        if isinstance(text, str):
            text = text.split()
        coded_text = []
        for word in text:
            coded_text.append(self.word2code(word))
        return coded_text

    def code2text(self, coded_text):
        if isinstance(coded_text, str):
            coded_text = coded_text.split()
        text = []
        for code in coded_text:
            text.append(self.code2word(code))
        return ' '.join(text)

    def decode_texts(self, dataframe=None, save=True):
        print("Decoding texts.")
        if dataframe is None:
            dataframe = self.corpus_dataframe
        if self.vocab_c2w == {}:
            if self.vocab_path != "":
                self.load_vocab(self.vocab_path)
            else:
                self.generate_vocab_c2w()
        result = dataframe.copy()
        for idx, row in dataframe.iterrows():
            result.iloc[idx]['question'] = self.code2text(row['question'])
            result.iloc[idx]['answer'] = self.code2text(row['answer'])
            if self.corpus_type == "ark":
                result.iloc[idx]['paragraph'] = self.code2text(row['paragraph'])
        if save:
            self.corpus_dataframe = result
            self.encoded = False
            print("\t Corpus is updated. ")
        return result

    #### LINGUISTIC PREPROCESSING

    def preprocess_text(self, text):
        ''' add spaces to punctuation, lemmatize and lower'''
        for punct in ['?', ',', '.', ':', '!', "\'", "\"", "(", ")", "[", "]", "%", "*"]:
            text = text.replace(punct, " " + punct + " ")
        text = text.replace("\'s", "'s")
        text = text.replace("  ", " ")
        text = text.split()
        if self.lemmatize_option:
            return [lemmatizer.lemmatize(word.lower()) for word in text]
        else:
            return [word.lower() for word in text]

    def lemmatize_texts(self, dataframe=None, save=True):
        print("Lemmatizing texts in corpus.")
        if dataframe is None:
            if self.encoded:
                self.decode_texts()
            dataframe = self.corpus_dataframe.copy()
        dataframe['question_lemmata'] = ''
        dataframe['answer_lemmata'] = ''
        for idx, row in dataframe.iterrows():
            print("Progress: %0.0f %%" % (100 * idx / len(dataframe)), end='\r')
            dataframe.iloc[idx]['question_lemmata'] = ' '.join(self.preprocess_text(row['question']))
            dataframe.iloc[idx]['answer_lemmata'] = ' '.join(self.preprocess_text(row['answer']))
            if self.corpus_type == "ark":
                dataframe.iloc[idx]['paragraph_lemmata'] = ' '.join(self.preprocess_text(row['paragraph']))
        if save:
            self.corpus_dataframe = dataframe
            print("\t Corpus is updated. ")
        return dataframe

    def pos_tag_texts(self, dataframe=None, save=True):
        print("POS tagging texts in corpus.")
        if dataframe is None:
            if self.encoded:
                self.decode_texts()
            dataframe = self.corpus_dataframe.copy()
        dataframe['question_pos'] = ''
        dataframe['answer_pos'] = ''
        for idx, row in dataframe.iterrows():
            print("Progress: %0.0f %%" % (100 * idx / len(dataframe)), end='\r')
            question_text = spacy_model(' '.join(self.preprocess_text(row['question'])))
            answer_text = spacy_model(' '.join(self.preprocess_text(row['answer'])))
            dataframe.iloc[idx]['question_pos'] = ' '.join([token.pos_ for token in question_text])
            dataframe.iloc[idx]['answer_pos'] = ' '.join([token.pos_ for token in answer_text])
            if self.corpus_type == "ark":
                dataframe.iloc[idx]['paragraph_pos'] = ' '.join(
                    [token.pos_ for token in spacy_model(self.preprocess_text(row['paragraph']))])
        if save:
            self.corpus_dataframe = dataframe
            print("\t Corpus is updated. ")
        return dataframe

    #### WORD2VEC

    def generate_word2vec(self, word2vec_filename, word2vec_vec_path="", dim=100, use_gensim=False, option="load"):
        ''' Train or load word2vec model and save it.'''
        # train or load model
        if option == "train":
            model = self.train_word2vec(word2vec_vec_path)
        else:
            if use_gensim:
                print("Loading custom word2vec model with gensim.")
                model = gensim.models.Word2Vec.load(word2vec_vec_path)
            else:
                print("Loading pre-trained Google word2vec model with gensim.")
                model = gensim.models.KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin",
                                                                        binary=True, unicode_errors='ignore')
        if len(self.vocab_w) == 0:
            self.vocab_w = self.generate_vocab_w()
        # save model
        with codecs.open(word2vec_filename, "w", "utf-8") as output_file:
            for idx, word in enumerate(self.vocab_w):
                print(idx / len(self.vocab_w))
                output_file.write(word + " ")
                try:
                    wv = model.wv[word]
                except Exception as e:
                    print("\t WARNING: word " + word + " not found, replacing with a random vector.")
                    wv = np.random.rand(dim)
                for elem in wv:
                    output_file.write(str(elem) + " ")
                output_file.write("\n")

    def train_word2vec(self, save_path, wv_size=100):
        print("Training word2vec model with gensim.")
        sentences = Sentences(df=self.corpus_dataframe, lemmatize_option=self.lemmatize_option)
        model = gensim.models.Word2Vec(iter=1, min_count=1, size=wv_size)
        model.build_vocab(sentences)
        if len(self.vocab_w) == 0:
            self.vocab_w = self.generate_vocab_w()
        model.train(sentences, total_examples=len(self.vocab_w), epochs=1)
        model.save(save_path)
        print("\t word2vec model was trained by gensim with dimension " + str(wv_size) + " and saved to " + save_path)
        self.wv_model = model
        return model
