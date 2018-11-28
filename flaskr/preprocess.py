import time
import logging
import pickle
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import operator
import sys
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
import enchant
import spacy

# sys.path.append("../../codes")
from loading_preprocessing_TC import *

DATASET_PATH = '../resources/datasets/'
SCORER_PATH = '../resources/datasets/semeval/scorer/'
DATA_PATH = '../out/data/semeval/'

d = enchant.Dict("en_US")
stemmer = EnglishStemmer()
nlp = spacy.load('en')

# text = Text("Helo!!! Here :))) How is Qatar? ", "answer", 1)
# print(text.tokenize())
# print(text.stemmatize())
# print(text.lemmatize())
# print(text.clean())
# print(text.pos_tag())
# print(text.ner())
# print(text.replace_ne())
# print(text.spell_check())

files = [DATASET_PATH + 'semeval/train/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml',
         DATASET_PATH + 'semeval/train/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml']
train_xml = read_xml(files)

files = [DATASET_PATH + 'semeval/dev/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml']
dev_xml = read_xml(files)

files = [SCORER_PATH + 'SemEval2017_task3_test/English/SemEval2017-task3-English-test-subtaskA.xml']
test_xml = read_xml(files)

files = [SCORER_PATH + 'SemEval2016_task3_test/English/SemEval2016-Task3-CQA-QL-test-subtaskA.xml']
test2016_xml = read_xml(files)

# TODO: Fix deprecation of pandas set_value
test2016, answer_texts_test2016 = xml2dataframe_NoLabels(test2016_xml, 'test2016')
test2016_, answer_texts_test2016_ = xml2dataframe_Labels(test2016_xml, 'test2016')
test2016_['answer_id'] = test2016_['answer_ids']
lst_col = 'answer_id'
test2016__expanded = pd.DataFrame({col: np.repeat(test2016_[col].values, test2016_[lst_col].str.len())
                                   for col in test2016_.columns.difference([lst_col])
                                   }).assign(**{lst_col: np.concatenate(test2016_[lst_col].values)})[
    test2016_.columns.tolist()]
test2016_ = test2016__expanded.merge(answer_texts_test2016_, on='answer_id', how='left')

dev, answer_texts_dev = xml2dataframe_NoLabels(dev_xml, 'dev')
print(dev.head())

dev_, _ = xml2dataframe_Labels(dev_xml, 'dev')
print(dev.head())

test, answer_texts_test = xml2dataframe_NoLabels(test_xml, 'test')
print(test.head())

test_, _ = xml2dataframe_Labels(test_xml, 'test')
print(test.head())

train, answer_texts_train = xml2dataframe_Labels(train_xml, 'train')
print(train.head())
print(answer_texts_train.head())

# Expanding dataset
# Leave only questions with > 0 correct answers
train = train[train.answer_ids.apply(len) > 0]
print(len(train))

train['answer_id'] = train['answer_ids']
lst_col = 'answer_id'
train_expanded = pd.DataFrame({col:np.repeat(train[col].values, train[lst_col].str.len())
       for col in train.columns.difference([lst_col])
       }).assign(**{lst_col:np.concatenate(train[lst_col].values)})[train.columns.tolist()]
train = train_expanded.merge(answer_texts_train, on = 'answer_id', how = 'left')
print(train.head())

pickle.dump(answer_texts_train, open(DATA_PATH + "answer_texts_train_NoLing.p", "wb"))
pickle.dump(answer_texts_dev, open(DATA_PATH + "answer_texts_dev_NoLing.p", "wb"))
pickle.dump(answer_texts_test, open(DATA_PATH + "answer_texts_test_NoLing.p", "wb"))
pickle.dump(train, open(DATA_PATH + "train_NoLing.p", "wb"))
pickle.dump(train, open(DATA_PATH + "train-expanded_NoLing.p", "wb"))
pickle.dump(dev, open(DATA_PATH + "dev-NoLabels_NoLing.p", "wb"))
pickle.dump(test, open(DATA_PATH + "test-NoLabels_NoLing.p", "wb"))
pickle.dump(dev_, open(DATA_PATH + "dev-Labels_NoLing.p", "wb"))
pickle.dump(test_, open(DATA_PATH + "test-Labels_NoLing.p", "wb"))

pickle.dump(answer_texts_test2016, open(DATA_PATH + "answer_texts_test2016_NoLing.p", "wb"))
pickle.dump(test2016, open(DATA_PATH + "test2016-NoLabels_NoLing.p", "wb"))
pickle.dump(test2016_, open(DATA_PATH + "test2016-Labels_NoLing.p", "wb"))

# Transforming dataset

transformations = [('heavy_clean',  Text.heavy_clean), ('clean',  Text.clean),  ('tokenize',  Text.tokenize), ('spell_check',  Text.spell_check),
                   ('stemmatize',  Text.stemmatize), ( 'lemmatize',  Text.lemmatize),
                   ('pos_tag',  Text.pos_tag), ('ner',  Text.ner), ('replace_ne',  Text.replace_ne)]

for idx, row in train.iterrows():
    train.set_value(idx, 'question', Text(row['question'], 'question', idx))
    train.set_value(idx, 'answer', Text(row['answer'], 'answer', idx))
for transformation in transformations:
    train = transform_dataset(train, transformation)

for dataset in [test, dev, test_, dev_]:
    for idx, row in dataset.iterrows():
        dataset.set_value(idx, 'question', Text(row['question'], 'question', idx))
    for transformation in transformations:
        dataset = transform_dataset(dataset, transformation)

for dataset in [answer_texts_train, answer_texts_dev, answer_texts_test]:
    for idx, row in dataset.iterrows():
        dataset.set_value(idx, 'answer', Text(row['answer'], 'answer', idx))
    for transformation in transformations:
        dataset = transform_dataset(dataset, transformation)

for dataset in [test2016, test2016_]:
    for idx, row in dataset.iterrows():
        dataset.set_value(idx, 'question', Text(row['question'], 'question', idx))
        if 'answer' in row.keys():
            dataset.set_value(idx, 'answer', Text(row['answer'], 'question', idx))
    for transformation in transformations:
        dataset = transform_dataset(dataset, transformation)

for dataset in [answer_texts_test2016]:
    for idx, row in dataset.iterrows():
        dataset.set_value(idx, 'answer', Text(row['answer'], 'answer', idx))
    for transformation in transformations:
        dataset = transform_dataset(dataset, transformation)

pickle.dump(answer_texts_train, open(DATA_PATH + "answer_texts_train_Ling.p", "wb"))
pickle.dump(answer_texts_dev, open(DATA_PATH + "answer_texts_dev_Ling.p", "wb"))
pickle.dump(answer_texts_test, open(DATA_PATH + "answer_texts_test_Ling.p", "wb"))
pickle.dump(train, open(DATA_PATH + "train-expanded_Ling.p", "wb"))
pickle.dump(dev, open(DATA_PATH + "dev-NoLabels_Ling.p", "wb"))
pickle.dump(test, open(DATA_PATH + "test-NoLabels_Ling.p", "wb"))
pickle.dump(dev_, open(DATA_PATH + "dev-Labels_Ling.p", "wb"))
pickle.dump(test_, open(DATA_PATH + "test-Labels_Ling.p", "wb"))

pickle.dump(answer_texts_test2016, open(DATA_PATH + "answer_texts_test2016_Ling.p", "wb"))
pickle.dump(test2016, open(DATA_PATH + "test2016-NoLabels_Ling.p", "wb"))
pickle.dump(test2016_, open(DATA_PATH + "test2016-Labels_Ling.p", "wb"))
