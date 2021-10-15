# Seed value using https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)


import math
import math

import pandas as pd
import tensorflow_text  # Necessary. Otherwise tensorflow.python.framework.errors_impl.NotFoundError
import tensorflow_hub as hub
import numpy as np

from sklearn.utils import class_weight
import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD

from typing import List, Text, Tuple
from operator import itemgetter

from keras.models import load_model

import random

# from data_handler.json_helper import select
import json

from pathlib import Path

#### Import Universal Encoder
# tensorflow version requires 2.1.0
USE_EN_EMBEDDINGS_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

embeddings_dict = {
    "en": hub.load(USE_EN_EMBEDDINGS_URL),
}

default_model_path = 'model/hk_en_0.h5'
default_mapping_path = 'mapping/hk_en_0.json'
model = load_model(default_model_path)
mapping = json.load(open(default_mapping_path, "r"))
embeddings = embeddings_dict['en']


def find_top_n_matches(context):
    prob_y = context['prob_y']
    mapping = context['mapping']
    n = context.get('n')
    if not n:
        n = len(mapping.keys())
    text = context.get('text')
    if not text:
        text = ''
    confidence_threshold = context.get('confidence_threshold')
    if not confidence_threshold:
        confidence_threshold = 0.5
    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)
    indices = list(reversed(argsort(prob_y[:])))
    indices = indices[:n]
    result = []
    for index in indices:
        confidence = prob_y[index]
        if confidence >= confidence_threshold:
            result.append({'text': text, 'label': mapping[str(index)], 'confidence': confidence})
    return {'result': result}

def classify_text(context):
    if context['method'] == 'keras':
        texts = [context['text']]
        embeddings(texts).numpy()
        encoded_texts = embeddings(texts).numpy()
        prob_y = model.predict_proba(encoded_texts)
        prob_y = prob_y[0].tolist()
        matches = find_top_n_matches({**context, **{'prob_y': prob_y, 'mapping': mapping, 
            'text': context['text']}})['result']
        return {'result': matches}

def find_document_labels(context):
    texts = split_text(context)['result']
    result = []
    for text in texts:
        label = classify_text({**context, **{"text": text}})['result']
        result.append(label)
    return {'result': result}
















































# from pathlib import Path
# 
# 
# '''
# first basic version; just load the already trained classifier.
# 
# 
# '''
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# def train_text_classifier_keras(context):
#     texts, labels = [x['text'] for x in context['data']], [x['label'] for x in context['data']]
#     from sklearn.model_selection import train_test_split    
#     train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=.2)
# 
# 
# def train_text_classifier_huggingface(context):
#     texts, labels = [x['text'] for x in context['data']], [x['label'] for x in context['data']]
#     from sklearn.model_selection import train_test_split    
#     train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=.2)
#     from transformers import DistilBertTokenizerFast
#     tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#     train_encodings = tokenizer(train_texts, truncation=True, padding=True)
#     val_encodings = tokenizer(val_texts, truncation=True, padding=True)
#     import torch
#     class TextClassificationDataset(torch.utils.data.Dataset):
#         def __init__(self, encodings, labels):
#             self.encodings = encodings
#             self.labels = labels
#         def __getitem__(self, idx):
#             item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#             item['labels'] = torch.tensor(self.labels[idx])
#             return item
#         def __len__(self):
#             return len(self.labels)
#     train_dataset = TextClassificationDataset(train_encodings, train_labels)
#     val_dataset = TextClassificationDataset(val_encodings, val_labels)
#     test_dataset = ITextClassificationDataset(test_encodings, test_labels)
#     from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
#     training_args = TrainingArguments(
#         output_dir='./results',          # output directory
#         num_train_epochs=3,              # total number of training epochs
#         per_device_train_batch_size=16,  # batch size per device during training
#         per_device_eval_batch_size=64,   # batch size for evaluation
#         warmup_steps=500,                # number of warmup steps for learning rate scheduler
#         weight_decay=0.01,               # strength of weight decay
#         logging_dir='./logs',            # directory for storing logs
#         logging_steps=10,
#     )
#     model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
#     trainer = Trainer(
#         model=model,                         # the instantiated 🤗 Transformers model to be trained
#         args=training_args,                  # training arguments, defined above
#         train_dataset=train_dataset,         # training dataset
#         eval_dataset=val_dataset             # evaluation dataset
#     )
#     trainer.train()
#     return {'result': model}
# 
# def train_text_classifier(context):
#     if context['method'] == 'huggingface':
#         return train_text_classifier_huggingface(context)
#     elif context['method'] == 'keras':
#         return train_text_classifier_keras(context)

