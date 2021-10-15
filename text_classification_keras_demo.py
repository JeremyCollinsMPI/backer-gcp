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
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Sequential
from keras import layers
from keras.optimizers import SGD

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
USE_ZH_EMBEDDINGS_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"

embeddings = {
    "en": hub.load(USE_EN_EMBEDDINGS_URL),
    "zh": hub.load(USE_ZH_EMBEDDINGS_URL),
}

def make_mapping(context):
    mapping = {}
    label_to_index_mapping = {}
    for i in range(len(context['unique_labels'])):
        mapping[str(i)] = context['unique_labels'][i]  
        label_to_index_mapping[context['unique_labels'][i]] = i
    if context['version']: 
        version = str(context['version'])
    else:
        version = '0'
    json.dump(mapping, open('mapping/mapping.json', 'w'), indent=4)
    return {'result': mapping, 'label_to_index_mapping': label_to_index_mapping}

def prepare_training_and_test_data(context):
    target_language = context["language"]
    if target_language == "en":
        embed = embeddings.get("en")
    else:
        embed = embeddings.get("zh")
        
    random.shuffle(context['data'])
    if not context.get('sample_training_ratio'):
       context['sample_training_ratio'] = 0.99
    assert 0 < context['sample_training_ratio'] < 1, "sample_training_ratio is invalid."

    sample_training_size = int(math.floor(context['sample_training_ratio']*len(context['data'])))
    sample_testing_size = int(math.floor((1-context['sample_training_ratio'])*len(context['data'])))

    print("sample_training_size: %d, ratio: %f" % (sample_training_size, context['sample_training_ratio']))
    print("sample_testing_size: %d, ratio: %f" % (sample_testing_size, (1-context['sample_training_ratio'])))

    assert sample_training_size + sample_testing_size <= len(
        context['data']), "Error: the sum of training (%d) and testing size (%d) is greater than the size of data (%d)" % (
        sample_training_size, sample_testing_size, len(context['data']))

    # training set
    train_x_text = [d["text"] for d in context['data'][0:sample_training_size]] 
    train_y_text = [d["label"] for d in context['data'][0:sample_training_size]]
    train_y_index = [context['label_to_index_mapping'][d["label"]] for d in context['data'][0:sample_training_size]]
    
    # test set    
    
    test_x_text = [d["text"] for d in context['data'][sample_training_size: sample_training_size + sample_testing_size]] 
    test_y_text = [d["label"] for d in context['data'][sample_training_size: sample_training_size + sample_testing_size]]
    test_y_index = [context['label_to_index_mapping'][d["label"]] for d in context['data'][sample_training_size: sample_training_size + sample_testing_size]]

    test_x_num = embed(test_x_text).numpy()

    train_y_num = list()
    for index in train_y_index:
        temp_list = [0] * len(context['unique_labels'])
        temp_list[index] = 1
        train_y_num.append(temp_list)
    train_y_num = np.array(train_y_num)

    test_y_num = list()
    for index in test_y_index:
        temp_list = [0] * len(context['unique_labels'])
        temp_list[index] = 1
        test_y_num.append(temp_list)
    test_y_num = np.array(test_y_num)
    size = 200
    train_x_num = embed(train_x_text[0:size])
    length = len(train_x_text)
    for i in range(size, len(train_x_text)+1, size):
        train_x_num = np.concatenate((train_x_num, embed(train_x_text[i:(min(i+size, length))]).numpy()))
    return {"train_x_num": train_x_num ,'train_x_text': train_x_text, 'train_y_text': train_y_text, 'train_y_num': train_y_num,
    'test_x_text': test_x_text, 'test_y_text': test_y_text, 'train_y_index': train_y_index, 'test_y_index': test_y_index, 
    'test_x_num': test_x_num, 'test_y_num': test_y_num}

def train_general_classifier(context):
    '''
    Structure of context is {'data': [{'text': 'hello', 'label': 'greeting''}]}
    '''
    unique_labels = np.unique([x['label'] for x in context['data']])
    context['unique_labels'] = unique_labels
    mapping_result = make_mapping(context)
    context['label_to_index_mapping'] = mapping_result['label_to_index_mapping']
    
    label_number = len(unique_labels)
    prepare_training_and_test_data_result = prepare_training_and_test_data(context)
    input_dim = len(prepare_training_and_test_data_result['test_x_num'].tolist()[0])

    keras_model = Sequential()
    keras_model.add(
        layers.Dense(768, input_dim=input_dim, kernel_initializer='glorot_normal', bias_initializer='zeros'))
    keras_model.add(layers.BatchNormalization())
    keras_model.add(layers.Activation('relu'))
    droprate = 0.3
    keras_model.add(layers.Dropout(droprate, noise_shape=None, seed=None))
    keras_model.add(layers.Dense(label_number, activation='sigmoid'))
    adagrad = keras.optimizers.Adagrad()
    keras_model.compile(optimizer=adagrad,
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'categorical_accuracy'])

    class_weights_orig = class_weight.compute_class_weight('balanced',
                                                           classes=np.unique(prepare_training_and_test_data_result['train_y_text']),
                                                           y=prepare_training_and_test_data_result['train_y_text'])
    directory = 'model'
    Path(directory).mkdir(parents=True, exist_ok=True)
    modelcallbacks = [EarlyStopping(patience=20, restore_best_weights=False),
                      ModelCheckpoint(filepath=context['keras_model_path'],
                                      save_best_only=True)]
    train_x_text = prepare_training_and_test_data_result['train_x_text']
    test_x_num = prepare_training_and_test_data_result['test_x_num']
    test_y_num = prepare_training_and_test_data_result['test_y_num']
    train_x_num = prepare_training_and_test_data_result['train_x_num']
    train_y_num = prepare_training_and_test_data_result['train_y_num']
    batch_size = 128
    steps_per_epoch = len(train_x_text)/batch_size

    history = keras_model.fit_generator(generator(train_x_num, train_y_num, 128),
                              epochs=2000, 
                              validation_data=(test_x_num, test_y_num), 
                              callbacks=modelcallbacks, 
                              class_weight=class_weights_orig,
                              steps_per_epoch=steps_per_epoch)
    
    return keras_model
    
def generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    while True:
        X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
        counter += 1
        yield X_batch,y_batch
        if counter >= number_of_batches:
            counter = 0

def start_train(region: str, lang: str, context=None):
    
    import pandas as pd
    
    if not context:
        context = {}
    
    form_types = ['card_servicing', 'ipo_servicing']
    
    special_intents = ["smartchat_transfer_to_agent", "smartchat_unclassified"]

    data = []
    
    for form_type in form_types:
        file_path = 'data/projects/smartchat_hk_dform/%s/%s/nlu.csv' % (form_type, lang,)
        df = pd.read_csv(file_path)
        for row in df.iterrows():
            data.append({'text': row[1]['sample'], 'label': row[1]['intent']})
    
#     for intents in special_intents:
#         file_path = 'data/projects/%s/%s/nlu.csv' % (intents, lang,)
#         df = pd.read_csv(file_path)
#         for row in df.iterrows():
#             data.append({'text': row[1]['sample'], 'label': row[1]['intent']})

    version = '0'
    keras_model_path = f'model/{region}_{lang}_{version}.h5'
    model = train_general_classifier({'data': data, 'region': region, 'language': lang,
                        'keras_model_path': keras_model_path, 'version': version})

def train_for_smarthub_data(context):
    import pandas as pd
        
    form_types = ['card_servicing', 'ipo_servicing']
    
    special_intents = ["smartchat_transfer_to_agent", "smartchat_unclassified"]

    data = []
    
    for form_type in form_types:
        file_path = 'data/smarthub/nlu.csv'
        df = pd.read_csv(file_path)
        for row in df.iterrows():
            data.append({'text': row[1]['sample'], 'label': row[1]['intent']})
    
    version = '0'
    keras_model_path = f'model/smarthub_model.h5'
    region = 'hk'
    lang = 'en'
    model = train_general_classifier({'data': data, 'region': region, 'language': lang,
                        'keras_model_path': keras_model_path, 'version': version, 'sample_training_ratio': 0.9})


def find_top_n_matches(prob_y: List[float], mapping, n=3):
    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)
    indices = list(reversed(argsort(prob_y[:])))
    indices = indices[:n]
    result = []
    for index in indices:
        result.append({'label': mapping[str(index)], 'confidence': prob_y[index]})
    return result

def classify_text(context):
    if context['method'] == 'keras':
        model_path = context.get('model_path')
        model = load_model(model_path)
        if context['language'] == 'en':
            embeddings_url = USE_EN_EMBEDDINGS_URL  
            embeddings = hub.load(embeddings_url)     
        with open(context['mapping_path'], "r") as f:
            mapping = json.load(f)
        texts = [context['text']]
        embeddings(texts).numpy()
        encoded_texts = embeddings(texts).numpy()
        prob_y = model.predict_proba(encoded_texts)
        prob_y = prob_y[0].tolist()
        matches = find_top_n_matches(prob_y, mapping, n=3)
        return {'result': matches}
        
        


if __name__ == "__main__":
#     start_train("hk", "en")
#     x = classify_text({"text": "how to check ipo result", "model_path": "model/hk_en_0.h5", 
#         "mapping_path": "mapping/hk_en_0.json", "language": "en", "method": "keras"})
#     print(x)
    train_for_smarthub_data({})
    
    
    
    
    
    
    
    
    
    
    
    
