import os

os.environ['TRANSFORMERS_CACHE'] = 'cache'

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering
import torch
nli_model = AutoModelForSequenceClassification.from_pretrained('valhalla/distilbart-mnli-12-1', cache_dir='cache')
tokenizer = AutoTokenizer.from_pretrained('valhalla/distilbart-mnli-12-1', cache_dir='cache')


qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad", cache_dir='cache')
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad", cache_dir='cache')

import tensorflow_text  # Necessary. Otherwise tensorflow.python.framework.errors_impl.NotFoundError
import tensorflow_hub as hub
from keras.models import load_model

#### Import Universal Encoder
# tensorflow version requires 2.1.0
USE_EN_EMBEDDINGS_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

embeddings_dict = {
    "en": hub.load(USE_EN_EMBEDDINGS_URL),
}
