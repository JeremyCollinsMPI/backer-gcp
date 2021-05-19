import os

os.environ['TRANSFORMERS_CACHE'] = 'cache'

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering
import torch
nli_model = AutoModelForSequenceClassification.from_pretrained('valhalla/distilbart-mnli-12-1', cache_dir='cache')
tokenizer = AutoTokenizer.from_pretrained('valhalla/distilbart-mnli-12-1', cache_dir='cache')


qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad", cache_dir='cache')
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad", cache_dir='cache')

