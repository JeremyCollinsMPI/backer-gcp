from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering
import torch
nli_model = AutoModelForSequenceClassification.from_pretrained('valhalla/distilbart-mnli-12-1')
tokenizer = AutoTokenizer.from_pretrained('valhalla/distilbart-mnli-12-1')


qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

