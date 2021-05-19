import os

os.environ['TRANSFORMERS_CACHE'] = 'cache'


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

def answer_question(question, text):
  inputs = qa_tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
  input_ids = inputs["input_ids"].tolist()[0]

  text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)
  answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)

  answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
  answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

  answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
  return answer