from query_functions import *
from example_flows import *
from question_answering import *

def test1():
  sources = ["US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day"]
  query = "feel good story"
  result = is_relevant(query, sources)
  print(result)
  
def test2():
  query = 'What do you have for oily skin?'
  product_descriptions = ['Neutrogena is good for oily skin and acne']
  product_names = ['Neutrogena']
  answer = find_most_relevant_products(query, product_names, product_descriptions)
  print(answer)

def test3():
  sources = ["US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day"]
  query = "feel good story"
  result = is_relevant(query, sources, use_api=True)
  print(result)

def test4():
  query = 'What do you have for oily skin?'
  product_descriptions = load_product_descriptions()
  product_names = ['Neutrogena', '1452', '1454']
  answer = find_most_relevant_products(query, product_names, product_descriptions)
  print(answer)

def test5():
  query = 'What do you have for cleaning the floor?'
  product_descriptions = load_product_descriptions()
  product_names = ['Neutrogena', '1452', '1454']
  answer = find_most_relevant_products(query, product_names, product_descriptions)
  print(answer)
  
def test6():
  question = "How many pretrained models are available in Transformers?"
  text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""
  print(answer_question(question, text))


