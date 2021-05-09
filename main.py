import os

from flask import Flask, request
from inference import *
from question_answering import *
from ner import *

app = Flask(__name__)

@app.route('/status', methods=['GET'])
def status():
  return {'statusCode': 200}
  
@app.route('/run', methods=['POST'])
def run_path():
  content = request.get_json(force=True)
  sentences = content['sentences']
  query = content['query']
  threshold = 0.25
  result = []
  for sentence in sentences:
    print(len(sentence))
    try:
      prob = evaluate_inference(sentence, 'This sentence is about' + query)
    except:
      continue
    print(sentence)
    print(prob)
    if prob > threshold:
      result.append(sentence)
  return {'result': result}
    
@app.route('/inference', methods=['POST'])
def inference_path():
  content = request.get_json(force=True)
  sentences = content['sentences']
  hypothesis = content['hypothesis']
  threshold = 0.8
  result = []
  for sentence in sentences:
    prob = evaluate_inference(sentence, hypothesis)
    if prob > threshold:
      result.append(sentence)
  return {'result': result}  

@app.route('/question_answering', methods=['POST'])
def question_answering_path():
  content = request.get_json(force=True)
  text = content['text']
  question = content['question']
  answer = answer_question(question, text)
  return {'result': answer}

@app.route('/ner', methods=['POST'])
def ner_path():
  content = request.get_json(force=True)
  text = content['text']
  return {'result': ner(text)}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
