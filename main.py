import os

from flask import Flask, request
from transformers import pipeline
from inference import *

app = Flask(__name__)

@app.route('/status', methods=['GET'])
def status():
  return {'statusCode': 200}
  
@app.route('/run', methods=['POST'])
def run_path():
  content = request.get_json(force=True)
  sentences = content['sentences']
  query = content['query']
  threshold = 0.5
  result = []
  for sentence in sentences:
    prob = evaluate_inference(sentence, 'This sentence is about' + query)
    if prob > threshold:
      result.append(sentence)
  return {'result': result}
    
@app.route('/inference', methods=['POST'])
def inference_path():
  content = request.get_json(force=True)
  sentences = content['sentences']
  hypothesis = content['hypothesis']
  threshold = 0.5
  result = []
  for sentence in sentences:
    prob = evaluate_inference(sentence, hypothesis)
    if prob > threshold:
      result.append(sentence)
  return {'result': result}  

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
