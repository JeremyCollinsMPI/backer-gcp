import os

from flask import Flask, request
from inference import *
from question_answering import *
from text_classification import classify_text

# from ner import *
import os

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

@app.route('/classify_intent', methods=['POST'])
def classify_intent_path():
  content = request.get_json(force=True)
  intents_and_examples = content['intents_and_examples']
  input = content['input']
  probs = []
  for example in intents_and_examples.keys():
    prob = evaluate_inference(input, example)
    probs.append([example, prob])
  probs = sorted(probs, key=lambda x:x[1], reverse=True)
  print(probs)
  best_example = probs[0][0]
  confidence = float(probs[0][1])
  intent = intents_and_examples[best_example]
  return {'result': {'name': intent, 'confidence': confidence}}

@app.route('/question_answering', methods=['POST'])
def question_answering_path():
  content = request.get_json(force=True)
  context = content
  answer_question_result = answer_question(context)
  score = float(answer_question_result['score'])
  if 'threshold' not in context.keys():
    context['threshold'] = 4
  if score > context['threshold']:
    return {'result': answer_question_result['result'], 'score': score}
  else:
    return {'result': 'NA', 'score': score}

@app.route('/classify_text', methods=['POST'])
def classify_text_path():
  content = request.get_json(force=True)
  context = content
  context['language'] = 'en'
  context['method'] = 'keras'
  classify_text_result = classify_text(context)
  return classify_text_result

@app.route('/cache_content')
def cache_content_path():
  return {'result': os.listdir('/root/.cache/huggingface/transformers/')}
# @app.route('/ner', methods=['POST'])
# def ner_path():
#   content = request.get_json(force=True)
#   text = content['text']
#   return {'result': ner(text)}

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080, threaded=False)
