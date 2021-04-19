import os

from flask import Flask
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
 
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
    temp = classifier(sequence_to_classify, candidate_labels)
    if temp['scores'][0] > threshold:
      result.append(sentence)
  return {'result': result}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))



