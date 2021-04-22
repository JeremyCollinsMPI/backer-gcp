# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')

def evaluate_inference(premise, hypothesis):
  # run through model pre-trained on MNLI
  x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')
  logits = nli_model(x)[0]
  # we throw away "neutral" (dim 1) and take the probability of
  # "entailment" (2) as the probability of the label being true 
  entail_contradiction_logits = logits[:,[0,2]]
  probs = entail_contradiction_logits.softmax(dim=1)
  prob_label_is_true = probs[:,1]
  return prob_label_is_true[0]

if __name__ == "__main__":
  premise = 'I want to see the world'
  hypothesis = 'This example is travel.'
  result = evaluate_inference(premise, hypothesis)
  print(result)
  if result > 0.5:
    print('yes')
  if result < 0.5:
    print('no')

