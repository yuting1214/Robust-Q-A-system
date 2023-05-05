import os
import time
import copy
import numpy as np
import collections
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, BertForQuestionAnswering
import evaluate

# Model Evaluation
class Answer:
    def __init__(self, raw_dataset_dict, Token_dataset_dict):
        self.example_to_features = collections.defaultdict(list)
        self.raw_dataset_dict = raw_dataset_dict
        self.Token_dataset_dict = Token_dataset_dict
        self.data_size = raw_dataset_dict.num_rows
        for idx, feature in tqdm(enumerate(Token_dataset_dict)):
            self.example_to_features[feature["example_id"]].append(idx)
            
    def predict_answer(self, test_dataloader, model, device, n_best=20, max_answer_length=30):
        # 1. Predict logits
        model.eval()
        start_logits = []
        end_logits = []
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(device)
                outputs = model(**batch)
                start_logits.append(outputs.start_logits.cpu().numpy())
                end_logits.append(outputs.end_logits.cpu().numpy())
        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        
        # 2. Transform logits into answers
        predicted_answers = []
        for example in tqdm(self.raw_dataset_dict):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in self.example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = self.Token_dataset_dict[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})
                
        self.predicted_answers = predicted_answers
        self.theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.raw_dataset_dict]
        return predicted_answers
    
    def eval_answer(self, metric_method):
        return metric_method(predictions=self.predicted_answers, references=self.theoretical_answers)

# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction_list, truth_list):
  size = len(prediction_list)
  result = []
  for idx in range(size):
    prediction = prediction_list[idx]
    truth = truth_list[idx]
    result.append(int(normalize_text(prediction) == normalize_text(truth)))
  return sum(result)/size

def compute_f1(prediction_list, truth_list):
    size = len(prediction_list)
    result = []
    for idx in range(size):
      prediction = prediction_list[idx]
      truth = truth_list[idx]
      pred_tokens = normalize_text(prediction).split()
      truth_tokens = normalize_text(truth).split() 
      # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
      if len(pred_tokens) == 0 or len(truth_tokens) == 0:
          return int(pred_tokens == truth_tokens)
      
      common_tokens = set(pred_tokens) & set(truth_tokens)
      
      # if there are no common tokens then f1 = 0
      if len(common_tokens) == 0:
          return 0
      
      prec = len(common_tokens) / len(pred_tokens)
      rec = len(common_tokens) / len(truth_tokens)
      value = 2 * (prec * rec) / (prec + rec)
      result.append(value)
    return sum(result)/size 