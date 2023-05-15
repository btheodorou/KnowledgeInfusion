import pickle
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import HALOConfig
from sklearn.metrics import r2_score
from symbolic import symbolic
from torch import nn
config = HALOConfig()
RUNS = 25
# (which past visits, 
# which positive codes from past visits, 
# which negative codes from past visits,
# which positive codes in current visit,
# which negative codes in the current visit,
# which output code in current visit,
# value to set output code to)

def evaluateDataset(dataset, rules):
  violationsPerRule = []
  for (past_visits, past_pos_codes, past_neg_codes, curr_pos_codes, curr_neg_codes, output_code, output_value) in tqdm(rules, leave=False):
    violations = 0
    for p in tqdm(dataset, leave=False):
      visits = [[]] + [[l + config.code_vocab_size for l in p['labels'].nonzero()[0]]] + p['visits']
      for i, v in enumerate(visits):
        pastSatisfied = False
        currSatisfied = False
        if not past_visits:
          pastSatisfied = True
        else:
          if past_visits == -1:
            past_codes = set([c for v in p['visits'][:i] for c in v])
            preds = logic_pred(past_codes)
            past_codes = decoder(past_codes, preds, True)
          else:
            visit_past_visits = [pi for pi in past_visits if (i > pi if pi >= 0 else i+pi >= 0)]
            past_codes = set([c for pi in visit_past_visits for c in (visits[pi] if pi >= 0 else visits[i+pi])])
            preds = logic_pred(past_codes)
            past_codes = decoder(past_codes, preds, True)
 
          if all([c in past_codes for c in past_pos_codes] + [c not in past_codes for c in past_neg_codes]):
            pastSatisfied = True

        if all([c in v for c in curr_pos_codes] + [c not in v for c in curr_neg_codes]):
          currSatisfied = True

        if pastSatisfied and currSatisfied:
          if (output_value and output_code not in v) or (not output_value and output_code in v): 
            violations += 1

    violationsPerRule.append(violations)
  results = {'Per Rule': violationsPerRule, 'Total Number': sum(violationsPerRule)}
  return results
  
# Extract and save statistics
logic_terms = []
for rule in config.rules: #where to get rules?
  if(rule[0]==[]):
    #this is (a and b and c) implies d <=> not (a and b and c) implies d <=> not a or not b or not c or d
    #so normal = negative; neg = positive; not = everything else
    if(rule[6]==0):
      rule[3].append(rule[5])
    else:
      rule[4].append(rule[5])
    logic_terms.append(symbolic.GEQConstant(rule[4], [i for i in range(config.total_vocab_size) if((i not in rule[4]) and (i not in rule[3]))], rule[3], 1, 0, 1))

logic_pred = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(config.total_vocab_size),
            nn.Linear(config.total_vocab_size, len(logic_terms))
        )        
decoder = symbolic.OrList(terms=logic_terms)

for i in tqdm(range(10)):
  # base_ehr_dataset = pickle.load(open(f'./results/baseDataset_{i}.pkl', 'rb'))
  # processed_ehr_dataset = pickle.load(open(f'./results/postProcessedDataset_{i}.pkl', 'rb'))
  consequence_ehr_dataset = pickle.load(open(f'./results/base_model.pkl', 'rb'))
  # loss_ehr_dataset = pickle.load(open(f'./results/lossBaselineDataset_{i}.pkl', 'rb'))
  # ccn_ehr_dataset = pickle.load(open(f'./results/ccnDataset_{i}.pkl', 'rb'))

  # base_violations = evaluateDataset(base_ehr_dataset, config.rules)
  # processed_violations = evaluateDataset(processed_ehr_dataset, config.rules)
  consequence_violations = evaluateDataset(consequence_ehr_dataset, config.rules)
  print(consequence_violations)
  # loss_violations = evaluateDataset(loss_ehr_dataset, config.rules)
  # ccn_violations = evaluateDataset(ccn_ehr_dataset, config.rules)
  # pickle.dump(base_violations, open(f'results/violation_stats/Base_Violation_Stats_{i}.pkl', 'wb'))
  # pickle.dump(processed_violations, open(f'results/violation_stats/Processed_Violation_Stats_{i}.pkl', 'wb'))
  pickle.dump(consequence_violations, open(f'results/violation_stats/ConSequence_Violation_Stats_{i}.pkl', 'wb'))
  # pickle.dump(loss_violations, open(f'results/violation_stats/Loss_Violation_Stats_{i}.pkl', 'wb'))
  # pickle.dump(ccn_violations, open(f'results/violation_stats/CCN_Violation_Stats_{i}.pkl', 'wb'))

# LAST 11 ARE STATIC