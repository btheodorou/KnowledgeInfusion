import pickle
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import HALOConfig
from sklearn.metrics import r2_score

config = HALOConfig()
base_ehr_dataset = pickle.load(open('./results/baseDataset.pkl', 'rb'))
# consequence_ehr_dataset = pickle.load(open('./results/conSequenceDataset.pkl', 'rb'))
# ccn_ehr_dataset = pickle.load(open('./results/ccnDataset.pkl', 'rb'))

def evaluateDataset(dataset, rules):
  violationsPerRule = []
  for (past_visits, past_pos_codes, past_neg_codes, curr_pos_codes, curr_neg_codes, output_code, output_value) in tqdm(rules):
    violations = 0
    for p in tqdm(dataset, leave=False):
      for i, v in enumerate([[] + [l + config.code_vocab_size for l in p['labels'].nonzero()[0]]] + p['visits']):
        pastSatisfied = False
        currSatisfied = False
        if not past_visits:
          pastSatisfied = True
        else:
          if past_visits == -1:
            past_codes = set([c for v in p['visits'][:i] for c in v])
          else:
            visit_past_visits = [pi for pi in past_visits if (i > pi if pi >= 0 else i+pi >= 0)]
            past_codes = set([c for pi in visit_past_visits for c in (p['visits'][pi] if pi >= 0 else p['visits'][i+pi])])
            
          if all([c in past_codes for c in past_pos_codes] + [c not in past_codes for c in past_neg_codes]):
            currSatisfied = True
        
        if all([c in v for c in curr_pos_codes] + [c not in v for c in curr_neg_codes]):
          currSatisfied = True
          
        if pastSatisfied and currSatisfied:
          if (output_value and output_code not in v) or (not output_value and output_code in v): 
            violations += 1
    
    violationsPerRule.append(violations)
  results = {'Per Rule': violationsPerRule, 'Total Number': sum(violationsPerRule)}
  return results
  

# Extract and save statistics
base_violations = evaluateDataset(base_ehr_dataset, config.rules)
# consequence_violations = evaluateDataset(consequence_ehr_dataset, config.rules)
# ccn_violations = evaluateDataset(ccn_ehr_dataset, config.rules)
# pickle.dump(base_violations, open('results/violation_stats/Base_Violation_Stats.pkl', 'wb'))
# pickle.dump(consequence_violations, open('results/violation_stats/ConSequence_Violation_Stats.pkl', 'wb'))
# pickle.dump(ccn_violations, open('results/violation_stats/CCN_Violation_Stats.pkl', 'wb'))
print(base_violations["Total Number"])
print(base_violations["Per Rule"])
# print(consequence_violations["Total Number"])
# print(consequence_violations["Per Rule"])
# print(ccn_violations["Total Number"])
# print(ccn_violations["Per Rule"])