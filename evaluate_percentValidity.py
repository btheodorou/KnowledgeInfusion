import pickle
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import HALOConfig
from sklearn.metrics import r2_score

config = HALOConfig()
base_ehr_dataset = pickle.load(open('./results/baseDataset.pkl', 'rb'))
processed_ehr_dataset = pickle.load(open('./results/postProcessedDataset.pkl', 'rb'))
consequence_ehr_dataset = pickle.load(open('./results/conSequenceDataset.pkl', 'rb'))
loss_ehr_dataset = pickle.load(open('./results/lossBaselineDataset.pkl', 'rb'))
ccn_ehr_dataset = pickle.load(open('./results/ccnDataset.pkl', 'rb'))

def evaluateDataset(dataset, rules):
  validCount = 0
  for p in tqdm(dataset, leave=False):
    visits = [[]] + [[l + config.code_vocab_size for l in p['labels'].nonzero()[0]]] + p['visits']
    valid = True
    for (past_visits, past_pos_codes, past_neg_codes, curr_pos_codes, curr_neg_codes, output_code, output_value) in rules:
      for i, v in enumerate(visits):
        pastSatisfied = False
        currSatisfied = False
        if not past_visits:
          pastSatisfied = True
        else:
          if past_visits == -1:
            past_codes = set([c for v in p['visits'][:i] for c in v])
          else:
            visit_past_visits = [pi for pi in past_visits if (i > pi if pi >= 0 else i+pi >= 0)]
            past_codes = set([c for pi in visit_past_visits for c in (visits[pi] if pi >= 0 else visits[i+pi])])
            
          if all([c in past_codes for c in past_pos_codes] + [c not in past_codes for c in past_neg_codes]):
            pastSatisfied = True
        
        if all([c in v for c in curr_pos_codes] + [c not in v for c in curr_neg_codes]):
          currSatisfied = True
          
        if pastSatisfied and currSatisfied:
          if (output_value and output_code not in v) or (not output_value and output_code in v): 
            valid = False
            break
      if not valid:
        break
      
    if valid:
      validCount += 1
    
  results = {'Percent Valid': validCount / len(dataset)}
  return results
  

# Extract and save statistics
base_validity = evaluateDataset(base_ehr_dataset, config.rules)
processed_validity = evaluateDataset(processed_ehr_dataset, config.rules)
consequence_validity = evaluateDataset(consequence_ehr_dataset, config.rules)
loss_validity = evaluateDataset(loss_ehr_dataset, config.rules)
ccn_validity = evaluateDataset(ccn_ehr_dataset, config.rules)
pickle.dump(base_validity, open('results/violation_stats/Base_Validity_Stats.pkl', 'wb'))
pickle.dump(processed_validity, open('results/violation_stats/Base_Validity_Stats.pkl', 'wb'))
pickle.dump(consequence_validity, open('results/violation_stats/ConSequence_Validity_Stats.pkl', 'wb'))
pickle.dump(loss_validity, open('results/violation_stats/Loss_Validity_Stats.pkl', 'wb'))
pickle.dump(ccn_validity, open('results/violation_stats/CCN_Validity_Stats.pkl', 'wb'))
print(base_validity["Percent Valid"])
print(processed_validity["Percent Valid"])
print(consequence_validity["Percent Valid"])
print(loss_validity["Percent Valid"])
print(ccn_validity["Percent Valid"])

# LAST 11