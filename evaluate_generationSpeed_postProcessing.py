from time import time
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from model import HALOModel
from config import HALOConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

config = HALOConfig()
NUM_GENERATIONS = 100000
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
rules = pickle.load(open('inpatient_data/rules.pkl', 'rb'))

def sample_sequence(model, length, context, batch_size, device='cuda', sample=True):
  empty = torch.zeros((1,1,config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
  context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
  prev = context.unsqueeze(1)
  context = None
  with torch.no_grad():
    for _ in range(length-1):
      prev = model.sample(torch.cat((prev,empty), dim=1), sample)
      if torch.sum(torch.sum(prev[:,:,config.code_vocab_size+config.label_vocab_size+1], dim=1).bool().int(), dim=0).item() == batch_size:
        break
  ehr = prev.cpu().detach().numpy()
  prev = None
  empty = None
  return ehr

def convert_ehr(ehrs, rules, index_to_code=None):
  ehr_outputs = []
  for i in range(len(ehrs)):
    ehr = ehrs[i]
    ehr_output = []
    labels_output = ehr[1][config.code_vocab_size:config.code_vocab_size+config.label_vocab_size]
    if index_to_code is not None:
      labels_output = [index_to_code[idx + config.code_vocab_size] for idx in np.nonzero(labels_output)[0]]
    for j in range(2, len(ehr)):
      visit = ehr[j]
      visit_output = []
      indices = np.nonzero(visit)[0]
      end = False
      for idx in indices:
        if idx < config.code_vocab_size: 
          visit_output.append(index_to_code[idx] if index_to_code is not None else idx)
        elif idx == config.code_vocab_size+config.label_vocab_size+1:
          end = True
      if visit_output != []:
        ehr_output.append(visit_output)
      if end:
        break
      
    p = {'visits': ehr_output, 'labels': labels_output}
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
      ehr_outputs.append(p)
      
  ehr = None
  ehr_output = None
  labels_output = None
  visit = None
  visit_output = None
  indices = None
  return ehr_outputs

model = HALOModel(config).to(device)
checkpoint = torch.load('./save/base_model', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])

# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
count = 0
stoken = np.zeros(config.total_vocab_size)
stoken[config.code_vocab_size+config.label_vocab_size] = 1
start = time()
pbar = tqdm(total=NUM_GENERATIONS)
while len(synthetic_ehr_dataset) < NUM_GENERATIONS:
  bs = min([NUM_GENERATIONS-len(synthetic_ehr_dataset), config.sample_batch_size])
  batch_synthetic_ehrs = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
  batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs, rules)
  synthetic_ehr_dataset += batch_synthetic_ehrs
  pbar.update(len(batch_synthetic_ehrs))
end = time()
pbar.close()

generationTime = end - start
secondsPerPatient = generationTime / NUM_GENERATIONS
print(f"Seconds Per Patient: {secondsPerPatient}")
pickle.dump(secondsPerPatient, open('./results/generationSpeeds/postProcessedSpeed.pkl', 'wb'))
pickle.dump(synthetic_ehr_dataset, open(f'./results/postProcessedDataset.pkl', 'wb'))