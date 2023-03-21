from time import time
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from ruleModels.graphModel import GraphModel
from config import HALOConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = HALOConfig(rules=pickle.load(open('inpatient_data/rules2.pkl', 'rb')))
NUM_GENERATIONS = 50000

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

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

def convert_ehr(ehrs, index_to_code=None):
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
    ehr_outputs.append({'visits': ehr_output, 'labels': labels_output})
  ehr = None
  ehr_output = None
  labels_output = None
  visit = None
  visit_output = None
  indices = None
  return ehr_outputs

model = GraphModel(config).to(device)
model.reset_rules(config)
model.to(device)
checkpoint = torch.load('./save/graph_model_newOrder', map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])

# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
count = 0
stoken = np.zeros(config.total_vocab_size)
stoken[config.code_vocab_size+config.label_vocab_size] = 1
start = time()
for i in tqdm(range(0, NUM_GENERATIONS, config.sample_batch_size)):
  bs = min([NUM_GENERATIONS-i, config.sample_batch_size])
  batch_synthetic_ehrs = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
  batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
  synthetic_ehr_dataset += batch_synthetic_ehrs
end = time()

generationTime = end - start
secondsPerPatient = generationTime / NUM_GENERATIONS
print(f"Seconds Per Patient: {secondsPerPatient}")
pickle.dump(secondsPerPatient, open('./results/generationSpeeds/graphNewOrderSpeed.pkl', 'wb'))
pickle.dump(synthetic_ehr_dataset, open(f'./results/graphNewOrderDataset.pkl', 'wb'))