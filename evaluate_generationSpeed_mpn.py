from time import time
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from model import HALOModel
from config import HALOConfig
from symbolic import symbolic
from torch import nn
RUNS = 1

config = HALOConfig()
NUM_GENERATIONS = 10000
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

        
        
def correct_with_mpn(vec, batch_size):
    preds = logic_pred(vec[:, -1, :])
    print(decoder(torch.tile(torch.arange(config.total_vocab_size, dtype=torch.float32), (batch_size, 1)), vec[:, -1, :], True))
    vec[:, -1, :] =decoder(torch.tile(torch.arange(config.total_vocab_size, dtype=torch.float32), (batch_size, 1)), vec[:, -1, :], True)[0]
    

    
def sample_sequence(model, length, context, batch_size, device='cuda', sample=True):
  empty = torch.zeros((1,1,config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
  context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
  prev = context.unsqueeze(1)
  context = None
  correct_with_mpn(prev, batch_size)
  print("hola")
  with torch.no_grad():
    for _ in range(length-1):
      prev = model.sample(torch.cat((prev,empty), dim=1), sample)
      correct_with_mpn(prev, batch_size)
      if torch.sum(torch.sum(prev[:,:,config.code_vocab_size+config.label_vocab_size+1], dim=1).bool().int(), dim=0).item() == batch_size:
        break
  ehr = prev.cpu().detach().numpy()
  prev = None
  empty = None
  preds = logic_pred(ehr)
  return ehr

def convert_ehr(ehrs, index_to_code=None):
  print("hello")
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

model = HALOModel(config).to(device)
#checkpoint = torch.load('./save/mpn_model', map_location=torch.device(device))
#model.load_state_dict(checkpoint['model'])
logic_terms = []
for rule in config.rules: #where to get rules?
  if(rule[0]==[]):
    #this is (a and b and c) implies d <=> not (a and b and c) implies d <=> not a or not b or not c or d
    #so normal = negative; neg = positive; not = everything else
    #improve this conversion later to account for repeats
    #wait this conversion sucks
    #(a and b and c implies d) and (b and q implies x) this'll be in cnf
    #use an automated solver to convert to dnf??
    #(a to b) and (c to d) and (f to g) = (not a or b) and (not a or d) and (not f or g)
    #so ends up being (not a and not a and not f) or (not a and not a and not g) or ... 
    for code in rule[4]:
        logic_terms.append(symbolic.GEQConstant([code], [], [], 1, 0, 1))
    for code in rule[3]:
        logic_terms.append(symbolic.GEQConstant([], [], [code], 1, 0, 1))
    if(rule[6]==0):
        logic_terms.append(symbolic.GEQConstant([], [], [rule[5]], 1, 0, 1))
    else:
        logic_terms.append(symbolic.GEQConstant([rule[5]], [], [], 1, 0, 1))
    
    

print("The length is",len(logic_terms))
logic_pred = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(config.total_vocab_size),
            nn.Linear(config.total_vocab_size, len(logic_terms))
        )        
decoder = symbolic.OrList(terms=logic_terms)
# Generate Synthetic EHR dataset
speeds = []
stoken = np.zeros(config.total_vocab_size)
stoken[config.code_vocab_size+config.label_vocab_size] = 1
for run in tqdm(range(RUNS)):
  SEED = run
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
  synthetic_ehr_dataset = []
  start = time()
  for i in tqdm(range(0, NUM_GENERATIONS, config.sample_batch_size), leave=False):
    bs = min([NUM_GENERATIONS-i, config.sample_batch_size])
    batch_synthetic_ehrs = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
    batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
    synthetic_ehr_dataset += batch_synthetic_ehrs
  end = time()

  generationTime = end - start
  secondsPerPatient = generationTime / NUM_GENERATIONS
  speeds.append(secondsPerPatient)
  pickle.dump(secondsPerPatient, open(f'./results/generationSpeeds/mpnSpeed_{run}.pkl', 'wb'))
  pickle.dump(synthetic_ehr_dataset, open(f'./results/mpnDataset_{run}.pkl', 'wb'))
print(f"Seconds Per Patient: {np.mean(speeds)} +/- {np.std(speeds) / np.sqrt(RUNS) * 1.96}")