from time import time
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from ruleModels.mpnModel import MultiPlexNetModel
from config import HALOConfig
RUNS = 1

config = HALOConfig()
NUM_GENERATIONS = 100
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
  return ehr.cpu().detach().numpy()

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

model = MultiPlexNetModel(config).to(device)
#checkpoint = torch.load('./save/mpn_model', map_location=torch.device(device))
#model.load_state_dict(checkpoint['model'])
#for outpatient
#dnf_rules = "(1817 & ~1169 & ~1818 & ~1819 & ~1820 & ~1821 & ~1822) | (1817 & ~1169 & ~1818 & ~1819 & ~1820 & ~1821 & ~1823) | (1818 & ~1169 & ~1817 & ~1819 & ~1820 & ~1821 & ~1822) | (1818 & ~1169 & ~1817 & ~1819 & ~1820 & ~1821 & ~1823) | (1819 & ~1169 & ~1817 & ~1818 & ~1820 & ~1821 & ~1822) | (1819 & ~1169 & ~1817 & ~1818 & ~1820 & ~1821 & ~1823) | (1820 & ~1169 & ~1817 & ~1818 & ~1819 & ~1821 & ~1822) | (1820 & ~1169 & ~1817 & ~1818 & ~1819 & ~1821 & ~1823) | (1821 & ~1169 & ~1817 & ~1818 & ~1819 & ~1820 & ~1822) | (1821 & ~1169 & ~1817 & ~1818 & ~1819 & ~1820 & ~1823) | (1596 & 1817 & 1823 & 512 & ~1818 & ~1819 & ~1820 & ~1821 & ~1822) | (1596 & 1818 & 1823 & 512 & ~1817 & ~1819 & ~1820 & ~1821 & ~1822) | (1596 & 1819 & 1823 & 512 & ~1817 & ~1818 & ~1820 & ~1821 & ~1822) | (1596 & 1820 & 1823 & 512 & ~1817 & ~1818 & ~1819 & ~1821 & ~1822) | (1596 & 1821 & 1823 & 512 & ~1817 & ~1818 & ~1819 & ~1820 & ~1822)"

#for inpatient:
dnf_rules = "(1158 & 1610 & 288 & 93 & ~1611 & ~1612 & ~1613 & ~1614) | (1158 & 1610 & 288 & 93 & ~1611 & ~1612 & ~1613 & ~1615) | (1158 & 1611 & 288 & 93 & ~1610 & ~1612 & ~1613 & ~1614) | (1158 & 1611 & 288 & 93 & ~1610 & ~1612 & ~1613 & ~1615) | (1158 & 1612 & 288 & 93 & ~1610 & ~1611 & ~1613 & ~1614) | (1158 & 1612 & 288 & 93 & ~1610 & ~1611 & ~1613 & ~1615) | (1158 & 1613 & 288 & 93 & ~1610 & ~1611 & ~1612 & ~1614) | (1158 & 1613 & 288 & 93 & ~1610 & ~1611 & ~1612 & ~1615) | (1158 & 1610 & 93 & ~1366 & ~1611 & ~1612 & ~1613 & ~1614) | (1158 & 1610 & 93 & ~1366 & ~1611 & ~1612 & ~1613 & ~1615) | (1158 & 1611 & 93 & ~1366 & ~1610 & ~1612 & ~1613 & ~1614) | (1158 & 1611 & 93 & ~1366 & ~1610 & ~1612 & ~1613 & ~1615) | (1158 & 1612 & 93 & ~1366 & ~1610 & ~1611 & ~1613 & ~1614) | (1158 & 1612 & 93 & ~1366 & ~1610 & ~1611 & ~1613 & ~1615) | (1158 & 1613 & 93 & ~1366 & ~1610 & ~1611 & ~1612 & ~1614) | (1158 & 1613 & 93 & ~1366 & ~1610 & ~1611 & ~1612 & ~1615) | (1610 & 288 & 93 & ~1297 & ~1611 & ~1612 & ~1613 & ~1614) | (1610 & 288 & 93 & ~1297 & ~1611 & ~1612 & ~1613 & ~1615) | (1611 & 288 & 93 & ~1297 & ~1610 & ~1612 & ~1613 & ~1614) | (1611 & 288 & 93 & ~1297 & ~1610 & ~1612 & ~1613 & ~1615) | (1612 & 288 & 93 & ~1297 & ~1610 & ~1611 & ~1613 & ~1614) | (1612 & 288 & 93 & ~1297 & ~1610 & ~1611 & ~1613 & ~1615) | (1613 & 288 & 93 & ~1297 & ~1610 & ~1611 & ~1612 & ~1614) | (1613 & 288 & 93 & ~1297 & ~1610 & ~1611 & ~1612 & ~1615) | (1610 & 93 & ~1297 & ~1366 & ~1611 & ~1612 & ~1613 & ~1614) | (1610 & 93 & ~1297 & ~1366 & ~1611 & ~1612 & ~1613 & ~1615) | (1611 & 93 & ~1297 & ~1366 & ~1610 & ~1612 & ~1613 & ~1614) | (1611 & 93 & ~1297 & ~1366 & ~1610 & ~1612 & ~1613 & ~1615) | (1612 & 93 & ~1297 & ~1366 & ~1610 & ~1611 & ~1613 & ~1614) | (1612 & 93 & ~1297 & ~1366 & ~1610 & ~1611 & ~1613 & ~1615) | (1613 & 93 & ~1297 & ~1366 & ~1610 & ~1611 & ~1612 & ~1614) | (1613 & 93 & ~1297 & ~1366 & ~1610 & ~1611 & ~1612 & ~1615) | (1158 & 1610 & 288 & ~1196 & ~1611 & ~1612 & ~1613 & ~1614 & ~514) | (1158 & 1610 & 288 & ~1196 & ~1611 & ~1612 & ~1613 & ~1615 & ~514) | (1158 & 1611 & 288 & ~1196 & ~1610 & ~1612 & ~1613 & ~1614 & ~514) | (1158 & 1611 & 288 & ~1196 & ~1610 & ~1612 & ~1613 & ~1615 & ~514) | (1158 & 1612 & 288 & ~1196 & ~1610 & ~1611 & ~1613 & ~1614 & ~514) | (1158 & 1612 & 288 & ~1196 & ~1610 & ~1611 & ~1613 & ~1615 & ~514) | (1158 & 1613 & 288 & ~1196 & ~1610 & ~1611 & ~1612 & ~1614 & ~514) | (1158 & 1613 & 288 & ~1196 & ~1610 & ~1611 & ~1612 & ~1615 & ~514) | (1158 & 1610 & ~1196 & ~1366 & ~1611 & ~1612 & ~1613 & ~1614 & ~514) | (1158 & 1610 & ~1196 & ~1366 & ~1611 & ~1612 & ~1613 & ~1615 & ~514) | (1158 & 1611 & ~1196 & ~1366 & ~1610 & ~1612 & ~1613 & ~1614 & ~514) | (1158 & 1611 & ~1196 & ~1366 & ~1610 & ~1612 & ~1613 & ~1615 & ~514) | (1158 & 1612 & ~1196 & ~1366 & ~1610 & ~1611 & ~1613 & ~1614 & ~514) | (1158 & 1612 & ~1196 & ~1366 & ~1610 & ~1611 & ~1613 & ~1615 & ~514) | (1158 & 1613 & ~1196 & ~1366 & ~1610 & ~1611 & ~1612 & ~1614 & ~514) | (1158 & 1613 & ~1196 & ~1366 & ~1610 & ~1611 & ~1612 & ~1615 & ~514) | (1610 & 288 & ~1196 & ~1297 & ~1611 & ~1612 & ~1613 & ~1614 & ~514) | (1610 & 288 & ~1196 & ~1297 & ~1611 & ~1612 & ~1613 & ~1615 & ~514) | (1611 & 288 & ~1196 & ~1297 & ~1610 & ~1612 & ~1613 & ~1614 & ~514) | (1611 & 288 & ~1196 & ~1297 & ~1610 & ~1612 & ~1613 & ~1615 & ~514) | (1612 & 288 & ~1196 & ~1297 & ~1610 & ~1611 & ~1613 & ~1614 & ~514) | (1612 & 288 & ~1196 & ~1297 & ~1610 & ~1611 & ~1613 & ~1615 & ~514) | (1613 & 288 & ~1196 & ~1297 & ~1610 & ~1611 & ~1612 & ~1614 & ~514) | (1613 & 288 & ~1196 & ~1297 & ~1610 & ~1611 & ~1612 & ~1615 & ~514) | (1610 & ~1196 & ~1297 & ~1366 & ~1611 & ~1612 & ~1613 & ~1614 & ~514) | (1610 & ~1196 & ~1297 & ~1366 & ~1611 & ~1612 & ~1613 & ~1615 & ~514) | (1611 & ~1196 & ~1297 & ~1366 & ~1610 & ~1612 & ~1613 & ~1614 & ~514) | (1611 & ~1196 & ~1297 & ~1366 & ~1610 & ~1612 & ~1613 & ~1615 & ~514) | (1612 & ~1196 & ~1297 & ~1366 & ~1610 & ~1611 & ~1613 & ~1614 & ~514) | (1612 & ~1196 & ~1297 & ~1366 & ~1610 & ~1611 & ~1613 & ~1615 & ~514) | (1613 & ~1196 & ~1297 & ~1366 & ~1610 & ~1611 & ~1612 & ~1614 & ~514) | (1613 & ~1196 & ~1297 & ~1366 & ~1610 & ~1611 & ~1612 & ~1615 & ~514)"


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