import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from common import *
from model import HALOModel
from config import HALOConfig

# Circuit imports
import sys
sys.path.append(os.path.join(sys.path[0],'hmc-utils'))
sys.path.append(os.path.join(sys.path[0],'hmc-utils', 'pypsdd'))
from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE

RUNS = 1

config = HALOConfig()
NUM_GENERATIONS = 10000
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# Create circuit object
cmpe = CircuitMPE('constraints/inpatient.vtree', 'constraints/inpatient.sdd')

# Create gating function
gate = DenseGatingFunction(cmpe.beta, gate_layers=[config.n_embd] + [256]*config.num_gates, num_reps=config.num_reps).to(device)

# Create the model
model = HALOModel(config).to(device)

state = torch.load('../../save/spl_model')
model.load_state_dict(state['model'])
gate.load_state_dict(state['gate'])

train_ehr_dataset = pickle.load(open('../../inpatient_data/trainDataset.pkl', 'rb'))
test_ehr_dataset = pickle.load(open('../../inpatient_data/testDataset.pkl', 'rb'))
train_c = set([c for p in train_ehr_dataset for v in p['visits'] for c in v])
test_ehr_dataset = [{'labels': p['labels'], 'visits': [[c for c in v if c in train_c] for v in p['visits'][:96]]} for p in test_ehr_dataset]

def get_batch(dataset, loc, batch_size):
    ehr = dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
    batch_lens = np.zeros((len(ehr)), dtype=np.int32)
    for i, p in enumerate(ehr):
        visits = p['visits']
        batch_lens[i] = len(visits) + 1
        for j, v in enumerate(visits):
            batch_ehr[i,j+2][v] = 1
        batch_ehr[i,1,config.code_vocab_size:config.code_vocab_size+config.label_vocab_size] = np.array(p['labels']) # Set the patient labels
        batch_ehr[i,len(visits)+1,config.code_vocab_size+config.label_vocab_size+1] = 1 # Set the final visit to have the end token
        batch_ehr[i,len(visits)+2:,config.code_vocab_size+config.label_vocab_size+2] = 1 # Set the rest to the padded visit token
    
    batch_ehr[:,0,config.code_vocab_size+config.label_vocab_size] = 1 # Set the first visits to be the start token
    return batch_ehr, batch_lens, int(batch_lens.sum())

model.eval()
gate.eval()
log_probability = 0
n_visits = 0
n_pos_codes = 0
n_total_codes = 0
with torch.no_grad():
    val_l = []
    for i in tqdm(range(0, len(test_ehr_dataset), config.batch_size)):
        batch_ehr, batch_lens, tot_visits = get_batch(test_ehr_dataset, i, config.batch_size)
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    
        output = model(batch_ehr)
        final_output = torch.zeros(tot_visits, config.n_embd).to(device)
        final_labels = torch.zeros(tot_visits, config.total_vocab_size).to(device)
        currStart = 0
        for j in range(batch_ehr.size(0)):
            final_output[currStart:currStart+batch_lens[j].item()] = output[j, :batch_lens[j].item()]
            final_labels[currStart:currStart+batch_lens[j].item()] = batch_ehr[j, 1:batch_lens[j].item()+1]
            currStart += batch_lens[j].item()

        thetas = gate(final_output)
        cmpe.set_params(thetas)
        nll = cmpe.cross_entropy(final_labels, log_space=True).sum()
        log_probability += -nll.item()
        n_visits += tot_visits
        n_pos_codes += final_labels.sum().cpu().item()
        n_total_codes += final_labels.size(0) * final_labels.size(1)

pp_visit = np.exp(-log_probability/n_visits)
pp_positive = np.exp(-log_probability/n_pos_codes)
pp_possible = np.exp(-log_probability/n_total_codes)

metrics_dict = {}
metrics_dict['Test Log Probability'] = log_probability
metrics_dict['Perplexity Per Visit'] = pp_visit
metrics_dict['Perplexity Per Positive Code'] = pp_positive
metrics_dict['Perplexity Per Possible Code'] = pp_possible
pickle.dump(metrics_dict, open("../../results/testing_stats/SPL_Metrics.pkl", "wb"))

print('Test Log Probability: ', log_probability)
print('Perplexity Per Visit: ', pp_visit)
print('Perplexity Per Positive Code: ', pp_positive)
print('Perplexity Per Possible Code: ', pp_possible)