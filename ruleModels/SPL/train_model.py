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

config = HALOConfig()
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# Set seed
seed_all_rngs(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load the datasets
train_ehr_dataset = pickle.load(open('../../inpatient_data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('../../inpatient_data/valDataset.pkl', 'rb'))

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


# Create circuit object
cmpe = CircuitMPE('constraints/inpatient.vtree', 'constraints/inpatient.sdd')
gate = DenseGatingFunction(cmpe.beta, gate_layers=[config.n_embd] + [256]*config.num_gates, num_reps=config.num_reps).to(device)

# Create the model
model = HALOModel(config).to(device)

optimizer = torch.optim.Adam(list(model.parameters()) + list(gate.parameters()), lr=config.lr)

global_loss = 1e10
for e in tqdm(range(config.epoch)):
    model.train()
    gate.train()
    np.random.shuffle(train_ehr_dataset)
    for i in range(0, len(train_ehr_dataset), config.batch_size):        
        batch_ehr, batch_lens, tot_visits = get_batch(train_ehr_dataset, i, config.batch_size)
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        optimizer.zero_grad()
    
        output = model(batch_ehr)
        final_output = torch.zeros(tot_visits, config.n_embd).to(device)
        final_labels = torch.zeros(tot_visits, config.total_vocab_size).to(device)
        currStart = 0
        for j in range(batch_ehr.size(0)):
            final_output[currStart:currStart+batch_lens[j]] = output[j, :batch_lens[j]]
            final_labels[currStart:currStart+batch_lens[j]] = batch_ehr[j, 1:batch_lens[j]+1]
            currStart += batch_lens[j].item()

        thetas = gate(final_output)
        cmpe.set_params(thetas)
        loss = cmpe.cross_entropy(final_labels, log_space=True).mean()
        
        loss.backward()
        optimizer.step()
        
        if i % (5000*config.batch_size) == 0:
            print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss))
            
    model.eval()
    gate.eval()
    with torch.no_grad():
        val_l = []
        for v_i in range(0, len(val_ehr_dataset), config.batch_size):
            batch_ehr, batch_lens, tot_visits = get_batch(val_ehr_dataset, v_i, config.batch_size)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            optimizer.zero_grad()
        
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
            loss = cmpe.cross_entropy(final_labels, log_space=True).mean()
            val_l.append(loss.cpu().detach().numpy())
    
        cur_val_loss = np.mean(val_l)
        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
        if cur_val_loss < global_loss:
            global_loss = cur_val_loss
            state = {
                'model': model.state_dict(),
                'gate': gate.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, '../../save/spl_model')