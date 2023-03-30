import os
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from model import HALOModel
from config import HALOConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

config = HALOConfig()

local_rank = -1
fp16 = False
if local_rank == -1:device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

train_ehr_dataset = pickle.load(open('./inpatient_data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('./inpatient_data/valDataset.pkl', 'rb'))

def get_batch(dataset, loc, batch_size):
  ehr = dataset[loc:loc+batch_size]
  batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
  batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
  for i, p in enumerate(ehr):
    visits = p['visits']
    for j, v in enumerate(visits):
      batch_ehr[i,j+2][v] = 1
      batch_mask[i,j+2] = 1
    batch_ehr[i,1,config.code_vocab_size:config.code_vocab_size+config.label_vocab_size] = np.array(p['labels']) # Set the patient labels
    batch_ehr[i,len(visits)+1,config.code_vocab_size+config.label_vocab_size+1] = 1 # Set the final visit to have the end token
    batch_ehr[i,len(visits)+2:,config.code_vocab_size+config.label_vocab_size+2] = 1 # Set the rest to the padded visit token
  
  batch_mask[:,1] = 1 # Set the mask to cover the labels
  batch_ehr[:,0,config.code_vocab_size+config.label_vocab_size] = 1 # Set the first visits to be the start token
  batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
  return batch_ehr, batch_mask

def shuffle_training_data(train_ehr_dataset):
  np.random.shuffle(train_ehr_dataset)

model = HALOModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists("./save/base_model"):
  print("Loading previous model")
  checkpoint = torch.load('./save/base_model', map_location=torch.device(device))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])

# Train
global_loss = 1e10
for e in tqdm(range(config.epoch)):
  shuffle_training_data(train_ehr_dataset)
  model.train()
  for i in range(0, len(train_ehr_dataset), config.batch_size):    
    batch_ehr, batch_mask = get_batch(train_ehr_dataset, i, config.batch_size)
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
    
    optimizer.zero_grad()
    loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
    loss.backward()
    optimizer.step()
    
    if i % (5000*config.batch_size) == 0:
      print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss))
    
  model.eval()
  with torch.no_grad():
    val_l = []
    for v_i in range(0, len(val_ehr_dataset), config.batch_size):
      batch_ehr, batch_mask = get_batch(val_ehr_dataset, v_i, config.batch_size)
      batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
      batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)

      val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
      val_l.append((val_loss).cpu().detach().numpy())
      
    cur_val_loss = np.mean(val_l)
    print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
    if cur_val_loss < global_loss:
      global_loss = cur_val_loss
      state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': i
        }
      torch.save(state, './save/base_model')
      print('\n------------ Save best model ------------\n')