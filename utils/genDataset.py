import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

mimic_dir = "./"
patientsFile = mimic_dir + 'PATIENTS.csv'
admissionFile = mimic_dir + "ADMISSIONS.csv"
diagnosisFile = mimic_dir + "DIAGNOSES_ICD.csv"
procedureFile = mimic_dir + "PROCEDURES_ICD.csv"

print("Loading CSVs Into Dataframes")
patientsDf = pd.read_csv(patientsFile, dtype=str).set_index("SUBJECT_ID")
patientsDf = patientsDf[['GENDER', 'DOB']]
patientsDf['DOB'] = pd.to_datetime(patientsDf['DOB'])
admissionDf = pd.read_csv(admissionFile, dtype=str)
admissionDf['ADMITTIME'] = pd.to_datetime(admissionDf['ADMITTIME'])
admissionDf = admissionDf.sort_values('ADMITTIME')
admissionDf = admissionDf.reset_index(drop=True)
diagnosisDf = pd.read_csv(diagnosisFile, dtype=str).set_index("HADM_ID")
diagnosisDf = diagnosisDf[diagnosisDf['ICD9_CODE'].notnull()]
diagnosisDf = diagnosisDf[['ICD9_CODE']]
procedureDf = pd.read_csv(procedureFile, dtype=str).set_index("HADM_ID")
procedureDf = procedureDf[procedureDf['ICD9_CODE'].notnull()]
procedureDf = procedureDf[['ICD9_CODE']]

print("Building Dataset")
data = {}
prevAdmitTime = {}
for row in tqdm(admissionDf.itertuples(), total=admissionDf.shape[0]):          
    #Extracting Admissions Table Info
    hadm_id = row.HADM_ID
    subject_id = row.SUBJECT_ID
    admit_time = row.ADMITTIME
      
    visit_count = (0 if subject_id not in data else len(data[subject_id]['visits'])) + 1
      
    # Extract the gender and age
    patientRow = patientsDf.loc[[subject_id]].iloc[0]  
    gender = patientRow.GENDER
    age = (admit_time.to_pydatetime() - patientRow.DOB.to_pydatetime()).days // 365
    if age < 20:
        age = "0-20"
    elif age < 40:
        age = "20-40"
    elif age < 60:
        age = "40-60"
    else:
        age = "60+"

    # Extracting the Diagnoses
    if hadm_id in diagnosisDf.index: 
        diagnoses = list(diagnosisDf.loc[[hadm_id]]["ICD9_CODE"])
        diagnosisDf.loc[[hadm_id]]
    else:
        diagnoses = []
    
    # Extracting the Procedures
    if hadm_id in procedureDf.index: 
        procedures = list(procedureDf.loc[[hadm_id]]["ICD9_CODE"])
    else:
        procedures = []
        
    # Building the hospital admission data point
    if subject_id not in data:
      data[subject_id] = {'labels': (age, gender), 'visits': [(diagnoses, procedures)]}
    else:
      data[subject_id]['visits'].append((diagnoses, procedures))

print("Converting Labels")
label_to_id = {}
for k in Counter([p['labels'][0] for p in data.values()]):
    label_to_id[('AGE', k)] = len(label_to_id)
for k in Counter([p['labels'][1] for p in data.values()]):
    label_to_id[('GENDER', k)] = len(label_to_id)
print(f"LABELS LENGTH: {len(label_to_id)}")

for p in data:
    newLabels = np.zeros(len(label_to_id))
    newLabels[label_to_id[('AGE', data[p]['labels'][0])]] = 1
    newLabels[label_to_id[('GENDER', data[p]['labels'][1])]] = 1
    data[p]['labels'] = newLabels

id_to_label = {v: k for k, v in label_to_id.items()}

print("Shortening Records")
MAX_LEN = 48
for p in data:
    data[p]['visits'] = data[p]['visits'][:MAX_LEN]

code_to_index = {}
for k in Counter([c[0:3] for p in data.values() for v in p['visits'] for c in v[0]]):
    code_to_index[('DIAGNOSIS ICD9_CODE', k)] = len(code_to_index)
print(f"POST-DIAGNOSIS VOCAB SIZE: {len(code_to_index)}")
for k in Counter([c[0:3] for p in data.values() for v in p['visits'] for c in v[1]]):
    code_to_index[('PROCEDURE ICD9_CODE', k)] = len(code_to_index)
print(f"POST-PROCEDURE VOCAB SIZE: {len(code_to_index)}")
print(f"FINAL VOCAB SIZE: {len(code_to_index)}")

index_to_code = {v: k for k, v in code_to_index.items()}

print("Converting Visits")
for p in data:
    new_visits = []
    for v in data[p]['visits']:
        new_visit = []
        for c in v[0]:
            new_visit.append(code_to_index[('DIAGNOSIS ICD9_CODE', c[0:3])])
        for c in v[1]:
            new_visit.append(code_to_index[('PROCEDURE ICD9_CODE', c[0:3])])
                
        new_visits.append(list(set(new_visit)))
        
    data[p]['visits'] = new_visits    

data = list(data.values())
print(f"MAX LEN: {max([len(p['visits']) for p in data])}")
print(f"AVG LEN: {np.mean([len(p['visits']) for p in data])}")

# Train-Val-Test Split
print("Splitting Datasets")
train_idx, test_idx = train_test_split(range(len(data)), test_size=0.1, random_state=4, shuffle=True)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=4, shuffle=True)
train_dataset = [data[i] for i in train_idx]
val_dataset = [data[i] for i in val_idx]
test_dataset = [data[i] for i in test_idx]

# Save Everything
print("Saving Everything")
print(len(id_to_label))
print(len(index_to_code))
pickle.dump(id_to_label, open("../inpatient_data/idToLabel.pkl", "wb"))
pickle.dump(index_to_code, open("../inpatient_data/indexToCode.pkl", "wb"))
pickle.dump(data, open("../inpatient_data/allData.pkl", "wb"))
pickle.dump(train_dataset, open("../inpatient_data/trainDataset.pkl", "wb"))
pickle.dump(val_dataset, open("../inpatient_data/valDataset.pkl", "wb"))
pickle.dump(test_dataset, open("../inpatient_data/testDataset.pkl", "wb"))