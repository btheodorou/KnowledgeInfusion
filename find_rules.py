import pickle
import itertools
from tqdm import tqdm
from config import HALOConfig
from collections import Counter

config = HALOConfig()
data = pickle.load(open('./inpatient_data/allData.pkl', 'rb'))

# (which past visits, which positive codes from past visits, which negative codes from past visits, which positive codes in current visit, which negative codes in the current visit, which output code in current visit, value to set output code to)

overallPrevalences = Counter([c for p in data for v in p['visits'] for c in set(v)])
rules = []

for l in range(config.label_vocab_size):
    subPop = [p for p in data if p['labels'][l] == 1]
    subCodes = [c for p in subPop for v in p['visits'] for c in set(v)]
    missingCodes = [c for c in overallPrevalences if c not in subCodes]
    missingCommon = [c for c in missingCodes if overallPrevalences[c] >= 500]
    for c in missingCommon:
        rules.append(([1], [config.code_vocab_size + l], [], [], [], c, 0))
print('RULE LENGTH: ', len(rules))

multiVisit = [p for p in data if len(p['visits']) > 1]
condCounts = {}
for p in multiVisit:
    soFar = set()
    for i, v in enumerate(p['visits']):
        for c in set(v):
            if c not in condCounts:
                condCounts[c] = Counter()
            for pc in soFar:
                condCounts[c][pc] += 1
        for c in set(v):
            soFar.add(c)
for c in condCounts:
    if overallPrevalences[c] >= 10:
        for pc in condCounts[c]:
            if condCounts[c][pc] == overallPrevalences[c]:
                rules.append((-1, [], [], [], [pc], c, 0))
print('RULE LENGTH: ', len(rules))
                
multiCodes = set([c for p in multiVisit for v in p['visits'][:-1] for c in set(v)])
for c in tqdm(multiCodes, leave=False):
    good = True
    repeated = 0
    if overallPrevalences[c] < 10:
        continue
    for p in multiVisit:
        if not good:
          continue
        seen = False
        for v in p['visits']:
            if c in set(v):
                if not seen:
                    seen = True
                else: 
                    repeated += 1
            else:
                if seen:
                    good = False
                    break
    if good and repeated > 5:
        rules.append(([-1], [], [], [c], [], c, 1))
print('RULE LENGTH: ', len(rules))

NUM_AGE_GROUPS = 4
NUM_GENDER_VARIABLES = 2
for l in range(NUM_AGE_GROUPS):
    for l2 in range(l+1, NUM_AGE_GROUPS):
        rules.append(([], [], [], [config.code_vocab_size+l], [], config.code_vocab_size+l2, 0))
for l in range(NUM_GENDER_VARIABLES):
    for l2 in range(l+1, NUM_GENDER_VARIABLES):
        rules.append(([], [], [], [config.code_vocab_size+NUM_AGE_GROUPS+l], [], config.code_vocab_size+NUM_AGE_GROUPS+l2, 0))
print('RULE LENGTH: ', len(rules))

pairCounts = {}
for p in data:
    for v in p['visits']:
        for cs in itertools.combinations(set(v),2):
            if cs not in pairCounts:
                pairCounts[cs] = 1
            else:
                pairCounts[cs] += 1
for cs in pairCounts:
    if pairCounts[cs] == overallPrevalences[cs[0]] and overallPrevalences[cs[0]] >= 10:
        rules.append(([], [], [], [cs[0]], [], cs[1], 1))
    elif pairCounts[cs] == overallPrevalences[cs[1]] and overallPrevalences[cs[1]] >= 10:
        rules.append(([], [], [], [cs[1]], [], cs[0], 1))
print('RULE LENGTH: ', len(rules))

pickle.dump(rules, open('./inpatient_data/rules.pkl', 'wb'))