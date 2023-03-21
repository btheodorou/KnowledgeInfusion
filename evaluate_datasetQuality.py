import pickle
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import HALOConfig
from sklearn.metrics import r2_score

config = HALOConfig()
train_ehr_dataset = pickle.load(open('./inpatient_data/trainDataset.pkl', 'rb'))
base_ehr_dataset = pickle.load(open('./results/baseDataset.pkl', 'rb'))
base_plus_ehr_dataset = pickle.load(open('./results/basePlusDataset.pkl', 'rb'))
graph_ehr_dataset = pickle.load(open('./results/graphDataset.pkl', 'rb'))
graph_plus_ehr_dataset = pickle.load(open('./results/graphPlusDataset.pkl', 'rb'))
graph_minus_ehr_dataset = pickle.load(open('./results/graphMinusDataset.pkl', 'rb'))
graph_tuned_ehr_dataset = pickle.load(open('./results/graphTunedDataset.pkl', 'rb'))
graph_newOrder_ehr_dataset = pickle.load(open('./results/graphNewOrderDataset.pkl', 'rb'))

def generate_statistics(d):
    stats = {}
    aggregate_stats = {}
    record_lens = [len(p['visits']) for p in d]
    visit_lens = [len(v) for p in d for v in p['visits']]
    avg_record_len = np.mean(record_lens)
    std_record_len = np.std(record_lens)
    avg_visit_len = np.mean(visit_lens)
    std_visit_len = np.std(visit_lens)
    aggregate_stats["Record Length Mean"] = avg_record_len
    aggregate_stats["Record Length Standard Deviation"] = std_record_len
    aggregate_stats["Visit Length Mean"] = avg_visit_len
    aggregate_stats["Visit Length Standard Deviation"] = std_visit_len
    stats["Aggregate"] = aggregate_stats

    code_stats = {}
    n_records = len(record_lens)
    n_visits = len(visit_lens)
    record_code_counts = {}
    visit_code_counts = {}
    record_bigram_counts = {}
    visit_bigram_counts = {}
    record_sequential_bigram_counts = {}
    visit_sequential_bigram_counts = {}
    for p in d:
        patient_codes = set()
        patient_bigrams = set()
        sequential_bigrams = set()
        for j in range(len(p['visits'])):
            v = p['visits'][j]
            for c in v:
                visit_code_counts[c] = 1 if c not in visit_code_counts else visit_code_counts[c] + 1
                patient_codes.add(c)
            for cs in itertools.combinations(v,2):
                cs = list(cs)
                cs.sort()
                cs = tuple(cs)
                visit_bigram_counts[cs] = 1 if cs not in visit_bigram_counts else visit_bigram_counts[cs] + 1
                patient_bigrams.add(cs)
            if j > 0:
                v0 = p['visits'][j-1]
                for c0 in v0:
                    for c in v:
                        sc = (c0, c)
                        visit_sequential_bigram_counts[sc] = 1 if sc not in visit_sequential_bigram_counts else visit_sequential_bigram_counts[sc] + 1
                        sequential_bigrams.add(sc)
        for c in patient_codes:
            record_code_counts[c] = 1 if c not in record_code_counts else record_code_counts[c] + 1
        for cs in patient_bigrams:
            record_bigram_counts[cs] = 1 if cs not in record_bigram_counts else record_bigram_counts[cs] + 1
        for sc in sequential_bigrams:
            record_sequential_bigram_counts[sc] = 1 if sc not in record_sequential_bigram_counts else record_sequential_bigram_counts[sc] + 1
    record_code_probs = {c: record_code_counts[c]/n_records for c in record_code_counts}
    visit_code_probs = {c: visit_code_counts[c]/n_visits for c in visit_code_counts}
    record_bigram_probs = {cs: record_bigram_counts[cs]/n_records for cs in record_bigram_counts}
    visit_bigram_probs = {cs: visit_bigram_counts[cs]/n_visits for cs in visit_bigram_counts}
    record_sequential_bigram_probs = {sc: record_sequential_bigram_counts[sc]/n_records for sc in record_sequential_bigram_counts}
    visit_sequential_bigram_probs = {sc: visit_sequential_bigram_counts[sc]/(n_visits - len(d)) for sc in visit_sequential_bigram_counts}
    code_stats["Per Record Code Probabilities"] = record_code_probs
    code_stats["Per Visit Code Probabilities"] = visit_code_probs
    code_stats["Per Record Bigram Probabilities"] = record_bigram_probs
    code_stats["Per Visit Bigram Probabilities"] = visit_bigram_probs
    code_stats["Per Record Sequential Visit Bigram Probabilities"] = record_sequential_bigram_probs
    code_stats["Per Visit Sequential Visit Bigram Probabilities"] = visit_sequential_bigram_probs
    stats["Probabilities"] = code_stats
    return stats
    
def generate_plots(stats1, stats2, label1, label2, types=["Per Record Code Probabilities", "Per Visit Code Probabilities", "Per Record Bigram Probabilities", "Per Visit Bigram Probabilities", "Per Record Sequential Visit Bigram Probabilities", "Per Visit Sequential Visit Bigram Probabilities"]):
    data1 = stats1["Probabilities"]
    data2 = stats2["Probabilities"]
    for t in types:
        probs1 = data1[t]
        probs2 = data2[t]
        keys = set(probs1.keys()).union(set(probs2.keys()))
        values1 = [probs1[k] if k in probs1 else 0 for k in keys]
        values2 = [probs2[k] if k in probs2 else 0 for k in keys]

        plt.clf()
        r2 = r2_score(values1, values2)
        plt.scatter(values1, values2, marker=".", alpha=0.66)
        maxVal = min(1.1 * max(max(values1), max(values2)), 1.0)
        # maxVal *= (0.3 if 'Sequential' in t else (0.45 if 'Code' in t else 0.3))
        plt.xlim([0,maxVal])
        plt.ylim([0,maxVal])
        plt.title(t)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.annotate("r-squared = {:.3f}".format(r2), (0.1*maxVal, 0.9*maxVal))
        plt.savefig(f"results/dataset_stats/plots/{label2}_{t}".replace(" ", "_"))

# Extract and save statistics
# train_ehr_stats = generate_statistics(train_ehr_dataset)
# base_ehr_stats = generate_statistics(base_ehr_dataset)
# base_plus_ehr_stats = generate_statistics(base_plus_ehr_dataset)
# graph_ehr_stats = generate_statistics(graph_ehr_dataset)
# graph_plus_ehr_stats = generate_statistics(graph_plus_ehr_dataset)
# graph_minus_ehr_stats = generate_statistics(graph_minus_ehr_dataset)
# graph_tuned_ehr_stats = generate_statistics(graph_minus_ehr_dataset)
graph_newOrder_ehr_stats = generate_statistics(graph_newOrder_ehr_dataset)
# pickle.dump(train_ehr_stats, open('results/dataset_stats/Train_Stats.pkl', 'wb'))
# pickle.dump(base_ehr_stats, open('results/dataset_stats/Base_Synthetic_Stats.pkl', 'wb'))
# pickle.dump(base_plus_ehr_stats, open('results/dataset_stats/Base_Plus_Synthetic_Stats.pkl', 'wb'))
# pickle.dump(graph_ehr_stats, open('results/dataset_stats/Graph_Synthetic_Stats.pkl', 'wb'))
# pickle.dump(graph_plus_ehr_stats, open('results/dataset_stats/Graph_Plus_Synthetic_Stats.pkl', 'wb'))
# pickle.dump(graph_minus_ehr_stats, open('results/dataset_stats/Graph_Minus_Synthetic_Stats.pkl', 'wb'))
# pickle.dump(graph_tuned_ehr_stats, open('results/dataset_stats/Graph_Tuned_Synthetic_Stats.pkl', 'wb'))
pickle.dump(graph_newOrder_ehr_stats, open('results/dataset_stats/Graph_New_Order_Synthetic_Stats.pkl', 'wb'))
# print(train_ehr_stats["Aggregate"])
# print(base_ehr_stats["Aggregate"])
# print(base_plus_ehr_stats["Aggregate"])
# print(graph_ehr_stats["Aggregate"])
# print(graph_plus_ehr_stats["Aggregate"])
# print(graph_minus_ehr_stats["Aggregate"])
# print(graph_tuned_ehr_stats["Aggregate"])
# print(graph_newOrder_ehr_stats["Aggregate"])
train_ehr_stats = pickle.load(open('results/dataset_stats/Train_Stats.pkl', 'rb'))
# base_ehr_stats = pickle.load(open('results/dataset_stats/Base_Synthetic_Stats.pkl', 'rb'))
# base_plus_ehr_stats = pickle.load(open('results/dataset_stats/Base_Plus_Synthetic_Stats.pkl', 'rb'))
# graph_ehr_stats = pickle.load(open('results/dataset_stats/Graph_Synthetic_Stats.pkl', 'rb'))
# graph_plus_ehr_stats = pickle.load(open('results/dataset_stats/Graph_Plus_Synthetic_Stats.pkl', 'rb'))
# graph_minus_ehr_stats = pickle.load(open('results/dataset_stats/Graph_Minus_Synthetic_Stats.pkl', 'rb'))
# graph_tuned_ehr_stats = pickle.load(open('results/dataset_stats/Graph_Tuned_Synthetic_Stats.pkl', 'rb'))
graph_newOrder_ehr_stats = pickle.load(open('results/dataset_stats/Graph_New_Order_Synthetic_Stats.pkl', 'rb'))

# Plot per-code statistics
# generate_plots(train_ehr_stats, base_ehr_stats, "Training Data", "Base Synthetic Data")
# generate_plots(train_ehr_stats, base_plus_ehr_stats, "Training Data", "Base Plus Synthetic Data")
# generate_plots(train_ehr_stats, graph_ehr_stats, "Training Data", "Graph Synthetic Data")
# generate_plots(train_ehr_stats, graph_plus_ehr_stats, "Training Data", "Graph Plus Synthetic Data")
# generate_plots(train_ehr_stats, graph_minus_ehr_stats, "Training Data", "Graph Minus Synthetic Data")
# generate_plots(train_ehr_stats, graph_tuned_ehr_stats, "Training Data", "Graph Tuned Synthetic Data")
generate_plots(train_ehr_stats, graph_newOrder_ehr_stats, "Training Data", "Graph New Synthetic Data")