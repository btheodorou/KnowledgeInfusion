import pickle
import numpy as np
from sklearn.metrics import r2_score

NUM_RUNS = 25

# Generation Speeds
base_speeds = [pickle.load(open(f'results/generationSpeeds/baseSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
processing_speeds = [pickle.load(open(f'results/generationSpeeds/postProcessedSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
consequence_speeds = [pickle.load(open(f'results/generationSpeeds/conSequenceSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
ccn_speeds = [pickle.load(open(f'results/generationSpeeds/ccnSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
loss_speeds = [pickle.load(open(f'results/generationSpeeds/lossBaselineSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
print(f'Base Speed: {np.mean(base_speeds)} \pm {np.std(base_speeds) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Post Processed Speed: {np.mean(processing_speeds)} \pm {np.std(processing_speeds) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'ConSequence Speed: {np.mean(consequence_speeds)} \pm {np.std(consequence_speeds) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'CCN Speed: {np.mean(ccn_speeds)} \pm {np.std(ccn_speeds) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Loss Baseline Speed: {np.mean(loss_speeds)} \pm {np.std(loss_speeds) / np.sqrt(NUM_RUNS) * 1.96}')

base_violations = [pickle.load(open(f'results/violation_stats/Base_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
processing_violations = [pickle.load(open(f'results/violation_stats/Post_Processed_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
consequence_violations = [pickle.load(open(f'results/violation_stats/ConSequence_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
ccn_violations = [pickle.load(open(f'results/violation_stats/CCN_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
loss_violations = [pickle.load(open(f'results/violation_stats/Loss_Baseline_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
print(f'Base Violations: {np.mean(base_violations)} \pm {np.std(base_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Post Processed Violations: {np.mean(processing_violations)} \pm {np.std(processing_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'ConSequence Violations: {np.mean(consequence_violations)} \pm {np.std(consequence_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'CCN Violations: {np.mean(ccn_violations)} \pm {np.std(ccn_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Loss Baseline Violations: {np.mean(loss_violations)} \pm {np.std(loss_violations) / np.sqrt(NUM_RUNS) * 1.96}')

base_static_violations = [sum(pickle.load(open(f'results/violation_stats/Base_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
processing_static_violations = [sum(pickle.load(open(f'results/violation_stats/Post_Processed_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
consequence_static_violations = [sum(pickle.load(open(f'results/violation_stats/ConSequence_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
ccn_static_violations = [sum(pickle.load(open(f'results/violation_stats/CCN_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
loss_static_violations = [sum(pickle.load(open(f'results/violation_stats/Loss_Baseline_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
print(f'Base Static Violations: {np.mean(base_static_violations)} \pm {np.std(base_static_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Post Processed Static Violations: {np.mean(processing_static_violations)} \pm {np.std(processing_static_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'ConSequence Static Violations: {np.mean(consequence_static_violations)} \pm {np.std(consequence_static_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'CCN Static Violations: {np.mean(ccn_static_violations)} \pm {np.std(ccn_static_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Loss Baseline Static Violations: {np.mean(loss_static_violations)} \pm {np.std(loss_static_violations) / np.sqrt(NUM_RUNS) * 1.96}')

base_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/Base_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
processing_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/Post_Processed_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
consequence_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/ConSequence_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
ccn_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/CCN_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
loss_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/Loss_Baseline_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
print(f'Base Temporal Violations: {np.mean(base_temporal_violations)} \pm {np.std(base_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Post Processed Temporal Violations: {np.mean(processing_temporal_violations)} \pm {np.std(processing_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'ConSequence Temporal Violations: {np.mean(consequence_temporal_violations)} \pm {np.std(consequence_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'CCN Temporal Violations: {np.mean(ccn_temporal_violations)} \pm {np.std(ccn_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Loss Baseline Temporal Violations: {np.mean(loss_temporal_violations)} \pm {np.std(loss_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96}')

base_validity = [pickle.load(open(f'results/violation_stats/Base_Validity_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
processing_validity = [pickle.load(open(f'results/violation_stats/Post_Processed_Validity_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
consequence_validity = [pickle.load(open(f'results/violation_stats/ConSequence_Validity_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
ccn_validity = [pickle.load(open(f'results/violation_stats/CCN_Validity_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
loss_validity = [pickle.load(open(f'results/violation_stats/Loss_Baseline_Validity_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
print(f'Base Validity: {np.mean(base_validity)} \pm {np.std(base_validity) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Post Processed Validity: {np.mean(processing_validity)} \pm {np.std(processing_validity) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'ConSequence Validity: {np.mean(consequence_validity)} \pm {np.std(consequence_validity) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'CCN Validity: {np.mean(ccn_validity)} \pm {np.std(ccn_validity) / np.sqrt(NUM_RUNS) * 1.96}')
print(f'Loss Baseline Validity: {np.mean(loss_validity)} \pm {np.std(loss_validity) / np.sqrt(NUM_RUNS) * 1.96}')

train_dataset_stats = pickle.load(open('results/dataset_stats/Train_Stats.pkl', 'rb'))
base_dataset_stats = [pickle.load(open(f'results/dataset_stats/Base_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
processing_dataset_stats = [pickle.load(open(f'results/dataset_stats/Processed_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
consequence_dataset_stats = [pickle.load(open(f'results/dataset_stats/ConSequence_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
ccn_dataset_stats = [pickle.load(open(f'results/dataset_stats/CCN_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
loss_dataset_stats = [pickle.load(open(f'results/dataset_stats/Loss_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]

def print_r2(label, stats1, stats2, types=["Per Visit Code Probabilities", "Per Visit Bigram Probabilities", "Per Visit Sequential Visit Bigram Probabilities"]):
    for t in types:
        values = []
        for i in range(NUM_RUNS):
            probs2 = stats2[i]["Probabilities"][t]
            probs1 = stats1[i]["Probabilities"][t]
            keys = set(probs1.keys()).union(set(probs2.keys()))
            values1 = [probs1[k] if k in probs1 else 0 for k in keys]
            values2 = [probs2[k] if k in probs2 else 0 for k in keys]
            values.append(r2_score(values1, values2))
        print(f'{label} {t}: {np.mean(values)} \pm {np.std(values) / np.sqrt(NUM_RUNS) * 1.96}')
        
print_r2("Base", train_dataset_stats, base_dataset_stats)
print_r2("Post Processed", train_dataset_stats, processing_dataset_stats)
print_r2("ConSequence", train_dataset_stats, consequence_dataset_stats)
print_r2("CCN", train_dataset_stats, ccn_dataset_stats)
print_r2("Loss Baseline", train_dataset_stats, loss_dataset_stats)