import pickle
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib

NUM_RUNS = 25
font = {#'family': 'normal',
        # 'weight': 'bold',
        'size': 12}
matplotlib.rc('font', **font)

# Training Speeds
base_speed = 1 + 30/60 + 38/3600
processing_speed = 1 + 30/60 + 38/3600
consequence_speed = 1 + 30/60 + 50/3600
ccn_speed = 4 + 38/60 + 13/3600
loss_speed = 1 + 44/60 + 16/3600
multiplex_speed = 3 + 44/60 + 53/3600
spl_speed = 2 + 46/60 + 9/3600

# Generation Speeds
base_speeds = [pickle.load(open(f'results/generationSpeeds/baseSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
processing_speeds = [pickle.load(open(f'results/generationSpeeds/postProcessedSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
consequence_speeds = [pickle.load(open(f'results/generationSpeeds/conSequenceSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
ccn_speeds = [pickle.load(open(f'results/generationSpeeds/ccnSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
mpn_speeds = [pickle.load(open(f'results/generationSpeeds/mpnSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
loss_speeds = [pickle.load(open(f'results/generationSpeeds/lossBaselineSpeed_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
base_avg_speed = np.mean(base_speeds)
processing_avg_speed = np.mean(processing_speeds)
consequence_avg_speed = np.mean(consequence_speeds)
ccn_avg_speed = np.mean(ccn_speeds)
loss_avg_speed = np.mean(loss_speeds)
multiplex_avg_speed = np.mean(mpn_speeds)
spl_avg_speed = 0
base_error = np.std(base_speeds) / np.sqrt(NUM_RUNS) * 1.96
processing_error = np.std(processing_speeds) / np.sqrt(NUM_RUNS) * 1.96
consequence_error = np.std(consequence_speeds) / np.sqrt(NUM_RUNS) * 1.96
ccn_error = np.std(ccn_speeds) / np.sqrt(NUM_RUNS) * 1.96
loss_error = np.std(loss_speeds) / np.sqrt(NUM_RUNS) * 1.96
multiplex_error = np.std(mpn_speeds) / np.sqrt(NUM_RUNS) * 1.96
spl_error = 0

models = ['Vanilla\nHALO', 'Post\nProcessed', 'ConSequence', 'CCN', 'Semantic\nLoss', 'MultiPlexNet', 'SPL']
gen_speeds = [base_avg_speed, processing_avg_speed, consequence_avg_speed, ccn_avg_speed, loss_avg_speed, multiplex_avg_speed, spl_avg_speed]
gen_err = [base_error, processing_error, consequence_error, ccn_error, loss_error, multiplex_error, spl_error]
train_speeds = [base_speed, processing_speed, consequence_speed, ccn_speed, loss_speed, multiplex_speed, spl_speed]
train_err = [0, 0, 0, 0, 0, 0, 0]

fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax2 = ax1.twinx()
ax1.set_title("Model Efficiency")
plt.xticks([1, 2, 3, 4, 5, 6, 7], models)
ax1.set_ylabel("Generation Speed (s/patient)")
ax2.set_ylabel("Training Speed (hours)")
ax1.set_xlabel("Model")
x = np.arange(len(models))
ax1.set_xticks(x)
ax1.set_xticklabels(models)
width = 0.25
bar1 = ax1.bar(x - width/1.5, gen_speeds, width, yerr=gen_err, label="Generation Speed")
bar2 = ax2.bar(x + width/1.5, train_speeds, width, yerr=train_err, label="Training Speed", color='orange')
bars = [bar1, bar2]
labels = [bar.get_label() for bar in bars]
plt.legend(bars, labels, loc="upper right")
plt.subplots_adjust(top=0.94, bottom=0.15)
plt.savefig('figures/inpatient_speeds.png')



base_violations = [pickle.load(open(f'results/violation_stats/Base_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
processing_violations = [pickle.load(open(f'results/violation_stats/Processed_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
consequence_violations = [pickle.load(open(f'results/violation_stats/ConSequence_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
ccn_violations = [pickle.load(open(f'results/violation_stats/CCN_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
loss_violations = [pickle.load(open(f'results/violation_stats/Loss_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
mpn_violations = [pickle.load(open(f'results/violation_stats/MPN_Violation_Stats_{i}.pkl', 'rb'))['Total Number'] for i in range(NUM_RUNS)]
base_avg_violations = np.mean(base_violations)
processing_avg_violations = np.mean(processing_violations)
consequence_avg_violations = np.mean(consequence_violations)
ccn_avg_violations = np.mean(ccn_violations)
loss_avg_violations = np.mean(loss_violations)
mpn_avg_violations = np.mean(mpn_violations)
base_error = np.std(base_violations) / np.sqrt(NUM_RUNS) * 1.96
processing_error = np.std(processing_violations) / np.sqrt(NUM_RUNS) * 1.96
consequence_error = np.std(consequence_violations) / np.sqrt(NUM_RUNS) * 1.96
ccn_error = np.std(ccn_violations) / np.sqrt(NUM_RUNS) * 1.96
loss_error = np.std(loss_violations) / np.sqrt(NUM_RUNS) * 1.96
mpn_error = np.std(mpn_violations) / np.sqrt(NUM_RUNS) * 1.96

base_static_violations = [sum(pickle.load(open(f'results/violation_stats/Base_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
processing_static_violations = [sum(pickle.load(open(f'results/violation_stats/Processed_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
consequence_static_violations = [sum(pickle.load(open(f'results/violation_stats/ConSequence_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
ccn_static_violations = [sum(pickle.load(open(f'results/violation_stats/CCN_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
loss_static_violations = [sum(pickle.load(open(f'results/violation_stats/Loss_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
mpn_static_violations = [sum(pickle.load(open(f'results/violation_stats/MPN_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][-11:]) for i in range(NUM_RUNS)]
base_avg_static_violations = np.mean(base_static_violations)
processing_avg_static_violations = np.mean(processing_static_violations)
consequence_avg_static_violations = np.mean(consequence_static_violations)
ccn_avg_static_violations = np.mean(ccn_static_violations)
loss_avg_static_violations = np.mean(loss_static_violations)
mpn_avg_static_violations = np.mean(mpn_static_violations)
base_static_error = np.std(base_static_violations) / np.sqrt(NUM_RUNS) * 1.96
processing_static_error = np.std(processing_static_violations) / np.sqrt(NUM_RUNS) * 1.96
consequence_static_error = np.std(consequence_static_violations) / np.sqrt(NUM_RUNS) * 1.96
ccn_static_error = np.std(ccn_static_violations) / np.sqrt(NUM_RUNS) * 1.96
loss_static_error = np.std(loss_static_violations) / np.sqrt(NUM_RUNS) * 1.96
mpn_static_error = np.std(mpn_static_violations) / np.sqrt(NUM_RUNS) * 1.96

base_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/Base_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
processing_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/Processed_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
consequence_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/ConSequence_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
ccn_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/CCN_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
loss_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/Loss_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
mpn_temporal_violations = [sum(pickle.load(open(f'results/violation_stats/MPN_Violation_Stats_{i}.pkl', 'rb'))['Per Rule'][:-11]) for i in range(NUM_RUNS)]
base_avg_temporal_violations = np.mean(base_temporal_violations)
processing_avg_temporal_violations = np.mean(processing_temporal_violations)
consequence_avg_temporal_violations = np.mean(consequence_temporal_violations)
ccn_avg_temporal_violations = np.mean(ccn_temporal_violations)
loss_avg_temporal_violations = np.mean(loss_temporal_violations)
mpn_avg_temporal_violations = np.mean(mpn_temporal_violations)
base_temporal_error = np.std(base_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96
processing_temporal_error = np.std(processing_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96
consequence_temporal_error = np.std(consequence_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96
ccn_temporal_error = np.std(ccn_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96
loss_temporal_error = np.std(loss_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96
mpn_temporal_error = np.std(mpn_temporal_violations) / np.sqrt(NUM_RUNS) * 1.96

base_validity = [pickle.load(open(f'results/violation_stats/Base_Validity_Stats_{i}.pkl', 'rb'))['Percent Valid'] for i in range(NUM_RUNS)]
processing_validity = [pickle.load(open(f'results/violation_stats/Processed_Validity_Stats_{i}.pkl', 'rb'))['Percent Valid'] for i in range(NUM_RUNS)]
consequence_validity = [pickle.load(open(f'results/violation_stats/ConSequence_Validity_Stats_{i}.pkl', 'rb'))['Percent Valid'] for i in range(NUM_RUNS)]
ccn_validity = [pickle.load(open(f'results/violation_stats/CCN_Validity_Stats_{i}.pkl', 'rb'))['Percent Valid'] for i in range(NUM_RUNS)]
loss_validity = [pickle.load(open(f'results/violation_stats/Loss_Validity_Stats_{i}.pkl', 'rb'))['Percent Valid'] for i in range(NUM_RUNS)]
mpn_validity = [pickle.load(open(f'results/violation_stats/MPN_Validity_Stats_{i}.pkl', 'rb'))['Percent Valid'] for i in range(NUM_RUNS)]
base_avg_validity = np.mean(base_validity)
processing_avg_validity = np.mean(processing_validity)
consequence_avg_validity = np.mean(consequence_validity)
ccn_avg_validity = np.mean(ccn_validity)
loss_avg_validity = np.mean(loss_validity)
mpn_avg_validity = np.mean(mpn_validity)
base_validity_error = np.std(base_validity) / np.sqrt(NUM_RUNS) * 1.96
processing_validity_error = np.std(processing_validity) / np.sqrt(NUM_RUNS) * 1.96
consequence_validity_error = np.std(consequence_validity) / np.sqrt(NUM_RUNS) * 1.96
ccn_validity_error = np.std(ccn_validity) / np.sqrt(NUM_RUNS) * 1.96
loss_validity_error = np.std(loss_validity) / np.sqrt(NUM_RUNS) * 1.96
mpn_validity_error = np.std(mpn_validity) / np.sqrt(NUM_RUNS) * 1.96

models = ['Vanilla\nHALO', 'Post\nProcessed', 'ConSequence', 'CCN', 'Semantic\nLoss', 'MultiPlexNet']
violations = [base_avg_violations, processing_avg_violations, consequence_avg_violations, ccn_avg_violations, loss_avg_violations, mpn_avg_violations]
violations_err = [base_error, processing_error, consequence_error, ccn_error, loss_error, mpn_error]
static_violations = [base_avg_static_violations, processing_avg_static_violations, consequence_avg_static_violations, ccn_avg_static_violations, loss_avg_static_violations, mpn_avg_static_violations]
static_violations_err = [base_static_error, processing_static_error, consequence_static_error, ccn_static_error, loss_static_error, mpn_static_error]
temporal_violations = [base_avg_temporal_violations, processing_avg_temporal_violations, consequence_avg_temporal_violations, ccn_avg_temporal_violations, loss_avg_temporal_violations, mpn_avg_temporal_violations]
temporal_violations_err = [base_temporal_error, processing_temporal_error, consequence_temporal_error, ccn_temporal_error, loss_temporal_error, mpn_temporal_error]
validity = [base_avg_validity, processing_avg_validity, consequence_avg_validity, ccn_avg_validity, loss_avg_validity, mpn_avg_validity]
validity_err = [base_validity_error, processing_validity_error, consequence_validity_error, ccn_validity_error, loss_validity_error, mpn_validity_error]

fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax2 = ax1.twinx()
ax1.set_title("Constraint Satisfaction")
plt.xticks([1, 2, 3, 4, 5], models)
ax1.set_ylabel("Number of Violations")
ax2.set_ylabel("Percent Validity")
ax1.set_xlabel("Model")
x = np.arange(len(models))
ax1.set_xticks(x)
ax1.set_xticklabels(models)
width = 0.1
bar1 = ax1.bar(x - 2.25 * width, violations, width, yerr=violations_err, label="Total Violations", color='blue')
bar2 = ax1.bar(x - 0.75 * width, static_violations, width, yerr=static_violations_err, label="Static Violations", color='orange')
bar3 = ax1.bar(x + 0.75 * width, temporal_violations, width, yerr=temporal_violations_err, label="Temporal Violations", color='green')
bar4 = ax2.bar(x + 2.25 * width, validity, width, yerr=validity_err, label="Percent Validity", color='red')
bars = [bar1, bar2, bar3, bar4]
labels = [bar.get_label() for bar in bars]
plt.legend(bars, labels, loc="upper left")
plt.savefig('figures/inpatient_validity.png')

train_dataset_stats = pickle.load(open('results/dataset_stats/Train_Stats.pkl', 'rb'))
base_dataset_stats = [pickle.load(open(f'results/dataset_stats/Base_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
processing_dataset_stats = [pickle.load(open(f'results/dataset_stats/Processed_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
consequence_dataset_stats = [pickle.load(open(f'results/dataset_stats/ConSequence_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
ccn_dataset_stats = [pickle.load(open(f'results/dataset_stats/CCN_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
loss_dataset_stats = [pickle.load(open(f'results/dataset_stats/Loss_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]
mpn_dataset_stats = [pickle.load(open(f'results/dataset_stats/MPN_Synthetic_Stats_{i}.pkl', 'rb')) for i in range(NUM_RUNS)]

def return_r2(stats1, stats2, types=["Per Visit Code Probabilities", "Per Visit Bigram Probabilities", "Per Visit Sequential Visit Bigram Probabilities"]):
    r2s = []
    for t in types:
        values = []
        for i in range(NUM_RUNS):
            probs2 = stats2[i]["Probabilities"][t]
            probs1 = stats1["Probabilities"][t]
            keys = set(probs1.keys()).union(set(probs2.keys()))
            values1 = [probs1[k] if k in probs1 else 0 for k in keys]
            values2 = [probs2[k] if k in probs2 else 0 for k in keys]
            values.append(r2_score(values1, values2))
        r2s.append((np.mean(values), np.std(values) / np.sqrt(NUM_RUNS) * 1.96))
    return r2s
        
base_r2 = return_r2(train_dataset_stats, base_dataset_stats)
processing_r2 = return_r2(train_dataset_stats, processing_dataset_stats)
consequence_r2 = return_r2(train_dataset_stats, consequence_dataset_stats)
ccn_r2 = return_r2(train_dataset_stats, ccn_dataset_stats)
loss_r2 = return_r2(train_dataset_stats, loss_dataset_stats)
mpn_r2 = return_r2(train_dataset_stats, mpn_dataset_stats)

models = ['Vanilla\nHALO', 'Post\nProcessed', 'ConSequence', 'CCN', 'Semantic\nLoss', 'MultiPlexNet']
individual = [base_r2[0][0], processing_r2[0][0], consequence_r2[0][0], ccn_r2[0][0], loss_r2[0][0], mpn_r2[0][0]]
individual_err = [base_r2[0][1], processing_r2[0][1], consequence_r2[0][1], ccn_r2[0][1], loss_r2[0][1], mpn_r2[0][1]]
coocurring = [base_r2[1][0], processing_r2[1][0], consequence_r2[1][0], ccn_r2[1][0], loss_r2[1][0], mpn_r2[1][0]]
coocurring_err = [base_r2[1][1], processing_r2[1][1], consequence_r2[1][1], ccn_r2[1][1], loss_r2[1][1], mpn_r2[1][1]]
sequential = [base_r2[2][0], processing_r2[2][0], consequence_r2[2][0], ccn_r2[2][0], loss_r2[2][0], mpn_r2[2][0]]
sequential_err = [base_r2[2][1], processing_r2[2][1], consequence_r2[2][1], ccn_r2[2][1], loss_r2[2][1], mpn_r2[2][1]]
fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax1.set_title("Generative Quality")
plt.xticks([1, 2, 3, 4, 5], models)
ax1.set_ylabel("$R^2$ Score")
ax1.set_xlabel("Model")
x = np.arange(len(models))
ax1.set_xticks(x)
ax1.set_xticklabels(models)
width = 0.15
bar1 = ax1.bar(x - 1.5 * width, individual, width, yerr=individual_err, label="Individual", color='blue')
bar2 = ax1.bar(x, coocurring, width, yerr=coocurring_err, label="Co-Occurring", color='orange')
bar3 = ax1.bar(x + 1.5 * width, sequential, width, yerr=sequential_err, label="Sequential", color='green')
bars = [bar1, bar2, bar3]
labels = [bar.get_label() for bar in bars]
plt.legend(bars, labels, loc="upper left")
plt.savefig('figures/inpatient_quality.png')