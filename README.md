# ConSequence

This is the source code for reproducing the inpatient dataset experiments found in the paper "ConSequence: Synthesizing Logically Constrained Sequences for Electronic Health Record Generation."

## Generating the Dataset
This code interfaces with the pubilc MIMIC-III ICU stay database. Before using the code, you will need to apply, complete training, and download the ADMISSIONS, DIAGNOSES_ICD, and PROCEDURES_ICD tables from <https://physionet.org>. From there, generate an empty directory `inpatient_data/`, edit the `mimic_dir` variable in the file `utils/genDataset.py`, and run that file. It will generate the core data files.
From there, run `utils/find_rules.py` to find and generate the rules for that dataset.

## Training a Model
Next, a model can be training by creating an empty `save/` directory and running the `train_consequence.py` script.

## Training Baseline Models
Next, any desired baseline models may be trained using the `train_{baseline_model}.py` scripts (with `train_model.py` being an unconstrained model).

## Generating Data
From there, generate datasets corresponding to each compared more using the `evaluate_generationSpeed_{model}.py` scripts. This will both benchmark the generation speed and output synthetic datasets. Before generating data create `results/` and `results/generationSpeeds/` directories

## Evaluating the Models and Data
Finally, the trained models and its synthetic data may be evaluated. Before beginning, create the following directory paths:
* `results/violation_stats/`
* `results/testing_stats/`
* `results/dataset_stats/`
* `results/dataset_stats/plots/`

After these directories are created, first run the `evaluate_testSet_{model}.py` scripts to generate modeling statistics, `evaluate_datasetQuality.py` script to generate synthetic data quality metrics, and the `evaluate_percentValidity.py` and `evaluate_ruleViolations.py` scripts to generate rule compliance metrics. All metrics will be both printed and saved.
