import pandas as pd
import json
import os
import glob
from os.path import join

liver_targets = pd.read_csv(
    "/vol/miltank/projects/ukbb/projects/risk_assessment/comorbidities/coding19_with_elixhauser_comorbidities.tsv",
    sep="\t",
)

liver_icds = liver_targets[liver_targets["Liver disease"] == 1]["coding"].to_list()

dir_path = "/vol/miltank/projects/ukbb/data/abdominal/liver_data/numpy/"
np_files = glob.glob(os.path.join(dir_path, "*.npy"))
ids = [os.path.splitext(os.path.basename(file))[0] for file in np_files]
ids = [int(id) for id in ids]

pheno_dir = "/vol/miltank/projects/ukbb/677795/phenos"
field_id = "41270-0.*"
pfiles = glob.glob(join(pheno_dir, f"{field_id}.csv"))

dfs = [pd.read_csv(pfile) for pfile in pfiles]
icd10_diagnoses = pd.concat(dfs, axis=1)
icd10_diagnoses = icd10_diagnoses[sorted(icd10_diagnoses.columns, key=lambda x: float(x.split(".")[1]))]

dfout = pd.read_csv(join(pheno_dir, "eid.csv"))
icd10_diagnoses = pd.concat([dfout, icd10_diagnoses], axis=1)

diagnosis_mask = icd10_diagnoses.iloc[:, 1:].isin(liver_icds).any(axis=1)
result_dict = dict(zip(icd10_diagnoses["eid"], diagnosis_mask.astype(int)))

pos_count = 0
for k, v in result_dict.items():
    if v:
        pos_count += v

print(f"Positive label ratio: {pos_count / len(result_dict):.4f}")

with open("liver_diagnosis_dict.json", "w") as f:
    json.dump(result_dict, f)

print("liver_diagnosis_dict.json created successfully.")
