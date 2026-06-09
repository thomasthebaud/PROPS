import pandas as pd
import numpy as np
from tqdm import tqdm

dfs=[]
dataset_list = ["CommonVoice_dev", "CommonVoice_test", "GigaSpeech_test", "CommonVoice_train", "GigaSpeech_train"]
dataset_list = ["CommonVoice_dev", "CommonVoice_test", "CommonVoice_train"]
columns_of_interest = ['pitch', 'age', 'gender', 'speaking_rate', 'speech_monotony', 'accent']
columns_of_interest = ['gender', 'age', 'accent']

for dataset in dataset_list:
    dfs.append(pd.read_csv(f'data/{dataset}/segments.csv'))
df = pd.concat(dfs)
df = df[['speaker']+columns_of_interest]
print(f"Loaded {len(df)} utterances")

descs = df[columns_of_interest]
descs.drop_duplicates(inplace=True)
descs=descs.fillna('unknown')
print(f"Found {len(descs)} unique descriptions")


combinations = {col:[] for col in descs.columns}
descs.loc[-1] = ['unknown']*len(descs.columns)
if len(columns_of_interest)==6:
    for pitch in set(descs['pitch']):
        for age in set(descs['age']):
            for gender in set(descs['gender']):
                for speaking_rate in set(descs['speaking_rate']):
                    for speech_monotony in set(descs['speech_monotony']):
                        for accent in set(descs['accent']):
                            combinations['pitch'].append(pitch)
                            combinations['age'].append(age)
                            combinations['gender'].append(gender)
                            combinations['speaking_rate'].append(speaking_rate)
                            combinations['speech_monotony'].append(speech_monotony)
                            combinations['accent'].append(accent)
elif len(columns_of_interest)==3:
    for age in set(descs['age']):
        for gender in set(descs['gender']):
            for accent in set(descs['accent']):
                combinations['age'].append(age)
                combinations['gender'].append(gender)
                combinations['accent'].append(accent)

combinations = pd.DataFrame(combinations)
print(f"With the fields: {descs.columns}")
print(f'Makes a total of {len(combinations)} combinations.')

combinations['valid']=True
for idx, row in tqdm(combinations.iterrows(), total=len(combinations), desc='finding represented combinations in the dataset'):
    valid_desc = descs.copy()
    for col in descs.columns:
        if row[col]!='unknown': 
            valid_desc = valid_desc[valid_desc[col]==row[col]]
            if len(valid_desc)==0:
                combinations.at[idx, 'valid']=False
                break

print(f"Only {len(combinations[combinations['valid']])}/{len(combinations)} are represented in the current dataset")
valid_combinations = combinations[combinations['valid']]

valid_combinations.pop('valid')
all_unknown = valid_combinations.copy()
for col in all_unknown.columns:
    all_unknown = all_unknown[all_unknown[col]=='unknown']
print('dropping index:', int(all_unknown.index[0]))
valid_combinations.drop(int(all_unknown.index[0]), inplace=True)
print("Removed the one 'all unknown' combination.")

csv_path = 'data/profile_combinations.csv'
valid_combinations.to_csv(csv_path, index=False)
print(f"Saved in {csv_path}")