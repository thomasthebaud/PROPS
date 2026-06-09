import os
import pandas as pd
from tqdm import tqdm
import mutagen
import numpy as np
import re

dataset, path = 'CommonVoice','/export/corpora7/CommonVoice-19.0/en/'

files = [i for i in os.listdir(path) if '.tsv' in i]
files = ['train', 'test', 'dev']
print(files)
dfs = []
for file in files:
    df = pd.read_csv(path+file+'.tsv', sep='\t')
    df['split'] = file
    dfs.append(df)
df = pd.concat(dfs)
df.pop('sentence_id')
df.pop('variant')
df.pop('locale')
df.pop('segment')
df.pop('sentence_domain')
df.pop('up_votes')
df.pop('down_votes')
print(df.shape)

df = df.dropna(subset=["age"])
print('droped missing ages', df.shape)

# df = df[df['age'].isin(['sixties', 'seventies', 'nineties', 'eighties', 'teens'])]
# a_to_a = {'sixties':65, 'seventies':75, 'nineties':95, 'eighties':85, 'teens':15, 'twenties':25}
# a_to_a = {'sixties':65, 'seventies':75, 'nineties':95, 'eighties':85, 'teens':15, 'twenties':25, 'fourties':45, 'fifties':55, 'thirties':35}
# df['age'] = [a_to_a[a] for a in df['age']]
# print(df.shape)
# print(set(df['age']))

duration_df = pd.read_csv(path+'clip_durations.tsv', sep='\t')
duration_df = duration_df[duration_df['clip'].isin(list(df['path']))]
durations_dict = {row['clip']:row['duration[ms]'] for idx, row in duration_df.iterrows()}

durations_list = []

for p in df['path']:
    durations_list.append(durations_dict[p]/1000 if p in durations_dict else 'NaN')

df['duration'] = durations_list
df = df[df['duration']!='NaN']
df = df[df['duration']>5]
print(df.shape)
print('duration = ', np.sum(df['duration'])/3600)


df['corpusid']=dataset
df['dataset']=dataset
df['source_type']='afv'
spk_to_id = {spk:'cv_'+str(idx) for idx, spk in enumerate(list(set(df['client_id'])))}
df['speaker'] = [spk_to_id[spk] for spk in df['client_id']]
df.pop('client_id')

df['storage_path'] = [path+'clips/'+p for p in df['path']]
df['video_id'] = df['path']
df.pop('path')
df['sample_freq']=16000

df = df[df['gender'].isin(['male_masculine', 'female_feminine'])]
gd = {'male_masculine':'male', 'female_feminine':'female'}
df['gender'] = [gd[g] for g in df['gender']]
# df['label'] = 'CTL'
df['id'] = ['cv_'+i.split('_')[-1][:-4] for i in df['video_id']]
df['transcript'] = df['sentence']
print(df.shape)
print(df.head())

df = df.dropna(subset='accents')
print('dropped missing accents', df.shape)
accents = set(list(df['accents']))

CATEGORIES = {
    "US": [
        "united states", "american", "california", "texas", "midwest",
        "new york", "boston", "chicago", "florida", "appalachian",
        "oregon", "washington", "rhode island", "pennsylvania",
        "philadelphia", "kentucky", "ohio", "new england"
    ],

    "England": [
        "england", "british", "london", "rp",
        "received pronunciation", "yorkshire", "liverpool",
        "lancashire", "cumbrian", "midlands",
        "sussex", "east anglian", "northumbrian",
        "home counties"
    ],

    "Scotland": ["scottish", "scotland"],
    "Ireland": ["irish", "northern irish", "belfast"],
    "Wales": ["welsh"],

    "Canada": ["canadian", "ontario", "toronto"],

    "Australia": ["australian", "australia", "sydney"],
    "New Zealand": ["new zealand", "kiwi"],

    "South Asia": [
        "india", "indian", "south asia", "pakistan",
        "sri lanka", "nepal", "nepalese",
        "bangladesh", "bangladeshi", "bengali",
        "south indian"
    ],

    "Southeast Asia": [
        "singapore", "singaporean", "malaysia",
        "malaysian", "filipino", "philippines",
        "indonesia", "indonesian", "javanese",
        "thai", "vietnam"
    ],

    "East Asia": [
        "chinese", "hong kong", "japanese",
        "hmong", "east asian", "tibetan"
    ],

    "Africa": [
        "african", "south african", "southern african",
        "nigerian", "nigeria", "kenyan",
        "ghanaian", "west african",
        "east african", "afrikaans"
    ],

    "Caribbean": [
        "west indies", "bermuda", "jamaica",
        "trinidad", "patois", "caribbean",
        "west indian"
    ],

    "Latin America": [
        "latin", "latino", "hispanic",
        "mexican", "argentinian",
        "brazilian", "brazillian",
        "central american"
    ],

    "French": ["french", "france"],

    "Germanic": [
        "german", "germany", "austrian",
        "swiss", "alemannic", "dutch"
    ],

    "Nordic": [
        "swedish", "scandinavian", "danish",
        "norwegian", "finnish", "icelandic"
    ],

    "Slavic / Eastern Europe": [
        "slavic", "polish", "russian",
        "ukrainian", "bulgarian", "czech",
        "croatian", "serbian", "slovak",
        "latvian", "romanian",
        "hungarian", "georgian",
        "kazakhstan"
    ],

    "Southern Europe": [
        "italian", "spanish",
        "catalan", "greek", "cretan"
    ],

    "Middle East": ["israeli", "turkish"],
}

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[_/,-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def categorize_accent(accent):
    a = normalize(accent)
    matches = []
    for category, keywords in CATEGORIES.items():
        if any(k in a for k in keywords):
            matches.append(category)
    # No match
    if len(matches) == 0:
        return "Other / unclear"
    # Single match
    if len(matches) == 1:
        return matches[0]
    return "multiple"


# Final mapping
accent_to_category = {
    accent: categorize_accent(accent)
    for accent in accents
}

# Reverse mapping
category_to_accents = {}

for accent, category in accent_to_category.items():
    category_to_accents.setdefault(category, []).append(accent)

# Summary
# print("\n===== CATEGORY COUNTS =====")
# for cat, vals in sorted(category_to_accents.items()):
#     print(f"{cat:25s} {len(vals)}")

df['accents'] = [accent_to_category[a] for a in (df['accents'])]
# print(set(list(df['accents'])))

df = df[~df['accents'].isin(['multiple', "Other / unclear"])]
print(f"After removing unclear accents: {df.shape}")
df['accent'] = df['accents']

for split in files:
    os.makedirs(f"data/{dataset}_{split}", exist_ok=True)
    df_ = df[df['split']==split]
    segments = df_[['id','gender','age', 'speaker', 'transcript', 'accent']]
    recordings = df_[['id','storage_path','duration','sample_freq']]
    segments.to_csv(f"data/{dataset}_{split}/segments.csv", index=False)
    recordings.to_csv(f"data/{dataset}_{split}/recordings.csv", index=False)
