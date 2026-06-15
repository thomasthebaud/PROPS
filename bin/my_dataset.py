from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import random
import os
import re
from tqdm.auto import tqdm


SBERT_EMBEDDING_CACHE = {}
XVECTOR_CACHE = {}

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "model"


class XVectorDataset(torch.utils.data.Dataset):
    @staticmethod
    def resolve_csv_path(value):
        path = Path(str(value))
        if path.suffix == '.csv' or path.exists():
            return path
        if '_' in str(value):
            dataset, split = str(value).rsplit('_', 1)
            split_path = Path('data') / dataset / f'{split}.csv'
            if split_path.exists():
                return split_path
        legacy_path = Path('data') / str(value) / 'segments.csv'
        if legacy_path.exists():
            return legacy_path
        return Path('data') / str(value)

    @staticmethod
    def xvector_dataset_name(df, csv_path):
        if 'dataset' in df.columns and 'split' in df.columns:
            dataset = str(df['dataset'].dropna().iloc[0]).strip() if not df['dataset'].dropna().empty else ''
            split = str(df['split'].dropna().iloc[0]).strip() if not df['split'].dropna().empty else ''
            if dataset and split:
                return f'{dataset}_{split}'
        if csv_path.name in {'train.csv', 'dev.csv', 'test.csv'}:
            return f'{csv_path.parent.name}_{csv_path.stem}'
        return csv_path.parent.name

    @staticmethod
    def add_xvector_dataset_column(df, fallback_dataset):
        if 'dataset' not in df.columns or 'split' not in df.columns:
            df['set'] = fallback_dataset
            return df

        dataset = df['dataset'].fillna('').astype(str).str.strip()
        split = df['split'].fillna('').astype(str).str.strip()
        row_dataset = dataset + '_' + split
        df['set'] = row_dataset.where(dataset.ne('') & split.ne(''), fallback_dataset)
        return df

    def __init__(
        self,
        csv_paths,
        profile_csv_path,
        prop_dropout=0.2,
        dataset_fractions=None,
        seed=1234,
        preload_embeddings=False,
        preload_xvectors=True,
        xvector_preload_workers=16,
        sbert_root='exp/SBERT_embs/Capspeech',
    ):
        self.prop_dropout = prop_dropout
        self.sbert_root = sbert_root
        self.fields = ['pitch', 'age', 'gender', 'speaking_rate', 'speech_monotony', 'accent']
        dfs=[]
        dataset_fractions = dataset_fractions or {}
        for dataset in csv_paths:
            csv_path = self.resolve_csv_path(dataset)
            df = pd.read_csv(csv_path)
            dataset_key = self.xvector_dataset_name(df, csv_path)
            fraction = dataset_fractions.get(str(dataset), dataset_fractions.get(dataset_key, 1.0))
            if fraction < 1.0:
                original_len = len(df)
                sample_size = max(1, int(round(original_len * fraction)))
                df = df.sample(n=sample_size, random_state=seed).sort_index()
                print(
                    f"Using {len(df)}/{original_len} utterances "
                    f"({fraction:.1%}) from {csv_path}.",
                    flush=True,
                )
            df = self.add_xvector_dataset_column(df, dataset_key)
            dfs.append(df)
        self.utterances = pd.concat(dfs)
        if 'storage_path' not in self.utterances.columns:
            self.utterances['storage_path'] = ''
        if 'speaker' not in self.utterances.columns:
            self.utterances['speaker'] = 'unknown'
        for field in self.fields:
            if field not in self.utterances.columns:
                self.utterances[field]='unknown'
        self.utterances = self.utterances[['id', 'speaker', 'set', 'storage_path']+self.fields]
        self.utterances = self.utterances.fillna('unknown')
        self.utterances = self.utterances.reset_index(drop=True)
        print(f"Loaded {len(self.utterances)} utterances.")

        self.profiles = pd.read_csv(profile_csv_path)
        self.profiles = self.profiles.fillna('not_computed')
        print(f"Loaded {len(self.profiles)} profiles.")
        self.profile_lookup = self.build_profile_lookup()
        self.sbert_embeddings = {}
        self.xvectors = {}
        self.missing_xvector_counts = Counter()
        self.xvectors_preloaded = False
        self.length = len(self.utterances)
        if preload_xvectors:
            self.preload_xvectors(num_workers=xvector_preload_workers)
        if preload_embeddings:
            self.preload_sbert_embeddings()

    def __len__(self):
        return self.length

    def select_profile(self, properties):
        # return a profile with missing props
        # drops props at random
        # new_props = {k: properties.get(k) if np.random.rand(1)>self.prop_dropout else 'unknown' for k in self.fields}
        # only keeps gender
        # new_props = {k: properties.get(k) if k=='gender' else 'unknown' for k in self.fields}
        # keep all properties
        new_props = {k: properties.get(k) for k in self.fields}
        
        # n_known = np.sum([1 if new_props.get(k)!='unknown' else 0 for k in self.fields])
        # if n_known==0: return self.select_profile(properties)
        # else: 
        return new_props

    def profile_key(self, profile):
        return tuple(profile[k] for k in self.fields)

    def build_profile_lookup(self):
        lookup = {}
        for index, profile in self.profiles.iterrows():
            key = self.profile_key(profile)
            lookup.setdefault(key, []).append(index)
        return lookup

    def candidate_xvector_paths(self, id, dataset, storage_path=''):
        xvector_dir = Path('exp/xvectors/ecapa_tdnn') / str(dataset) / 'xvectors'
        candidates = [
            xvector_dir / f'{safe_name(id)}.npz',
            xvector_dir / f'{id}.npz',
        ]
        storage_path = str(storage_path).strip()
        if storage_path and storage_path != 'unknown':
            candidates.append(xvector_dir / f'{safe_name(Path(storage_path).stem)}.npz')
        deduped = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    def load_xvector(self, id, dataset, storage_path=''):
        # dim 192
        candidates = self.candidate_xvector_paths(id, dataset, storage_path)
        xvector_path = next((path for path in candidates if path.exists()), candidates[0])
        xvector_path = str(xvector_path)
        if xvector_path in XVECTOR_CACHE:
            return XVECTOR_CACHE[xvector_path]
        try:
            data_file = np.load(xvector_path)
        except FileNotFoundError as exc:
            candidate_text = ', '.join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"Missing xvector for id={id!r}, dataset={dataset!r}. Tried: {candidate_text}"
            ) from exc
        with data_file as data:
            xvector = np.asarray(data['xvector'], dtype=np.float32)
        norm = np.linalg.norm(xvector)
        if norm > 0:
            xvector = xvector / norm
        XVECTOR_CACHE[xvector_path] = torch.from_numpy(xvector)
        return XVECTOR_CACHE[xvector_path]

    def preload_xvectors(self, num_workers=16):
        rows = list(self.utterances[['id', 'set', 'storage_path']].itertuples(index=True, name=None))

        def load_one(row):
            idx, utterance_id, dataset, storage_path = row
            try:
                return idx, dataset, self.load_xvector(utterance_id, dataset, storage_path)
            except FileNotFoundError:
                return idx, dataset, None

        if num_workers <= 1:
            iterator = map(load_one, rows)
        else:
            executor = ThreadPoolExecutor(max_workers=num_workers)
            iterator = executor.map(load_one, rows)

        try:
            for idx, dataset, xvector in tqdm(
                iterator,
                total=len(rows),
                desc=f"Loading x-vectors ({max(1, num_workers)} threads)",
                unit="xvec",
            ):
                if xvector is None:
                    self.missing_xvector_counts[dataset] += 1
                    continue
                self.xvectors[idx] = xvector
        finally:
            if num_workers > 1:
                executor.shutdown(wait=True)
        self.xvectors_preloaded = True
        self.print_missing_xvector_summary()

    def print_missing_xvector_summary(self):
        if not self.missing_xvector_counts:
            print("Missing xvectors by dataset: none", flush=True)
            return
        print("Missing xvectors by dataset:", flush=True)
        for dataset, count in sorted(self.missing_xvector_counts.items()):
            print(f"  {dataset}: {count}", flush=True)

    def find_description(self, profile):
        matching_indices = self.profile_lookup.get(self.profile_key(profile), [])
        if len(matching_indices)==0:
            print("Warning: empty dataframe")
            print(profile)
            exit()
        desc_num = np.random.randint(1,10)#keep desc0 for testing
        index = matching_indices[0]
        return self.profiles.at[index, f'desc{desc_num}'], desc_num, index

    def load_sbert_emb(self, desc_num, index):
        # dim 384
        sbert_path = f'{self.sbert_root}/{index}/desc{desc_num}.npz'
        if sbert_path in SBERT_EMBEDDING_CACHE:
            return SBERT_EMBEDDING_CACHE[sbert_path]
        if not os.path.exists(sbert_path):
            SBERT_EMBEDDING_CACHE[sbert_path] = torch.zeros(384)
            return SBERT_EMBEDDING_CACHE[sbert_path]
        with np.load(sbert_path) as data:
            sbert_emb = data['embedding']
        SBERT_EMBEDDING_CACHE[sbert_path] = torch.from_numpy(sbert_emb)
        return SBERT_EMBEDDING_CACHE[sbert_path]

    def preload_sbert_embeddings(self):
        total = len(self.profiles) * 10
        progress = tqdm(total=total, desc="Loading SBERT embeddings", unit="emb")
        for index in self.profiles.index:
            for desc_num in range(1,10):
                self.sbert_embeddings[(index, desc_num)] = self.load_sbert_emb(desc_num, index)
                progress.update(1)
        progress.close()
        

    def __getitem__(self, idx):
        # Get utterance
        utterance = self.utterances.iloc[idx]
        # Properties of this utterance
        properties = {k: utterance.get(k, 'unknown') for k in self.fields}
        # get x-vector
        xvector = self.xvectors.get(idx)
        if xvector is None:
            xvector = self.load_xvector(utterance['id'], utterance['set'], utterance.get('storage_path', ''))
        # select a profile (by dropping at random some properties)
        selected_profile = self.select_profile(properties)
        # get description
        description, desc_num, index = self.find_description(selected_profile)
        # get SBERT embeddings
        embedding = self.sbert_embeddings.get((index, desc_num))
        if embedding is None:
            embedding = self.load_sbert_emb(desc_num, index)

        return xvector, embedding, torch.tensor(index, dtype=torch.long)


class ProfileDataset(XVectorDataset):
    def __init__(
        self,
        csv_paths,
        profile_csv_path,
        prop_dropout=0.2,
        dataset_fractions=None,
        seed=1234,
        preload_embeddings=False,
        preload_xvectors=True,
        xvector_preload_workers=16,
        sbert_root='exp/SBERT_embs/Capspeech',
    ):
        super().__init__(
            csv_paths=csv_paths,
            profile_csv_path=profile_csv_path,
            prop_dropout=prop_dropout,
            dataset_fractions=dataset_fractions,
            seed=seed,
            preload_embeddings=preload_embeddings,
            preload_xvectors=preload_xvectors,
            xvector_preload_workers=xvector_preload_workers,
            sbert_root=sbert_root,
        )
        self.profile_utterance_lookup = self.build_profile_utterance_lookup()
        self.available_profile_indices = [
            profile_index
            for profile_index in self.profiles.index
            if self.profile_utterance_lookup.get(profile_index)
        ]
        self.profile_embedding_rows = [
            (profile_index, desc_num)
            for profile_index in self.available_profile_indices
            for desc_num in range(1, 10)
        ]
        self.length = len(self.profile_embedding_rows)
        print(
            f"Loaded {len(self.available_profile_indices)} profiles with matching x-vectors "
            f"and {self.length} profile-description embedding rows.",
            flush=True,
        )

    def normalize_property(self, value):
        if pd.isna(value):
            return 'unknown'
        return str(value).strip() or 'unknown'

    def utterance_key(self, utterance):
        return tuple(self.normalize_property(utterance.get(field, 'unknown')) for field in self.fields)

    def build_utterance_key_lookup(self):
        lookup = {}
        for utterance_index, utterance in self.utterances.iterrows():
            if self.xvectors_preloaded and utterance_index not in self.xvectors:
                continue
            lookup.setdefault(self.utterance_key(utterance), []).append(utterance_index)
        return lookup

    def matching_utterance_indices(self, profile, utterance_key_lookup):
        known_values = {
            field: value
            for field in self.fields
            if (value := self.normalize_property(profile.get(field, 'unknown'))) not in {'unknown', 'not_computed'}
        }
        matching_indices = []
        for utterance_key, utterance_indices in utterance_key_lookup.items():
            if all(utterance_key[self.fields.index(field)] == value for field, value in known_values.items()):
                matching_indices.extend(utterance_indices)
        return matching_indices

    def build_profile_utterance_lookup(self):
        lookup = {}
        utterance_key_lookup = self.build_utterance_key_lookup()
        for profile_index, profile in tqdm(
            self.profiles.iterrows(),
            total=len(self.profiles),
            desc="Matching profiles to x-vectors",
            unit="profile",
        ):
            matching_indices = self.matching_utterance_indices(profile, utterance_key_lookup)
            if matching_indices:
                lookup[profile_index] = matching_indices
        return lookup

    def __getitem__(self, idx):
        profile_index, desc_num = self.profile_embedding_rows[idx]
        utterance_idx = random.choice(self.profile_utterance_lookup[profile_index])
        utterance = self.utterances.iloc[utterance_idx]

        xvector = self.xvectors.get(utterance_idx)
        if xvector is None:
            xvector = self.load_xvector(utterance['id'], utterance['set'], utterance.get('storage_path', ''))

        embedding = self.sbert_embeddings.get((profile_index, desc_num))
        if embedding is None:
            embedding = self.load_sbert_emb(desc_num, profile_index)

        return xvector, embedding, torch.tensor(profile_index, dtype=torch.long)

if __name__=='__main__':
    mydataset = ProfileDataset(["CommonVoice_test", "GigaSpeech_test"], 'data/profile_prompts.csv')
    for data in mydataset:
        xv, emb, _profile_index = data
        print(xv.shape, emb.shape)
        break
