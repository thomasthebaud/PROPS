from pathlib import Path
import pandas as pd

datasets = ["CommonVoice", "GigaSpeech", "MLS-en", "Emilia-en"]
splits = ["train", "dev", "test"]
output_root = Path("data") / "Capspeech"
output_root.mkdir(parents=True, exist_ok=True)

for split in splits:
    frames = []
    for dataset_name in datasets:
        path = Path("data") / dataset_name / f"{split}.csv"
        if not path.exists():
            print(f"WARNING: missing {path}; skipping")
            continue
        df = pd.read_csv(path)
        df.insert(0, "dataset", dataset_name)
        frames.append(df)
    out = output_root / f"{split}.csv"
    if not frames:
        pd.DataFrame().to_csv(out, index=False)
        print(f"Wrote 0 rows to {out}")
        continue
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged.to_csv(out, index=False)
    print(f"Wrote {len(merged)} rows to {out}")