#!/usr/bin/env python3
"""Extract ECAPA-TDNN speaker embeddings for one PROPS dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm


DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
torch: Any = None
torchaudio: Any = None
np: Any = None
EncoderClassifier: Any = None


def load_runtime_dependencies() -> None:
    global torch
    global torchaudio
    global np
    global EncoderClassifier

    import numpy as numpy_module
    import torch as torch_module
    import torchaudio as torchaudio_module

    try:
        from speechbrain.inference.speaker import EncoderClassifier as encoder_classifier
    except ImportError:  # SpeechBrain < 1.0
        from speechbrain.pretrained import EncoderClassifier as encoder_classifier

    np = numpy_module
    torch = torch_module
    torchaudio = torchaudio_module
    EncoderClassifier = encoder_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SpeechBrain ECAPA-TDNN xvectors for data/<dataset_name>."
    )
    parser.add_argument("dataset_name", nargs="?", help="Dataset directory under --data-root.")
    parser.add_argument("--dataset-name", dest="dataset_name_opt", help="Dataset directory under --data-root.")
    parser.add_argument("--data-root", default="data", type=Path, help="Directory containing dataset folders.")
    parser.add_argument("--exp-root", default=Path("exp/xvectors"), type=Path, help="Root output directory.")
    parser.add_argument("--model-name", default="ecapa_tdnn", help="Directory name under --exp-root.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="SpeechBrain model source.")
    parser.add_argument(
        "--savedir",
        type=Path,
        default=None,
        help="Where SpeechBrain caches the model. Defaults to exp/speechbrain_models/<model-name>.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    parser.add_argument("--sample-rate", default=16000, type=int, help="Target sample rate for ECAPA.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs instead of resuming.")
    parser.add_argument("--limit", type=int, default=None, help="Extract only the first N rows; useful for smoke tests.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress log interval.")
    args = parser.parse_args()

    args.dataset_name = args.dataset_name_opt or args.dataset_name
    if not args.dataset_name:
        parser.error("dataset_name is required, e.g. GigaSpeech_test")

    return args


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an 'id' column")
        return {row["id"]: row for row in reader}


def read_segments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an 'id' column")
        return list(reader)


def read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "id" in reader.fieldnames and "xvector_path" in reader.fieldnames:
            for row in reader:
                xvector_path = row.get("xvector_path", "")
                if xvector_path and Path(xvector_path).exists():
                    done.add(row["id"])
    return done


def load_waveform(path: Path, sample_rate: int, device: str) -> tuple[torch.Tensor, float]:
    waveform, source_sample_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError("audio file is empty")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, sample_rate)

    waveform = waveform.squeeze(0).to(device)
    duration = float(waveform.numel()) / float(sample_rate)
    return waveform, duration


def extract_embedding(
    classifier: EncoderClassifier,
    audio_path: Path,
    sample_rate: int,
    device: str,
) -> tuple[Any, float]:
    waveform, duration = load_waveform(audio_path, sample_rate, device)
    batch = waveform.unsqueeze(0)
    wav_lens = torch.ones(1, device=device)

    with torch.no_grad():
        embedding = classifier.encode_batch(batch, wav_lens=wav_lens)

    vector = embedding.squeeze().detach().cpu().float().numpy()
    if vector.size == 0:
        raise ValueError("model returned an empty embedding")
    return vector, duration


def open_metadata_writer(path: Path, append: bool) -> tuple[Any, csv.DictWriter]:
    fieldnames = ["id", "storage_path", "xvector_path", "duration", "xvector_dim"]
    file_obj = path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
    if not append:
        writer.writeheader()
    return file_obj, writer


def write_failure_header(path: Path, append: bool) -> tuple[Any, csv.DictWriter]:
    file_obj = path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file_obj, fieldnames=["id", "storage_path", "error"])
    if not append:
        writer.writeheader()
    return file_obj, writer


def metadata_is_appendable(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return bool(reader.fieldnames and "id" in reader.fieldnames and "xvector_path" in reader.fieldnames)


def xvector_npz_path(output_dir: Path, segment_id: str) -> Path:
    return output_dir / "xvectors" / f"{safe_name(segment_id)}.npz"


def save_xvector(path: Path, segment_id: str, vector: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, id=segment_id, xvector=np.asarray(vector, dtype=np.float32))
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    load_runtime_dependencies()

    dataset_dir = args.data_root / args.dataset_name
    recordings_path = dataset_dir / "recordings.csv"
    segments_path = dataset_dir / "segments.csv"

    if not recordings_path.exists():
        raise FileNotFoundError(f"Missing {recordings_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"Missing {segments_path}")

    output_dir = args.exp_root / safe_name(args.model_name) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    xvector_path = output_dir / "xvector.csv"
    failures_path = output_dir / "failures.csv"

    if args.overwrite:
        xvector_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)

    recordings = read_csv_by_id(recordings_path)
    segments = read_segments(segments_path)
    if args.limit is not None:
        segments = segments[: args.limit]

    done_ids = read_done_ids(xvector_path)
    device = resolve_device(args.device)
    savedir = args.savedir or (args.exp_root.parent / "speechbrain_models" / safe_name(args.model_name))

    print(f"dataset={args.dataset_name}", flush=True)
    print(f"segments={len(segments)} already_done={len(done_ids)}", flush=True)
    print(f"model={args.source} model_name={args.model_name}", flush=True)
    print(f"device={device} output_dir={output_dir}", flush=True)

    classifier = EncoderClassifier.from_hparams(
        source=args.source,
        savedir=str(savedir),
        run_opts={"device": device},
    )
    classifier.eval()

    args_path = output_dir / "extract_args.json"
    args_path.write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "data_root": str(args.data_root),
                "model_name": args.model_name,
                "source": args.source,
                "savedir": str(savedir),
                "sample_rate": args.sample_rate,
                "device": device,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    xvector_file = None
    xvector_writer = None
    failure_file, failure_writer = write_failure_header(failures_path, failures_path.exists() and not args.overwrite)
    append_metadata = metadata_is_appendable(xvector_path) and not args.overwrite
    extracted = 0
    failed = 0

    try:
        for index, segment in tqdm(enumerate(segments, start=1), desc=f'Extracting x-vectors with {args.model_name}', total=len(segments)):
            segment_id = segment["id"]
            if segment_id in done_ids:
                continue

            recording = recordings.get(segment_id)
            storage_path = recording.get("storage_path", "") if recording else ""
            if not storage_path:
                failure_writer.writerow(
                    {"id": segment_id, "storage_path": storage_path, "error": "missing recording/storage_path"}
                )
                failure_file.flush()
                failed += 1
                continue

            try:
                vector, audio_duration = extract_embedding(
                    classifier=classifier,
                    audio_path=Path(storage_path),
                    sample_rate=args.sample_rate,
                    device=device,
                )
                npz_path = xvector_npz_path(output_dir, segment_id)
                save_xvector(npz_path, segment_id, vector)
                if xvector_writer is None:
                    xvector_file, xvector_writer = open_metadata_writer(
                        xvector_path,
                        append=append_metadata,
                    )
                duration = segment.get("duration") or recording.get("duration") or f"{audio_duration:.6f}"
                row = {
                    "id": segment_id,
                    "storage_path": storage_path,
                    "xvector_path": str(npz_path),
                    "duration": duration,
                    "xvector_dim": int(vector.shape[0]),
                }
                xvector_writer.writerow(row)
                xvector_file.flush()
                extracted += 1
            except Exception as exc:  # Keep long dataset jobs moving.
                failure_writer.writerow({"id": segment_id, "storage_path": storage_path, "error": repr(exc)})
                failure_file.flush()
                failed += 1

            # if args.log_every > 0 and index % args.log_every == 0:
            #     print(f"processed={index}/{len(segments)} extracted={extracted} failed={failed}", flush=True)
    finally:
        if xvector_file is not None:
            xvector_file.close()
        failure_file.close()

    print(f"done extracted={extracted} failed={failed} skipped={len(done_ids)}", flush=True)
    print(f"wrote {xvector_path}", flush=True)
    if failed:
        print(f"failures logged in {failures_path}", flush=True)
    return 0 if extracted or len(done_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
