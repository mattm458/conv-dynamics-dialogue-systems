import os
from datetime import datetime
from os import path
from typing import Final, Optional

import pandas as pd
import torch
import torchtext
from pandas import DataFrame
from torchtext.data import get_tokenizer
from tqdm import tqdm

from cdmodel.preprocessing.consts import FEATURES, FEATURES_NORM_BY_CONV_SPEAKER
from cdmodel.preprocessing.datasets import Dataset
from cdmodel.preprocessing.normalize import norm_by_conv_speaker


def _segment_export(data: DataFrame, out_dir: str) -> None:
    segment_path: Final[str] = path.join(out_dir, "segments")

    # Have we processed segment data already?
    if path.exists(segment_path):
        print("segments directory exists, skipping...")
        return
    else:
        os.mkdir(segment_path)

    for id, group in tqdm(data.groupby("id"), desc="Exporting segments"):
        torch.save(group.to_dict(orient="list"), path.join(segment_path, f"{id}.pt"))


def _embeddings(data: DataFrame, out_dir: str) -> None:
    embeddings_path: Final[str] = path.join(out_dir, "embeddings")

    # Have we processed embeddings already?
    if path.exists(embeddings_path):
        print("embeddings directory exists, skipping...")
        return

    os.mkdir(embeddings_path)

    # Load GloVe embeddings
    vec = torchtext.vocab.GloVe(name="6B", dim=50)
    tokenizer = get_tokenizer("basic_english")

    # Look up and save embeddings for each conversation
    for id, text in tqdm(data.groupby("id")["transcript"], desc="Exporting embeddings"):
        tokenized = [tokenizer(x) for x in text]
        vectorized = [vec.get_vecs_by_tokens(x) for x in tokenized]
        lengths = torch.LongTensor([len(x) for x in vectorized])

        vectorized_padded = torch.nn.utils.rnn.pad_sequence(
            vectorized, batch_first=True
        )

        torch.save(vectorized_padded, path.join(embeddings_path, f"{id}-embeddings.pt"))
        torch.save(lengths, path.join(embeddings_path, f"{id}-lengths.pt"))


def _normalize(data: DataFrame, out_dir: str) -> DataFrame:
    # If we performed normalization already, there will be a "data-norm.csv" file
    # in the output directory. Otherwise, we have to perform normalization.
    data_norm_csv_path: Final[str] = path.join(out_dir, "data-norm.csv")
    if path.exists(data_norm_csv_path):
        print("data-norm.csv exists, loading...")
        return pd.read_csv(data_norm_csv_path)

    data_norm = norm_by_conv_speaker(data, features=FEATURES)
    data_norm.to_csv(data_norm_csv_path, index=False)

    return data_norm


def _extract_features(dataset: Dataset, out_dir: str) -> DataFrame:
    # If we performed feature extraction already, there will be a "data.csv" file
    # in the output directory. Otherwise, we have to perform feature extraction.
    data_csv_path: Final[str] = path.join(out_dir, "data.csv")
    if path.exists(data_csv_path):
        print("data.csv exists, loading...")
        return pd.read_csv(data_csv_path)
    
    features = dataset.extract_features()
    print(features)

    data = pd.DataFrame(features)
    data.to_csv(data_csv_path, index=False)

    return data


def preprocess(
    dataset_name: str,
    dataset_dir: str,
    out_dir: Optional[str],
    n_jobs: Optional[int],
) -> None:
    # Determine if we support the dataset for preprocessing. If so, instantiate the
    # corresponding preprocessing object
    dataset: Optional[Dataset] = None
    if dataset_name == "switchboard":
        from cdmodel.preprocessing.datasets import SwitchboardDataset

        dataset = SwitchboardDataset(dataset_dir=dataset_dir)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented")

    # If the caller did not specify an output directory, create a new one with the current
    # timestamp to distinguish it from previous runs.
    if out_dir is None:
        out_dir = f"{dataset_name} {datetime.now()}"

    # Try creating the directory.
    # If it already exists, we assume we're resuming an interrupted preprocessing session.
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        print(f"Output directory {out_dir} already exists, resuming...")

    # Step 1: Basic feature extraction and normalization
    data = _extract_features(dataset=dataset, out_dir=out_dir)
    data_norm = _normalize(data=data, out_dir=out_dir)

    # Step 2: Embeddings
    _embeddings(data=data_norm, out_dir=out_dir)

    # Step 3: Segment export
    _segment_export(data=data_norm, out_dir=out_dir)

    # ID assignment: All speakers
    torch.save(
        dict((k, i + 1) for (i, k) in enumerate(data.speaker_id.unique())),
        path.join(out_dir, "speaker-ids-all.pt"),
    )

    #print(dataset.get_conversations_with_min_speaker_repeat(min_repeat=3))

