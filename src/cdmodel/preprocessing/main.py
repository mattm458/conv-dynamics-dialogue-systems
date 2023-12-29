import os
from os import path
from typing import Final, Optional

import pandas as pd
import torch
import torchtext
from pandas import DataFrame
from torchtext.data import get_tokenizer
from tqdm import tqdm

from cdmodel.preprocessing.datasets import Dataset
from cdmodel.preprocessing.normalize import norm_by_conv_speaker
from datetime import datetime

from cdmodel.preprocessing.consts import FEATURES, FEATURES_NORM_BY_CONV_SPEAKER


def _segment_export(df: DataFrame, out_dir: str) -> None:
    """
    Export conversational segment data to Torch .pt files organized by conversation.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing conversational segment data to export.
    out_dir : str
        A path to the directory where segment data should be saved.
    """

    # If we performed segment export already, there will be a "segments" directory
    # in the output directory. If not, we have to perform segment export.
    segment_path: Final[str] = path.join(out_dir, "segments")
    if path.exists(segment_path):
        print("segments directory exists, skipping...")
        return

    os.mkdir(segment_path)

    for id, group in tqdm(df.groupby("id"), desc="Exporting segments"):
        torch.save(group.to_dict(orient="list"), path.join(segment_path, f"{id}.pt"))


def __extract_embeddings(df: DataFrame, out_dir: str) -> None:
    """
    Perform embedding extraction on text transcripts contained in the given DataFrame.

    Currently, embedding extraction produces 50d GloVe embeddings.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing dialogue transcripts.
    out_dir : str
        A path to the directory where GloVe embeddings should be saved.
    """

    # If we performed embedding extraction already, there will be an "embedddings" directory
    # in the output directory. If not, we have to perform embedding extraction.
    embeddings_path: Final[str] = path.join(out_dir, "embeddings")
    if path.exists(embeddings_path):
        print("embeddings directory exists, skipping...")
        return

    os.mkdir(embeddings_path)

    # Load GloVe embeddings
    vec = torchtext.vocab.GloVe(name="6B", dim=50)
    tokenizer = get_tokenizer("basic_english")

    # Look up and save embeddings for each conversation
    for id, text in tqdm(df.groupby("id")["transcript"], desc="Exporting embeddings"):
        tokenized = [tokenizer(x) for x in text]
        vectorized = [vec.get_vecs_by_tokens(x) for x in tokenized]
        lengths = torch.LongTensor([len(x) for x in vectorized])

        vectorized_padded = torch.nn.utils.rnn.pad_sequence(
            vectorized, batch_first=True
        )

        torch.save(vectorized_padded, path.join(embeddings_path, f"{id}-embeddings.pt"))
        torch.save(lengths, path.join(embeddings_path, f"{id}-lengths.pt"))


def __normalize(df: DataFrame, out_dir: str) -> DataFrame:
    """
    Perform normalization on a given set of speech features in a DataFrame.

    Currently, the following types of normalization are performed:

    * 3 standard deviations above and below the median scaled to [-1, 1], per [Raitio et al., 2020](https://arxiv.org/abs/2009.06775).
        * By speaker and conversation

    A CSV file titled `data-norm.csv` containing the results of normalization
    will be saved in the given `out_dir`.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing speech features to normalize
    out_dir : str
        A path to the directory where a CSV file should be saved.

    Returns
    -------
    DataFrame
        A DataFrame containing all original columns, plus new columns with normalized features.
    """

    # If we performed normalization already, there will be a "data-norm.csv" file
    # in the output directory. If not, we have to perform normalization.
    data_norm_csv_path: Final[str] = path.join(out_dir, "data-norm.csv")
    if path.exists(data_norm_csv_path):
        print("data-norm.csv exists, loading...")
        return pd.read_csv(data_norm_csv_path)

    # Perform normalization and save the results.
    data_norm = norm_by_conv_speaker(df, features=FEATURES)
    data_norm.to_csv(data_norm_csv_path, index=False)

    return data_norm


def __extract_features(dataset: Dataset, out_dir: str) -> DataFrame:
    """
    Perform basic feature extraction on a given Dataset object.

    A CSV file titled `data.csv` containing the results of feature extraction
    will be saved in the given `out_dir`.

    Parameters
    ----------
    dataset : Dataset
        A Dataset object capable of performing feature extraction.
    out_dir : str
        A path to the directory where a CSV file should be saved.

    Returns
    -------
    DataFrame
        A DataFrame containing the results of preprocessing.
    """

    # If we performed feature extraction already, there will be a "data.csv" file
    # in the output directory. If not, we have to perform feature extraction.
    data_csv_path: Final[str] = path.join(out_dir, "data.csv")
    if path.exists(data_csv_path):
        print("data.csv exists, loading...")
        return pd.read_csv(data_csv_path)

    # Perform feature extraction and save the results
    df = pd.DataFrame(dataset.extract_features())
    df.to_csv(data_csv_path, index=False)

    return df


def preprocess(
    dataset_name: str,
    dataset_dir: str,
    out_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> None:
    """
    Preprocess a dataset for use in a conversational dynamics model.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to preprocess. Currently, only `switchboard` is supported.
    dataset_dir : str
        Path to the dataset's root directory.
    out_dir : str, optional
        A path to the directory where the results of preprocessing will be saved incrementally.
        If the directory does not exist, it will be created.
        If `None`, preprocessing output will be saved in `preprocessed/{dataset} {datetime}/`.

        If a preprocessing job was interrupted, and `out_dir` is set to the directory where the interrupted
        job was saving output, then preprocessing will resume from where it left off.

        Defaults to `None`.
    n_jobs : int, optional
        When executing parallelized operations, how many jobs to run at once. If `None`,
        jobs will run in series.

        Defaults to `None`.

    Raises
    ------
    NotImplementedError
        Raised if an unsupported dataset is given as `dataset_name`.
    """

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
    data = __extract_features(dataset=dataset, out_dir=out_dir)
    data_norm = __normalize(df=data, out_dir=out_dir)

    # Step 2: Embeddings
    __extract_embeddings(df=data_norm, out_dir=out_dir)

    # Step 3: Segment export
    _segment_export(df=data_norm, out_dir=out_dir)

    # ID assignment: All speakers
    torch.save(
        dict((k, i + 1) for (i, k) in enumerate(data.speaker_id.unique())),
        path.join(out_dir, "speaker-ids-all.pt"),
    )

    # TODO: Finish this
    # print(dataset.get_conversations_with_min_speaker_repeat(min_repeat=3))
