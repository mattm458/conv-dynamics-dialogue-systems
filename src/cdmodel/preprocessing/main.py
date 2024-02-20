import os
from datetime import datetime
from os import path
from typing import Final, Optional

import pandas as pd
import torch
import torchtext
import ujson
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer
from tqdm import tqdm

from cdmodel.consts import FEATURES, FEATURES_NORM_BY_CONV_SPEAKER
from cdmodel.data.manifest import validate_df, write_manifest
from cdmodel.preprocessing.datasets import Dataset
from cdmodel.preprocessing.normalize import norm_by_conv_speaker


def _segment_export(df: DataFrame, out_dir: str) -> None:
    """
    Export conversational segment data to JSON files organized by conversation.

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
        print("segments directory exists, overwriting...")
    else:
        os.mkdir(segment_path)

    for id, group in tqdm(df.groupby("id"), desc="Exporting segments"):
        side_dict = group.groupby("side")["speaker_id"].unique().to_dict()
        group_dict = group.to_dict(orient="series")

        for k in group_dict:
            group_dict[k] = group_dict[k].tolist()

        group_dict["side_a_speaker_id"] = int(side_dict["A"][0])
        group_dict["side_b_speaker_id"] = int(side_dict["B"][0])

        with open(path.join(segment_path, f"{id}.json"), "w") as outfile:
            ujson.dump(group_dict, outfile)


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
        return pd.read_csv(data_norm_csv_path, engine="pyarrow")

    # Perform normalization and save the results.
    data_norm = pd.concat(
        [
            df,
            norm_by_conv_speaker(df, features=FEATURES),
        ],
        axis=1,
    )
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
        return pd.read_csv(data_csv_path, engine="pyarrow")

    # Perform feature extraction and save the results
    df = pd.DataFrame(dataset.extract_features())
    df.to_csv(data_csv_path, index=False)

    return df


def write_da(df: DataFrame, da_col: str, out_dir: str) -> None:
    df_da = pd.DataFrame(
        (da, i + 1) for (i, da) in enumerate(df[da_col][df[da_col].notnull()].unique())
    )
    if len(df_da) > 0:
        df_da.columns = pd.Index(["da", "idx"])
        df_da.to_csv(path.join(out_dir, f"{da_col}.csv"), index=False)


def preprocess(
    dataset_name: str,
    dataset_dir: str,
    out_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
    debug: bool = False,
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
    debug : bool
        Whether to run preprocessing in debug mode. Debug mode involves preprocessing a small subset of
        the dataset and occasionally outputting more information about the process. Defaults to `False`.

    Raises
    ------
    NotImplementedError
        Raised if an unsupported dataset is given as `dataset_name`.

    DatasetVersionError
        Raised if preprocessing is resumed on a dataset too old for current preprocessing code.
    """

    # Determine if we support the dataset for preprocessing. If so, instantiate the
    # corresponding preprocessing object
    dataset: Dataset
    if dataset_name == "switchboard":
        from cdmodel.preprocessing.datasets import SwitchboardDataset

        dataset = SwitchboardDataset(dataset_dir=dataset_dir, debug=debug)
    elif dataset_name == "fisher":
        from cdmodel.preprocessing.datasets import FisherDataset

        dataset = FisherDataset(dataset_dir=dataset_dir, debug=debug, n_jobs=8)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented")

    # If the caller did not specify an output directory, create a new one with the current
    # timestamp to distinguish it from previous runs.
    if out_dir is None:
        out_dir = f"{dataset_name} {datetime.now()}"

    print(f"Saving to {out_dir}")

    # Try creating the directory.
    # If it already exists, we assume we're resuming an interrupted preprocessing session.
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        print(f"Output directory {out_dir} already exists, resuming...")

    write_manifest(out_dir)

    # Step 1: Basic feature extraction and normalization
    df = __extract_features(dataset=dataset, out_dir=out_dir)
    df = __normalize(df=df, out_dir=out_dir)

    validate_df(df)

    # Write dataset properties
    with open(path.join(out_dir, "properties.json"), "w") as outfile:
        ujson.dump(dataset.get_properties(), outfile)

    # Write dialogue act indices
    write_da(df, "da_consolidated", out_dir=out_dir)
    write_da(df, "da_category", out_dir=out_dir)

    # Step 2: Embeddings
    __extract_embeddings(df=df, out_dir=out_dir)

    # Step 3: Segment export
    _segment_export(df=df, out_dir=out_dir)

    # ID assignment: All speakers
    write_id_mapping(set(df.speaker_id.unique()), out_dir, "speaker-ids-all")

    conversation_ids = list(df.id.unique())
    write_split(conversation_ids, out_dir, "all")

    # ID assignment: Dialogue acts only
    da_conversation_ids = set(df[df.da.notnull()].id.unique())
    if len(da_conversation_ids) > 0:
        write_id_mapping(
            set(df[df.id.isin(da_conversation_ids)].speaker_id.unique()),
            out_dir,
            "speaker-ids-da",
        )

    min_repeat_conversation_ids = get_conversation_ids_with_min_speaker_repeat(
        df, min_repeat=3
    )
    if len(min_repeat_conversation_ids) > 0:
        write_id_mapping(
            set(df[df.id.isin(min_repeat_conversation_ids)].speaker_id.unique()),
            out_dir,
            "speaker-ids-min-repeat",
        )


def write_split(conversation_ids: list[int], out_dir: str, name: str) -> None:
    train_ids, test_ids = train_test_split(
        conversation_ids, train_size=0.8, random_state=42
    )
    test_ids, val_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

    pd.DataFrame(train_ids).to_csv(
        path.join(out_dir, f"train-{name}.csv"), header=False, index=False
    )
    pd.DataFrame(val_ids).to_csv(
        path.join(out_dir, f"val-{name}.csv"), header=False, index=False
    )
    pd.DataFrame(test_ids).to_csv(
        path.join(out_dir, f"test-{name}.csv"), header=False, index=False
    )


def get_conversation_ids_with_min_speaker_repeat(
    df: DataFrame, min_repeat: int
) -> list[int]:
    df_calls = df.groupby(["id", "speaker_id", "side"]).size().reset_index()
    call_counts = df_calls.speaker_id.value_counts()

    eligible_callers = set(call_counts.index)

    prev_eligible_caller_size = len(eligible_callers)

    # Here, we iteratively reconstruct df_call_pairs by removing callers with fewer than 3 examples.
    # We have to do this iteratively because an almost-rare caller (with only 3 instances) might converse
    # with a rare caller (2 or fewer instances) and is removed, bringing their total count down to just 2.
    # This iteration solves that problem and results in a dataset where we're guaranteed
    # there are at least 3 instances of each caller.
    while True:
        # Get all the speakers from side A and B separately that are in the set of eligible callers
        df_calls_a = df_calls[
            (df_calls.side == "A") & (df_calls.speaker_id.isin(eligible_callers))
        ].set_index("id")
        df_calls_b = df_calls[
            (df_calls.side == "B") & (df_calls.speaker_id.isin(eligible_callers))
        ].set_index("id")

        # Inner join sides A and B. This has the effect of removing conversations where one speaker was eligible, but the other was not.
        df_call_pairs = df_calls_a.join(
            df_calls_b, lsuffix="_a", rsuffix="_b", how="inner"
        )

        # Since some conversations may have disappeared, determine the new set of eligible callers according to min_repeat
        call_counts = pd.concat(
            [df_call_pairs.speaker_id_a, df_call_pairs.speaker_id_b]
        ).value_counts()
        call_counts = call_counts[call_counts >= min_repeat]
        eligible_callers = set(call_counts.index)

        # If nothing changed, then we found a complete set of calls where all callers were present at least min_repeat times.
        if len(eligible_callers) == prev_eligible_caller_size:
            break

        # Otherwise, try again
        prev_eligible_caller_size = len(eligible_callers)

    return list(set(df[df.speaker_id.isin(eligible_callers)].id))


def write_id_mapping(speaker_ids: set, out_dir: str, name: str) -> None:
    """
    Generate a speaker ID mapping and write it to a CSV file.

    Parameters
    ----------
    speaker_ids : set
        A set of speaker IDs to map.
    out_dir : str
        A path to the directory where the mapping should be saved.
    filename : str
        A name for the mapping.
    """
    df = pd.DataFrame((k, i + 1) for (i, k) in enumerate(speaker_ids))
    df = df.set_axis(pd.Index(["speaker_id", "idx"]), axis=1)
    df.to_csv(path.join(out_dir, f"{name}.csv"), index=False)
