import csv
import itertools
import os
import re
from functools import partial
from os import path

import pandas as pd
import parselmouth
import torch
import torchtext
from pqdm.processes import pqdm
from sklearn.model_selection import train_test_split
from speech_utils.preprocessing.feature_extraction import extract_features
from torchtext.data import get_tokenizer
from tqdm import tqdm
from unidecode import unidecode


def __get_transcript_paths(fisher_dir):
    transcript_paths = []

    for dir in ["fe_03_p1_tran", "fe_03_p2_tran"]:
        trans_path = path.join(fisher_dir, dir, "data", "trans")
        for filepath, _, files in os.walk(trans_path):
            for file in files:
                if ".txt" not in file:
                    continue
                transcript_id = file[:-4]
                transcript_paths.append(
                    (transcript_id, path.join(filepath, file)))

    return transcript_paths


def __clean_text(text):
    text = text.lower()
    text = unidecode(text)
    text = text.replace("((", "").replace("))", "")
    text = text.replace("[laughter]", "")
    text = text.replace("[mn]", "")
    text = text.replace("[cough]", "")
    text = text.replace("[noise]", "")
    text = text.replace("[sigh]", "")
    text = text.replace("[lipsmack]", "")
    text = text.replace("[laugh]", "")
    text = text.replace("[sneeze]", "")
    text = text.replace("[breath]", "")
    text = text.replace("[pause]", "")
    text = text.replace("[[skip]]", "")

    text = " ".join(text.split())

    if text == "":
        return None

    return text


__parse_transcript_regex = re.compile("(\d+\.\d+) (\d+\.\d+) ([AB]): (.+)")


def __parse_transcript(transcript_id, transcript_path):
    lines = []

    with open(transcript_path) as infile:
        for line in infile:
            line = line.strip()

            result = __parse_transcript_regex.match(line)
            if result is None:
                continue

            start, end, speaker, text = result.groups()

            start = float(start)
            end = float(end)
            text = __clean_text(text)

            if text is None:
                continue

            lines.append((transcript_id, start, end, speaker, text))

    return lines


def __feature_extraction(transcript, fisher_dir):
    id = transcript[0][0]

    wav_path = path.join(fisher_dir, "wav", f"{id}.wav")
    sound = parselmouth.Sound(wav_path)
    sound_a, sound_b = sound.extract_all_channels()

    data = []
    for id, start, end, speaker, text in transcript:
        if speaker == "A":
            sound_speaker = sound_a
        else:
            sound_speaker = sound_b

        features = extract_features(
            sound=sound_speaker, transcript=text, start=start, end=end
        )

        if features is None:
            continue

        features["id"] = id
        features["start"] = start
        features["end"] = end
        features["speaker"] = speaker
        features["text"] = text

        data.append(features)

    return data


def __save_embeddings(transcript, tokenizer, vec, embedding_out_dir):
    id = transcript[0]["id"]

    embeddings_all = []
    lengths = []
    for line in transcript:
        text = line["text"]
        vectorized = vec.get_vecs_by_tokens(tokenizer(text))
        embeddings_all.append(vectorized)
        lengths.append(len(vectorized))

    embeddings_all = torch.nn.utils.rnn.pad_sequence(
        embeddings_all, batch_first=True)
    lengths = torch.LongTensor(lengths)

    torch.save(embeddings_all, path.join(
        embedding_out_dir, f"{id}-embeddings.pt"))
    torch.save(lengths, path.join(embedding_out_dir, f"{id}-lengths.pt"))


def __do_feature_extract_and_embedding(
    transcript, fisher_dir, tokenizer, vec, embedding_out_dir
):
    results = __feature_extraction(transcript, fisher_dir=fisher_dir)
    __save_embeddings(results, tokenizer, vec, embedding_out_dir)

    return results


FEATURES = ['duration', 'duration_vcd', 'pitch_mean', 'pitch_5', 'pitch_95',
            'pitch_range', 'intensity_mean', 'intensity_mean_vcd', 'jitter',
            'shimmer', 'nhr', 'nhr_vcd', 'rate', 'rate_vcd']
FEATURES_ZSCORE = [f"{x}_zscore" for x in FEATURES]


def run(fisher_dir, embedding_out_dir):
    print("Loading word embeddings...")
    vec = torchtext.vocab.GloVe(name="6B", dim=50)
    tokenizer = get_tokenizer("basic_english")

    print("Parsing transcripts...")
    transcript_paths = __get_transcript_paths(fisher_dir)
    transcripts = [__parse_transcript(id, p)
                   for id, p in tqdm(transcript_paths)]

    print("Feature extraction and embeddings...")
    do = partial(
        __do_feature_extract_and_embedding,
        fisher_dir=fisher_dir,
        tokenizer=tokenizer,
        vec=vec,
        embedding_out_dir=embedding_out_dir,
    )
    results = pqdm(transcripts, do, n_jobs=16)

    print("Writing to CSV...")
    results_all = list(itertools.chain(*results))
    df = pd.DataFrame(results_all)

    df[FEATURES_ZSCORE] = df.groupby(['id', 'speaker'])[FEATURES].transform(
        lambda x: (x-x.mean())/x.std())

    df = df.sort_values(['id', 'start'])
    train_ids, test_ids = train_test_split(
        df.id.unique(), random_state=9001, train_size=0.8)
    test_ids, val_ids = train_test_split(
        test_ids, test_size=0.5, random_state=9001)

    df[df.id.isin(train_ids)].to_csv(
        "transcript-train.csv",
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )
    df[df.id.isin(val_ids)].to_csv(
        "transcript-val.csv",
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )
    df[df.id.isin(test_ids)].to_csv(
        "transcript-test.csv",
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )
