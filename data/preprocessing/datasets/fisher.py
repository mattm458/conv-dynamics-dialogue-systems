import os
import re
from os import path
import torchtext

from unidecode import unidecode


def get_tokens():
    return []


def __get_transcript_paths(fisher_dir):
    transcript_paths = []

    for dir in ["fe_03_p1_tran", "fe_03_p2_tran"]:
        trans_path = path.join(fisher_dir, dir, "data", "trans")
        for filepath, _, files in os.walk(trans_path):
            for file in files:
                if not ("fe_" in file and ".txt" in file):
                    continue
                transcript_paths.append(path.join(filepath, file))

    return transcript_paths


__parse_transcript_regex = re.compile("(\d+\.\d+) (\d+\.\d+) ([AB]): (.+)")


def __clean_text(text):
    text = text.lower()
    text = unidecode(text)
    text = text.replace("((", "").replace("))", "")
    text=text.replace('[laughter]','')
    text=text.replace('[mn]','')
    text=text.replace('[cough]','')
    text=text.replace('[noise]','')
    text=text.replace('[sigh]','')
    text=text.replace('[lipsmack]','')
    text=text.replace('[laugh]','')
    text=text.replace('[sneeze]','')
    text=text.replace('[breath]','')
    text=text.replace('[pause]','')
    text=text.replace('[[skip]]','')






    if '[' in text:
        print(text)

    return text


def __parse_transcript(transcript_path):
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


def run(fisher_dir):
    vec = torchtext.vocab.GloVe(name='6B', dim=50)

    transcript_paths = __get_transcript_paths(fisher_dir)
    for transcript_path in transcript_paths:
        __parse_transcript(transcript_path)
