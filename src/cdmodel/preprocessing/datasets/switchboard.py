import os
import re
from os import path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from cdmodel.preprocessing.datasets.dataset import (
    ConversationFile,
    Dataset,
    Segment,
    Segmentation,
)

## PROCESSING CODE FROM https://github.com/emeinhardt/switchboard-lm/
interrupted_word_pattern = ".*-$"
resumed_word_pattern = "^-.*"


def isInterrupted(wordform):
    return re.match(interrupted_word_pattern, wordform) is not None


def isResumed(wordform):
    return re.match(resumed_word_pattern, wordform) is not None


def isBroken(wordform):
    return isInterrupted(wordform) or isResumed(wordform)


def hasBrokenWords(speech):
    speech_word_seq = speech.split(" ")
    broken_words = list(filter(isBroken, speech_word_seq))
    return len(broken_words) > 0


def remove_broken_words(speech, insertUnk=False):
    if insertUnk:
        replacement = unk
    else:
        replacement = ""

    speech_word_seq = speech.split(" ")
    speech_word_seq_fixed = " ".join(replace(speech_word_seq, isBroken, (replacement,)))
    speech_fixed = " ".join([w for w in speech_word_seq_fixed.split(" ") if len(w) > 0])
    return speech_fixed


def hasCurlyBraces(wordform):
    return "{" in wordform or "}" in wordform


def isCurlyBraced(wordform):
    if len(wordform) == 0:
        return False
    return wordform[0] == "{" and wordform[-1] == "}"


def removeCurlyBraces(wordform):
    if not isCurlyBraced(wordform):
        return wordform
    return wordform[1:-1]


def remove_curly_braces(speech):
    speech_word_seq = speech.split(" ")
    speech_word_seq_fixed = " ".join(list(map(removeCurlyBraces, speech_word_seq)))
    speech_fixed = " ".join([w for w in speech_word_seq_fixed.split(" ") if len(w) > 0])
    return speech_fixed


def hasUnderscore(wordform):
    return "_" in wordform


def fixUnderscore(wordform):
    if not hasUnderscore(wordform):
        return wordform
    fixed = wordform.replace("_1", "")
    return fixed


def fix_underscores(speech):
    speech_word_seq = speech.split(" ")
    speech_word_seq_filtered = [
        w for w in speech_word_seq if w != "<b_aside>" and w != "<e_aside>"
    ]
    speech_word_seq_fixed = " ".join(list(map(fixUnderscore, speech_word_seq_filtered)))
    speech_fixed = " ".join([w for w in speech_word_seq_fixed.split(" ") if len(w) > 0])
    return speech_fixed


def lowercase(speech):
    return speech.lower()


def removeNonSpeech(word):
    if len(word) > 0 and word[0] != "[" and word[-1] != "]":
        return word
    else:
        return None


def process_word(word):
    word = lowercase(word)
    word = fixUnderscore(word)
    word = remove_curly_braces(word)

    if isBroken(word):
        return None

    word = removeNonSpeech(word)
    return word


def _get_conversation_speaker_ids(dataset_dir: str) -> dict[tuple[int, str], int]:
    speaker_ids: dict[tuple[int, str], int] = {}

    df_calls = pd.read_csv(
        path.join(dataset_dir, "tables", "call_con_tab.csv"), header=None
    )
    df_calls.columns = pd.Index(
        [
            "conversation_no",
            "conversation_side",
            "caller_no",
            "phone_number",
            "length",
            "ivi_no",
            "remarks",
            "active",
        ]
    )

    for _, row in df_calls.iterrows():
        speaker_ids[(row.conversation_no, row.conversation_side)] = row.caller_no

    return speaker_ids


class SwitchboardDataset(Dataset):
    def __init__(self, dataset_dir: str, segmentation: str = "turn", n_jobs: int = 8):
        super().__init__(
            dataset_dir=dataset_dir,
            # sr=8000,
            # f_max=None,
            segmentation=segmentation,
            n_jobs=n_jobs,
        )

    def get_speaker_gender(self) -> dict[int, str]:
        speaker_gender: dict[int, str] = {}

        df_callers = pd.read_csv(
            path.join(self.dataset_dir, "tables", "caller_tab.csv"), header=None
        )
        df_callers.columns = pd.Index(
            [
                "caller_no",
                "pin",
                "target",
                "sex",
                "birth_year",
                "dialect_area",
                "education",
                "ti",
                "payment_type",
                "amt_pd",
                "con",
                "remarks",
                "calls_deleted",
                "speaker_partition",
            ]
        )

        df_callers.sex = ["m" if x == "MALE" else "f" for x in df_callers.sex]

        for _, row in df_callers.iterrows():
            speaker_gender[row.caller_no] = row.sex

        return speaker_gender

    def get_all_conversations(self) -> dict[int, list[ConversationFile]]:
        speaker_ids = _get_conversation_speaker_ids(self.dataset_dir)

        # A set containing all conversation IDs
        conversation_ids: set[int] = set()

        # A dictionary mapping conversation IDs and speaker ID to ts corresponding speaker file
        conversation_speaker: dict[tuple[int, str], ConversationFile] = {}

        # First, we collect all the conversation files
        dirs = list(os.walk(path.join(self.dataset_dir, "audio")))
        for root, dirs, files in tqdm(dirs, desc="Loading conversations"):
            for filename in files:
                if ".wav" not in filename:
                    continue

                # Switchboard files are of the format 'sw<id>.<speaker_id>.wav'
                # Ensure it is in the correct format before continuing
                filename_split = filename.split(".")
                if len(filename_split) != 3:
                    continue
                id_str, side, _ = filename_split

                # Extract the integer ID from the ID string
                # Switchboard IDs are of the format 'sw<id>'abs
                id: int = int(id_str[2:])

                # At this point, we successfully found a Switchboard audio file.
                # Put together the data for the output dictionary and add it:
                conversation_file = ConversationFile(
                    path=path.join(root, filename),
                    side=side,
                    speaker_id=speaker_ids[(id, side)],
                    channel=0,  # All Switchboard files are mono
                )

                conversation_ids.add(id)
                conversation_speaker[(id, side)] = conversation_file

        # Next, we merge the speaker-specific conversation files into a single dictionary
        conversations: dict[int, list[ConversationFile]] = {}
        for id in conversation_ids:
            conversations[id] = [
                conversation_speaker[id, "A"],
                conversation_speaker[id, "B"],
            ]

        return conversations

    def filter_conversations(
        self, conversations: dict[int, list[ConversationFile]]
    ) -> dict[int, list[ConversationFile]]:
        # # DEBUG - REMOVE
        # out = {}
        # out[4345] = conversations[4345]
        # out[3029] = conversations[4785]
        # return out
        return conversations

    def get_segmented_transcripts(
        self, conversations: dict[int, list[ConversationFile]]
    ) -> dict[int, list[Segment]]:
        if self.segmentation == Segmentation.IPU:
            raise Exception("IPU segmentation currently not available!")

        segments: dict[int, list[Segment]] = {}

        for id, conversation_files in tqdm(
            conversations.items(), desc="Loading transcripts"
        ):
            ipu_segmentation: list[Segment] = []

            for speaker_conversation_file in conversation_files:
                id_str = str(id)

                # swb_ms98 transcriptions have the following directory structure:
                #
                #   <dataset dir>/swb_ms98_transcriptions/<first 2 numbers of ID>/<id>/
                #
                # Inside the directory path shown above are the transcript and word-level files
                transcript_path = path.join(
                    self.dataset_dir,
                    "swb_ms98_transcriptions",
                    id_str[:2],
                    id_str,
                    f"sw{id_str}{speaker_conversation_file.side}-ms98-a-word.text",
                )

                # Transcripts have the following structure:
                #
                #   <id> <start time> <end time> <word>
                speaker_segmentation: list[Segment] = []

                with open(transcript_path) as infile:
                    for line_str in infile:
                        line = line_str.strip().split()
                        start = float(line[1])
                        end = float(line[2])
                        word = " ".join(line[3:])

                        word = process_word(word)
                        if word is None:
                            continue

                        speaker_segmentation.append(
                            Segment(
                                transcript=word,
                                side=speaker_conversation_file.side,
                                speaker_id=speaker_conversation_file.speaker_id,
                                start=start,
                                end=end,
                            )
                        )

                speaker_ipu_segmentation = _words_to_ipu_single_speaker(
                    speaker_segmentation
                )
                ipu_segmentation += speaker_ipu_segmentation

            # Sort the word segmentation by start time
            ipu_segmentation.sort(key=lambda x: x.start)
            segments[id] = _segments_to_turn(ipu_segmentation)

        return segments


# Assuming a single speaker, convert a list of words to a list of IPUs
def _words_to_ipu_single_speaker(word_segmentation: list[Segment]) -> list[Segment]:
    first = True
    prev_end: float = 0.0

    ipu_words: list[Segment] = []
    ipu_segmentation: list[Segment] = []

    for word in word_segmentation:
        if not first and word.start - prev_end >= 0.05:
            ipu_segmentation.append(
                Segment(
                    transcript=" ".join([x.transcript for x in ipu_words]),
                    side=ipu_words[0].side,
                    speaker_id=ipu_words[0].speaker_id,
                    start=ipu_words[0].start,
                    end=ipu_words[-1].end,
                )
            )

            ipu_words = []

        prev_end = word.end
        ipu_words.append(word)
        first = False

    if len(ipu_words) > 0:
        ipu_segmentation.append(
            Segment(
                transcript=" ".join([x.transcript for x in ipu_words]),
                side=ipu_words[0].side,
                speaker_id=ipu_words[0].speaker_id,
                start=ipu_words[0].start,
                end=ipu_words[-1].end,
            )
        )

    return ipu_segmentation


# Assuming mixed speakers, convert a list of segments to a list of segment-maximal turns
def _segments_to_turn(word_segmentation: list[Segment]) -> list[Segment]:
    prev_side: str = word_segmentation[0].side

    turn_words: list[Segment] = []
    turn_segmentation: list[Segment] = []

    for word in word_segmentation:
        if word.side != prev_side:
            turn_segmentation.append(
                Segment(
                    transcript=" ".join([x.transcript for x in turn_words]),
                    side=prev_side,
                    speaker_id=turn_words[0].speaker_id,
                    start=turn_words[0].start,
                    end=turn_words[-1].end,
                )
            )

            turn_words = []

        prev_side = word.side
        turn_words.append(word)

    if len(turn_words) > 0:
        turn_segmentation.append(
            Segment(
                transcript=" ".join([x.transcript for x in turn_words]),
                side=turn_words[0].side,
                speaker_id=turn_words[0].speaker_id,
                start=turn_words[0].start,
                end=turn_words[-1].end,
            )
        )

    return turn_segmentation
