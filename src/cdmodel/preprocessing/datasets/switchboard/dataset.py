import os
import re
from os import path
from typing import Optional

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from cdmodel.preprocessing.datasets.dataset import (
    ConversationFile,
    Dataset,
    Segment,
    Segmentation,
)
from cdmodel.preprocessing.datasets.switchboard.da import (
    expand_terminals_da,
    get_dialogue_acts,
    get_terminals,
)
from cdmodel.preprocessing.datasets.switchboard.transcript_processing import (
    process_word,
)


def _get_call_con(dataset_dir: str) -> DataFrame:
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

    return df_calls


def _get_conversation_speaker_ids(dataset_dir: str) -> dict[tuple[int, str], int]:
    speaker_ids: dict[tuple[int, str], int] = {}

    df_calls = _get_call_con(dataset_dir=dataset_dir)

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

    def apply_dialogue_acts(
        self, transcripts: dict[int, list[Segment]]
    ) -> dict[int, list[Segment]]:
        transcripts_out: dict[int, list[Segment]] = {}

        for id, segments in transcripts.items():
            segments_out: list[Segment] = []

            try:
                da_a = get_dialogue_acts(id, "A", self.dataset_dir)
                da_b = get_dialogue_acts(id, "B", self.dataset_dir)

                terminals_a = get_terminals(id, "A", self.dataset_dir)
                terminals_b = get_terminals(id, "B", self.dataset_dir)
            except Exception as e:
                continue

            expand_terminals_da(da_a, terminals_a)
            expand_terminals_da(da_b, terminals_b)

            for segment in segments:
                if segment.side == "A":
                    side_terminals = terminals_a.values()
                else:
                    side_terminals = terminals_b.values()

                segment_da = set(
                    x["nitetype"]
                    for x in side_terminals
                    if not ((x["end"] < segment.start) or (x["start"] > segment.end))
                )

                segments_out.append(
                    Segment(
                        transcript=segment.transcript,
                        side=segment.side,
                        speaker_id=segment.speaker_id,
                        start=segment.start,
                        end=segment.end,
                        da=segment_da,
                    )
                )

                transcripts_out[id] = segments_out

        return transcripts_out

    def get_conversations_with_min_speaker_repeat(self, min_repeat: int) -> list[int]:
        conversation_ids: list[int] = []

        df_calls = _get_call_con(self.dataset_dir)
        call_counts = df_calls.caller_no.value_counts()
        eligible_callers = set(call_counts.index)

        prev_eligible_caller_size = len(eligible_callers)

        # Here, we iteratively reconstruct df_call_pairs by removing callers with fewer than 3 examples.
        # We have to do this iteratively because an almost-rare caller (with only 3 instances) might converse
        # with a rare caller (2 or fewer instances) and is removed, bringing their total count down to just 2.
        # This iteration solves that problem and results in a dataset where we're guaranteed
        # there are at least 3 instances of each caller.
        while True:
            df_calls_a = df_calls[
                (df_calls.conversation_side == "A")
                & (df_calls.caller_no.isin(eligible_callers))
            ].set_index("conversation_no")
            df_calls_b = df_calls[
                (df_calls.conversation_side == "B")
                & (df_calls.caller_no.isin(eligible_callers))
            ].set_index("conversation_no")

            df_call_pairs = df_calls_a.join(
                df_calls_b, lsuffix="_a", rsuffix="_b", how="inner"
            )

            call_counts = pd.concat(
                [df_call_pairs.caller_no_a, df_call_pairs.caller_no_b]
            ).value_counts()
            call_counts = call_counts[call_counts >= 3]
            eligible_callers = set(call_counts.index)

            if len(eligible_callers) == prev_eligible_caller_size:
                break

            prev_eligible_caller_size = len(eligible_callers)

        return list(
            set(df_calls[df_calls.caller_no.isin(eligible_callers)].conversation_no)
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
        dirs_walk = os.walk(path.join(self.dataset_dir, "audio"))
        for root, _, files in tqdm(dirs_walk, desc="Loading conversations"):
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
        # DEBUG - REMOVE
        out = {}
        out[4345] = conversations[4345]
        out[3029] = conversations[4785]
        return out
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
