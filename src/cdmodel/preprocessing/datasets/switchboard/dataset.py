import os
from collections import Counter, defaultdict
from os import path
from typing import Final

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from cdmodel.preprocessing.datasets.dataset import (
    ConversationStub,
    Dataset,
    DatasetPropertiesDict,
    Segment,
    Segmentation,
)
from cdmodel.preprocessing.datasets.switchboard.dialogue_acts import (
    load_dialogue_acts,
    load_terminals,
    pair_terminals_das,
)
from cdmodel.preprocessing.datasets.switchboard.switchboard_utils import (
    load_call_metadata,
    load_caller_metadata,
    pair_conversations_speakers,
)
from cdmodel.preprocessing.datasets.switchboard.transcript_processing import (
    process_word,
)


class SwitchboardDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        segmentation: str = "turn",
        n_jobs: int = 8,
        debug: bool = False,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            segmentation=segmentation,
            n_jobs=n_jobs,
            debug=debug,
        )

        self.call_metadata: Final[DataFrame] = load_call_metadata(self.dataset_dir)
        self.caller_metadata: Final[DataFrame] = load_caller_metadata(self.dataset_dir)

    def get_properties(self) -> DatasetPropertiesDict:
        return {"has_da": True}

    def apply_dialogue_acts(
        self, conversation_segments: dict[int, list[Segment]]
    ) -> tuple[dict[int, list[Segment]], Counter[str]]:
        # Dictionary for DA-annotated conversation segments
        conversations_out: defaultdict[int, list[Segment]] = defaultdict(lambda: [])

        # Counter for dialogue acts so we can see which are the most common
        da_counter: Counter[str] = Counter()

        for conversation_id, segments in tqdm(
            conversation_segments.items(), desc="Processing dialogue acts"
        ):
            # Load NXT-Switchboard dialogue acts and terminals
            try:
                das_a = load_dialogue_acts(conversation_id, "A", self.dataset_dir)
                das_b = load_dialogue_acts(conversation_id, "B", self.dataset_dir)

                terminals_a = load_terminals(conversation_id, "A", self.dataset_dir)
                terminals_b = load_terminals(conversation_id, "B", self.dataset_dir)
            except FileNotFoundError:
                # If loading fails, add the conversation segments to the output without
                # dialogue acts.
                #
                # Failure can occur if the conversation is not annotated by NXT-Switchboard,
                # or if the user running the preprocessing script does not have the NXT-
                # switchboard annotations.
                conversations_out[conversation_id] = segments
                continue

            # Pair the terminals and dialogue acts
            das_terminals_a = pair_terminals_das(terminals_a, das_a)
            das_terminals_b = pair_terminals_das(terminals_b, das_b)

            # Update the counter with dialogue acts from both sides of the conversation
            da_counter.update(da.nitetype for (_, da) in das_terminals_a.values())
            da_counter.update(da.nitetype for (_, da) in das_terminals_b.values())

            for segment in segments:
                # Identify the side of the conversational segment and get the appropriate
                # DA/terminal pairs
                if segment.side == "A":
                    conversation_side_das_terminals = das_terminals_a.values()
                else:
                    conversation_side_das_terminals = das_terminals_b.values()

                # Construct a set of all dialogue acts contained in the segment
                segment_da = set(
                    da.nitetype
                    for (terminal, da) in conversation_side_das_terminals
                    if not (
                        terminal.end < segment.start or terminal.start > segment.end
                    )
                )

                # Add the segment to the output conversation with new dialogue act annotations
                conversations_out[conversation_id].append(
                    Segment(
                        transcript=segment.transcript,
                        side=segment.side,
                        speaker_id=segment.speaker_id,
                        start=segment.start,
                        end=segment.end,
                        da=segment_da,
                    )
                )

        return dict(conversations_out), da_counter

    def get_conversations_with_min_speaker_repeat(self, min_repeat: int) -> list[int]:
        conversation_ids: list[int] = []

        call_counts = self.call_metadata.caller_no.value_counts()
        eligible_callers = set(call_counts.index)

        prev_eligible_caller_size = len(eligible_callers)

        # Here, we iteratively reconstruct df_call_pairs by removing callers with fewer than 3 examples.
        # We have to do this iteratively because an almost-rare caller (with only 3 instances) might converse
        # with a rare caller (2 or fewer instances) and is removed, bringing their total count down to just 2.
        # This iteration solves that problem and results in a dataset where we're guaranteed
        # there are at least 3 instances of each caller.
        while True:
            df_calls_a = self.call_metadata[
                (self.call_metadata.conversation_side == "A")
                & (self.call_metadata.caller_no.isin(eligible_callers))
            ].set_index("conversation_no")
            df_calls_b = self.call_metadata[
                (self.call_metadata.conversation_side == "B")
                & (self.call_metadata.caller_no.isin(eligible_callers))
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
            set(
                self.call_metadata[
                    self.call_metadata.caller_no.isin(eligible_callers)
                ].conversation_no
            )
        )

    def get_speaker_gender(self) -> dict[int, str]:
        speaker_gender: dict[int, str] = {}

        self.caller_metadata.sex = [
            "m" if x == "MALE" else "f" for x in self.caller_metadata.sex
        ]

        for _, row in self.caller_metadata.iterrows():
            speaker_gender[row.caller_no] = row.sex

        return speaker_gender

    def load_conversation_stubs(self) -> dict[int, list[ConversationStub]]:
        speaker_ids = pair_conversations_speakers(self.call_metadata)

        # A set containing all conversation IDs
        conversation_ids: set[int] = set()

        # A dictionary mapping conversation IDs and speaker ID to ts corresponding speaker file
        conversation_speaker: dict[tuple[int, str], ConversationStub] = {}

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
                conversation_file = ConversationStub(
                    path=path.join(root, filename),
                    side=side,
                    speaker_id=speaker_ids[(id, side)],
                    channel=0,  # All Switchboard files are mono
                )

                conversation_ids.add(id)
                conversation_speaker[(id, side)] = conversation_file

        # Next, we merge the speaker-specific conversation files into a single dictionary
        conversations: dict[int, list[ConversationStub]] = {}
        for id in conversation_ids:
            conversations[id] = [
                conversation_speaker[id, "A"],
                conversation_speaker[id, "B"],
            ]

        return conversations

    def filter_conversation_stubs(
        self, conversation_stubs: dict[int, list[ConversationStub]]
    ) -> dict[int, list[ConversationStub]]:
        # If in debug mode, output a small subset of conversations to make
        # preprocessing go faster.
        if self.debug:
            return {4345: conversation_stubs[4345], 3029: conversation_stubs[4785]}

        return conversation_stubs

    def load_conversation_segments(
        self, conversation_stubs: dict[int, list[ConversationStub]]
    ) -> dict[int, list[Segment]]:
        if self.segmentation == Segmentation.IPU:
            raise Exception("IPU segmentation currently not available!")

        segments: dict[int, list[Segment]] = {}

        for id, stubs in tqdm(conversation_stubs.items(), desc="Loading transcripts"):
            ipu_segmentation: list[Segment] = []

            for stub in stubs:
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
                    f"sw{id_str}{stub.side}-ms98-a-word.text",
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
                                side=stub.side,
                                speaker_id=stub.speaker_id,
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
