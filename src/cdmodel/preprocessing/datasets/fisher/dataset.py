import os
import re
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


class FisherDataset(Dataset):
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

        self.filelist_df: Final[DataFrame] = pd.concat(
            [
                pd.read_csv(
                    path.join(self.dataset_dir, "meta", "fe_03_p1_calldata.tbl")
                ),
                pd.read_csv(
                    path.join(self.dataset_dir, "meta", "fe_03_p2_calldata.tbl")
                ),
            ]
        )

        self.conv_speakers: dict[int, dict[str, int]] = {}
        for _, row in self.filelist_df.iterrows():
            self.conv_speakers[row.CALL_ID] = {"A": row.APIN, "B": row.BPIN}

    def get_properties(self) -> DatasetPropertiesDict:
        return {"has_da": False}

    def load_conversation_stubs(self) -> dict[int, list[ConversationStub]]:
        output: Final[dict[int, list[ConversationStub]]] = {}

        for _, row in self.filelist_df.iterrows():
            wav_path: str = path.join(
                self.dataset_dir, "wav", f"fe_03_{str(row.CALL_ID).zfill(5)}.wav"
            )

            output[row.CALL_ID] = [
                ConversationStub(
                    path=wav_path, side="A", speaker_id=row.APIN, channel=0
                ),
                ConversationStub(
                    path=wav_path, side="B", speaker_id=row.BPIN, channel=1
                ),
            ]

        return output

    def load_conversation_segments(
        self, conversation_stubs: dict[int, list[ConversationStub]]
    ) -> dict[int, list[Segment]]:
        line_regex = re.compile(r"^(\d*\.?\d*) (\d*\.?\d*) ([AB]): (.*)$")
        noise_regex = re.compile(r"(\[[a-z]*\])")
        # incomplete_word_regex = re.compile(r"[a-z]+-|-[a-z]+")
        # incomplete_word_regex = re.compile(r"[a-z]+-")
        # parenthesis_regex = re.compile(r"\(\( [a-z ]+ \)\)")

        space_regex = re.compile(r" +")

        output: Final[dict[int, list[Segment]]] = {}

        for id, _ in tqdm(conversation_stubs.items(), desc="Loading transcripts"):
            id_str = str(id).zfill(5)

            p: int = 1 if id <= 5850 else 2

            transcript_path = path.join(
                self.dataset_dir,
                f"fe_03_p{p}_tran",
                "data",
                "trans",
                id_str[:3],
                f"fe_03_{id_str}.txt",
            )

            speaker_segmentation: list[Segment] = []

            with open(transcript_path) as infile:
                for line_raw in infile:
                    line = line_raw.strip()

                    # Remove empty lines
                    if len(line) == 0:
                        continue

                    match = line_regex.match(line)
                    if match is None:
                        continue

                    start, end, speaker, text = match.groups()

                    text = text.lower()
                    text = text.replace("(( ))", "")
                    text = text.replace("(( ", "")
                    text = text.replace(" ))", "")

                    text = noise_regex.sub("", text)
                    # text = incomplete_word_regex.sub("", text)
                    text = space_regex.sub(" ", text)
                    # text = parenthesis_regex.sub("", text)
                    text = text.strip()

                    if len(text) == 0:
                        continue

                    speaker_segmentation.append(
                        Segment(
                            transcript=text,
                            side=speaker,
                            speaker_id=self.conv_speakers[id][speaker],
                            start=float(start),
                            end=float(end),
                            da=None,
                        )
                    )

            output[id] = speaker_segmentation

        return output

    def filter_conversation_stubs(
        self, conversation_stubs: dict[int, list[ConversationStub]]
    ) -> dict[int, list[ConversationStub]]:
        if self.debug:
            return {
                1: conversation_stubs[1],
                2: conversation_stubs[2],
                3: conversation_stubs[3],
                4: conversation_stubs[4],
                5: conversation_stubs[5],
                6: conversation_stubs[6],
                7: conversation_stubs[7],
                8: conversation_stubs[8],
                9: conversation_stubs[9],
                10: conversation_stubs[10],
            }
        else:
            return conversation_stubs
