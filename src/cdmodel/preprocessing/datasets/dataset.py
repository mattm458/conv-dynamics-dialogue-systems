from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from functools import partial
from typing import NamedTuple, Optional

import parselmouth
import torch

# import torchaudio
from pqdm.processes import pqdm

# from speech_utils.audio.transforms.mel_spectrogram import TacotronMelSpectrogram
from speech_utils.preprocessing.feature_extraction import extract_features

from cdmodel.preprocessing.datasets.da import da_consolidate_category, da_vote


class ConversationStub(NamedTuple):
    path: str
    side: str
    speaker_id: int
    channel: int


class Segment(NamedTuple):
    transcript: str
    side: str
    speaker_id: int
    start: float
    end: float
    da: Optional[set[str]] = None


class Segmentation(Enum):
    TURN = "turn"
    IPU = "ipu"


# This function performs feature extraction on all segments within a conversation.
def _process_transcript(
    conversation_id: int,
    conversation_segments: list[Segment],
    conversation_stubs: dict[int, list[ConversationStub]],
    speaker_gender: Optional[dict] = None,
    da_precedence: Optional[list[str]] = None,
) -> list[dict]:
    conversation_files = conversation_stubs[conversation_id]

    # Open the conversation files
    conversation_audio: dict[str, parselmouth.Sound] = {}
    for conversation_file in conversation_files:
        conversation_audio[conversation_file.side] = parselmouth.Sound(
            conversation_file.path
        )

    # Final output list
    output: list[dict] = []

    # Sets containing the conversation sides and speaker IDs for
    # determining partner at each segment
    sides: set[str] = set(conversation_audio.keys())
    speaker_ids: set[int] = set(x.speaker_id for x in conversation_segments)

    for segment in conversation_segments:
        features = segment._asdict()
        extracted_features = extract_features(
            transcript=segment.transcript,
            sound=conversation_audio[segment.side],
            start=segment.start,
            end=segment.end,
        )

        if extracted_features is None:
            continue

        # Assemble the final feature dictionary
        features = features | extracted_features
        features["id"] = conversation_id
        features["side_partner"] = list(sides - {segment.side})[0]
        features["speaker_id_partner"] = list(speaker_ids - {segment.speaker_id})[0]

        if speaker_gender is not None:
            features["gender"] = speaker_gender[segment.speaker_id]
            features["gender_partner"] = speaker_gender[features["speaker_id_partner"]]

        # Assemble all DAs into the final feature object
        if da_precedence is not None:
            if segment.da is not None and len(segment.da) > 0:
                da_consolidated = da_vote(segment.da, da_precedence)
                features["da_consolidated"] = da_consolidated
                features["da_category"] = da_consolidate_category(da_consolidated)

            da_dict = dict(
                [
                    (
                        f"da_{da}",
                        1 if segment.da is not None and da in segment.da else 0,
                    )
                    for da in da_precedence
                ]
            )
            features |= da_dict

        del features["da"]

        output.append(features)

    return output


from matplotlib import pyplot as plt

# Compute the Mel spectrograms of conversation-initial segments from each speaker
# def _get_first_spectrograms(
#     transcript: list[Segment],
#     conversation: list[ConversationFile],
#     expected_sr: int,
#     mel_spectrogram: torchaudio.transforms.MelSpectrogram,
# ) -> dict[int, torch.Tensor]:
#     conversation_dict: dict[int, ConversationFile] = {}
#     for x in conversation:
#         conversation_dict[x.speaker_id] = x

#     output: dict[int, torch.Tensor] = {}

#     for t in transcript:
#         if t.speaker_id not in output:
#             path = conversation_dict[t.speaker_id].path
#             audio, sr = torchaudio.load(path)
#             if sr != expected_sr:
#                 raise Exception(
#                     f"Audio file {path} does not have the dataset-specified sample rate of {expected_sr}, has {sr} instead!"
#                 )

#             audio = audio[0, round(t.start * sr) : round(t.end * sr)]
#             output[t.speaker_id] = mel_spectrogram(audio)

#         if len(output) == 2:
#             break

#     return output


# Compute the Mel spectrogram of conversation-initial segments from each speaker
# in each conversation
# def _get_all_first_spectrograms(
#     transcripts: dict[int, list[Segment]],
#     conversations: dict[int, list[ConversationFile]],
#     expected_sr: int,
#     mel_spectrogram: torchaudio.transforms.MelSpectrogram,
# ) -> dict[int, dict[int, torch.Tensor]]:
#     output: dict[int, dict[int, torch.Tensor]] = {}

#     for id, transcript in transcripts.items():
#         output[id] = _get_first_spectrograms(
#             transcript, conversations[id], expected_sr, mel_spectrogram
#         )

#     return output


class Dataset(ABC):
    def __init__(
        self,
        dataset_dir: str,
        segmentation: str = "turn",
        n_jobs: int = 8,
        debug: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.segmentation: Segmentation = Segmentation(segmentation)
        self.n_jobs = n_jobs
        self.debug = debug

    def extract_features(self) -> list[dict]:
        """
        Perform feature extraction on the dataset.

        Returns
        -------
        list[dict]
            A list of dictionaries. Each dictionary represents a segment of the
            conversation, and contains features extracted from that segment.
        """
        # Get conversation stub data
        conversation_stubs = self.load_conversation_stubs()

        # Filter any unwanted conversation stubs
        conversation_stubs = self.filter_conversation_stubs(
            conversation_stubs=conversation_stubs
        )

        # Load and segment transcripts from the conversation stubs
        conversation_segments = self.load_conversation_segments(
            conversation_stubs=conversation_stubs
        )

        # If dialogue acts are stored separately, extract them here
        conversation_segments, da_counts = self.apply_dialogue_acts(
            conversation_segments=conversation_segments
        )
        da_precedence = [da for (da, _) in da_counts.most_common()]

        # Preprocess using pqdm to run in parallel
        processed = pqdm(
            conversation_segments.items(),
            partial(
                _process_transcript,
                speaker_gender=self.get_speaker_gender(),
                conversation_stubs=conversation_stubs,
                da_precedence=da_precedence,
            ),
            n_jobs=self.n_jobs,
            argument_type="args",
            exception_behaviour="immediate",
        )

        # Flatten the returned list of extracted features
        return [item for sublist in processed for item in sublist]

    @abstractmethod
    def load_conversation_stubs(self) -> dict[int, list[ConversationStub]]:
        pass

    def filter_conversation_stubs(
        self, conversation_stubs: dict[int, list[ConversationStub]]
    ) -> dict[int, list[ConversationStub]]:
        return conversation_stubs

    @abstractmethod
    def load_conversation_segments(
        self, conversation_stubs: dict[int, list[ConversationStub]]
    ) -> dict[int, list[Segment]]:
        pass

    def apply_dialogue_acts(
        self, conversation_segments: dict[int, list[Segment]]
    ) -> tuple[dict[int, list[Segment]], Counter[str]]:
        """
        Apply dialogue act annotations to extracted conversation data, if available.

        Parameters
        ----------
        conversations : dict[int, list[Segment]]
            A dictionary mapping conversation ID to a list of Segment objects

        Returns
        -------
        tuple[dict[int, list[Segment]], Counter[str]]
            A tuple containing a new dictionary mapping conversation ID to a list of Segment objects.
            Segments will be annotated with dialogue acts where available.

            Additionally, the tuple contains a Counter object containing the number of times each
            dialogue act appears in the conversations.
        """
        return conversation_segments, Counter()

    def get_speaker_gender(self) -> dict[int, str]:
        """
        Return a dictionary mapping speaker ID to the speaker's gender, for speakers where
        this information is available. If gender information is not available for a speaker,
        their ID and gender will not be in the returned dictionary.

        Returns
        -------
        Optional[dict[int, str]]
            A dictionary mapping speaker ID to gender.
        """
        return {}
