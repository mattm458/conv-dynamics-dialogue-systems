from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import NamedTuple, Optional

import parselmouth
import torch

# import torchaudio
from pqdm.processes import pqdm

# from speech_utils.audio.transforms.mel_spectrogram import TacotronMelSpectrogram
from speech_utils.preprocessing.feature_extraction import extract_features


class ConversationFile(NamedTuple):
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
def _process_trascript(
    conversation_id: int,
    transcript: list[Segment],
    conversations: dict[int, list[ConversationFile]],
    speaker_gender: Optional[dict] = None,
) -> list[dict]:
    conversation_files = conversations[conversation_id]

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
    speaker_ids: set[int] = set(x.speaker_id for x in transcript)

    # Get all of the dialogue acts
    da_all: set[str] = set()
    has_da = False
    for segment in transcript:
        if segment.da is not None:
            has_da = True
            da_all = da_all.union(segment.da)

    for segment in transcript:
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
        if has_da:
            da_dict = dict(
                [
                    (
                        f"da_{da}",
                        1 if segment.da is not None and da in segment.da else 0,
                    )
                    for da in da_all
                ]
            )
            features |= da_dict

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
        # sr: int,
        # f_max=Optional[float],
        segmentation: str = "turn",
        n_jobs: int = 8,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        # self.sr = sr
        self.segmentation: Segmentation = Segmentation(segmentation)
        self.n_jobs = n_jobs
        # self.mel_spectrogram = TacotronMelSpectrogram(sample_rate=sr, f_max=f_max)

    def extract_features(self) -> list[dict]:
        # Retrieve and filter all the conversations we want to use
        conversations = self.get_all_conversations()
        conversations = self.filter_conversations(conversations=conversations)

        # Get the transcripts according to the segmentation method we initialized with the dataset class
        transcripts = self.get_segmented_transcripts(conversations=conversations)

        # If dialogue acts are stored separately, extract them here
        transcripts = self.apply_dialogue_acts(transcripts=transcripts)

        # first_spectrograms = _get_all_first_spectrograms(
        #     transcripts,
        #     conversations,
        #     expected_sr=self.sr,
        #     mel_spectrogram=self.mel_spectrogram,
        # )

        # If speaker gender is available, get it for use in preprocessing
        speaker_gender = self.get_speaker_gender()

        # Preprocess using pqdm to run in parallel
        processed = pqdm(
            transcripts.items(),
            partial(
                _process_trascript,
                speaker_gender=speaker_gender,
                conversations=conversations,
            ),
            n_jobs=self.n_jobs,
            argument_type="args",
            exception_behaviour="immediate",
        )

        # Flatten the returned list of extracted features
        processed = [item for sublist in processed for item in sublist]
        return processed  # , first_spectrograms

    @abstractmethod
    def get_all_conversations(self) -> dict[int, list[ConversationFile]]:
        pass

    @abstractmethod
    def filter_conversations(
        self, conversations: dict[int, list[ConversationFile]]
    ) -> dict[int, list[ConversationFile]]:
        pass

    @abstractmethod
    def get_segmented_transcripts(
        self, conversations: dict[int, list[ConversationFile]]
    ) -> dict[int, list[Segment]]:
        pass

    @abstractmethod
    def apply_dialogue_acts(
        self, transcripts: dict[int, list[Segment]]
    ) -> dict[int, list[Segment]]:
        pass

    def get_speaker_gender(self) -> Optional[dict[int, str]]:
        return None
