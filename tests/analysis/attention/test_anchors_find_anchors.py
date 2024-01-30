import pytest
import torch

from cdmodel.analysis.attention.anchors import Anchor, AnchorTimestep, find_anchors


@pytest.fixture
def att_scores_end_1():
    return [
        torch.tensor([1.0]),
        torch.tensor([0.1, 0.9]),
        torch.tensor([0.05, 0.9, 0.05]),
        torch.tensor([0.1, 0.1, 0.1, 0.7]),
        torch.tensor([0.1, 0.1, 0.1, 0.6, 0.1]),
        torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
    ]


@pytest.fixture
def att_scores_empty_1(att_scores_end_1):
    return [torch.tensor([])] + att_scores_end_1


@pytest.fixture
def anchors_end_1(att_scores_end_1):
    return [
        Anchor(
            highest_scoring_segment=0,
            timesteps=[
                AnchorTimestep(predict_timestep=0, scores=att_scores_end_1[0]),
            ],
        ),
        Anchor(
            highest_scoring_segment=1,
            timesteps=[
                AnchorTimestep(predict_timestep=1, scores=att_scores_end_1[1]),
                AnchorTimestep(predict_timestep=2, scores=att_scores_end_1[2]),
            ],
        ),
        Anchor(
            highest_scoring_segment=3,
            timesteps=[
                AnchorTimestep(predict_timestep=3, scores=att_scores_end_1[3]),
                AnchorTimestep(predict_timestep=4, scores=att_scores_end_1[4]),
            ],
        ),
        Anchor(
            highest_scoring_segment=5,
            timesteps=[
                AnchorTimestep(predict_timestep=5, scores=att_scores_end_1[5]),
            ],
        ),
    ]


@pytest.fixture
def anchors_empty_1(att_scores_end_1):
    return [
        None,
        Anchor(
            highest_scoring_segment=0,
            timesteps=[
                AnchorTimestep(predict_timestep=1, scores=att_scores_end_1[0]),
            ],
        ),
        Anchor(
            highest_scoring_segment=1,
            timesteps=[
                AnchorTimestep(predict_timestep=2, scores=att_scores_end_1[1]),
                AnchorTimestep(predict_timestep=3, scores=att_scores_end_1[2]),
            ],
        ),
        Anchor(
            highest_scoring_segment=3,
            timesteps=[
                AnchorTimestep(predict_timestep=4, scores=att_scores_end_1[3]),
                AnchorTimestep(predict_timestep=5, scores=att_scores_end_1[4]),
            ],
        ),
        Anchor(
            highest_scoring_segment=5,
            timesteps=[
                AnchorTimestep(predict_timestep=6, scores=att_scores_end_1[5]),
            ],
        ),
    ]


@pytest.fixture
def att_scores_end_2():
    return [
        torch.tensor([1.0]),
        torch.tensor([0.1, 0.9]),
        torch.tensor([0.05, 0.9, 0.05]),
        torch.tensor([0.1, 0.1, 0.1, 0.7]),
        torch.tensor([0.1, 0.1, 0.1, 0.6, 0.1]),
        torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
        torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1]),
    ]


@pytest.fixture
def anchors_end_2(att_scores_end_2):
    return [
        Anchor(
            highest_scoring_segment=0,
            timesteps=[
                AnchorTimestep(predict_timestep=0, scores=att_scores_end_2[0]),
            ],
        ),
        Anchor(
            highest_scoring_segment=1,
            timesteps=[
                AnchorTimestep(predict_timestep=1, scores=att_scores_end_2[1]),
                AnchorTimestep(predict_timestep=2, scores=att_scores_end_2[2]),
            ],
        ),
        Anchor(
            highest_scoring_segment=3,
            timesteps=[
                AnchorTimestep(predict_timestep=3, scores=att_scores_end_2[3]),
                AnchorTimestep(predict_timestep=4, scores=att_scores_end_2[4]),
            ],
        ),
        Anchor(
            highest_scoring_segment=5,
            timesteps=[
                AnchorTimestep(predict_timestep=5, scores=att_scores_end_2[5]),
                AnchorTimestep(predict_timestep=6, scores=att_scores_end_2[6]),
            ],
        ),
    ]


def test_find_anchors_end_1(att_scores_end_1, anchors_end_1):
    anchors_out = find_anchors(scores_all=att_scores_end_1)

    assert anchors_out == anchors_end_1


def test_find_anchors_end_2(att_scores_end_2, anchors_end_2):
    anchors_out = find_anchors(scores_all=att_scores_end_2)

    assert anchors_out == anchors_end_2


def test_find_anchors_empty_1(att_scores_empty_1, anchors_empty_1):
    anchors_out = find_anchors(scores_all=att_scores_empty_1)

    assert anchors_out == anchors_empty_1
